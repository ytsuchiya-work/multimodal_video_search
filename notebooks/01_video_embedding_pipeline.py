# Databricks notebook source
# MAGIC %md
# MAGIC # 動画Embeddingパイプライン
# MAGIC
# MAGIC Cosmos-Embed1-448pを使って、YouTubeからダウンロードした動画のembeddingを生成し、Delta Tableに保存する。
# MAGIC
# MAGIC **実行要件**: GPU Cluster (A10G / T4 / A100)

# COMMAND ----------

# MAGIC %pip install decord pillow einops
# MAGIC # torch/transformers/accelerate/safetensors/huggingface_hub are pre-installed in GPU ML runtime 15.4

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import io
import numpy as np
from datetime import datetime

from decord import VideoReader, cpu
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    ArrayType, FloatType, TimestampType
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.video_embeddings"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/thumbnails"

SEGMENT_DURATION = 30  # seconds
FRAMES_PER_SEGMENT = 8

DOWNLOAD_DIR = "/tmp/videos"
THUMBNAIL_DIR = "/tmp/thumbnails"

print(f"Table: {TABLE_NAME}")
print(f"Volume: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## カタログ・スキーマ・Volume作成

# COMMAND ----------

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
except Exception as e:
    print(f"Schema already exists or error: {e}")
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.thumbnails")
except Exception as e:
    print(f"Volume already exists or error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## YouTubeから動画をダウンロード

# COMMAND ----------

MAX_VIDEOS = 10

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

# Databricks Japanチャンネルの動画リスト (UC Volumeにアップロード済み)
VIDEO_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

video_list = [
    {"id": "TLpGLZkas70", "title": "Genieスペースを使用した自然言語でデータ分析デモ", "duration": 220},
    {"id": "887Y7q4lR8c", "title": "最速でDatabricksを始める！Express Setupで簡単サインアップ", "duration": 298},
    {"id": "mNn-0jDLfLg", "title": "データサイエンス編02_Databricksで初めての機械学習モデルを構築する", "duration": 276},
    {"id": "_7HSZsYpiek", "title": "３分でわかるDelta Live Tables", "duration": 232},
    {"id": "7fC6h46gC0s", "title": "3分で分かるデータブリックス・ワークスペース", "duration": 265},
    {"id": "6ABXeFwz4aM", "title": "3分で分かるDatabricks SQL", "duration": 192},
]

video_list = video_list[:MAX_VIDEOS]
print(f"対象動画数: {len(video_list)}")
for v in video_list:
    print(f"  - {v['id']}: {v['title']} ({v['duration']}s)")

# COMMAND ----------

# UC Volumeから動画をローカルにコピー
import shutil

downloaded_videos = []

for video_info in video_list:
    vid_id = video_info["id"]
    src_path = f"{VIDEO_VOLUME_PATH}/{vid_id}.mp4"
    output_path = os.path.join(DOWNLOAD_DIR, f"{vid_id}.mp4")

    if os.path.exists(output_path):
        print(f"スキップ (既存): {vid_id}")
        downloaded_videos.append({**video_info, "path": output_path})
        continue

    if os.path.exists(src_path):
        shutil.copy2(src_path, output_path)
        downloaded_videos.append({**video_info, "path": output_path})
        print(f"コピー完了: {vid_id} - {video_info['title']}")
    else:
        print(f"ファイル未検出: {src_path}")

print(f"\n準備完了: {len(downloaded_videos)}/{len(video_list)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cosmos モデル直接ロード
# MAGIC
# MAGIC cosmos-video-encoder Model Serving は GPU ML ランタイムの依存関係との非互換により
# MAGIC 一貫して DEPLOYMENT_FAILED するため、パイプラインクラスター上で直接モデルをロードする。
# MAGIC GPU ML ランタイム 15.4 には torch/transformers/accelerate/safetensors が
# MAGIC プリインストールされており、serving 環境より安定している。

# COMMAND ----------

from huggingface_hub import snapshot_download
import torch
from transformers import AutoModel, AutoProcessor

COSMOS_LOCAL_DIR = "/tmp/cosmos_embed1_model"

py_files = [f for f in os.listdir(COSMOS_LOCAL_DIR) if f.endswith(".py")] if os.path.exists(COSMOS_LOCAL_DIR) else []
if not py_files:
    print("Cosmos-Embed1-448p をダウンロード中 (初回のみ、約2.4GB)...")
    snapshot_download(
        repo_id="nvidia/Cosmos-Embed1-448p",
        local_dir=COSMOS_LOCAL_DIR,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot"],
    )
    print(f"ダウンロード完了: {COSMOS_LOCAL_DIR}")
else:
    print(f"既存モデルを再利用: {COSMOS_LOCAL_DIR}")

_COSMOS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_COSMOS_DTYPE = torch.float16 if _COSMOS_DEVICE == "cuda" else torch.float32
print(f"デバイス: {_COSMOS_DEVICE}, dtype: {_COSMOS_DTYPE}")

# low_cpu_mem_usage=True causes pos_embed shape mismatch (init_empty_weights uses default
# image_size=224 → 257 positions, but 448p weights have 1025 positions). Load normally.
_cosmos_model = AutoModel.from_pretrained(
    COSMOS_LOCAL_DIR,
    trust_remote_code=True,
).to(_COSMOS_DEVICE, dtype=_COSMOS_DTYPE)
_cosmos_model.eval()
_cosmos_processor = AutoProcessor.from_pretrained(COSMOS_LOCAL_DIR, trust_remote_code=True)
print("Cosmos モデルロード完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedding生成

# COMMAND ----------

def extract_segment_frames(video_path, start_time, end_time, num_frames=8):
    """動画セグメントからフレームを均等抽出"""
    reader = VideoReader(video_path, ctx=cpu(0))
    fps = reader.get_avg_fps()
    total_frames = len(reader)

    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)

    if end_frame <= start_frame:
        return None

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()
    frames = reader.get_batch(frame_indices).asnumpy()
    return frames


def save_thumbnail(frames, segment_id, output_dir):
    """セグメントの中央フレームをサムネイルとして保存"""
    mid_idx = len(frames) // 2
    img = Image.fromarray(frames[mid_idx])
    img = img.resize((320, 180))
    thumb_path = os.path.join(output_dir, f"{segment_id}.jpg")
    img.save(thumb_path, "JPEG", quality=80)
    return thumb_path


def compute_video_embedding(frames):
    """フレーム列からビデオembeddingを計算 (Cosmos モデル直接推論)"""
    frames_np = np.array(frames)  # (T, H, W, C)
    # BTCHW フォーマットに変換
    batch = np.transpose(np.expand_dims(frames_np, 0), (0, 1, 4, 2, 3))
    video_inputs = _cosmos_processor(videos=batch).to(_COSMOS_DEVICE, dtype=_COSMOS_DTYPE)

    with torch.no_grad():
        video_emb = _cosmos_model.get_video_embeddings(**video_inputs)

    if hasattr(video_emb, 'video_embeds'):
        emb_tensor = video_emb.video_embeds
    elif torch.is_tensor(video_emb):
        emb_tensor = video_emb
    elif hasattr(video_emb, 'visual_proj'):
        emb_tensor = video_emb.visual_proj
    else:
        emb_tensor = video_emb[0] if hasattr(video_emb, '__getitem__') else video_emb

    return emb_tensor.cpu().float().numpy().flatten().tolist()

# COMMAND ----------

all_segments = []

for video_info in downloaded_videos:
    vid_id = video_info["id"]
    video_path = video_info["path"]
    title = video_info["title"]
    duration = video_info["duration"]
    youtube_url = f"https://www.youtube.com/watch?v={vid_id}"

    print(f"\n処理中: {title} ({duration}s)")

    num_segments = max(1, int(duration / SEGMENT_DURATION))

    for seg_idx in range(num_segments):
        start_time = seg_idx * SEGMENT_DURATION
        end_time = min((seg_idx + 1) * SEGMENT_DURATION, duration)
        segment_id = f"{vid_id}_seg{seg_idx:04d}"

        try:
            frames = extract_segment_frames(video_path, start_time, end_time, FRAMES_PER_SEGMENT)
            if frames is None:
                continue

            embedding = compute_video_embedding(frames)
            thumb_path = save_thumbnail(frames, segment_id, THUMBNAIL_DIR)

            all_segments.append({
                "video_id": vid_id,
                "segment_id": segment_id,
                "title": title,
                "channel_name": "Databricks Japan",
                "youtube_url": youtube_url,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "embedding": embedding,
                "thumbnail_path": f"{VOLUME_PATH}/{segment_id}.jpg",
                "created_at": datetime.now().isoformat(),
            })

            if (seg_idx + 1) % 5 == 0:
                print(f"  セグメント {seg_idx + 1}/{num_segments} 完了")

        except Exception as e:
            print(f"  エラー (seg {seg_idx}): {str(e)[:100]}")
            continue

    print(f"  完了: {title} - {num_segments}セグメント")

print(f"\n全セグメント数: {len(all_segments)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Tableに保存

# COMMAND ----------

schema = StructType([
    StructField("video_id", StringType(), False),
    StructField("segment_id", StringType(), False),
    StructField("title", StringType(), False),
    StructField("channel_name", StringType(), False),
    StructField("youtube_url", StringType(), False),
    StructField("start_time", DoubleType(), False),
    StructField("end_time", DoubleType(), False),
    StructField("embedding", ArrayType(FloatType()), False),
    StructField("thumbnail_path", StringType(), True),
    StructField("created_at", StringType(), False),
])

df = spark.createDataFrame(all_segments, schema=schema)
df = df.withColumn("created_at", F.to_timestamp("created_at"))

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)

print(f"保存完了: {TABLE_NAME}")
print(f"レコード数: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## サムネイルをVolumeにコピー

# COMMAND ----------

import shutil

dbutils.fs.mkdirs(VOLUME_PATH.replace("/Volumes/", "dbfs:/Volumes/"))

for segment in all_segments:
    seg_id = segment["segment_id"]
    src = os.path.join(THUMBNAIL_DIR, f"{seg_id}.jpg")
    dst = f"/Volumes/{CATALOG}/{SCHEMA}/thumbnails/{seg_id}.jpg"
    if os.path.exists(src):
        shutil.copy2(src, dst)

print(f"サムネイルコピー完了: {len(all_segments)} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 確認

# COMMAND ----------

display(spark.table(TABLE_NAME).select("video_id", "segment_id", "title", "start_time", "end_time", "youtube_url").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 検証

# COMMAND ----------

row_count = spark.table(TABLE_NAME).count()
print(f"=== 検証結果 ===")
print(f"テーブル行数: {row_count}")
assert row_count > 0, f"ERROR: {TABLE_NAME} にデータが存在しません"

import pyspark.sql.functions as F2
sample = spark.table(TABLE_NAME).select("segment_id", "embedding").limit(1).collect()
emb = sample[0]["embedding"]
assert emb is not None and len(emb) == 768, f"ERROR: embedding次元が不正: {len(emb) if emb else 'None'}"
assert any(v != 0.0 for v in emb), "ERROR: embeddingが全てゼロです"

videos_with_data = spark.table(TABLE_NAME).select("video_id").distinct().count()
print(f"処理済み動画数: {videos_with_data}")
print(f"embedding次元: {len(emb)}")
print("NOTEBOOK 01 VERIFIED OK")
