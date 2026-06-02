# Databricks notebook source
# MAGIC %md
# MAGIC # マルチモーダル検索パイプライン
# MAGIC
# MAGIC 動画の音声文字起こし（Whisper）+ 画像フレームCLIP embeddingを生成し、
# MAGIC マルチモーダル検索用のDelta Tableに保存する。
# MAGIC
# MAGIC **実行要件**: GPU Cluster (T4以上) + ffmpeg

# COMMAND ----------

# MAGIC %pip install openai-whisper decord pillow

# COMMAND ----------

import os
import base64
import io
import json
import shutil
import subprocess
import numpy as np
import requests
import torch
from datetime import datetime
from decord import VideoReader, cpu
from PIL import Image
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    ArrayType, FloatType
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
TABLE_NAME = f"{CATALOG}.{SCHEMA}.multimodal_segments"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/thumbnails"
VIDEO_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

SEGMENT_DURATION = 5  # 5秒セグメント
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOWNLOAD_DIR = "/tmp/videos"
AUDIO_DIR = "/tmp/audio"
THUMBNAIL_DIR = "/tmp/thumbnails_mm"

print(f"Device: {DEVICE}")
print(f"Table: {TABLE_NAME}")

# COMMAND ----------

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

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
# MAGIC ## 動画をローカルにコピー

# COMMAND ----------

video_list = [
    {"id": "TLpGLZkas70", "title": "Genieスペースを使用した自然言語でデータ分析デモ", "duration": 220},
    {"id": "887Y7q4lR8c", "title": "最速でDatabricksを始める！Express Setupで簡単サインアップ", "duration": 298},
    {"id": "mNn-0jDLfLg", "title": "データサイエンス編02_Databricksで初めての機械学習モデルを構築する", "duration": 276},
    {"id": "_7HSZsYpiek", "title": "３分でわかるDelta Live Tables", "duration": 232},
    {"id": "7fC6h46gC0s", "title": "3分で分かるデータブリックス・ワークスペース", "duration": 265},
    {"id": "6ABXeFwz4aM", "title": "3分で分かるDatabricks SQL", "duration": 192},
]

downloaded_videos = []
for video_info in video_list:
    vid_id = video_info["id"]
    src_path = f"{VIDEO_VOLUME_PATH}/{vid_id}.mp4"
    output_path = os.path.join(DOWNLOAD_DIR, f"{vid_id}.mp4")
    if os.path.exists(output_path):
        downloaded_videos.append({**video_info, "path": output_path})
        continue
    if os.path.exists(src_path):
        shutil.copy2(src_path, output_path)
        downloaded_videos.append({**video_info, "path": output_path})
        print(f"コピー完了: {vid_id}")
    else:
        print(f"ファイル未検出: {src_path}")

print(f"準備完了: {len(downloaded_videos)}/{len(video_list)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルロード

# COMMAND ----------

# Whisper: 音声文字起こし専用 (ローカルロード)
# Note: Whisperはローカルで実行。音声データbase64が~13MBとなりModel Servingの10MB制限を超えるため。
print("Whisperモデルロード中...")
import whisper
whisper_model = whisper.load_model("base", device=DEVICE)
print("Whisperロード完了")

# CLIP / multilingual-e5: Model Serving エンドポイントで実行
CLIP_ENDPOINT = "clip-encoder"
TEXT_EMBED_ENDPOINT = "multilingual-e5-embedder"
HOST = spark.conf.get("spark.databricks.workspaceUrl")
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
ENDPOINT_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}
print(f"CLIP endpoint: {CLIP_ENDPOINT}")
print(f"Text embed endpoint: {TEXT_EMBED_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 音声文字起こし + Embedding生成

# COMMAND ----------

all_segments = []

for video_info in downloaded_videos:
    vid_id = video_info["id"]
    video_path = video_info["path"]
    title = video_info["title"]
    duration = video_info["duration"]
    youtube_url = f"https://www.youtube.com/watch?v={vid_id}"
    print(f"\n処理中: {title} ({duration}s)")

    # 音声抽出
    audio_path = os.path.join(AUDIO_DIR, f"{vid_id}.wav")
    if not os.path.exists(audio_path):
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], capture_output=True)

    # Whisper文字起こし (全体)
    print(f"  文字起こし中...")
    result = whisper_model.transcribe(audio_path, language="ja", verbose=False)
    segments_data = result.get("segments", [])

    # 動画フレーム読み込み
    reader = VideoReader(video_path, ctx=cpu(0))
    fps = reader.get_avg_fps()
    total_frames = len(reader)

    # 5秒セグメントに再構成
    num_segments = max(1, int(duration / SEGMENT_DURATION))

    for seg_idx in range(num_segments):
        start_time = seg_idx * SEGMENT_DURATION
        end_time = min((seg_idx + 1) * SEGMENT_DURATION, duration)
        segment_id = f"{vid_id}_mm{seg_idx:04d}"

        try:
            # このセグメントに対応する文字起こしテキストを収集
            transcript_parts = []
            for seg in segments_data:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                if seg_start < end_time and seg_end > start_time:
                    transcript_parts.append(seg.get("text", "").strip())
            transcript = " ".join(transcript_parts).strip()

            # フレーム抽出 (セグメント中央)
            mid_time = (start_time + end_time) / 2
            frame_idx = min(int(mid_time * fps), total_frames - 1)
            frame = reader[frame_idx].asnumpy()
            img = Image.fromarray(frame)

            # サムネイル保存
            thumb = img.resize((320, 180))
            thumb_path = os.path.join(THUMBNAIL_DIR, f"{segment_id}.jpg")
            thumb.save(thumb_path, "JPEG", quality=80)

            # CLIP image embedding (clip-encoder serving endpoint)
            img_buf = io.BytesIO()
            img.save(img_buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(img_buf.getvalue()).decode()
            clip_resp = requests.post(
                f"https://{HOST}/serving-endpoints/{CLIP_ENDPOINT}/invocations",
                headers=ENDPOINT_HEADERS,
                json={"dataframe_records": [{"type": "image", "content": img_b64}]},
                timeout=30,
            )
            clip_resp.raise_for_status()
            image_emb = clip_resp.json()["predictions"]["embedding"][0]

            # テキスト embedding (multilingual-e5-embedder serving endpoint)
            if transcript:
                e5_resp = requests.post(
                    f"https://{HOST}/serving-endpoints/{TEXT_EMBED_ENDPOINT}/invocations",
                    headers=ENDPOINT_HEADERS,
                    json={"dataframe_records": [{"text": transcript}]},
                    timeout=30,
                )
                e5_resp.raise_for_status()
                text_emb = e5_resp.json()["predictions"]["embedding"][0]
            else:
                text_emb = [0.0] * 1024

            all_segments.append({
                "video_id": vid_id,
                "segment_id": segment_id,
                "title": title,
                "youtube_url": youtube_url,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "transcript": transcript if transcript else "",
                "text_embedding": text_emb,
                "image_embedding": image_emb,
                "thumbnail_path": f"{VOLUME_PATH}/{segment_id}.jpg",
                "created_at": datetime.now().isoformat(),
            })

        except Exception as e:
            print(f"  エラー (seg {seg_idx}): {str(e)[:100]}")
            continue

    print(f"  完了: {title} - {num_segments}セグメント")

print(f"\n全セグメント数: {len(all_segments)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Tableに保存

# COMMAND ----------

if all_segments:
    schema = StructType([
        StructField("video_id", StringType(), False),
        StructField("segment_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("youtube_url", StringType(), False),
        StructField("start_time", DoubleType(), False),
        StructField("end_time", DoubleType(), False),
        StructField("transcript", StringType(), True),
        StructField("text_embedding", ArrayType(FloatType()), False),
        StructField("image_embedding", ArrayType(FloatType()), False),
        StructField("thumbnail_path", StringType(), True),
        StructField("created_at", StringType(), False),
    ])
    df = spark.createDataFrame(all_segments, schema=schema)
    df = df.withColumn("created_at", F.to_timestamp("created_at"))
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE_NAME)

    # CDF有効化
    spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    print(f"保存完了: {TABLE_NAME} ({len(all_segments)} rows)")

    # サムネイルコピー
    for segment in all_segments:
        seg_id = segment["segment_id"]
        src = os.path.join(THUMBNAIL_DIR, f"{seg_id}.jpg")
        dst = f"/Volumes/{CATALOG}/{SCHEMA}/thumbnails/{seg_id}.jpg"
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print("サムネイルコピー完了")
else:
    print("ERROR: セグメントが0件です")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 確認

# COMMAND ----------

display(spark.table(TABLE_NAME).select(
    "video_id", "segment_id", "title", "start_time", "end_time", "transcript"
).limit(20))

# COMMAND ----------

row_count = spark.table(TABLE_NAME).count()
print(f"テーブル行数: {row_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 検証

# COMMAND ----------

print(f"=== 検証結果 ===")
print(f"テーブル行数: {row_count}")
assert row_count > 0, f"ERROR: {TABLE_NAME} にデータが存在しません"

sample = spark.table(TABLE_NAME).select("segment_id", "text_embedding", "image_embedding").limit(1).collect()
row = sample[0]
text_emb = row["text_embedding"]
img_emb = row["image_embedding"]
assert text_emb is not None and len(text_emb) == 1024, f"ERROR: text_embedding次元が不正: {len(text_emb) if text_emb else 'None'}"
assert img_emb is not None and len(img_emb) == 512, f"ERROR: image_embedding次元が不正: {len(img_emb) if img_emb else 'None'}"

videos_with_data = spark.table(TABLE_NAME).select("video_id").distinct().count()
print(f"処理済み動画数: {videos_with_data}")
print(f"text_embedding次元: {len(text_emb)}")
print(f"image_embedding次元: {len(img_emb)}")
print("NOTEBOOK 04 VERIFIED OK")
