# Databricks notebook source
# MAGIC %md
# MAGIC # GCS 公開サンプル動画ダウンロード
# MAGIC
# MAGIC Google Cloud Storage の公開バケット (`gtv-videos-bucket/sample/`) から
# MAGIC サンプル動画をダウンロードし、UC Volume に保存する。
# MAGIC
# MAGIC **注意**: GCS の JSON listing API はバケット単位の読み取り権限が必要なため 401 になる。
# MAGIC オブジェクト単位の公開 URL は認証不要でダウンロードできるため、既知 URL を直接使用する。

# COMMAND ----------

CATALOG    = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA     = "multimodal_video_search"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

# gtv-videos-bucket/sample/ の公開サンプル動画 URL 一覧
# storage.googleapis.com は 403 になるため、CDN 経由の commondatastorage.googleapis.com を使用
GCS_BASE = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample"
SAMPLE_VIDEOS = [
    "BigBuckBunny.mp4",
    "ElephantsDream.mp4",
    "ForBiggerBlazes.mp4",
    "ForBiggerEscapes.mp4",
    "ForBiggerFun.mp4",
    "ForBiggerJoyrides.mp4",
    "ForBiggerMeltdowns.mp4",
    "Sintel.mp4",
    "SubaruOutbackOnStreetAndDirt.mp4",
    "TearsOfSteel.mp4",
    "VolkswagenGTIReview.mp4",
    "WeAreGoingOnBullrun.mp4",
    "WhatCareers.mp4",
]

all_videos = [{"filename": f, "url": f"{GCS_BASE}/{f}"} for f in SAMPLE_VIDEOS]

print(f"ダウンロード対象: {len(all_videos)} 件")
for v in all_videos:
    print(f"  {v['url']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC Volume にダウンロード

# COMMAND ----------

import os
import requests

os.makedirs(VOLUME_PATH, exist_ok=True)

results = []
for v in all_videos:
    dest = os.path.join(VOLUME_PATH, v["filename"])

    if os.path.exists(dest):
        size_mb = round(os.path.getsize(dest) / 1024 / 1024, 1)
        print(f"[SKIP]  {v['filename']}  (既存 {size_mb} MB)")
        results.append({"filename": v["filename"], "size_mb": size_mb, "status": "skipped", "dest": dest})
        continue

    print(f"[DL]    {v['filename']} ...", end=" ", flush=True)
    try:
        resp = requests.get(v["url"], stream=True, timeout=300)
        resp.raise_for_status()

        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        os.rename(tmp, dest)

        size_mb = round(os.path.getsize(dest) / 1024 / 1024, 1)
        print(f"OK ({size_mb} MB)")
        results.append({"filename": v["filename"], "size_mb": size_mb, "status": "downloaded", "dest": dest})

    except Exception as e:
        if os.path.exists(dest + ".tmp"):
            os.unlink(dest + ".tmp")
        print(f"ERROR: {type(e).__name__}: {e}")
        results.append({"filename": v["filename"], "size_mb": 0, "status": "error", "dest": "", "error": str(e)})
        raise  # 最初のエラーで即停止して error_trace に詳細を出す

# COMMAND ----------

# MAGIC %md
# MAGIC ## 結果サマリ

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(results)[["filename", "size_mb", "status", "dest"]]
display(df)

downloaded = [r for r in results if r["status"] == "downloaded"]
skipped    = [r for r in results if r["status"] == "skipped"]
errors     = [r for r in results if r["status"] == "error"]

print(f"\nダウンロード完了: {len(downloaded)} 件")
print(f"スキップ (既存):  {len(skipped)} 件")
print(f"エラー:           {len(errors)} 件")

assert len(errors) == 0, f"ダウンロードエラー: {[e['filename'] for e in errors]}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume に保存されたファイルを確認

# COMMAND ----------

saved = sorted(f for f in os.listdir(VOLUME_PATH) if f.endswith(".mp4"))
print(f"Volume 内の MP4 ファイル ({len(saved)} 件): {VOLUME_PATH}")
for f in saved:
    size_mb = round(os.path.getsize(os.path.join(VOLUME_PATH, f)) / 1024 / 1024, 1)
    print(f"  {f}  ({size_mb} MB)")
