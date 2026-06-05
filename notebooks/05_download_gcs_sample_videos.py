# Databricks notebook source
# MAGIC %md
# MAGIC # GCS 公開サンプル動画ダウンロード
# MAGIC
# MAGIC Google Cloud Storage の公開バケットからサンプル動画を一覧取得し、
# MAGIC UC Volume (`/Volumes/.../videos`) に保存する。
# MAGIC
# MAGIC **対象バケット**
# MAGIC
# MAGIC | バケット | プレフィックス | 内容 |
# MAGIC |---------|--------------|------|
# MAGIC | `gtv-videos-bucket` | `sample/` | Google 公式デモ用 MP4（Big Buck Bunny 等） |
# MAGIC
# MAGIC 認証不要（公開バケット）。GCS JSON API でオブジェクト一覧を取得し、
# MAGIC 動画ファイルのみフィルタしてダウンロードする。

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA  = "multimodal_video_search"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

# ダウンロード対象バケット定義 (バケット名, プレフィックス) のリスト
# 公開バケットを追加する場合はここに追記する
GCS_SOURCES = [
    ("gtv-videos-bucket", "sample/"),
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## GCS バケットからファイル一覧を取得

# COMMAND ----------

import requests

def list_gcs_videos(bucket: str, prefix: str = "") -> list[dict]:
    """GCS JSON API で公開バケット内の動画ファイルを全件列挙する。"""
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    videos = []
    page_token = None

    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            name: str = item["name"]
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext in VIDEO_EXTENSIONS:
                size_bytes = int(item.get("size", 0))
                videos.append({
                    "bucket":   bucket,
                    "name":     name,
                    "filename": name.split("/")[-1],
                    "size_mb":  round(size_bytes / 1024 / 1024, 1),
                    "url":      f"https://storage.googleapis.com/{bucket}/{name}",
                })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return videos


all_videos = []
for bucket, prefix in GCS_SOURCES:
    print(f"\n=== {bucket}/{prefix} ===")
    videos = list_gcs_videos(bucket, prefix)
    for v in videos:
        print(f"  {v['filename']}  ({v['size_mb']} MB)  {v['url']}")
    all_videos.extend(videos)

print(f"\n合計: {len(all_videos)} 件")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC Volume にダウンロード

# COMMAND ----------

import os

os.makedirs(VOLUME_PATH, exist_ok=True)

results = []
for v in all_videos:
    dest = os.path.join(VOLUME_PATH, v["filename"])

    if os.path.exists(dest):
        actual_mb = round(os.path.getsize(dest) / 1024 / 1024, 1)
        print(f"[SKIP]  {v['filename']}  (既存 {actual_mb} MB)")
        results.append({**v, "status": "skipped", "dest": dest})
        continue

    print(f"[DL]    {v['filename']}  ({v['size_mb']} MB)  ...", end=" ", flush=True)
    try:
        resp = requests.get(v["url"], stream=True, timeout=300)
        resp.raise_for_status()

        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        os.rename(tmp, dest)

        actual_mb = round(os.path.getsize(dest) / 1024 / 1024, 1)
        print(f"OK ({actual_mb} MB)")
        results.append({**v, "status": "downloaded", "dest": dest})

    except Exception as e:
        if os.path.exists(dest + ".tmp"):
            os.unlink(dest + ".tmp")
        print(f"ERROR: {e}")
        results.append({**v, "status": "error", "error": str(e)})

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

saved = sorted(os.listdir(VOLUME_PATH))
print(f"Volume 内のファイル ({len(saved)} 件): {VOLUME_PATH}")
for f in saved:
    size_mb = round(os.path.getsize(os.path.join(VOLUME_PATH, f)) / 1024 / 1024, 1)
    print(f"  {f}  ({size_mb} MB)")
