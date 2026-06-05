# Databricks notebook source
# MAGIC %md
# MAGIC # 公開サンプル動画ダウンロード
# MAGIC
# MAGIC Blender Foundation の公式ダウンロードサーバーからクリエイティブ・コモンズ動画を
# MAGIC UC Volume (`/Volumes/.../videos`) に保存する。
# MAGIC
# MAGIC **GCS (`gtv-videos-bucket`) が使えない理由**
# MAGIC `storage.googleapis.com` および `commondatastorage.googleapis.com` はともに
# MAGIC AWS/Databricks のクラスタ IP からのリクエストを 403 Forbidden で拒否する
# MAGIC (YouTube と同様のクラウド IP ブロック)。
# MAGIC
# MAGIC **代替ソース: Blender Foundation `download.blender.org`**
# MAGIC IP 制限なし・認証不要・Creative Commons ライセンス。

# COMMAND ----------

CATALOG    = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA     = "multimodal_video_search"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

# Blender Foundation 公式サンプル動画 (Creative Commons)
# サイズは低解像度版を優先し、ダウンロード時間を短縮する
SAMPLE_VIDEOS = [
    {
        "filename": "BigBuckBunny_320x180.mp4",
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
        "title": "Big Buck Bunny (320x180, ~64MB)",
        "license": "CC BY 3.0",
    },
    {
        "filename": "BigBuckBunny_480p.mov",
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov",
        "title": "Big Buck Bunny (480p, ~237MB)",
        "license": "CC BY 3.0",
    },
    {
        "filename": "TearsOfSteel_720p.mov",
        "url": "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        "title": "Tears of Steel (720p, ~354MB)",
        "license": "CC BY 3.0",
    },
]

print(f"ダウンロード対象: {len(SAMPLE_VIDEOS)} 件")
for v in SAMPLE_VIDEOS:
    print(f"  {v['filename']}  {v['url']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UC Volume にダウンロード

# COMMAND ----------

import os
import requests

os.makedirs(VOLUME_PATH, exist_ok=True)

results = []
for v in SAMPLE_VIDEOS:
    dest = os.path.join(VOLUME_PATH, v["filename"])

    if os.path.exists(dest):
        size_mb = round(os.path.getsize(dest) / 1024 / 1024, 1)
        print(f"[SKIP]  {v['filename']}  (既存 {size_mb} MB)")
        results.append({"filename": v["filename"], "size_mb": size_mb, "status": "skipped", "dest": dest})
        continue

    print(f"[DL]    {v['filename']} ...", end=" ", flush=True)
    try:
        resp = requests.get(v["url"], stream=True, timeout=600)
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 結果サマリ

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(results)
display(df)

downloaded = [r for r in results if r["status"] == "downloaded"]
skipped    = [r for r in results if r["status"] == "skipped"]
errors     = [r for r in results if r["status"] == "error"]

print(f"\nダウンロード完了: {len(downloaded)} 件")
print(f"スキップ (既存):  {len(skipped)} 件")
print(f"エラー:           {len(errors)} 件")

if errors:
    for e in errors:
        print(f"  ERROR: {e['filename']} — {e.get('error','')}")

assert len(downloaded) + len(skipped) > 0, "1件もダウンロードできませんでした"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume に保存されたファイルを確認

# COMMAND ----------

saved = sorted(f for f in os.listdir(VOLUME_PATH) if f.lower().endswith((".mp4", ".mov")))
print(f"Volume 内の動画ファイル ({len(saved)} 件): {VOLUME_PATH}")
for f in saved:
    size_mb = round(os.path.getsize(os.path.join(VOLUME_PATH, f)) / 1024 / 1024, 1)
    print(f"  {f}  ({size_mb} MB)")
