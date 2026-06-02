# Databricks notebook source
# MAGIC %md
# MAGIC # 動画ダウンロード
# MAGIC
# MAGIC yt-dlp を使って YouTube から動画をダウンロードし、UC Volume に保存する。
# MAGIC
# MAGIC **フォールバック**: yt-dlp によるダウンロードが失敗した場合（クラウド環境でのボット検出など）、
# MAGIC ノートブックと同じリポジトリの `video_files/` ディレクトリにある MP4 ファイルを使用する。

# COMMAND ----------

# MAGIC %pip install yt-dlp

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import shutil
import subprocess

CATALOG = "classic_stable_ytcy_catalog"
SCHEMA = "multimodal_video_search"
VIDEO_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/videos"

# リポジトリ内 video_files/ ディレクトリのパスをノートブックの場所から動的に解決
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = os.path.dirname(os.path.dirname(notebook_path))  # notebooks/ の親 = リポジトリルート
FALLBACK_VIDEO_DIR = f"/Workspace{repo_root}/video_files"

print(f"Volume パス:    {VIDEO_VOLUME_PATH}")
print(f"フォールバック: {FALLBACK_VIDEO_DIR}")

video_list = [
    {"id": "TLpGLZkas70", "title": "Genieスペースを使用した自然言語でデータ分析デモ"},
    {"id": "887Y7q4lR8c", "title": "最速でDatabricksを始める！Express Setupで簡単サインアップ"},
    {"id": "mNn-0jDLfLg", "title": "データサイエンス編02_Databricksで初めての機械学習モデルを構築する"},
    {"id": "_7HSZsYpiek", "title": "３分でわかるDelta Live Tables"},
    {"id": "7fC6h46gC0s", "title": "3分で分かるデータブリックス・ワークスペース"},
    {"id": "6ABXeFwz4aM", "title": "3分で分かるDatabricks SQL"},
]

# COMMAND ----------

downloaded = []
fallback_copied = []
skipped = []
failed = []

for video in video_list:
    vid_id = video["id"]
    dst_path = f"{VIDEO_VOLUME_PATH}/{vid_id}.mp4"

    # Volume に既に存在する場合はスキップ
    if os.path.exists(dst_path):
        print(f"スキップ (既存): {vid_id}")
        skipped.append(vid_id)
        continue

    # --- Step 1: YouTube からダウンロードを試みる ---
    url = f"https://www.youtube.com/watch?v={vid_id}"
    tmp_path = f"/tmp/{vid_id}.mp4"

    print(f"ダウンロード中: {vid_id} - {video['title']}")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", tmp_path,
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode == 0 and os.path.exists(tmp_path):
        shutil.copy2(tmp_path, dst_path)
        os.remove(tmp_path)
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"  完了 (YouTube): {vid_id} ({size_mb:.1f} MB)")
        downloaded.append(vid_id)
        continue

    # --- Step 2: video_files/ からのフォールバックコピー ---
    print(f"  YouTube ダウンロード失敗: {result.stderr[:200]}")
    fallback_path = os.path.join(FALLBACK_VIDEO_DIR, f"{vid_id}.mp4")

    if os.path.exists(fallback_path):
        shutil.copy2(fallback_path, dst_path)
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"  完了 (フォールバック): {vid_id} ({size_mb:.1f} MB) <- {fallback_path}")
        fallback_copied.append(vid_id)
    else:
        print(f"  失敗: {vid_id} (YouTube ダウンロード失敗、フォールバックファイルも未検出: {fallback_path})")
        failed.append(vid_id)

# COMMAND ----------

print(f"\n=== ダウンロード結果 ===")
print(f"YouTube ダウンロード: {len(downloaded)} 件 - {downloaded}")
print(f"フォールバックコピー: {len(fallback_copied)} 件 - {fallback_copied}")
print(f"スキップ (既存):      {len(skipped)} 件 - {skipped}")
print(f"失敗:                {len(failed)} 件 - {failed}")

if failed:
    raise Exception(
        f"以下の動画の取得に失敗しました: {failed}\n"
        f"YouTube ダウンロードが失敗し、フォールバック先 ({FALLBACK_VIDEO_DIR}) にもファイルがありません。\n"
        f"MP4 ファイルを {FALLBACK_VIDEO_DIR}/ または {VIDEO_VOLUME_PATH}/ に手動配置してください。"
    )

total = len(downloaded) + len(fallback_copied) + len(skipped)
if total == 0:
    raise Exception("動画が1本もありません。処理を中止します。")

print(f"\n処理対象動画: {total} 本")
print("DOWNLOAD COMPLETE")
