# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search 設定
# MAGIC
# MAGIC Delta TableのembeddingカラムにVector Search Indexを作成する。
# MAGIC 作成するIndex:
# MAGIC - `video_embeddings_index` (Cosmos 768次元, video_embeddings テーブル)
# MAGIC - `multimodal_text_index` (multilingual-e5-large 1024次元, multimodal_segments テーブル)
# MAGIC - `multimodal_image_index` (CLIP 512次元, multimodal_segments テーブル)

# COMMAND ----------

import time
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url = f"https://{host}/api/2.0/vector-search"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"

VS_ENDPOINT_NAME = "video-search-endpoint"

VIDEO_TABLE = f"{CATALOG}.{SCHEMA}.video_embeddings"
MM_TABLE = f"{CATALOG}.{SCHEMA}.multimodal_segments"

VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.video_embeddings_index"
MM_TEXT_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_text_index"
MM_IMAGE_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_image_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ソーステーブルのデータ確認

# COMMAND ----------

video_count = spark.table(VIDEO_TABLE).count()
mm_count = spark.table(MM_TABLE).count()
print(f"=== ソーステーブル確認 ===")
print(f"video_embeddings: {video_count} 行")
print(f"multimodal_segments: {mm_count} 行")

assert video_count > 0, f"ERROR: {VIDEO_TABLE} にデータがありません。01_video_embedding_pipeline を先に実行してください。"
assert mm_count > 0, f"ERROR: {MM_TABLE} にデータがありません。04_multimodal_pipeline を先に実行してください。"
print("ソーステーブル確認 OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Endpoint 確認

# COMMAND ----------

resp = requests.get(f"{base_url}/endpoints", headers=headers)
existing_endpoints = [ep["name"] for ep in resp.json().get("endpoints", [])]

if VS_ENDPOINT_NAME not in existing_endpoints:
    print(f"Endpoint作成中: {VS_ENDPOINT_NAME}")
    resp = requests.post(
        f"{base_url}/endpoints",
        headers=headers,
        json={"name": VS_ENDPOINT_NAME, "endpoint_type": "STANDARD"},
    )
    print(f"Response: {resp.status_code} - {resp.text[:200]}")
else:
    print(f"Endpoint既存: {VS_ENDPOINT_NAME}")

for i in range(60):
    resp = requests.get(f"{base_url}/endpoints/{VS_ENDPOINT_NAME}", headers=headers)
    ep_data = resp.json()
    state = ep_data.get("endpoint_status", {}).get("state", "UNKNOWN")
    if state == "ONLINE":
        print(f"Endpoint ONLINE: {VS_ENDPOINT_NAME}")
        break
    print(f"  待機中... ({i+1}/60) - 状態: {state}")
    time.sleep(10)
else:
    raise Exception("ERROR: Endpointがタイムアウトしました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index作成ヘルパー

# COMMAND ----------

def create_or_sync_index(index_name, source_table, embedding_col, embedding_dim, columns_to_sync):
    resp = requests.get(
        f"{base_url}/indexes",
        headers=headers,
        params={"endpoint_name": VS_ENDPOINT_NAME},
    )
    existing_indexes = [idx.get("name", "") for idx in resp.json().get("vector_indexes", [])]

    if index_name not in existing_indexes:
        print(f"Index作成中: {index_name}")
        create_body = {
            "name": index_name,
            "endpoint_name": VS_ENDPOINT_NAME,
            "primary_key": "segment_id",
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": source_table,
                "pipeline_type": "TRIGGERED",
                "embedding_vector_columns": [
                    {"name": embedding_col, "embedding_dimension": embedding_dim}
                ],
                "columns_to_sync": columns_to_sync,
            },
        }
        resp = requests.post(f"{base_url}/indexes", headers=headers, json=create_body)
        print(f"  作成レスポンス: {resp.status_code} - {resp.text[:200]}")
        # Wait for index to exist before triggering sync
        for i in range(30):
            r = requests.get(f"{base_url}/indexes/{index_name}", headers=headers)
            if r.status_code == 200:
                break
            time.sleep(5)

    # Always trigger sync (TRIGGERED pipeline doesn't auto-sync on creation)
    print(f"同期トリガー: {index_name}")
    resp = requests.post(f"{base_url}/indexes/{index_name}/sync", headers=headers)
    print(f"  同期レスポンス: {resp.status_code} - {resp.text[:100]}")

    # 同期完了まで待機
    for i in range(120):
        resp = requests.get(f"{base_url}/indexes/{index_name}", headers=headers)
        idx_data = resp.json()
        status = idx_data.get("status", {})
        ready = status.get("ready", False)
        indexed_count = status.get("indexed_row_count", 0)
        if ready:
            print(f"  Index READY: {index_name} ({indexed_count} rows indexed)")
            return indexed_count
        msg = status.get("message", "syncing...")[:80]
        print(f"  同期中... ({i+1}/120) - {msg} | indexed: {indexed_count}")
        time.sleep(10)
    raise Exception(f"ERROR: {index_name} 同期タイムアウト")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. video_embeddings_index 作成・同期

# COMMAND ----------

video_cols = ["video_id", "segment_id", "title", "channel_name",
              "youtube_url", "start_time", "end_time", "thumbnail_path", "embedding"]
video_indexed = create_or_sync_index(
    VS_INDEX_NAME, VIDEO_TABLE, "embedding", 768, video_cols
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. multimodal_text_index 作成・同期

# COMMAND ----------

mm_cols = ["video_id", "segment_id", "title", "youtube_url",
           "start_time", "end_time", "transcript", "thumbnail_path",
           "text_embedding", "image_embedding"]
text_indexed = create_or_sync_index(
    MM_TEXT_INDEX, MM_TABLE, "text_embedding", 1024, mm_cols
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. multimodal_image_index 作成・同期

# COMMAND ----------

image_indexed = create_or_sync_index(
    MM_IMAGE_INDEX, MM_TABLE, "image_embedding", 512, mm_cols
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 検証

# COMMAND ----------

import numpy as np

print(f"=== Vector Search Index 検証 ===")
print(f"video_embeddings_index: {video_indexed} rows indexed")
print(f"multimodal_text_index:  {text_indexed} rows indexed")
print(f"multimodal_image_index: {image_indexed} rows indexed")

assert video_indexed > 0, f"ERROR: video_embeddings_index にデータが同期されていません"
assert text_indexed > 0, f"ERROR: multimodal_text_index にデータが同期されていません"
assert image_indexed > 0, f"ERROR: multimodal_image_index にデータが同期されていません"

# テスト検索
for index_name, dim in [(VS_INDEX_NAME, 768), (MM_TEXT_INDEX, 1024), (MM_IMAGE_INDEX, 512)]:
    test_vector = np.random.randn(dim).tolist()
    resp = requests.post(
        f"{base_url}/indexes/{index_name}/query",
        headers=headers,
        json={
            "columns": ["segment_id", "title", "start_time"],
            "query_vector": test_vector,
            "num_results": 3,
        },
    )
    result = resp.json()
    rows = result.get("result", {}).get("data_array", [])
    print(f"  {index_name.split('.')[-1]}: テスト検索 {resp.status_code}, {len(rows)} 件ヒット")
    assert resp.status_code == 200, f"ERROR: {index_name} テスト検索失敗: {resp.status_code} {resp.text[:200]}"

print("NOTEBOOK 02 VERIFIED OK")
