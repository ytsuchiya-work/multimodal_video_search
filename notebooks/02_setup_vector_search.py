# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search 設定
# MAGIC
# MAGIC Delta TableのembeddingカラムにVector Search Indexを作成する。

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
TABLE_NAME = f"{CATALOG}.{SCHEMA}.video_embeddings"

VS_ENDPOINT_NAME = "video-search-endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.video_embeddings_index"

EMBEDDING_DIMENSION = 768
EMBEDDING_COLUMN = "embedding"
PRIMARY_KEY = "segment_id"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Search Endpoint 作成

# COMMAND ----------

# 既存Endpoint確認
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

# Endpointがオンラインになるまで待機
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
    print("WARNING: Endpointがタイムアウトしました。後で確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Sync Index 作成

# COMMAND ----------

# 既存Index確認
resp = requests.get(
    f"{base_url}/indexes",
    headers=headers,
    params={"endpoint_name": VS_ENDPOINT_NAME},
)
existing_indexes = [idx.get("name", "") for idx in resp.json().get("vector_indexes", [])]

if VS_INDEX_NAME not in existing_indexes:
    print(f"Index作成中: {VS_INDEX_NAME}")
    create_body = {
        "name": VS_INDEX_NAME,
        "endpoint_name": VS_ENDPOINT_NAME,
        "primary_key": PRIMARY_KEY,
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": TABLE_NAME,
            "pipeline_type": "TRIGGERED",
            "embedding_vector_columns": [
                {
                    "name": EMBEDDING_COLUMN,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                }
            ],
            "columns_to_sync": [
                "video_id", "segment_id", "title", "channel_name",
                "youtube_url", "start_time", "end_time", "thumbnail_path",
                "embedding"
            ],
        },
    }
    resp = requests.post(
        f"{base_url}/indexes",
        headers=headers,
        json=create_body,
    )
    print(f"Response: {resp.status_code} - {resp.text[:300]}")
else:
    print(f"Index既存: {VS_INDEX_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index同期状態の確認

# COMMAND ----------

for i in range(60):
    try:
        resp = requests.get(f"{base_url}/indexes/{VS_INDEX_NAME}", headers=headers)
        idx_data = resp.json()
        status = idx_data.get("status", {})
        ready = status.get("ready", False)
        if ready:
            print(f"Index READY: {VS_INDEX_NAME}")
            break
        msg = status.get("message", "syncing...")
        print(f"  同期中... ({i+1}/60) - {msg}")
    except Exception as e:
        print(f"  確認中... ({i+1}/60) - {str(e)[:80]}")
    time.sleep(10)
else:
    print("WARNING: Index同期がタイムアウトしました。後で確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## テスト検索

# COMMAND ----------

import numpy as np

test_vector = np.random.randn(EMBEDDING_DIMENSION).tolist()

resp = requests.post(
    f"{base_url}/indexes/{VS_INDEX_NAME}/query",
    headers=headers,
    json={
        "columns": ["segment_id", "title", "youtube_url", "start_time", "end_time"],
        "query_vector": test_vector,
        "num_results": 5,
    },
)

print("テスト検索結果:")
result = resp.json()
for row in result.get("result", {}).get("data_array", []):
    print(f"  {row}")
