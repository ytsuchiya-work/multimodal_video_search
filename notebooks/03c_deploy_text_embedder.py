# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving: multilingual-e5-large デプロイ
# MAGIC
# MAGIC intfloat/multilingual-e5-large をMLflow pyfunc として登録し、GPU Serving Endpoint にデプロイする。
# MAGIC マルチモーダル検索のテキストクエリ embedding 生成に使用 (1024次元)。
# MAGIC
# MAGIC **入力**: テキスト文字列
# MAGIC **出力**: 1024次元 embedding ベクトル (L2正規化済み)

# COMMAND ----------

# MAGIC %pip install mlflow sentence-transformers torch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from sentence_transformers import SentenceTransformer

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.multilingual_e5_embedder"
SERVING_ENDPOINT_NAME = "multilingual-e5-embedder"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

LOCAL_MODEL_DIR = "/tmp/multilingual_e5_large"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("モデルをダウンロード中...")
model = SentenceTransformer("intfloat/multilingual-e5-large")
model.save(LOCAL_MODEL_DIR)
print(f"保存完了: {LOCAL_MODEL_DIR}")

# COMMAND ----------

class MultilingualE5Embedder(mlflow.pyfunc.PythonModel):
    """multilingual-e5-large テキストエンコーダー。
    入力: テキスト文字列 ("query: " プレフィックスは自動付与)
    出力: 1024次元 L2正規化済み embedding
    """

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(context.artifacts["model_dir"], device=device)

    def predict(self, context, model_input):
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        elif isinstance(model_input, dict):
            texts = model_input.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
        else:
            texts = [str(model_input)]

        # e5 モデルはクエリに "query: " プレフィックスが必要
        prefixed = [f"query: {t}" if not t.startswith("query:") else t for t in texts]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return {"embedding": embeddings.tolist()}

# COMMAND ----------

input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1024), "embedding")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name="multilingual-e5-embedder-deploy"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=MultilingualE5Embedder(),
        artifacts={"model_dir": LOCAL_MODEL_DIR},
        pip_requirements=["sentence-transformers>=2.2.0", "torch>=2.0.0", "numpy"],
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example={"text": ["データブリックスとは何ですか？"]},
    )

print(f"登録完了: {MODEL_NAME}")

# COMMAND ----------

import requests, json, time

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version)).version

endpoint_url = f"https://{host}/api/2.0/serving-endpoints"
resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)

served_entity = {
    "entity_name": MODEL_NAME,
    "entity_version": str(latest_version),
    "workload_size": "Small",
    "workload_type": "GPU_SMALL",
    "scale_to_zero_enabled": True,
}

if resp.status_code == 200:
    resp = requests.put(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/config", headers=headers,
                        json={"served_entities": [served_entity]})
else:
    resp = requests.post(endpoint_url, headers=headers,
                         json={"name": SERVING_ENDPOINT_NAME, "config": {"served_entities": [served_entity]}})
print(f"Response: {resp.status_code}")

for i in range(120):
    state = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers).json().get("state", {})
    if state.get("ready") == "READY":
        print(f"Endpoint READY: {SERVING_ENDPOINT_NAME}")
        break
    print(f"  待機中... ({i+1}/120)")
    time.sleep(15)

# テスト
resp = requests.post(
    f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/invocations",
    headers=headers,
    json={"dataframe_records": [{"text": "データ分析のデモンストレーション"}]},
)
result = resp.json()
if "predictions" in result:
    emb = result["predictions"]["embedding"][0]
    print(f"テスト成功: embedding 次元 = {len(emb)}")
else:
    print(f"レスポンス: {json.dumps(result)[:200]}")
