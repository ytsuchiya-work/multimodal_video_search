# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving: テキストエンコーダーデプロイ
# MAGIC
# MAGIC Cosmos-Embed1-448pのテキストエンコーダー部分をMLflow pyfuncモデルとして登録し、
# MAGIC GPU Serving Endpointにデプロイする。
# MAGIC
# MAGIC モデルをartifactに含め、Serving環境でのHuggingFaceダウンロードを不要にする。

# COMMAND ----------

# MAGIC %pip install mlflow transformers einops torch torchvision
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from transformers import AutoProcessor, AutoModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.cosmos_text_encoder"
SERVING_ENDPOINT_NAME = "cosmos-text-encoder"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをローカルにダウンロード・保存

# COMMAND ----------

LOCAL_MODEL_DIR = "/tmp/cosmos_embed1_model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("モデルをHuggingFaceからダウンロード中...")
model = AutoModel.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)

model.save_pretrained(LOCAL_MODEL_DIR, safe_serialization=False)
processor.save_pretrained(LOCAL_MODEL_DIR)
print(f"モデル保存完了: {LOCAL_MODEL_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Pyfunc モデル定義

# COMMAND ----------

class CosmosTextEncoder(mlflow.pyfunc.PythonModel):
    """Cosmos-Embed1-448pのテキストエンコーダーをラップしたMLflowモデル"""

    def load_context(self, context):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        model_path = context.artifacts["model_dir"]
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device, dtype=self.dtype)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    def predict(self, context, model_input):
        import pandas as pd
        import numpy as np
        import torch

        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        elif isinstance(model_input, dict):
            texts = model_input.get("text", [])
            if isinstance(texts, str):
                texts = [texts]
        else:
            texts = [str(model_input)]

        text_inputs = self.processor(text=texts).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            text_emb = self.model.get_text_embeddings(**text_inputs)

        embeddings = text_emb.text_proj.cpu().numpy().tolist()

        return {"embeddings": embeddings}

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをMLflowに登録 (artifactとしてモデルファイルを含む)

# COMMAND ----------

input_schema = Schema([ColSpec("string", "text")])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 768), "embeddings")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

pip_requirements = [
    "transformers>=4.40.0",
    "einops",
    "torch>=2.0.0",
    "torchvision",
    "numpy",
]

with mlflow.start_run(run_name="cosmos-text-encoder-with-artifacts"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CosmosTextEncoder(),
        artifacts={"model_dir": LOCAL_MODEL_DIR},
        pip_requirements=pip_requirements,
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example={"text": ["データブリックスのデモ"]},
    )

print(f"モデル登録完了: {MODEL_NAME}")
print(f"Run ID: {model_info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving Endpoint 作成

# COMMAND ----------

import requests
import json
import time

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# 最新バージョンを取得
from mlflow import MlflowClient
client = MlflowClient()
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(model_versions, key=lambda v: int(v.version)).version
print(f"最新バージョン: {latest_version}")

# COMMAND ----------

# Serving Endpoint作成/更新
endpoint_url = f"https://{host}/api/2.0/serving-endpoints"

# 既存確認
resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)
if resp.status_code == 200:
    print(f"既存Endpoint更新: {SERVING_ENDPOINT_NAME}")
    resp = requests.put(
        f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/config",
        headers=headers,
        json={
            "served_entities": [
                {
                    "entity_name": MODEL_NAME,
                    "entity_version": str(latest_version),
                    "workload_size": "Small",
                    "workload_type": "GPU_MEDIUM",
                    "scale_to_zero_enabled": True,
                }
            ]
        },
    )
    print(f"Update response: {resp.status_code}")
else:
    print(f"新規Endpoint作成: {SERVING_ENDPOINT_NAME}")
    resp = requests.post(
        endpoint_url,
        headers=headers,
        json={
            "name": SERVING_ENDPOINT_NAME,
            "config": {
                "served_entities": [
                    {
                        "entity_name": MODEL_NAME,
                        "entity_version": str(latest_version),
                        "workload_size": "Small",
                        "workload_type": "GPU_MEDIUM",
                        "scale_to_zero_enabled": True,
                    }
                ]
            },
        },
    )
    print(f"Create response: {resp.status_code} - {resp.text[:300]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Ready待機

# COMMAND ----------

for i in range(120):
    resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)
    ep_data = resp.json()
    state = ep_data.get("state", {})
    ready = state.get("ready", "NOT_READY")
    config_update = state.get("config_update", "UNKNOWN")
    if ready == "READY":
        print(f"Endpoint READY: {SERVING_ENDPOINT_NAME}")
        break
    if config_update == "UPDATE_FAILED":
        print(f"ERROR: デプロイ失敗")
        pending = ep_data.get("pending_config", {})
        for entity in pending.get("served_entities", []):
            print(f"  {entity.get('state', {})}")
        break
    print(f"  待機中... ({i+1}/120) - ready={ready}, config_update={config_update}")
    time.sleep(15)
else:
    print("WARNING: Endpointデプロイがタイムアウトしました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## テスト推論

# COMMAND ----------

resp = requests.post(
    f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/invocations",
    headers=headers,
    json={"dataframe_records": [{"text": "データ分析のデモンストレーション"}]},
)

result = resp.json()
if "predictions" in result:
    emb = result["predictions"]["embeddings"][0]
    print(f"テスト成功! Embedding次元: {len(emb)}")
    print(f"先頭5要素: {emb[:5]}")
else:
    print(f"レスポンス: {json.dumps(result, indent=2)[:500]}")
