# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving: CLIP エンコーダーデプロイ
# MAGIC
# MAGIC openai/clip-vit-base-patch32 をMLflow pyfunc として登録し、GPU Serving Endpoint にデプロイする。
# MAGIC テキストと画像の両方をエンコード可能。マルチモーダル検索の画像クエリ embedding に使用 (512次元)。
# MAGIC
# MAGIC **入力**: `type` ("text" or "image") + `content` (テキスト文字列 or base64 JPEG)
# MAGIC **出力**: 512次元 L2正規化済み embedding

# COMMAND ----------

# All required packages (mlflow, transformers, pillow, torch) are pre-installed in GPU ML runtime 15.4

# COMMAND ----------

import os
import mlflow
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from transformers import CLIPModel, CLIPProcessor

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.clip_encoder"
SERVING_ENDPOINT_NAME = "clip-encoder"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

LOCAL_MODEL_DIR = "/tmp/clip_vit_base_patch32"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

print("CLIPモデルをダウンロード中...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained(LOCAL_MODEL_DIR)
processor.save_pretrained(LOCAL_MODEL_DIR)
print(f"保存完了: {LOCAL_MODEL_DIR}")

# COMMAND ----------

class CLIPEncoder(mlflow.pyfunc.PythonModel):
    """CLIP (clip-vit-base-patch32) エンコーダー。
    type="text"  → テキストクエリを 512次元 embedding に変換
    type="image" → base64 JPEG 画像を 512次元 embedding に変換
    """

    def load_context(self, context):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = context.artifacts["model_dir"]
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def predict(self, context, model_input):
        import base64
        import io
        import numpy as np
        import torch
        import pandas as pd
        from PIL import Image

        if isinstance(model_input, pd.DataFrame):
            types = model_input["type"].tolist()
            contents = model_input["content"].tolist()
        elif isinstance(model_input, dict):
            types = model_input.get("type", [])
            contents = model_input.get("content", [])
            if isinstance(types, str):
                types = [types]
                contents = [contents]
        else:
            raise ValueError(f"Unsupported input type: {type(model_input)}")

        results = []
        for enc_type, content in zip(types, contents):
            with torch.no_grad():
                if enc_type == "text":
                    inputs = self.processor(text=[content], return_tensors="pt", padding=True).to(self.device)
                    features = self.model.get_text_features(**inputs)
                elif enc_type == "image":
                    img_bytes = base64.b64decode(content)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                    features = self.model.get_image_features(**inputs)
                else:
                    raise ValueError(f"Unknown type: {enc_type}. Must be 'text' or 'image'.")

                # transformers >= 4.46 may return ModelOutput instead of tensor
                if not torch.is_tensor(features):
                    if hasattr(features, 'image_embeds') and features.image_embeds is not None:
                        features = features.image_embeds
                    elif hasattr(features, 'text_embeds') and features.text_embeds is not None:
                        features = features.text_embeds
                    elif hasattr(features, 'pooler_output') and features.pooler_output is not None:
                        features = features.pooler_output
                    else:
                        features = next(v for v in features.values() if torch.is_tensor(v))

                features = features / features.norm(dim=-1, keepdim=True)
                results.append(features.cpu().numpy().flatten().tolist())

        return {"embedding": results}

# COMMAND ----------

input_schema = Schema([ColSpec("string", "type"), ColSpec("string", "content")])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 512), "embedding")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name="clip-encoder-deploy"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CLIPEncoder(),
        artifacts={"model_dir": LOCAL_MODEL_DIR},
        pip_requirements=["transformers>=4.30.0,<5.0.0", "pillow", "numpy"],
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example={"type": ["text"], "content": ["データブリックスの機械学習機能"]},
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

# テスト (text)
resp = requests.post(
    f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/invocations",
    headers=headers,
    json={"dataframe_records": [{"type": "text", "content": "データブリックスのデモ動画"}]},
)
result = resp.json()
if "predictions" in result:
    emb = result["predictions"]["embedding"][0]
    print(f"テキストテスト成功: embedding 次元 = {len(emb)}")
else:
    print(f"レスポンス: {json.dumps(result)[:200]}")

