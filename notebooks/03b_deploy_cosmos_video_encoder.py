# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving: Cosmos ビデオエンコーダーデプロイ
# MAGIC
# MAGIC Cosmos-Embed1-448p のビデオエンコーダー部分を MLflow pyfunc モデルとして登録し、
# MAGIC GPU Serving Endpoint にデプロイする。
# MAGIC
# MAGIC **入力**: base64 エンコードされた JPEG フレーム画像のリスト (8フレーム/セグメント)
# MAGIC **出力**: 768次元ビデオ embedding

# COMMAND ----------

# MAGIC %pip install einops
# MAGIC # mlflow/transformers/pillow/torch are pre-installed in GPU ML runtime 15.4

# COMMAND ----------

import os
import mlflow
import torch
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from transformers import AutoProcessor, AutoModel

# COMMAND ----------

CATALOG = spark.conf.get("bundle.variable.catalog", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.cosmos_video_encoder"
SERVING_ENDPOINT_NAME = "cosmos-video-encoder"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをローカルにダウンロード・保存
# MAGIC テキストエンコーダーと同じモデル (Cosmos-Embed1-448p) を使用。
# MAGIC 既にダウンロード済みであれば再利用する。

# COMMAND ----------

LOCAL_MODEL_DIR = "/tmp/cosmos_embed1_model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    print("モデルをHuggingFaceからダウンロード中...")
    model = AutoModel.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)
    model.save_pretrained(LOCAL_MODEL_DIR, safe_serialization=False)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    print(f"モデル保存完了: {LOCAL_MODEL_DIR}")
else:
    print(f"既存モデルを再利用: {LOCAL_MODEL_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Pyfunc モデル定義

# COMMAND ----------

class CosmosVideoEncoder(mlflow.pyfunc.PythonModel):
    """Cosmos-Embed1-448p のビデオエンコーダー。
    入力: base64 JPEG フレームのリスト (1セグメント = 8フレーム)
    出力: 768次元 embedding ベクトル
    """

    def load_context(self, context):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # float16 instead of bfloat16: T4 GPU (compute capability 7.5) does not support bfloat16
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_path = context.artifacts["model_dir"]
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device, dtype=self.dtype)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def predict(self, context, model_input):
        import base64
        import io
        import numpy as np
        import torch
        import pandas as pd
        from PIL import Image

        if isinstance(model_input, pd.DataFrame):
            frames_b64_rows = model_input["frames"].tolist()
        elif isinstance(model_input, dict):
            frames_b64_rows = model_input.get("frames", [])
            if isinstance(frames_b64_rows[0], str):
                # 単一セグメントの場合: フレームのリストがそのまま渡される
                frames_b64_rows = [frames_b64_rows]
        else:
            raise ValueError(f"Unsupported input type: {type(model_input)}")

        results = []
        for frames_b64 in frames_b64_rows:
            frames = []
            for f_b64 in frames_b64:
                img_bytes = base64.b64decode(f_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                frames.append(np.array(img))

            frames_np = np.array(frames)  # (T, H, W, C)
            # BTCHW フォーマットに変換
            batch = np.transpose(np.expand_dims(frames_np, 0), (0, 1, 4, 2, 3))
            video_inputs = self.processor(videos=batch).to(self.device, dtype=self.dtype)

            with torch.no_grad():
                video_emb = self.model.get_video_embeddings(**video_inputs)

            embedding = video_emb.visual_proj.cpu().float().numpy().flatten().tolist()
            results.append(embedding)

        return {"embedding": results}

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow に登録

# COMMAND ----------

input_schema = Schema([ColSpec("string", "frames")])  # JSON配列文字列 or array
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 768), "embedding")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

pip_requirements = [
    "transformers>=4.40.0",
    "einops",
    "torch>=2.0.0",
    "torchvision",
    "pillow",
    "numpy",
]

with mlflow.start_run(run_name="cosmos-video-encoder-deploy"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CosmosVideoEncoder(),
        artifacts={"model_dir": LOCAL_MODEL_DIR},
        pip_requirements=pip_requirements,
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example={"frames": [["<base64_jpeg_frame_1>", "<base64_jpeg_frame_2>"]]},
    )

print(f"モデル登録完了: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving Endpoint 作成・更新

# COMMAND ----------

import requests, json, time

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

from mlflow import MlflowClient
client = MlflowClient()
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(model_versions, key=lambda v: int(v.version)).version
print(f"最新バージョン: {latest_version}")

endpoint_url = f"https://{host}/api/2.0/serving-endpoints"
resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)

served_entity = {
    "entity_name": MODEL_NAME,
    "entity_version": str(latest_version),
    "workload_size": "Small",
    "workload_type": "GPU_MEDIUM",
    "scale_to_zero_enabled": True,
}

if resp.status_code == 200:
    print(f"既存 Endpoint 更新: {SERVING_ENDPOINT_NAME}")
    resp = requests.put(
        f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/config",
        headers=headers,
        json={"served_entities": [served_entity]},
    )
else:
    print(f"新規 Endpoint 作成: {SERVING_ENDPOINT_NAME}")
    resp = requests.post(
        endpoint_url,
        headers=headers,
        json={"name": SERVING_ENDPOINT_NAME, "config": {"served_entities": [served_entity]}},
    )
print(f"Response: {resp.status_code} - {resp.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ready 待機

# COMMAND ----------

for i in range(120):
    resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)
    state = resp.json().get("state", {})
    if state.get("ready") == "READY":
        print(f"Endpoint READY: {SERVING_ENDPOINT_NAME}")
        break
    if state.get("config_update") == "UPDATE_FAILED":
        raise Exception("デプロイ失敗")
    print(f"  待機中... ({i+1}/120) - {state.get('ready')}")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テスト推論

# COMMAND ----------

import base64, io
from PIL import Image
import numpy as np

# ダミー画像フレームを生成してテスト
dummy_frames = []
for _ in range(8):
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG", quality=85)
    dummy_frames.append(base64.b64encode(buf.getvalue()).decode())

resp = requests.post(
    f"{endpoint_url}/{SERVING_ENDPOINT_NAME}/invocations",
    headers=headers,
    json={"dataframe_records": [{"frames": dummy_frames}]},
)
result = resp.json()
if "predictions" in result:
    emb = result["predictions"]["embedding"][0]
    print(f"テスト成功: embedding 次元 = {len(emb)}")
else:
    print(f"レスポンス: {json.dumps(result, indent=2)[:300]}")
