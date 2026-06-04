# Databricks notebook source
# DBTITLE 1,Cell 1
# MAGIC %md
# MAGIC # Model Serving: Cosmos ビデオエンコーダーデプロイ
# MAGIC
# MAGIC Cosmos-Embed1-448p のビデオエンコーダー部分を MLflow pyfunc モデルとして登録し、
# MAGIC GPU Serving Endpoint にデプロイする。
# MAGIC
# MAGIC **入力**: base64 エンコードされた JPEG フレーム画像のリスト (8フレーム/セグメント)
# MAGIC **出力**: 768次元ビデオ embedding
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### デプロイ時の修正履歴
# MAGIC
# MAGIC | セル | 問題 | 修正内容 |
# MAGIC |------|------|----------|
# MAGIC | Cell 8 | Q-Former/BERT 層が float16 非対応 (`expected Float but found Half`) | `self.dtype` を `torch.float32` に固定 |
# MAGIC | Cell 10 | `transformers 5.x` で `find_pruneable_heads_and_indices` が削除済み | `transformers>=4.40.0,<5.0.0` にバージョン制限 |
# MAGIC | Cell 10 | `pandas` が pip\_requirements に未記載 | `"pandas"` を追加 |
# MAGIC | Cell 10 | `torch`/`torchvision` が GPU コンテナのプリインストール版と競合し pip 全体が失敗 | pip\_requirements から削除 |
# MAGIC | Cell 10 | `.cache/huggingface/` メタデータが不要な WSFS ルックアップを誘発 | `shutil.rmtree` で除外してからログ |
# MAGIC | Cell 14 | 旧バージョンが READY のまま新バージョン反映前に break | `config_update == "NOT_UPDATING"` 条件を追加 |
# MAGIC | Cell 16 | invocations URL に `/api/2.0/` プレフィックス不要 | `https://{host}/serving-endpoints/{name}/invocations` に修正 |

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
from huggingface_hub import snapshot_download

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

# Use snapshot_download instead of save_pretrained:
# save_pretrained saves weights only; trust_remote_code models also need the
# custom modeling_*.py files, which snapshot_download includes.
# Without them, the serving endpoint (no HuggingFace internet access) fails with exit code 1.
py_files = [f for f in os.listdir(LOCAL_MODEL_DIR) if f.endswith(".py")] if os.path.exists(LOCAL_MODEL_DIR) else []
if not py_files:
    print("モデルをHuggingFaceからダウンロード中 (snapshot_download)...")
    snapshot_download(
        repo_id="nvidia/Cosmos-Embed1-448p",
        local_dir=LOCAL_MODEL_DIR,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.ot"],
    )
    print(f"モデル保存完了: {LOCAL_MODEL_DIR}")
    print("Pythonファイル:", [f for f in os.listdir(LOCAL_MODEL_DIR) if f.endswith(".py")])
else:
    print(f"既存モデルを再利用 (py files: {py_files}): {LOCAL_MODEL_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Pyfunc モデル定義

# COMMAND ----------

# DBTITLE 1,Cell 8
class CosmosVideoEncoder(mlflow.pyfunc.PythonModel):
    """Cosmos-Embed1-448p のビデオエンコーダー。
    入力: base64 JPEG フレームのリスト (1セグメント = 8フレーム)
    出力: 768次元 embedding ベクトル
    """

    def load_context(self, context):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # float32 for stability: Q-Former/BERT layers raise "expected Float but found Half" with float16
        self.dtype = torch.float32

        model_path = context.artifacts["model_dir"]
        # torch_dtype at load time avoids double-memory (float32 load + float16 convert)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
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

            # get_video_embeddings returns a tensor directly (not an object with attributes)
            if hasattr(video_emb, 'video_embeds'):
                emb_tensor = video_emb.video_embeds
            elif hasattr(video_emb, 'visual_proj'):
                emb_tensor = video_emb.visual_proj
            elif torch.is_tensor(video_emb):
                emb_tensor = video_emb
            else:
                emb_tensor = video_emb[0] if hasattr(video_emb, '__getitem__') else video_emb
            embedding = emb_tensor.cpu().float().numpy().flatten().tolist()
            results.append(embedding)

        return {"embedding": results}

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow に登録

# COMMAND ----------

# DBTITLE 1,Cell 10
input_schema = Schema([ColSpec("string", "frames")])  # JSON配列文字列 or array
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 768), "embedding")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

pip_requirements = [
    "transformers>=4.40.0,<5.0.0",  # model uses find_pruneable_heads_and_indices (removed in 5.x)
    "accelerate>=0.20.0",  # required for low_cpu_mem_usage and large model loading
    "safetensors>=0.3.0",  # model uses safetensors shards
    "einops",
    "pillow",
    "numpy",
    "huggingface_hub>=0.19.0",
    "pandas",
]

def _build_input_example():
    """Build input_example with valid base64 JPEG frames for MLflow health-check."""
    import base64, io
    import numpy as np
    from PIL import Image
    frames = []
    for _ in range(2):
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    return {"frames": [frames]}

# Remove .cache dir (snapshot_download metadata - not needed for inference)
import shutil
_cache_path = os.path.join(LOCAL_MODEL_DIR, ".cache")
if os.path.isdir(_cache_path):
    shutil.rmtree(_cache_path)
    print(f".cache 削除済み: {_cache_path}")

with mlflow.start_run(run_name="cosmos-video-encoder-deploy"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CosmosVideoEncoder(),
        artifacts={"model_dir": LOCAL_MODEL_DIR},
        pip_requirements=pip_requirements,
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example=_build_input_example(),
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
    "workload_type": "GPU_SMALL",  # GPU_MEDIUM causes DEPLOYMENT_ABORTED (capacity) in ap-northeast-1
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

# DBTITLE 1,Cell 14
for i in range(120):
    resp = requests.get(f"{endpoint_url}/{SERVING_ENDPOINT_NAME}", headers=headers)
    state = resp.json().get("state", {})
    if state.get("ready") == "READY" and state.get("config_update") == "NOT_UPDATING":
        print(f"Endpoint READY: {SERVING_ENDPOINT_NAME}")
        break
    if state.get("config_update") == "UPDATE_FAILED":
        # Include failure details from pending_config for diagnostics
        detail = ""
        pending = resp.json().get("pending_config", {})
        for entity in pending.get("served_entities", []):
            msg = entity.get("state", {}).get("deployment_state_message", "")
            if msg:
                detail = msg
                break
        raise Exception(f"デプロイ失敗: {detail or 'see service logs'}")
    print(f"  待機中... ({i+1}/120) - {state.get('ready')}")
    time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テスト推論

# COMMAND ----------

# DBTITLE 1,Cell 16
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

# Invocations API uses /serving-endpoints/ (not /api/2.0/serving-endpoints/)
scoring_url = f"https://{host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"
resp = requests.post(
    scoring_url,
    headers=headers,
    json={"dataframe_records": [{"frames": dummy_frames}]},
)
result = resp.json()
if "predictions" in result:
    emb = result["predictions"]["embedding"][0]
    print(f"テスト成功: embedding 次元 = {len(emb)}")
else:
    print(f"レスポンス: {json.dumps(result, indent=2)[:300]}")
