import os
import json
import logging
import subprocess
import tempfile
import requests as http_requests
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Search with Cosmos-Embed1")

CATALOG = os.environ.get("CATALOG", "classic_stable_ytcy_catalog")
SCHEMA = "multimodal_video_search"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.video_embeddings_index"
VS_ENDPOINT_NAME = os.environ.get("VS_ENDPOINT_NAME", "video-search-endpoint")
GPU_CLUSTER_ID = os.environ.get("GPU_CLUSTER_ID", "0525-034450-fs01avtr")
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

w = WorkspaceClient()
DATABRICKS_HOST = w.config.host.rstrip("/")


def get_db_headers():
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers


class SearchRequest(BaseModel):
    query: str
    num_results: int = 10


class SearchResult(BaseModel):
    segment_id: str
    title: str
    youtube_url: str
    start_time: float
    end_time: float
    score: float
    thumbnail_url: str
    channel_name: Optional[str] = None


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "index": VS_INDEX_NAME,
        "cluster": GPU_CLUSTER_ID,
        "host_configured": bool(DATABRICKS_HOST),
        "warehouse_id": WAREHOUSE_ID,
    }


@app.get("/api/cluster/status")
async def cluster_status():
    """GPUクラスタの状態を返す"""
    try:
        resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.0/clusters/get",
            headers=get_db_headers(),
            params={"cluster_id": GPU_CLUSTER_ID},
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "state": data.get("state"),
            "cluster_name": data.get("cluster_name"),
            "state_message": data.get("state_message", ""),
        }
    except Exception as e:
        logger.error(f"Cluster status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cluster/start")
async def cluster_start():
    """GPUクラスタを起動する"""
    try:
        resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/clusters/start",
            headers=get_db_headers(),
            json={"cluster_id": GPU_CLUSTER_ID},
        )
        if resp.status_code == 200:
            return {"status": "starting"}
        detail = resp.json().get("message", resp.text)
        raise HTTPException(status_code=resp.status_code, detail=detail)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cluster start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_videos(request: SearchRequest):
    """テキストクエリで動画セグメントを検索"""
    try:
        query_embedding = get_text_embedding(request.query)

        resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{VS_INDEX_NAME}/query",
            headers=get_db_headers(),
            json={
                "columns": [
                    "segment_id", "title", "channel_name",
                    "youtube_url", "start_time", "end_time", "thumbnail_path"
                ],
                "query_vector": query_embedding,
                "num_results": request.num_results,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        search_results = []
        for row in data.get("result", {}).get("data_array", []):
            segment_id = row[0]
            score = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            search_results.append(SearchResult(
                segment_id=segment_id,
                title=row[1] or "",
                channel_name=row[2] or "Databricks Japan",
                youtube_url=row[3] or "",
                start_time=float(row[4] or 0),
                end_time=float(row[5] or 0),
                score=score,
                thumbnail_url=f"/api/thumbnail/{segment_id}",
            ))

        return {"query": request.query, "results": search_results}

    except Exception as e:
        logger.error(f"Search error: {e}")
        err_msg = str(e)
        if "not currently ready" in err_msg or "ClusterNotReady" in err_msg or "Terminated" in err_msg:
            raise HTTPException(status_code=503, detail="GPUクラスタが停止中です。起動完了までお待ちください（数分かかります）。")
        raise HTTPException(status_code=500, detail=err_msg)


@app.post("/api/search/multimodal")
async def search_multimodal(request: SearchRequest):
    """マルチモーダル検索: テキスト(e5) + CLIP画像検索のスコア統合"""
    try:
        text_embedding = get_multimodal_embedding(request.query, "text")
        clip_embedding = get_multimodal_embedding(request.query, "clip")

        text_resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{MM_TEXT_INDEX}/query",
            headers=get_db_headers(),
            json={
                "columns": [
                    "segment_id", "title", "youtube_url",
                    "start_time", "end_time", "transcript", "thumbnail_path"
                ],
                "query_vector": text_embedding,
                "num_results": request.num_results * 2,
            },
        )
        text_resp.raise_for_status()
        text_data = text_resp.json()

        image_resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{MM_IMAGE_INDEX}/query",
            headers=get_db_headers(),
            json={
                "columns": [
                    "segment_id", "title", "youtube_url",
                    "start_time", "end_time", "transcript", "thumbnail_path"
                ],
                "query_vector": clip_embedding,
                "num_results": request.num_results * 2,
            },
        )
        image_resp.raise_for_status()
        image_data = image_resp.json()

        text_scores = {}
        text_meta = {}
        for row in text_data.get("result", {}).get("data_array", []):
            sid = row[0]
            score = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            text_scores[sid] = score
            text_meta[sid] = row

        image_scores = {}
        image_meta = {}
        for row in image_data.get("result", {}).get("data_array", []):
            sid = row[0]
            score = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            image_scores[sid] = score
            image_meta[sid] = row

        all_segment_ids = set(text_scores.keys()) | set(image_scores.keys())
        fused = []
        for sid in all_segment_ids:
            t_score = text_scores.get(sid, 0.0)
            i_score = image_scores.get(sid, 0.0)
            combined = 0.6 * t_score + 0.4 * i_score
            meta = text_meta.get(sid) or image_meta.get(sid)
            fused.append((sid, combined, meta))

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sid, score, row in fused[: request.num_results]:
            results.append({
                "segment_id": sid,
                "title": row[1] or "",
                "channel_name": "Databricks Japan",
                "youtube_url": row[2] or "",
                "start_time": float(row[3] or 0),
                "end_time": float(row[4] or 0),
                "transcript": row[5] or "",
                "score": score,
                "thumbnail_url": f"/api/thumbnail/{sid}",
            })

        return {"query": request.query, "results": results}

    except Exception as e:
        logger.error(f"Multimodal search error: {e}")
        err_msg = str(e)
        if "not currently ready" in err_msg or "ClusterNotReady" in err_msg or "Terminated" in err_msg:
            raise HTTPException(status_code=503, detail="GPUクラスタが停止中です。起動完了までお待ちください。")
        raise HTTPException(status_code=500, detail=err_msg)


@app.get("/api/videos")
async def list_videos():
    """登録済み動画の一覧を取得"""
    try:
        resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers=get_db_headers(),
            json={
                "warehouse_id": WAREHOUSE_ID,
                "statement": f"""
                    SELECT DISTINCT video_id, title, channel_name, youtube_url,
                           MIN(start_time) as min_start,
                           MAX(end_time) as max_end,
                           COUNT(*) as segment_count
                    FROM {CATALOG}.{SCHEMA}.video_embeddings
                    GROUP BY video_id, title, channel_name, youtube_url
                    ORDER BY title
                """,
                "wait_timeout": "30s",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        videos = []
        rows = data.get("result", {}).get("data_array", [])
        for row in rows:
            videos.append({
                "video_id": row[0],
                "title": row[1],
                "channel_name": row[2],
                "youtube_url": row[3],
                "duration": float(row[5] or 0),
                "segment_count": int(row[6] or 0),
            })

        return {"videos": videos}

    except Exception as e:
        logger.error(f"List videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/thumbnail/{segment_id}")
async def get_thumbnail(segment_id: str):
    """セグメントのサムネイル画像を返す"""
    volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/thumbnails/{segment_id}.jpg"
    try:
        file_headers = w.config.authenticate()
        resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.0/fs/files{volume_path}",
            headers=file_headers,
        )
        if resp.status_code == 200:
            return Response(content=resp.content, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Thumbnail not found")


NOTEBOOK_PATH = "/Workspace/Users/yusuke.tsuchiya@databricks.com/video-search-cosmos/_text_embed_notebook"
MULTIMODAL_EMBED_NOTEBOOK = "/Workspace/Users/yusuke.tsuchiya@databricks.com/video-search-cosmos/_multimodal_embed_notebook"

MM_TEXT_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_text_index"
MM_IMAGE_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_image_index"


def _ensure_embedding_notebook():
    """テキストembedding計算用のノートブックが存在することを確認"""
    import base64

    notebook_code = '''# Databricks notebook source
import torch, json
from transformers import AutoProcessor, AutoModel

query_text = dbutils.widgets.get("query_text")

model = AutoModel.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)
model = model.to("cuda", dtype=torch.float16).eval()
processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)

inputs = processor(text=[query_text]).to("cuda", dtype=torch.float16)
with torch.no_grad():
    emb = model.get_text_embeddings(**inputs)
result = emb.text_proj.cpu().numpy().flatten().tolist()
dbutils.notebook.exit(json.dumps(result))
'''
    nb_b64 = base64.b64encode(notebook_code.encode()).decode()
    http_requests.post(
        f"{DATABRICKS_HOST}/api/2.0/workspace/import",
        headers=get_db_headers(),
        json={
            "path": NOTEBOOK_PATH,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": nb_b64,
            "overwrite": True,
        },
    )


def _ensure_multimodal_embed_notebook():
    """マルチモーダル検索用embedding計算ノートブック (e5 + CLIP)"""
    import base64

    notebook_code = '''# Databricks notebook source
import torch, json
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

query_text = dbutils.widgets.get("query_text")
embed_type = dbutils.widgets.get("embed_type")

if embed_type == "text":
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    text_input = f"query: {query_text}"
    emb = model.encode(text_input, normalize_embeddings=True).tolist()
    dbutils.notebook.exit(json.dumps(emb))
elif embed_type == "clip":
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    emb = text_features.cpu().numpy().flatten().tolist()
    dbutils.notebook.exit(json.dumps(emb))
else:
    dbutils.notebook.exit(json.dumps({"error": "invalid embed_type"}))
'''
    nb_b64 = base64.b64encode(notebook_code.encode()).decode()
    http_requests.post(
        f"{DATABRICKS_HOST}/api/2.0/workspace/import",
        headers=get_db_headers(),
        json={
            "path": MULTIMODAL_EMBED_NOTEBOOK,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": nb_b64,
            "overwrite": True,
        },
    )


# Create notebooks on startup
_ensure_embedding_notebook()
_ensure_multimodal_embed_notebook()


def get_multimodal_embedding(text: str, embed_type: str) -> list:
    """GPU上でマルチモーダルembeddingを計算 (e5 text or CLIP text)"""
    import time

    submit_resp = http_requests.post(
        f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit",
        headers=get_db_headers(),
        json={
            "run_name": f"multimodal-embed-{embed_type}",
            "existing_cluster_id": GPU_CLUSTER_ID,
            "notebook_task": {
                "notebook_path": MULTIMODAL_EMBED_NOTEBOOK,
                "base_parameters": {"query_text": text, "embed_type": embed_type},
            },
        },
    )
    submit_resp.raise_for_status()
    run_id = submit_resp.json()["run_id"]

    for _ in range(90):
        time.sleep(2)
        status_resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get",
            headers=get_db_headers(),
            params={"run_id": run_id},
        )
        status_resp.raise_for_status()
        run_data = status_resp.json()
        state = run_data.get("state", {})
        life_cycle = state.get("life_cycle_state", "")
        result_state = state.get("result_state", "")

        if life_cycle == "TERMINATED":
            if result_state == "SUCCESS":
                output_resp = http_requests.get(
                    f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output",
                    headers=get_db_headers(),
                    params={"run_id": run_id},
                )
                output_resp.raise_for_status()
                notebook_output = output_resp.json().get("notebook_output", {})
                result_str = notebook_output.get("result", "")
                return json.loads(result_str)
            else:
                error_msg = state.get("state_message", "Unknown error")
                raise ValueError(f"Embedding計算失敗: {error_msg}")
        elif life_cycle in ("INTERNAL_ERROR", "SKIPPED"):
            raise ValueError(f"実行失敗: {state.get('state_message', life_cycle)}")

    raise TimeoutError("Embedding計算がタイムアウトしました (3分)")


def get_text_embedding(text: str) -> list:
    """GPUクラスタ上でCosmos-Embed1テキストembeddingを計算 (Jobs Run Submit API)"""
    import time

    submit_resp = http_requests.post(
        f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit",
        headers=get_db_headers(),
        json={
            "run_name": "text-embedding-query",
            "existing_cluster_id": GPU_CLUSTER_ID,
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "base_parameters": {"query_text": text},
            },
        },
    )
    submit_resp.raise_for_status()
    run_id = submit_resp.json()["run_id"]

    for _ in range(90):
        time.sleep(2)
        status_resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get",
            headers=get_db_headers(),
            params={"run_id": run_id},
        )
        status_resp.raise_for_status()
        run_data = status_resp.json()
        state = run_data.get("state", {})
        life_cycle = state.get("life_cycle_state", "")
        result_state = state.get("result_state", "")

        if life_cycle == "TERMINATED":
            if result_state == "SUCCESS":
                output_resp = http_requests.get(
                    f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output",
                    headers=get_db_headers(),
                    params={"run_id": run_id},
                )
                output_resp.raise_for_status()
                notebook_output = output_resp.json().get("notebook_output", {})
                result_str = notebook_output.get("result", "")
                return json.loads(result_str)
            else:
                error_msg = state.get("state_message", "Unknown error")
                raise ValueError(f"Embedding計算失敗: {error_msg}")
        elif life_cycle in ("INTERNAL_ERROR", "SKIPPED"):
            raise ValueError(f"実行失敗: {state.get('state_message', life_cycle)}")

    raise TimeoutError("テキストembedding計算がタイムアウトしました (3分)")


@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str):
    """UC Volume の動画をブラウザに配信"""
    source_path = f"/Volumes/{CATALOG}/{SCHEMA}/videos/{video_id}.mp4"
    file_headers = w.config.authenticate()
    resp = http_requests.get(
        f"{DATABRICKS_HOST}/api/2.0/fs/files{source_path}",
        headers=file_headers,
        stream=True,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"動画が見つかりません: {video_id}")

    file_size = int(resp.headers.get("content-length", 0))
    return StreamingResponse(
        resp.iter_content(65536),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        },
    )


CLIPS_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/clips"


class ClipRequest(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    save_to_volume: bool = True
    clip_name: Optional[str] = None


def _get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


@app.post("/api/clip")
async def create_clip(request: ClipRequest):
    """動画の指定時間範囲を切り出してVolume保存 + ダウンロード提供"""
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    if request.end_time - request.start_time > 300:
        raise HTTPException(status_code=400, detail="クリップは最大5分までです")

    video_id = request.video_id
    start = request.start_time
    end = request.end_time
    clip_id = f"{video_id}_clip_{int(start)}_{int(end)}"
    clip_filename = request.clip_name or clip_id

    try:
        source_path = f"/Volumes/{CATALOG}/{SCHEMA}/videos/{video_id}.mp4"
        file_headers = w.config.authenticate()
        resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.0/fs/files{source_path}",
            headers=file_headers,
            stream=True,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"動画ファイルが見つかりません: {video_id}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src_file:
            for chunk in resp.iter_content(chunk_size=8192):
                src_file.write(chunk)
            src_path = src_file.name

        out_path = src_path.replace(".mp4", f"_clip.mp4")
        ffmpeg_path = _get_ffmpeg_path()
        duration = end - start
        cmd = [
            ffmpeg_path, "-y",
            "-i", src_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        os.unlink(src_path)

        if result.returncode != 0:
            if os.path.exists(out_path):
                os.unlink(out_path)
            raise HTTPException(status_code=500, detail=f"ffmpegエラー: {result.stderr.decode()[:200]}")

        if request.save_to_volume:
            with open(out_path, "rb") as f:
                clip_data = f.read()
            upload_headers = w.config.authenticate()
            upload_headers["Content-Type"] = "application/octet-stream"
            upload_resp = http_requests.put(
                f"{DATABRICKS_HOST}/api/2.0/fs/files{CLIPS_VOLUME_PATH}/{clip_filename}.mp4",
                headers=upload_headers,
                data=clip_data,
            )
            if upload_resp.status_code not in (200, 201, 204):
                logger.warning(f"Volume upload failed: {upload_resp.status_code} {upload_resp.text[:100]}")

        os.rename(out_path, os.path.join(tempfile.gettempdir(), f"{clip_id}.mp4"))

        return {
            "clip_id": clip_id,
            "download_url": f"/api/clip/download/{clip_id}",
            "volume_path": f"{CLIPS_VOLUME_PATH}/{clip_filename}.mp4" if request.save_to_volume else None,
            "source_volume_path": f"/Volumes/{CATALOG}/{SCHEMA}/videos/{video_id}.mp4",
            "duration": round(end - start, 1),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clip error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clip/download/{clip_id}")
async def download_clip(clip_id: str):
    """クリップ動画をダウンロード"""
    local_path = os.path.join(tempfile.gettempdir(), f"{clip_id}.mp4")
    if os.path.exists(local_path):
        def iter_file():
            with open(local_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return StreamingResponse(
            iter_file(),
            media_type="video/mp4",
            headers={"Content-Disposition": f'attachment; filename="{clip_id}.mp4"'},
        )

    try:
        file_headers = w.config.authenticate()
        resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.0/fs/files{CLIPS_VOLUME_PATH}/{clip_id}.mp4",
            headers=file_headers,
        )
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{clip_id}.mp4"'},
            )
        raise HTTPException(status_code=404, detail="クリップが見つかりません")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail="クリップが見つかりません")


# Static files (React build output)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
