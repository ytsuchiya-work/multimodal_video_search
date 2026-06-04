import os
import json
import logging
import subprocess
import tempfile
import uuid
import requests as http_requests
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
MM_TEXT_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_text_index"
MM_IMAGE_INDEX = f"{CATALOG}.{SCHEMA}.multimodal_image_index"
VS_ENDPOINT_NAME = os.environ.get("VS_ENDPOINT_NAME", "video-search-endpoint")
COSMOS_ENDPOINT_NAME = os.environ.get("COSMOS_ENDPOINT_NAME", "cosmos-video-encoder")
TEXT_EMBED_ENDPOINT_NAME = os.environ.get("TEXT_EMBED_ENDPOINT_NAME", "multilingual-e5-embedder")
CLIP_ENDPOINT_NAME = os.environ.get("CLIP_ENDPOINT_NAME", "clip-encoder")
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

w = WorkspaceClient()
DATABRICKS_HOST = w.config.host.rstrip("/")

_ENDPOINT_TIMEOUT = 300  # GPU cold-start can take 3-5 min after scale-to-zero

# In-memory store for async search tasks
search_tasks: dict = {}

ENDPOINT_INFO = {
    COSMOS_ENDPOINT_NAME: {
        "display_name": "Cosmos-Embed1 ビデオエンコーダー",
        "description": "NVIDIA Cosmos-Embed1-448p — 動画セグメント(8フレーム)を768次元embeddingに変換。映像の視覚的内容を理解し、Cosmos検索で動画を検索する際に使用。",
        "model": "nvidia/Cosmos-Embed1-448p",
        "dimension": 768,
        "usage": "Cosmos検索",
    },
    TEXT_EMBED_ENDPOINT_NAME: {
        "display_name": "Multilingual E5 テキストエンコーダー",
        "description": "intfloat/multilingual-e5-large — 日英中など多言語テキスト・字幕を1024次元embeddingに変換。マルチモーダル検索のテキスト側で使用。",
        "model": "intfloat/multilingual-e5-large",
        "dimension": 1024,
        "usage": "マルチモーダル検索（テキスト）",
    },
    CLIP_ENDPOINT_NAME: {
        "display_name": "CLIP 画像エンコーダー",
        "description": "openai/clip-vit-base-patch32 — テキストと画像を同一の512次元embedding空間に変換。マルチモーダル検索の画像(フレーム)側で使用。",
        "model": "openai/clip-vit-base-patch32",
        "dimension": 512,
        "usage": "マルチモーダル検索（画像）",
    },
}


def get_db_headers():
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    return headers


class SearchRequest(BaseModel):
    query: str
    num_results: int = 10


# ── Endpoint management ──────────────────────────────────────────────────────

@app.get("/api/endpoints")
async def list_endpoints():
    results = []
    for name, info in ENDPOINT_INFO.items():
        try:
            resp = http_requests.get(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{name}",
                headers=get_db_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                s = resp.json().get("state", {})
                ready = s.get("ready", "UNKNOWN")
                config_update = s.get("config_update", "UNKNOWN")
                entities = resp.json().get("config", {}).get("served_entities", [])
                version = entities[0].get("entity_version", "?") if entities else "?"
            else:
                ready, config_update, version = "UNKNOWN", "UNKNOWN", "?"
        except Exception as e:
            logger.warning(f"Endpoint status check failed for {name}: {e}")
            ready, config_update, version = "UNKNOWN", "UNKNOWN", "?"
        results.append({
            "name": name,
            **info,
            "ready": ready,
            "config_update": config_update,
            "version": version,
        })
    return {"endpoints": results}


def _do_warmup(endpoint_name: str):
    if endpoint_name == TEXT_EMBED_ENDPOINT_NAME:
        payload = {"dataframe_records": [{"text": "warmup ping"}]}
    else:
        payload = {"dataframe_records": [{"type": "text", "content": "warmup ping"}]}
    try:
        resp = http_requests.post(
            f"{DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations",
            headers=get_db_headers(),
            json=payload,
            timeout=_ENDPOINT_TIMEOUT,
        )
        logger.info(f"Warmup {endpoint_name}: {resp.status_code}")
    except Exception as e:
        logger.error(f"Warmup {endpoint_name} error: {e}")


@app.post("/api/endpoints/{endpoint_name}/warmup")
async def warmup_endpoint(endpoint_name: str, background_tasks: BackgroundTasks):
    if endpoint_name not in ENDPOINT_INFO:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {endpoint_name}")
    background_tasks.add_task(_do_warmup, endpoint_name)
    return {"status": "warmup_started", "endpoint": endpoint_name}


# ── Health / cluster status ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "cosmos_endpoint": COSMOS_ENDPOINT_NAME,
        "text_embed_endpoint": TEXT_EMBED_ENDPOINT_NAME,
        "clip_endpoint": CLIP_ENDPOINT_NAME,
    }


@app.get("/api/cluster/status")
async def cluster_status():
    try:
        endpoints = [COSMOS_ENDPOINT_NAME, TEXT_EMBED_ENDPOINT_NAME, CLIP_ENDPOINT_NAME]
        states = {}
        for ep in endpoints:
            resp = http_requests.get(
                f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{ep}",
                headers=get_db_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                s = resp.json().get("state", {})
                states[ep] = s.get("ready", "UNKNOWN")
        all_ready = all(v == "READY" for v in states.values())
        return {
            "state": "RUNNING" if all_ready else "STARTING",
            "cluster_name": ", ".join(endpoints),
            "state_message": str(states),
        }
    except Exception as e:
        logger.error(f"Endpoint status error: {e}")
        return {"state": "UNKNOWN", "cluster_name": "endpoints", "state_message": str(e)}


@app.post("/api/cluster/start")
async def cluster_start():
    return {"status": "auto_scaling", "message": "Model Serving endpoints scale automatically."}


# ── Async search ─────────────────────────────────────────────────────────────

@app.get("/api/search/result/{task_id}")
async def get_search_result(task_id: str):
    result = search_tasks.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


def _run_cosmos_search(task_id: str, query: str, num_results: int):
    try:
        query_embedding = get_text_embedding(query)
        resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{VS_INDEX_NAME}/query",
            headers=get_db_headers(),
            json={
                "columns": ["segment_id", "title", "channel_name", "youtube_url",
                            "start_time", "end_time", "thumbnail_path"],
                "query_vector": query_embedding,
                "num_results": num_results,
            },
        )
        resp.raise_for_status()
        results = []
        for row in resp.json().get("result", {}).get("data_array", []):
            score = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            results.append({
                "segment_id": row[0],
                "title": row[1] or "",
                "channel_name": row[2] or "Databricks Japan",
                "youtube_url": row[3] or "",
                "start_time": float(row[4] or 0),
                "end_time": float(row[5] or 0),
                "score": score,
                "thumbnail_url": f"/api/thumbnail/{row[0]}",
            })
        search_tasks[task_id] = {"status": "done", "query": query, "results": results}
    except Exception as e:
        logger.error(f"Cosmos search {task_id} failed: {e}")
        search_tasks[task_id] = {"status": "error", "query": query, "error": str(e)}


def _run_multimodal_search(task_id: str, query: str, num_results: int):
    try:
        text_embedding = get_multimodal_embedding(query, "text")
        clip_embedding = get_multimodal_embedding(query, "clip")

        text_resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{MM_TEXT_INDEX}/query",
            headers=get_db_headers(),
            json={
                "columns": ["segment_id", "title", "youtube_url", "start_time",
                            "end_time", "transcript", "thumbnail_path"],
                "query_vector": text_embedding,
                "num_results": num_results * 2,
            },
        )
        text_resp.raise_for_status()

        image_resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{MM_IMAGE_INDEX}/query",
            headers=get_db_headers(),
            json={
                "columns": ["segment_id", "title", "youtube_url", "start_time",
                            "end_time", "transcript", "thumbnail_path"],
                "query_vector": clip_embedding,
                "num_results": num_results * 2,
            },
        )
        image_resp.raise_for_status()

        text_scores, text_meta = {}, {}
        for row in text_resp.json().get("result", {}).get("data_array", []):
            sid = row[0]
            text_scores[sid] = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            text_meta[sid] = row

        image_scores, image_meta = {}, {}
        for row in image_resp.json().get("result", {}).get("data_array", []):
            sid = row[0]
            image_scores[sid] = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            image_meta[sid] = row

        fused = []
        for sid in set(text_scores) | set(image_scores):
            combined = 0.6 * text_scores.get(sid, 0.0) + 0.4 * image_scores.get(sid, 0.0)
            meta = text_meta.get(sid) or image_meta.get(sid)
            fused.append((sid, combined, meta))
        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for sid, score, row in fused[:num_results]:
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
        search_tasks[task_id] = {"status": "done", "query": query, "results": results}
    except Exception as e:
        logger.error(f"Multimodal search {task_id} failed: {e}")
        search_tasks[task_id] = {"status": "error", "query": query, "error": str(e)}


@app.post("/api/search")
async def search_videos(request: SearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    search_tasks[task_id] = {"status": "pending", "query": request.query}
    background_tasks.add_task(_run_cosmos_search, task_id, request.query, request.num_results)
    return {"task_id": task_id, "status": "pending"}


@app.post("/api/search/multimodal")
async def search_multimodal(request: SearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    search_tasks[task_id] = {"status": "pending", "query": request.query}
    background_tasks.add_task(_run_multimodal_search, task_id, request.query, request.num_results)
    return {"task_id": task_id, "status": "pending"}


# ── Video listing ─────────────────────────────────────────────────────────────

@app.get("/api/videos")
async def list_videos():
    try:
        resp = http_requests.post(
            f"{DATABRICKS_HOST}/api/2.0/sql/statements",
            headers=get_db_headers(),
            json={
                "warehouse_id": WAREHOUSE_ID,
                "statement": (
                    f"SELECT DISTINCT video_id, title, channel_name, youtube_url, "
                    f"MIN(start_time) as min_start, MAX(end_time) as max_end, COUNT(*) as segment_count "
                    f"FROM {CATALOG}.{SCHEMA}.video_embeddings "
                    f"GROUP BY video_id, title, channel_name, youtube_url ORDER BY title"
                ),
                "wait_timeout": "30s",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        videos = []
        for row in data.get("result", {}).get("data_array", []):
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


# ── Thumbnail ─────────────────────────────────────────────────────────────────

@app.get("/api/thumbnail/{segment_id}")
async def get_thumbnail(segment_id: str):
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


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_text_embedding(text: str) -> list:
    resp = http_requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/{COSMOS_ENDPOINT_NAME}/invocations",
        headers=get_db_headers(),
        json={"dataframe_records": [{"type": "text", "content": text}]},
        timeout=_ENDPOINT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["predictions"]["embedding"][0]


def get_multimodal_embedding(text: str, embed_type: str) -> list:
    if embed_type == "text":
        payload = {"dataframe_records": [{"text": text}]}
        endpoint = TEXT_EMBED_ENDPOINT_NAME
    elif embed_type == "clip":
        payload = {"dataframe_records": [{"type": "text", "content": text}]}
        endpoint = CLIP_ENDPOINT_NAME
    else:
        raise ValueError(f"Unknown embed_type: {embed_type}")
    resp = http_requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/{endpoint}/invocations",
        headers=get_db_headers(),
        json=payload,
        timeout=_ENDPOINT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["predictions"]["embedding"][0]


# ── Video streaming ───────────────────────────────────────────────────────────

@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str):
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
        headers={"Content-Length": str(file_size), "Accept-Ranges": "bytes"},
    )


# ── Clip creation ─────────────────────────────────────────────────────────────

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
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    if request.end_time - request.start_time > 300:
        raise HTTPException(status_code=400, detail="クリップは最大5分までです")

    video_id = request.video_id
    start, end = request.start_time, request.end_time
    clip_id = f"{video_id}_clip_{int(start)}_{int(end)}"
    clip_filename = request.clip_name or clip_id

    try:
        source_path = f"/Volumes/{CATALOG}/{SCHEMA}/videos/{video_id}.mp4"
        file_headers = w.config.authenticate()
        resp = http_requests.get(
            f"{DATABRICKS_HOST}/api/2.0/fs/files{source_path}",
            headers=file_headers, stream=True,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=404, detail=f"動画ファイルが見つかりません: {video_id}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as src_file:
            for chunk in resp.iter_content(chunk_size=8192):
                src_file.write(chunk)
            src_path = src_file.name

        out_path = src_path.replace(".mp4", "_clip.mp4")
        ffmpeg_path = _get_ffmpeg_path()
        cmd = [ffmpeg_path, "-y", "-i", src_path, "-ss", str(start), "-t", str(end - start),
               "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", out_path]
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
                headers=upload_headers, data=clip_data,
            )
            if upload_resp.status_code not in (200, 201, 204):
                logger.warning(f"Volume upload failed: {upload_resp.status_code}")

        os.rename(out_path, os.path.join(tempfile.gettempdir(), f"{clip_id}.mp4"))
        return {
            "clip_id": clip_id,
            "download_url": f"/api/clip/download/{clip_id}",
            "volume_path": f"{CLIPS_VOLUME_PATH}/{clip_filename}.mp4" if request.save_to_volume else None,
            "duration": round(end - start, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clip error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clip/download/{clip_id}")
async def download_clip(clip_id: str):
    local_path = os.path.join(tempfile.gettempdir(), f"{clip_id}.mp4")
    if os.path.exists(local_path):
        def iter_file():
            with open(local_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        return StreamingResponse(
            iter_file(), media_type="video/mp4",
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
                content=resp.content, media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{clip_id}.mp4"'},
            )
        raise HTTPException(status_code=404, detail="クリップが見つかりません")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="クリップが見つかりません")


# ── Static files ──────────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
