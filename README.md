# Video Search with Cosmos-Embed1 + マルチモーダル検索

NVIDIA Cosmos-Embed1-448p を使用した動画セマンティック検索 + 音声文字起こし・画像フレームによるマルチモーダル検索アプリケーション。自然言語テキストで動画の特定シーンを検索できる。

## 概要

Databricks Japan の YouTube チャンネル動画を処理し、2つの異なる検索モードを提供する:

### 1. Cosmos検索モード
- 動画を30秒セグメントに分割
- NVIDIA Cosmos-Embed1-448p で video-text joint embedding (768次元) を生成
- テキストクエリと動画内容の意味的類似度で検索

### 2. マルチモーダル検索モード
- 動画を5秒セグメントに分割
- **音声文字起こし**: Whisper (base) で日本語音声をテキスト化 → multilingual-e5-large (1024次元) でテキスト embedding
- **画像フレーム**: 各セグメント中央フレームを CLIP (openai/clip-vit-base-patch32, 512次元) で画像 embedding
- **スコア統合**: text_score × 0.6 + image_score × 0.4 で融合ランキング

### アーキテクチャ

```
[UC Volume: 動画ファイル]
        │
        ├──[GPU Job: Cosmos-Embed1 video encoder] → [Delta Table] → [VS Index (768次元)]
        │    └─ クラスター上で直接推論 (Model Serving 非使用: 問題9参照)
        │
        └──[GPU Job: Whisper(ローカル) + CLIP + e5 endpoint呼出し]
               → [Delta Table] → [VS Index (text 1024次元)]
                               → [VS Index (image 512次元)]

[Model Serving Endpoints]
   ├─ cosmos-video-encoder   ※ DEPLOYMENT_FAILED (詳細は問題9) → pipeline 内直接推論に変更
   ├─ multilingual-e5-embedder (GPU_SMALL, scale_to_zero) → テキスト embedding (1024次元) ✅
   └─ clip-encoder           (GPU_SMALL, scale_to_zero) → 画像/テキスト embedding (512次元) ✅

[ユーザー] → [React UI (タブ切替)]
                  │
                  ├─ Cosmos検索     → [cosmos-video-encoder endpoint] → [VS Query]
                  └─ マルチモーダル検索 → [e5 endpoint + clip endpoint] → [VS Query x2] → [スコア統合]
                  │
                  ▼
          [検索結果 + サムネイル + YouTubeリンク + 文字起こし]
```

#### Whisper をローカル実行する理由

Whisper のみ Model Serving を使わずに GPU ジョブ内でローカル実行している。音声ファイルを base64 エンコードすると約 13MB になり、Model Serving の ペイロードサイズ上限 (約 10MB) を超えるため。Whisper の文字起こし結果 (テキスト) は小さいので、downstream の e5 embedding 計算には serving endpoint を使用できる。

## 使用技術

| カテゴリ | 技術 |
|---------|------|
| Video Embedding | [NVIDIA Cosmos-Embed1-448p](https://huggingface.co/nvidia/Cosmos-Embed1-448p) (768次元) |
| 音声文字起こし | OpenAI Whisper (base model, GPU上で直接実行) |
| テキスト Embedding | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (1024次元) |
| 画像 Embedding | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) (512次元) |
| データ基盤 | Databricks Unity Catalog (Delta Table, Volumes) |
| ベクトル検索 | Databricks Vector Search (Delta Sync Index, Cosine similarity) |
| GPU計算 (パイプライン) | Databricks Job Cluster (g4dn.xlarge, SINGLE_USER) |
| GPU計算 (検索クエリ) | Databricks Model Serving (GPU_MEDIUM/GPU_SMALL, scale_to_zero) |
| Backend | FastAPI + Databricks SDK (Python) |
| Frontend | React + Vite |
| デプロイ | Databricks Apps |
| 動画処理 | decord (フレーム抽出), ffmpeg (音声抽出), Pillow (サムネイル) |

## リソース構成

### Databricks ワークスペース

| 項目 | 値 |
|------|-----|
| ワークスペース | `fevm-classic-stable-ytcy.cloud.databricks.com` |
| アプリ URL | `https://multimodal-video-search-7474645908464260.aws.databricksapps.com` |

### Databricks リソース

| リソース | 名前/ID | 用途 |
|---------|---------|------|
| Unity Catalog | `classic_stable_ytcy_catalog.multimodal_video_search` | カタログ・スキーマ |
| Delta Table | `classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings` | Cosmos embedding + メタデータ |
| Delta Table | `classic_stable_ytcy_catalog.multimodal_video_search.multimodal_segments` | 文字起こし + CLIP/e5 embedding |
| UC Volume | `/Volumes/classic_stable_ytcy_catalog/multimodal_video_search/videos` | 動画ファイル格納 |
| UC Volume | `/Volumes/classic_stable_ytcy_catalog/multimodal_video_search/thumbnails` | サムネイル画像格納 |
| UC Volume | `/Volumes/classic_stable_ytcy_catalog/multimodal_video_search/clips` | クリップ動画格納 |
| Vector Search Endpoint | `video-search-endpoint` | ベクトル検索エンドポイント |
| Vector Search Index | `classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings_index` | Cosmos Delta Sync Index (768次元) |
| Vector Search Index | `classic_stable_ytcy_catalog.multimodal_video_search.multimodal_text_index` | テキスト Delta Sync Index (1024次元) |
| Vector Search Index | `classic_stable_ytcy_catalog.multimodal_video_search.multimodal_image_index` | 画像 Delta Sync Index (512次元) |
| Model Serving Endpoint | `cosmos-video-encoder` (GPU_MEDIUM, scale_to_zero) | Cosmos text/video embedding (768次元) |
| Model Serving Endpoint | `multilingual-e5-embedder` (GPU_SMALL, scale_to_zero) | テキスト query embedding (1024次元) |
| Model Serving Endpoint | `clip-encoder` (GPU_SMALL, scale_to_zero) | 画像/テキスト query embedding (512次元) |
| SQL Warehouse | `e351c2d1b16eae95` (Serverless Starter Warehouse) | メタデータクエリ用 |
| Databricks App | `multimodal-video-search` | Webアプリホスティング |
| Databricks Job | `multimodal-video-search-pipeline` | 全ノートブックの実行パイプライン |

### Delta Table スキーマ

#### Cosmos Embedding テーブル

```sql
CREATE TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings (
  video_id STRING,
  segment_id STRING,        -- "{video_id}_seg{NNNN}"
  title STRING,
  channel_name STRING,
  youtube_url STRING,
  start_time DOUBLE,        -- セグメント開始秒
  end_time DOUBLE,          -- セグメント終了秒
  embedding ARRAY<FLOAT>,   -- 768次元ベクトル (Cosmos-Embed1)
  thumbnail_path STRING,
  created_at TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
```

#### マルチモーダルセグメントテーブル

```sql
CREATE TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_segments (
  video_id STRING,
  segment_id STRING,              -- "{video_id}_mm{NNNN}"
  title STRING,
  youtube_url STRING,
  start_time DOUBLE,
  end_time DOUBLE,
  transcript STRING,              -- Whisper文字起こしテキスト
  text_embedding ARRAY<FLOAT>,    -- multilingual-e5-large (1024次元)
  image_embedding ARRAY<FLOAT>,   -- CLIP (512次元)
  thumbnail_path STRING,
  created_at TIMESTAMP
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
```

## ファイル構成

```
multimodal-video-search/
├── README.md
├── databricks.yml                           # Asset Bundle定義 (アプリ + ジョブを一元管理)
├── notebooks/
│   ├── 00_download_videos.py            # YouTube → UC Volume 動画ダウンロード (フォールバック付き)
│   ├── 01_video_embedding_pipeline.py   # Cosmos embedding生成パイプライン (endpoint呼出し)
│   ├── 02_setup_vector_search.py        # Vector Search Index作成 (3インデックス)
│   ├── 03b_deploy_cosmos_video_encoder.py  # Model Serving: cosmos-video-encoder デプロイ
│   ├── 03c_deploy_text_embedder.py         # Model Serving: multilingual-e5-embedder デプロイ
│   ├── 03d_deploy_clip_encoder.py          # Model Serving: clip-encoder デプロイ
│   └── 04_multimodal_pipeline.py        # マルチモーダル: Whisper(ローカル) + endpoint呼出し
└── app/
    ├── app.yaml                         # Databricks Apps設定
    ├── main.py                          # FastAPI Backend
    ├── requirements.txt                 # Python依存パッケージ
    ├── frontend/                        # React Frontend (ソース)
    │   ├── package.json
    │   ├── vite.config.js
    │   ├── index.html
    │   └── src/
    │       ├── App.jsx                  # メイン: タブ切替 + 検索ロジック
    │       ├── main.jsx
    │       └── components/
    │           ├── SearchBar.jsx
    │           ├── ResultGrid.jsx
    │           └── VideoCard.jsx        # transcript表示対応
    └── static/                          # ビルド済みフロントエンド
```

## 実施手順

### 前提条件

- Databricks ワークスペース `fevm-classic-stable-ytcy.cloud.databricks.com` へのアクセス権
- GPU クラスタ (g4dn.xlarge 以上) の利用権限
- `databricks` CLI のインストールと `fevm-classic-stable-ytcy` プロファイルの認証設定

### Step 1: 動画ファイルをUC Volumeにアップロード

```bash
# ローカルで動画をダウンロード (yt-dlp使用)
yt-dlp -f "best[height<=720]" -o "%(id)s.mp4" <YouTube_URL>

# UC Volumeにアップロード
databricks --profile fevm-classic-stable-ytcy fs cp ./TLpGLZkas70.mp4 \
  /Volumes/classic_stable_ytcy_catalog/multimodal_video_search/videos/TLpGLZkas70.mp4
```

### Step 2: Model Serving エンドポイントのデプロイ

`notebooks/03b_deploy_cosmos_video_encoder.py`, `03c_deploy_text_embedder.py`, `03d_deploy_clip_encoder.py` を順に GPU クラスタで実行する。

各スクリプトは HuggingFace からモデルをダウンロードし、MLflow pyfunc として Unity Catalog に登録し、GPU Serving Endpoint を作成する。

| ノートブック | エンドポイント名 | ワークロード | モデル |
|------------|--------------|------------|------|
| `03b_deploy_cosmos_video_encoder.py` | `cosmos-video-encoder` | GPU_MEDIUM | nvidia/Cosmos-Embed1-448p |
| `03c_deploy_text_embedder.py` | `multilingual-e5-embedder` | GPU_SMALL | intfloat/multilingual-e5-large |
| `03d_deploy_clip_encoder.py` | `clip-encoder` | GPU_SMALL | openai/clip-vit-base-patch32 |

エンドポイントの Ready 状態確認まで各スクリプト内で自動待機 (最大 30 分)。

> **注意**: `cosmos-video-encoder` はモデルサイズが大きいため (`~2GB+`)、初回デプロイに 20〜30 分かかる可能性がある。

> **⚠️ GPU ML ランタイムでの pip install について (重要)**  
> `15.4.x-gpu-ml-scala2.12` ランタイムには torch / transformers / mlflow 等が収録済みのため、これらを `%pip install` に含めると **transitive dependency の解決で 40〜90 分ハングする**。各ノートブックの pip install は以下の最小構成に保つこと:
> - `03b`: `%pip install einops` のみ
> - `03c`: `%pip install sentence-transformers --no-deps` のみ
> - `03d`: pip install なし  
> 詳細は [問題 7](#問題-7-gpu-ml-ランタイムで-pip-install-が-4090-分ハングする) を参照。

### Step 3: Cosmos Embedding パイプライン実行

GPU クラスタ (`video-search-gpu`) 上で `notebooks/01_video_embedding_pipeline.py` を実行する。

ノートブックはワークスペースにアップロード済み:
`/Users/yusuke.tsuchiya@databricks.com/multimodal-video-search/notebooks/`

処理内容:
1. UC Volume から動画をローカルにコピー
2. 30秒セグメントに分割、各セグメントから8フレーム均等抽出
3. フレームを JPEG base64 化して `cosmos-video-encoder` endpoint を呼び出し video embedding (768次元) を取得
4. Delta Table に保存
5. サムネイル画像を UC Volume にコピー

### Step 4: マルチモーダルパイプライン実行

GPU クラスタ上で `notebooks/04_multimodal_pipeline.py` を実行する。

処理内容:
1. UC Volume から動画をローカルにコピー
2. ffmpeg で音声を抽出 (16kHz WAV)
3. Whisper (base) で日本語音声を文字起こし (ローカル実行)
4. 5秒セグメントに分割し、対応する文字起こしテキストを割り当て
5. `multilingual-e5-embedder` endpoint でテキスト embedding (1024次元) を取得
6. 各セグメント中央フレームを `clip-encoder` endpoint に送り image embedding (512次元) を取得
7. Delta Table (`multimodal_segments`) に保存
8. サムネイル画像を UC Volume にコピー

> **注意**: Whisper はペイロードサイズ制約 (音声~13MB > Model Serving 10MB 上限) のためローカル実行。CLIP / e5 は embedding 化後のテキスト/画像のみ送信するため serving endpoint を使用できる。

### Step 5: Vector Search Index 作成

`notebooks/02_setup_vector_search.py` を実行する。

作成されるインデックス:
- `video_embeddings_index`: Cosmos embedding (768次元)
- `multimodal_text_index`: テキスト embedding (1024次元)
- `multimodal_image_index`: 画像 embedding (512次元)

初回プロビジョニングには15-30分程度かかる。

### Step 6: アプリとジョブのデプロイ (Databricks Asset Bundle)

アプリとジョブは `databricks.yml` で一元管理されている。変更を反映する際は以下を実行する。

```bash
# フロントエンドをビルド (変更がある場合)
cd app/frontend
npm install && npm run build
cp -r dist/* ../static/
# (変更はGitHub経由でワークスペースのGit Folderに自動同期される)

# バンドル検証
databricks bundle validate -t prod

# アプリ + ジョブを一括デプロイ
databricks bundle deploy -t prod

# アプリ起動 (初回または停止後)
databricks --profile fevm-classic-stable-ytcy apps start multimodal-video-search

# アプリコードを再デプロイ (ソースコード変更時)
databricks --profile fevm-classic-stable-ytcy apps deploy multimodal-video-search \
  --source-code-path "/Workspace/Users/yusuke.tsuchiya@databricks.com/multimodal_video_search/app"

# パイプラインジョブを実行 (必要に応じて)
databricks --profile fevm-classic-stable-ytcy jobs run-now 874014520497877
```

### Step 7: 動作確認

1. ブラウザで `https://multimodal-video-search-7474645908464260.aws.databricksapps.com` にアクセス
2. Databricks OAuth でログイン
3. **Cosmos検索タブ**: テキストで動画内容を検索 (30秒単位)
4. **マルチモーダル検索タブ**: テキストで音声・画像を横断検索 (5秒単位、文字起こし付き)
5. サムネイルクリックで YouTube の該当時刻にジャンプ

## 検索の仕組み

### Cosmos検索
1. ユーザーがテキストクエリを入力
2. `cosmos-video-encoder` Model Serving endpoint がテキスト query embedding (768次元) を返却
3. Vector Search Index に対してコサイン類似度で検索
4. 類似度スコアの高いセグメントを返却

### マルチモーダル検索
1. ユーザーがテキストクエリを入力
2. 2種類のModel Serving endpointを並行呼出し:
   - `multilingual-e5-embedder` → テキスト query embedding (1024次元)
   - `clip-encoder` → 画像検索用 query embedding (512次元)
3. それぞれの Vector Search Index に対してコサイン類似度で検索
4. スコア統合: `combined_score = 0.6 × text_score + 0.4 × image_score`
5. 統合スコア降順でソートし結果を返却
6. 文字起こしテキストのスニペットも結果に含む

## app.yaml 設定

```yaml
command:
  - uvicorn
  - main:app
  - --host
  - "0.0.0.0"
  - --port
  - "8000"

env:
  - name: CATALOG
    value: "classic_stable_ytcy_catalog"
  - name: DATABRICKS_WAREHOUSE_ID
    value: "e351c2d1b16eae95"
  - name: VS_ENDPOINT_NAME
    value: "video-search-endpoint"
  - name: COSMOS_ENDPOINT_NAME
    value: "cosmos-video-encoder"
  - name: TEXT_EMBED_ENDPOINT_NAME
    value: "multilingual-e5-embedder"
  - name: CLIP_ENDPOINT_NAME
    value: "clip-encoder"

resources:
  - name: sql-warehouse
    sql_warehouse:
      id: "e351c2d1b16eae95"
      permission: CAN_USE
  - name: vector-search-endpoint
    vector_search_endpoint:
      name: video-search-endpoint
      permission: CAN_USE
```

## デプロイ時のトラブルシューティング記録

このプロジェクトを実際にデプロイする過程で遭遇した問題とその解決策をまとめる。同じ環境で再デプロイする際の参考にしてほしい。

---

### 問題 1: クラウド環境で YouTube 動画をダウンロードできない

**現象**  
Databricks クラスタ上で yt-dlp を実行すると、全ての YouTube 動画ダウンロードが失敗する。

```
ERROR: Sign in to confirm you're not a bot. This helps protect our community.
```

**原因**  
YouTube がクラウド IP (AWS など) からのアクセスをボットとして検出してブロックする。Databricks クラスタは AWS 上で動作するため、yt-dlp でのダウンロードが一切できない。

**解決策**  
ローカル PC で動画をダウンロードし、Databricks CLI で UC Volume に手動アップロードする。

```bash
# ローカルでダウンロード
yt-dlp -f "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]" \
  --merge-output-format mp4 -o "%(id)s.mp4" <YouTube_URL>

# UC Volume にアップロード
databricks --profile <profile> fs cp ./TLpGLZkas70.mp4 \
  dbfs:/Volumes/classic_stable_ytcy_catalog/multimodal_video_search/videos/TLpGLZkas70.mp4
```

---

### 問題 2: パイプラインジョブが `CREATE CATALOG` で失敗する

**現象**  
`01_video_embedding_pipeline.py` ノートブックが以下のエラーで失敗する。

```
AnalysisException: [INVALID_STATE] Metastore storage root URL does not exist.
```

エラー箇所: `spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")`

**原因**  
`IF NOT EXISTS` を付けていても、対象カタログがすでに存在する場合に Databricks のメタストアがストレージルート URL の検証を試みてエラーになることがある。カタログの初期作成時に設定されたストレージロケーションが期待通りでない場合に発生する。

**解決策**  
カタログの DDL 文を削除する（カタログは事前に UI から作成しておく）。スキーマ・Volume の DDL は `try/except` でラップして既存リソースへの冪等アクセスを保証する。

```python
# 変更前
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.thumbnails")

# 変更後: CREATE CATALOG を削除、残りは try/except でラップ
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
except Exception as e:
    print(f"Schema already exists or error: {e}")
try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.thumbnails")
except Exception as e:
    print(f"Volume already exists or error: {e}")
```

---

### 問題 3: クラスタの `data_security_mode` と Unity Catalog アクセスの両立

**現象と経緯**  
クラスタの `data_security_mode` をめぐって複数の問題が連鎖的に発生した。

**① SINGLE_USER (作成者のみ) → SP がアタッチできない**  
クラスタ作成時のデフォルト設定 `SINGLE_USER` ではクラスタ所有者（`yusuke.tsuchiya@databricks.com`）しかジョブを実行できない。アプリの SP がジョブを投入しようとすると以下のエラーが発生する。

```
PERMISSION_DENIED: Single-user check failed: user '269e8acf-...' attempted to run
a command on single-user cluster, but the single user is 'yusuke.tsuchiya@databricks.com'
```

**② NONE (制限なし) → UC Volume にアクセスできない**  
`data_security_mode: NONE` に変更すると SP はクラスタにアタッチできるが、Unity Catalog の FUSE マウント (`/Volumes/...`) が機能しなくなる。`os.path.exists("/Volumes/...")` が常に `False` を返し、動画ファイルが見つからない。

**③ USER_ISOLATION (共有) → GPU ML ランタイムが非対応**  
複数ユーザーが使え UC にも対応する `USER_ISOLATION` モードを試みると以下のエラーが発生する。

```
INVALID_PARAMETER_VALUE: Spark version 15.4.x-gpu-ml-scala2.12 does not support Table Access Control
```

GPU ML ランタイムは Table ACL (USER_ISOLATION の前提) に非対応。

**解決策**  
クラスタの `existing_cluster_id` を直接指定する代わりに、ジョブに **job_cluster** (ジョブ実行時に自動作成されるクラスタ) を定義する。`data_security_mode: SINGLE_USER` のまま `single_user_name` を未指定にすると、ジョブ実行者のアイデンティティがシングルユーザーとして自動設定される。これにより:

- 人間ユーザーがジョブを手動実行 → その人のクラスタとして UC Volume にアクセス可
- SP がジョブを実行 → SP のクラスタとして UC Volume にアクセス可

```json
// ジョブ定義 (jobs reset API)
{
  "job_clusters": [{
    "job_cluster_key": "gpu_cluster",
    "new_cluster": {
      "spark_version": "15.4.x-gpu-ml-scala2.12",
      "node_type_id": "g4dn.xlarge",
      "num_workers": 0,
      "data_security_mode": "SINGLE_USER"
    }
  }],
  "tasks": [{
    "job_cluster_key": "gpu_cluster",
    ...
  }]
}
```

> **注意**: job_cluster はジョブ実行ごとに新規作成されるため、起動に 10〜15 分かかる。長期運用では既存クラスタを SP 専用の `SINGLE_USER` で作成し、`single_user_name` に SP の application_id を指定する方法も有効。

---

### 問題 4: Vector Search クエリで 403 Forbidden が発生する

**現象**  
アプリが Vector Search インデックスをクエリすると 403 エラーが発生する。

```
403 Client Error: Forbidden for url:
https://.../api/2.0/vector-search/indexes/.../video_embeddings_index/query
```

人間ユーザーのトークンでは同じクエリが 200 で成功する。

**原因と解決策（複数の要因が重なっていた）**

| 原因 | 確認方法 | 対処 |
|-----|---------|------|
| VS エンドポイントへの `CAN_USE` 権限がない | `GET /api/2.0/permissions/vector-search-endpoints/<id>` で SP の権限を確認 | `PATCH /api/2.0/permissions/vector-search-endpoints/<id>` で SP に `CAN_USE` を付与 |
| Unity Catalog の `USE CATALOG` 権限がない | `SHOW GRANTS ON CATALOG <name>` で確認 | `GRANT USE CATALOG ON CATALOG ... TO \`<sp_id>\`` を実行 |
| Unity Catalog の `USE SCHEMA` 権限がない | `SHOW GRANTS ON SCHEMA <name>` で確認 | `GRANT USE SCHEMA ON SCHEMA ... TO \`<sp_id>\`` を実行 |
| Delta Table への `SELECT` 権限がない (`video_embeddings`) | `SHOW GRANTS ON TABLE <name>` で確認 | `GRANT SELECT ON TABLE ... TO \`<sp_id>\`` を実行 |
| Delta Table への `SELECT` 権限がない (`multimodal_segments`) | 同上 | 同上 (初期設定で `video_embeddings` のみ付与していた) |

> **ポイント**: `app.yaml` の `resources` セクションで VS エンドポイントを宣言しても、SP への `CAN_USE` は自動付与されない。デプロイ後に手動で付与する必要がある。

---

### 問題 5: `02_setup_vector_search.py` が `multimodal_text_index` / `multimodal_image_index` を作成しない

**現象**  
Vector Search インデックスが `video_embeddings_index` のみ作成され、マルチモーダル検索で使用する `multimodal_text_index` と `multimodal_image_index` が存在しない。アプリのマルチモーダル検索で 404 エラーが発生する。

**原因**  
`02_setup_vector_search.py` の初期実装が `video_embeddings_index` のみを対象としており、`multimodal_segments` テーブルに対するインデックスが未実装だった。

**解決策**  
ノートブックに3つのインデックスすべての作成ロジックを追加し、各インデックスのソーステーブル行数確認・作成/同期・テスト検索による検証を行うように改修した。

| インデックス名 | ソーステーブル | 次元 | 用途 |
|-------------|------------|-----|-----|
| `video_embeddings_index` | `video_embeddings` | 768 | Cosmos 動画検索 |
| `multimodal_text_index` | `multimodal_segments` | 1024 | 音声文字起こし全文検索 |
| `multimodal_image_index` | `multimodal_segments` | 512 | 画像フレーム類似検索 |

---

### 問題 6: パイプライン実行時にローカルでモデルをロードすると非現実的なメモリ・時間コストがかかる

**現象**  
初期実装では `01_video_embedding_pipeline.py` が Cosmos-Embed1-448p (~2GB) をジョブクラスタ上に直接ロードしていた。また `04_multimodal_pipeline.py` は CLIP と multilingual-e5-large も同様にローカルロードしていた。パイプラインを修正・再実行するたびに毎回モデルのロードが必要で時間がかかる。加えてアプリ (FastAPI) がクエリ時にも同じモデルをロードしようとすると、GPU クラスタが必要になるためアプリの応答性が著しく悪化する。

**解決策**  
モデルを独立した **Model Serving Endpoint** として切り出し、パイプラインとアプリ両方から HTTP で呼び出す設計に変更した。

```
変更前: GPU Job → モデルをローカルロード → embedding生成
変更後: GPU Job → HTTP → Model Serving Endpoint → embedding生成
```

パイプラインノートブックから以下を削除:
- `from transformers import AutoProcessor, AutoModel` (Cosmos)
- `from transformers import CLIPModel, CLIPProcessor` (CLIP)
- `from sentence_transformers import SentenceTransformer` (e5)
- `%pip install torch transformers sentence-transformers ...`

代わりに serving endpoint への HTTP 呼び出しに置き換え:

```python
def compute_video_embedding(frames):
    frames_b64 = [base64.b64encode(...).decode() for frame in frames]
    resp = requests.post(
        f"https://{HOST}/serving-endpoints/cosmos-video-encoder/invocations",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"dataframe_records": [{"frames": frames_b64}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["predictions"]["embedding"][0]
```

> **Whisper の例外**: 音声ファイルを base64 エンコードすると ~13MB になり、Model Serving のペイロード上限 (~10MB) を超える。Whisper のみジョブクラスタ上でローカル実行を継続する。

---

### 問題 7: GPU ML ランタイムで `%pip install` が 40〜90 分ハングする

**現象**  
Model Serving デプロイ用ノートブック (03b/03c/03d) をジョブで実行すると、クラスタは正常起動するがその後 40〜90 分間一切の進捗がなくなる。MLflow の実行記録も登録モデルも生成されない。

```
# 第1バッチ (93分後にキャンセル): torch を明示的にインストール
%pip install mlflow transformers einops pillow torch torchvision

# 第2バッチ (41分後にキャンセル): torch を除外したが同様にハング
%pip install mlflow transformers einops pillow
%pip install mlflow sentence-transformers
```

**原因**  
GPU ML ランタイム (`15.4.x-gpu-ml-scala2.12`) には torch / torchvision がランタイムに同梱されている。ただし pip のパッケージリストに登録されていないケースがあり、`transformers` や `sentence-transformers` をインストールしようとすると pip がこれらの **transitive dependency として torch を PyPI からインストールしようとしてハング**する。

torch を明示的にインストールリストに含めても除外しても、`transformers` や `sentence-transformers` 経由で同じ問題が発生する。

**解決策**  
**ML ランタイムに収録済みのパッケージを `%pip install` リストから完全に除外し、本当に不足しているパッケージのみをインストールする**。torch に transitive dependency を持つパッケージは `--no-deps` フラグでインストールして依存関係の解決をスキップする。

| ノートブック | 変更前 (ハング) | 変更後 (正常) |
|------------|--------------|-------------|
| `03b_deploy_cosmos_video_encoder.py` | `%pip install mlflow transformers einops pillow` | `%pip install einops` |
| `03c_deploy_text_embedder.py` | `%pip install mlflow sentence-transformers` | `%pip install sentence-transformers --no-deps` |
| `03d_deploy_clip_encoder.py` | `%pip install mlflow transformers pillow` | *(pip install なし)* |

変更後: クラスタ起動からモデル登録まで **2 分以内** で完了 (変更前は 40〜90 分以上ハング)。

**GPU ML ランタイム 15.4.x-gpu-ml-scala2.12 に収録済みのパッケージ**:

| 収録済み (インストール不要) | 未収録 (インストール必要) |
|--------------------------|----------------------|
| `torch`, `torchvision` | `einops` |
| `transformers` | `sentence-transformers` |
| `mlflow` (Databricks版) | |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | |
| `pillow`, `requests`, `tqdm` | |
| `huggingface-hub` | |

> **原則**: GPU ML ランタイムを使うノートブックでは `%pip install torch`, `%pip install transformers` などを書いてはいけない。torch に依存するパッケージをインストールする場合は `--no-deps` を付ける。

---

### 問題 8: ノートブック実行結果の確認手段がなく問題の特定が困難

**現象**  
パイプラインジョブが途中で失敗しても、どのステップで何が起きたか把握しにくかった。特に Delta Table に実際にデータが書き込まれているかどうかが確認できなかった。

**解決策**  
各ノートブックの末尾に検証セルを追加した。処理結果が期待通りでない場合に早期に `AssertionError` を発生させて問題箇所を明示する。

```python
# 01_video_embedding_pipeline.py の検証例
row_count = spark.table(TABLE_NAME).count()
assert row_count > 0, f"ERROR: {TABLE_NAME} にデータが存在しません"

sample = spark.table(TABLE_NAME).select("segment_id", "embedding").limit(1).collect()
emb = sample[0]["embedding"]
assert len(emb) == 768, f"ERROR: embedding次元が不正: {len(emb)}"
assert any(v != 0.0 for v in emb), "ERROR: embeddingが全てゼロです"
print("NOTEBOOK 01 VERIFIED OK")
```

### 問題 9: cosmos-video-encoder Model Serving Endpoint が一貫して DEPLOYMENT_FAILED する

**現象**  
`cosmos-video-encoder` Serving Endpoint を GPU_MEDIUM / GPU_SMALL でデプロイしても、全バージョン (v1〜v8) で以下のいずれかに終わる:

| エラー | 原因 |
|--------|------|
| `DEPLOYMENT_ABORTED` | GPU_MEDIUM の ap-northeast-1 リージョン容量不足 |
| `DEPLOYMENT_FAILED: Model server failed to load the model. Exit code 1.` | モデルサーバーがランタイムエラーで終了 |

ログが Serving 環境からアクセス不可のため直接原因の特定は困難だったが、以下の複合要因と考えられる:

1. **trust_remote_code Python ファイルの欠落** (v1〜v3): `model.save_pretrained()` はモデル重みのみ保存し、`trust_remote_code=True` で必要なカスタム `modeling_*.py` は保存しない。Serving 環境には HuggingFace へのインターネットアクセスがないため、起動時にカスタムコードを取得できず失敗。→ `snapshot_download()` に変更して修正。
2. **pip_requirements の依存関係不足** (v4〜v8): Serving 環境は `pip_requirements` から scratch でインストールする。GPU ML ランタイムにはプリインストールされている `accelerate`, `safetensors` 等が Serving 環境では未インストール。→ `pip_requirements` に追加して修正するも依然 DEPLOYMENT_FAILED。
3. **Serving 環境固有の制約**: GPU ML ランタイムの pre-installed packages のバージョンや環境設定と、Serving コンテナの間に非互換がある可能性。ログ取得不可のため詳細不明。

**代替実装: パイプラインクラスター上での直接推論**

Model Serving を使わず、パイプラインジョブのクラスター (GPU ML 15.4.x, g4dn.xlarge) 上で Cosmos モデルを直接ロードして推論する方式に変更した。

```
変更前: 01_video_embedding_pipeline.py → cosmos-video-encoder endpoint → 768次元 embedding
変更後: 01_video_embedding_pipeline.py → snapshot_download + AutoModel.from_pretrained (直接推論) → 768次元 embedding
```

**変更点** (`01_video_embedding_pipeline.py`):

```python
# 変更前: endpoint 呼び出し
def compute_video_embedding(frames):
    frames_b64 = [base64_encode(f) for f in frames]
    resp = requests.post(f"https://{HOST}/serving-endpoints/cosmos-video-encoder/invocations", ...)
    return resp.json()["predictions"]["embedding"][0]

# 変更後: モデル直接推論 (ノートブック起動時に一度だけロード)
# ※ low_cpu_mem_usage=True は使わない (問題10参照)
_cosmos_model = AutoModel.from_pretrained(COSMOS_LOCAL_DIR, trust_remote_code=True).to("cuda", dtype=torch.float16)

def compute_video_embedding(frames):
    batch = np.transpose(np.expand_dims(np.array(frames), 0), (0, 1, 4, 2, 3))
    video_inputs = _cosmos_processor(videos=batch).to("cuda", dtype=torch.float16)
    with torch.no_grad():
        video_emb = _cosmos_model.get_video_embeddings(**video_inputs)
    return video_emb.cpu().float().numpy().flatten().tolist()
```

**メリット**:
- GPU ML ランタイムの安定した依存関係を使用 (serving 環境の非互換問題を回避)
- Serving endpoint の cold start 待ち不要
- ログが通常のノートブック出力として確認可能

**デメリット**:
- パイプライン実行時に初回 `snapshot_download` で ~2.4GB のダウンロードが発生 (約5〜10分)
- パイプライン実行中は GPU メモリをモデルが占有する
- リアルタイム検索クエリ時の Cosmos 検索機能は別途対応が必要 (現状は pipeline で事前計算済みの embedding を使用するため問題なし)

### 問題 10: `low_cpu_mem_usage=True` で Cosmos モデルの `pos_embed` shape mismatch が発生する

**現象**  
`01_video_embedding_pipeline.py` でパイプラインを実行すると、モデルロード時に以下のエラーが発生する:

```
ValueError: Trying to set a tensor of shape torch.Size([1, 257, 1408]) in "pos_embed"
(which has shape torch.Size([1, 1025, 1408])), this look incorrect.
```

**原因**  
`AutoModel.from_pretrained(..., low_cpu_mem_usage=True)` は内部で `accelerate` の `init_empty_weights()` を使用する。これはモデルを**デフォルト config の値**で空初期化してから weights を書き込む方式。

Cosmos-Embed1-448p の ViT backbone は `image_size=448` (32×32 パッチ = 1025 positions) だが、`init_empty_weights` がデフォルト config の `image_size=224` (16×16 パッチ = 257 positions) で `pos_embed` を初期化してしまい、実際の weights (1025 positions) とサイズ不一致になる。

| | `image_size` | pos_embed shape |
|---|---|---|
| デフォルト config (誤) | 224 | `[1, 257, 1408]` (= 1 + 16×16) |
| 実際の 448p weights (正) | 448 | `[1, 1025, 1408]` (= 1 + 32×32) |

**解決策**  
`low_cpu_mem_usage=True` と `torch_dtype` を `from_pretrained` から除去し、ロード後に `.to(device, dtype)` で変換する:

```python
# NG: low_cpu_mem_usage=True → pos_embed shape mismatch
_cosmos_model = AutoModel.from_pretrained(
    COSMOS_LOCAL_DIR, trust_remote_code=True,
    torch_dtype=torch.float16, low_cpu_mem_usage=True  # ← これが原因
).to(_COSMOS_DEVICE)

# OK: 通常ロード後に dtype 変換
_cosmos_model = AutoModel.from_pretrained(
    COSMOS_LOCAL_DIR, trust_remote_code=True,
).to(_COSMOS_DEVICE, dtype=_COSMOS_DTYPE)
```

メモリ使用量は増えるが (一時的に float32 でロード → float16 に変換)、g4dn.xlarge (16GB RAM, 16GB VRAM) では問題なく動作する。

---

## 制限事項・注意点

- Model Serving endpoint は `scale_to_zero_enabled: True` のため、長時間未使用後の初回クエリは cold start が発生し応答に30-60秒かかることがある
- マルチモーダル検索は2つの endpoint を呼び出すため、Cosmos検索より応答時間が長い
- Whisper は Model Serving 非対応 (音声 base64 ~13MB がペイロード上限 ~10MB を超える)。GPU ジョブ内でローカル実行
- YouTube からの動画ダウンロードはクラウド環境ではボット検出でブロックされるため、ローカルでダウンロードして UC Volume に手動アップロードする
- Cosmos-Embed1-448p (GPU_MEDIUM) はモデルサイズが大きく、初回デプロイに20-30分かかる
- Vector Search Index の初回プロビジョニングには15-30分程度かかる
- パイプライン用 GPU ジョブは `job_cluster` (SINGLE_USER) で定義する。g4dn.xlarge (T4 GPU) では bfloat16 が使えないため Cosmos pipeline は float16 を使用
- **GPU ML ランタイムでは `%pip install torch / transformers / mlflow` を書いてはいけない** (transitive 依存解決がハングする)。ランタイム未収録のパッケージのみをインストールし、torch に依存するものは `--no-deps` を使う
- クラスタイベントに `METASTORE_DOWN` が定期的に表示されることがあるが、ドライバ起動後 ~6 分で発生する周期的なヘルスチェックイベントであり致命的ではない。SQL Warehouse 経由でメタストアにアクセスできれば問題ない
- ジョブ実行中の notebook_output は `jobs/runs/get-output` API で取得できない (run 完了後のみ返される)。`execution_duration: 0` も実行中は 0 を返す仕様であり、正常動作の証拠にはならない

## アプリ SP への権限設定 (新規デプロイ時の必須手順)

Databricks Apps の Service Principal は、デフォルトでは各リソースへのアクセス権を持たない。アプリをデプロイした後、以下の権限を手動で付与する必要がある。

> **補足**: アプリの SP の application_id は `databricks apps get <app_name>` の `service_principal_client_id` フィールドで確認できる。

### 1. Vector Search エンドポイントへの CAN_USE 権限 + Unity Catalog 権限

SP が Vector Search インデックスをクエリするには、エンドポイントへの `CAN_USE` 権限に加え、Unity Catalog のカタログ・スキーマ・テーブルへのアクセス権が必要。`app.yaml` の `resources` 宣言だけでは自動付与されない。

```bash
# (a) VS エンドポイントへの CAN_USE
ENDPOINT_ID=$(databricks --profile fevm-classic-stable-ytcy api get \
  "/api/2.0/vector-search/endpoints/video-search-endpoint" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

databricks --profile fevm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/vector-search-endpoints/${ENDPOINT_ID}" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_USE"
  }]
}'

# (b) Unity Catalog 権限 (SQL Warehouse 経由で実行)
databricks --profile fevm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT USE CATALOG ON CATALOG classic_stable_ytcy_catalog TO `<app_sp_application_id>`"
}'
databricks --profile fevm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT USE SCHEMA ON SCHEMA classic_stable_ytcy_catalog.multimodal_video_search TO `<app_sp_application_id>`"
}'
databricks --profile fevm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings TO `<app_sp_application_id>`"
}'
databricks --profile fevm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_segments TO `<app_sp_application_id>`"
}'
```

### 2. Model Serving エンドポイントへの CAN_QUERY 権限

SP が embedding 計算の serving endpoint を呼び出すには `CAN_QUERY` 権限が必要。

```bash
# 各エンドポイントに CAN_QUERY を付与
for ENDPOINT in cosmos-video-encoder multilingual-e5-embedder clip-encoder; do
  databricks --profile fevm-classic-stable-ytcy api patch \
    "/api/2.0/permissions/serving-endpoints/${ENDPOINT}" --json "{
    \"access_control_list\": [{
      \"service_principal_name\": \"<app_sp_application_id>\",
      \"permission_level\": \"CAN_QUERY\"
    }]
  }"
done
```

### 権限設定が不足している場合のエラー

| エラーメッセージ | 原因 | 対処 |
|--------------|------|------|
| `403 Forbidden for url: .../vector-search/indexes/.../query` | SP が Vector Search エンドポイントの CAN_USE 権限なし、または Unity Catalog (USE CATALOG / USE SCHEMA / SELECT) 権限なし | 手順 1 を実施 |
| `403 Forbidden for url: .../serving-endpoints/.../invocations` | SP が Model Serving endpoint の CAN_QUERY 権限なし | 手順 2 を実施 |
| `404 Not Found for url: .../vector-search/indexes/.../query` | Vector Search インデックスが未作成 | Step 5 (Vector Search Index 作成) を実施 |
| `os.path.exists("/Volumes/...")` が常に `False` | `data_security_mode: NONE` クラスタは UC Volume の FUSE マウントを提供しない | パイプラインに `job_cluster` (SINGLE_USER) を使用する |
| `Spark version ... does not support Table Access Control` | GPU ML ランタイムは `USER_ISOLATION` モード非対応 | `SINGLE_USER` + job_cluster で対処 |
