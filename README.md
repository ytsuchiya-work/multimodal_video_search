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
        ├──[GPU: Cosmos-Embed1 embedding] → [Delta Table] → [Vector Search Index (768次元)]
        │
        └──[GPU: Whisper + CLIP + e5]     → [Delta Table] → [VS Index (text 1024次元)]
                                                           → [VS Index (image 512次元)]

[ユーザー] → [React UI (タブ切替)]
                  │
                  ├─ Cosmos検索 → [GPU: Cosmos text encoder] → [VS Query]
                  └─ マルチモーダル検索 → [GPU: e5 + CLIP text encoder] → [VS Query x2] → [スコア統合]
                  │
                  ▼
          [検索結果 + サムネイル + YouTubeリンク + 文字起こし]
```

## 使用技術

| カテゴリ | 技術 |
|---------|------|
| Video Embedding | [NVIDIA Cosmos-Embed1-448p](https://huggingface.co/nvidia/Cosmos-Embed1-448p) (768次元) |
| 音声文字起こし | OpenAI Whisper (base model, GPU上で直接実行) |
| テキスト Embedding | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (1024次元) |
| 画像 Embedding | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) (512次元) |
| データ基盤 | Databricks Unity Catalog (Delta Table, Volumes) |
| ベクトル検索 | Databricks Vector Search (Delta Sync Index, Cosine similarity) |
| GPU計算 | Databricks Cluster (g4dn.xlarge) + Jobs Run Submit API |
| Backend | FastAPI + Databricks SDK (Python) |
| Frontend | React + Vite |
| デプロイ | Databricks Apps |
| 動画処理 | decord (フレーム抽出), ffmpeg (音声抽出), Pillow (サムネイル) |

## リソース構成

### Databricks ワークスペース

| 項目 | 値 |
|------|-----|
| ワークスペース | `fevm-classic-stable-ytcy.cloud.databricks.com` |
| アプリ URL | `https://video-search-cosmos-7474645908464260.aws.databricksapps.com` |

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
| GPU Cluster | `0602-041404-xp8crh90` (video-search-gpu, g4dn.xlarge) | Embedding計算用 |
| SQL Warehouse | `e351c2d1b16eae95` (Serverless Starter Warehouse) | メタデータクエリ用 |
| Databricks App | `video-search-cosmos` | Webアプリホスティング |

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
├── databricks.yml                           # Asset Bundle定義
├── notebooks/
│   ├── 01_video_embedding_pipeline.py   # Cosmos embedding生成パイプライン
│   ├── 02_setup_vector_search.py        # Vector Search Index作成
│   ├── 03_deploy_text_encoder.py        # (オプション) Model Servingデプロイ
│   └── 04_multimodal_pipeline.py        # マルチモーダル: Whisper + CLIP + e5
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
- `databricks` CLI のインストールと `fe-vm-classic-stable-ytcy` プロファイルの認証設定

### Step 1: 動画ファイルをUC Volumeにアップロード

```bash
# ローカルで動画をダウンロード (yt-dlp使用)
yt-dlp -f "best[height<=720]" -o "%(id)s.mp4" <YouTube_URL>

# UC Volumeにアップロード
databricks --profile fe-vm-classic-stable-ytcy fs cp ./TLpGLZkas70.mp4 \
  /Volumes/classic_stable_ytcy_catalog/multimodal_video_search/videos/TLpGLZkas70.mp4
```

### Step 2: Cosmos Embedding パイプライン実行

GPU クラスタ (`video-search-gpu`) 上で `notebooks/01_video_embedding_pipeline.py` を実行する。

ノートブックはワークスペースにアップロード済み:
`/Users/yusuke.tsuchiya@databricks.com/multimodal-video-search/notebooks/`

処理内容:
1. UC Volume から動画をローカルにコピー
2. 30秒セグメントに分割、各セグメントから8フレーム均等抽出
3. Cosmos-Embed1-448p で video embedding (768次元) を計算
4. Delta Table に保存
5. サムネイル画像を UC Volume にコピー

### Step 3: マルチモーダルパイプライン実行

GPU クラスタ上で `notebooks/04_multimodal_pipeline.py` を実行する。

処理内容:
1. UC Volume から動画をローカルにコピー
2. ffmpeg で音声を抽出 (16kHz WAV)
3. Whisper (base) で日本語音声を文字起こし
4. 5秒セグメントに分割し、対応する文字起こしテキストを割り当て
5. multilingual-e5-large でテキスト embedding (1024次元) を生成
6. 各セグメント中央フレームから CLIP image embedding (512次元) を生成
7. Delta Table (`multimodal_segments`) に保存
8. サムネイル画像を UC Volume にコピー

> **注意**: g4dn.xlarge (T4 GPU) を使用する場合は `torch.float16` を指定する (bfloat16非対応のため)。

### Step 4: Vector Search Index 作成

`notebooks/02_setup_vector_search.py` を実行する。

作成されるインデックス:
- `video_embeddings_index`: Cosmos embedding (768次元)
- `multimodal_text_index`: テキスト embedding (1024次元)
- `multimodal_image_index`: 画像 embedding (512次元)

初回プロビジョニングには15-30分程度かかる。

### Step 5: アプリのデプロイ

```bash
# フロントエンドをビルド (変更がある場合)
cd app/frontend
npm install && npm run build
cp -r dist/* ../static/

# ワークスペースにアップロード
databricks --profile fe-vm-classic-stable-ytcy workspace import-dir ./app \
  /Users/yusuke.tsuchiya@databricks.com/multimodal-video-search/app --overwrite

# Databricks App をデプロイ
databricks --profile fe-vm-classic-stable-ytcy apps deploy video-search-cosmos \
  --source-code-path "/Workspace/Users/yusuke.tsuchiya@databricks.com/multimodal-video-search/app"
```

### Step 6: 動作確認

1. ブラウザで `https://video-search-cosmos-7474645908464260.aws.databricksapps.com` にアクセス
2. Databricks OAuth でログイン
3. **Cosmos検索タブ**: テキストで動画内容を検索 (30秒単位)
4. **マルチモーダル検索タブ**: テキストで音声・画像を横断検索 (5秒単位、文字起こし付き)
5. サムネイルクリックで YouTube の該当時刻にジャンプ

## 検索の仕組み

### Cosmos検索
1. ユーザーがテキストクエリを入力
2. GPU 上で Cosmos-Embed1 のテキストエンコーダーが query embedding (768次元) を生成
3. Vector Search Index に対してコサイン類似度で検索
4. 類似度スコアの高いセグメントを返却

### マルチモーダル検索
1. ユーザーがテキストクエリを入力
2. GPU 上で2種類の embedding を並行計算:
   - multilingual-e5-large でテキスト query embedding (1024次元)
   - CLIP text encoder で画像検索用 query embedding (512次元)
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
  - name: GPU_CLUSTER_ID
    value: "0602-041404-xp8crh90"

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

## 制限事項・注意点

- テキスト embedding 計算は GPU クラスタを使用するため、クラスタが停止している場合は検索に失敗する (アプリ上でクラスタ起動ボタンあり)
- 初回検索時はモデルロードに30-60秒程度かかる
- マルチモーダル検索は2つの embedding を計算するため、Cosmos検索より応答時間が長い
- YouTube からの動画ダウンロードはクラウド環境ではボット検出でブロックされるため、ローカルでダウンロードして UC Volume に手動アップロードする
- g4dn.xlarge (T4 GPU) では bfloat16 が使えないため float16 を使用 (A10G/A100 では bfloat16 可)
- Vector Search Index の初回プロビジョニングには15-30分程度かかる
- GPU クラスタは Service Principal の single-user モードで動作し、アプリの SP が embedding 計算ジョブを実行する

## アプリ SP への権限設定 (新規デプロイ時の必須手順)

Databricks Apps の Service Principal は、デフォルトでは各リソースへのアクセス権を持たない。アプリをデプロイした後、以下の権限を手動で付与する必要がある。

### 1. ノートブックディレクトリへの CAN_MANAGE 権限

アプリは起動時に embedding 計算用ノートブックを `/Workspace/Users/<user>/video-search-cosmos/` に動的作成する。SP がそのディレクトリを読み書きできるよう権限を付与する。

```bash
# ディレクトリの object_id を取得
databricks --profile fe-vm-classic-stable-ytcy api get \
  "/api/2.0/workspace/get-status?path=/Users/yusuke.tsuchiya@databricks.com/video-search-cosmos"

# CAN_MANAGE を付与 (<directory_object_id> は上記で取得した object_id)
databricks --profile fe-vm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/directories/<directory_object_id>" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_MANAGE"
  }]
}'
```

> **補足**: アプリの SP の application_id は `databricks apps get <app_name>` の `service_principal_client_id` フィールドで確認できる。

### 2. Vector Search エンドポイントへの CAN_USE 権限

SP が Vector Search インデックスをクエリするには、エンドポイントへの `CAN_USE` 権限が必要。`app.yaml` の `resources` 宣言だけでは自動付与されない場合があるため、明示的に付与する。

```bash
# エンドポイントの ID を取得
ENDPOINT_ID=$(databricks --profile fe-vm-classic-stable-ytcy api get \
  "/api/2.0/vector-search/endpoints/video-search-endpoint" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# CAN_USE を付与
databricks --profile fe-vm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/vector-search-endpoints/${ENDPOINT_ID}" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_USE"
  }]
}'
```

### 3. GPU クラスタへの CAN_ATTACH_TO 権限

アプリは Jobs Run Submit API を使って SP の権限で GPU クラスタにジョブを投入する。SP がクラスタにアタッチできるよう権限を付与する。

```bash
databricks --profile fe-vm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/clusters/<cluster_id>" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_ATTACH_TO"
  }]
}'
```

> **補足**: クラスタを起動する場合は `CAN_RESTART`、クラスタ設定も変更する場合は `CAN_MANAGE` が必要。

### 権限設定が不足している場合のエラー

### 3. GPU クラスタの data_security_mode

GPU クラスタを `SINGLE_USER` モードで作成すると、作成者のみが実行できる制約がかかる。SP が Jobs Run Submit でジョブを投入できるよう、クラスタは `NONE`（アイソレーションなし）モードで作成・設定する。

```bash
# クラスタ作成時に data_security_mode: "NONE" を指定
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/clusters/create" --json '{
  "cluster_name": "video-search-gpu",
  "spark_version": "15.4.x-gpu-ml-scala2.12",
  "node_type_id": "g4dn.xlarge",
  "num_workers": 0,
  "data_security_mode": "NONE",
  "spark_conf": {"spark.databricks.cluster.profile": "singleNode", "spark.master": "local[*]"},
  "custom_tags": {"ResourceClass": "SingleNode"},
  "autotermination_minutes": 60
}'

# 既存クラスタを変更する場合
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/clusters/edit" --json '{
  "cluster_id": "<cluster_id>",
  ...,
  "data_security_mode": "NONE"
}'
```

### 権限設定が不足している場合のエラー

| エラーメッセージ | 原因 | 対処 |
|--------------|------|------|
| `Unable to access the notebook ... lacks the required permissions` | ノートブックディレクトリへの権限なし | 手順 1 を実施 |
| `job run-as ... lacks 'Attach' permissions on the underlying cluster` | GPU クラスタへの Attach 権限なし | 手順 2 を実施 |
| `Single-user check failed: user '...' attempted to run a command on single-user cluster` | クラスタが SINGLE_USER モードで作成されており SP が実行不可 | 手順 3 を実施: クラスタを `data_security_mode: NONE` に変更 |
| `404 Not Found for url: .../vector-search/indexes/.../query` | Vector Search インデックスが未作成 | Step 4 (Vector Search Index 作成) を実施 |
| `403 Forbidden for url: .../vector-search/indexes/.../query` | SP が Vector Search エンドポイントへの CAN_USE 権限なし | 手順 2 を実施 |
