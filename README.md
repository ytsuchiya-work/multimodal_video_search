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
        ├──[Job: 01_video_embedding_pipeline] → HTTP → [cosmos-video-encoder endpoint]
        │                                     → [Delta Table] → [VS Index (768次元)]
        │
        └──[GPU Job: 04_multimodal_pipeline]
               Whisper(ローカル) + CLIP endpoint + e5 endpoint呼出し
               → [Delta Table] → [VS Index (text 1024次元)]
                               → [VS Index (image 512次元)]

[Model Serving Endpoints]
   ├─ cosmos-video-encoder   (GPU_MEDIUM, scale_to_zero) → Cosmos video embedding (768次元) ✅
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

### Embedding の対応関係

各インデックスは **構築時と同じモデル・同じ空間** でクエリする必要がある。以下にその対応をまとめる。

#### Cosmos検索

| フェーズ | 対象 | モデル | 次元 | 備考 |
|---------|------|--------|------|------|
| **構築時** (パイプライン) | 動画セグメント (8フレーム/30秒) | Cosmos-Embed1-448p `get_video_embeddings()` | 768 | 映像の視覚的内容をビデオembeddingに変換 |
| **クエリ時** (アプリ) | テキスト検索クエリ | Cosmos-Embed1-448p `get_text_embeddings()` | 768 | 同一モデルの text encoder — ビデオembeddingと同一空間に写像 |

Cosmos-Embed1 は video-text joint embedding モデルであり、**テキストとビデオを同一の embedding 空間に射影する**ように学習されている。クエリ時にテキストを同モデルでエンコードすることで、テキストと動画の意味的類似度が計算できる。

```
[パイプライン]  動画フレーム × 8 → cosmos-video-encoder (get_video_embeddings) → 768次元 → video_embeddings_index
[クエリ]        テキスト         → cosmos-video-encoder (get_text_embeddings)  → 768次元 → コサイン類似度検索
```

#### マルチモーダル検索

| インデックス | 構築時の対象 | 構築時のモデル | クエリ時のモデル | 次元 | 対応の根拠 |
|------------|------------|--------------|--------------|------|-----------|
| `multimodal_text_index` | 音声文字起こし (Whisper) | multilingual-e5-large `encode(transcript)` | multilingual-e5-large `encode(query)` | 1024 | 同一モデル・同一空間 |
| `multimodal_image_index` | 動画フレーム中央1枚 | CLIP `get_image_features(frame)` | CLIP `get_text_features(query)` | 512 | CLIP はテキスト・画像を同一空間に射影するよう学習 → テキストで画像インデックスを検索できる |

```
[パイプライン]  字幕テキスト → multilingual-e5-embedder → 1024次元 → multimodal_text_index
               フレーム画像 → clip-encoder (image)      →  512次元 → multimodal_image_index

[クエリ]        テキスト     → multilingual-e5-embedder → 1024次元 → multimodal_text_index  コサイン類似度
               テキスト     → clip-encoder (text)       →  512次元 → multimodal_image_index コサイン類似度
                                                                    ↓
                                          combined_score = 0.6 × text_score + 0.4 × image_score
```

> **CLIP のクロスモーダル性について**: CLIP (Contrastive Language-Image Pretraining) はテキストエンコーダーと画像エンコーダーを **対照学習** で同一の embedding 空間に揃えるよう訓練したモデル。インデックス構築時は画像エンコーダー、クエリ時はテキストエンコーダーを使うが、両者は同一空間にあるためコサイン類似度の比較が成立する。

---

### Cosmos検索 (フロー)
1. ユーザーがテキストクエリを入力
2. `cosmos-video-encoder` endpoint がテキスト query embedding (768次元) を返却
3. `video_embeddings_index` に対してコサイン類似度で検索
4. 類似度スコアの高いセグメントを返却

### マルチモーダル検索 (フロー)
1. ユーザーがテキストクエリを入力
2. 2種類の embedding endpoint を **並行** 呼び出し:
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
      permission: CAN_MANAGE        # CAN_USE ではインデックスクエリに 403 が発生する (問題13参照)
  - name: cosmos-encoder
    serving_endpoint:
      name: cosmos-video-encoder
      permission: CAN_QUERY
  - name: text-embedder
    serving_endpoint:
      name: multilingual-e5-embedder
      permission: CAN_QUERY
  - name: clip-encoder-resource
    serving_endpoint:
      name: clip-encoder
      permission: CAN_QUERY
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

### 問題 9: cosmos-video-encoder Model Serving Endpoint デプロイの試行錯誤

**経緯**  
`cosmos-video-encoder` Serving Endpoint は v1〜v8 で以下の問題が発生していた:

| エラー | 原因 | 対処 |
|--------|------|------|
| `DEPLOYMENT_ABORTED` | GPU_MEDIUM の ap-northeast-1 リージョン容量不足 | GPU_SMALL に変更 |
| `DEPLOYMENT_FAILED: Exit code 1` | `trust_remote_code` Python ファイルの欠落 | `snapshot_download()` に変更 |
| `DEPLOYMENT_FAILED: Exit code 1` | pip_requirements 不足 (`accelerate`, `safetensors`, `pandas` など) | 追加して解消 |
| Q-Former/BERT 層 float16 エラー | `expected Float but found Half` | `self.dtype = torch.float32` に固定 |

**現在の実装**

上記の修正を全て適用した結果、`cosmos-video-encoder` エンドポイントは **READY** 状態で稼働している。`01_video_embedding_pipeline.py` はエンドポイント経由で推論を行う。

```python
def compute_video_embedding(frames):
    frames_b64 = [base64.b64encode(Image.fromarray(f).tobytes()).decode() for f in frames]
    resp = requests.post(
        f"https://{HOST}/serving-endpoints/cosmos-video-encoder/invocations",
        headers=ENDPOINT_HEADERS,
        json={"dataframe_records": [{"frames": frames_b64}]},
        timeout=120,
    )
    return resp.json()["predictions"]["embedding"][0]
```

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
- パイプライン用ジョブは `job_cluster` (SINGLE_USER, GPU ML 15.4.x, g4dn.xlarge) で定義する。04_multimodal_pipeline で Whisper をローカル実行するため GPU が必要。01_video_embedding_pipeline は cosmos-video-encoder エンドポイントを呼び出すため GPU 不要だが、同じクラスタで実行している
- **GPU ML ランタイムでは `%pip install torch / transformers / mlflow` を書いてはいけない** (transitive 依存解決がハングする)。ランタイム未収録のパッケージのみをインストールし、torch に依存するものは `--no-deps` を使う
- クラスタイベントに `METASTORE_DOWN` が定期的に表示されることがあるが、ドライバ起動後 ~6 分で発生する周期的なヘルスチェックイベントであり致命的ではない。SQL Warehouse 経由でメタストアにアクセスできれば問題ない
- ジョブ実行中の notebook_output は `jobs/runs/get-output` API で取得できない (run 完了後のみ返される)。`execution_duration: 0` も実行中は 0 を返す仕様であり、正常動作の証拠にはならない

## アプリ SP への権限設定 (新規デプロイ時の必須手順)

Databricks Apps の Service Principal は、デフォルトでは各リソースへのアクセス権を持たない。アプリをデプロイした後、以下の権限を手動で付与する必要がある。

> **補足**: アプリの SP の application_id は `databricks apps get <app_name>` の `service_principal_client_id` フィールドで確認できる。

### 1. Vector Search エンドポイント + Unity Catalog 権限 (完全版)

SP が Vector Search インデックスをクエリするには以下の **7つ** の権限が全て必要。`app.yaml` の `resources` 宣言だけでは自動付与されない。

> **重要**: Vector Search の Delta Sync インデックスは Unity Catalog にオブジェクトとして登録される。**ソーステーブルへの SELECT だけでは不足**で、インデックスオブジェクト自体にも `GRANT SELECT` が必要。

```bash
SP_ID="<app_sp_application_id>"
WH_ID="<warehouse_id>"

# (a) VS エンドポイントへの CAN_MANAGE (CAN_USE では 403 が発生する)
ENDPOINT_ID=$(databricks --profile fevm-classic-stable-ytcy api get \
  "/api/2.0/vector-search/endpoints/video-search-endpoint" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

databricks --profile fevm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/vector-search-endpoints/${ENDPOINT_ID}" --json "{
  \"access_control_list\": [{
    \"service_principal_name\": \"${SP_ID}\",
    \"permission_level\": \"CAN_MANAGE\"
  }]
}"

# (b) Unity Catalog 権限 (SQL Warehouse 経由で実行)
# ソーステーブル + VS インデックスオブジェクトの両方に SELECT が必要
for STMT in \
  "GRANT USE CATALOG ON CATALOG classic_stable_ytcy_catalog TO \`${SP_ID}\`" \
  "GRANT USE SCHEMA ON SCHEMA classic_stable_ytcy_catalog.multimodal_video_search TO \`${SP_ID}\`" \
  "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings TO \`${SP_ID}\`" \
  "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_segments TO \`${SP_ID}\`" \
  "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings_index TO \`${SP_ID}\`" \
  "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_text_index TO \`${SP_ID}\`" \
  "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_image_index TO \`${SP_ID}\`"
do
  curl -s -X POST "https://fevm-classic-stable-ytcy.cloud.databricks.com/api/2.0/sql/statements" \
    -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
    -d "{\"warehouse_id\": \"${WH_ID}\", \"statement\": \"${STMT}\", \"wait_timeout\": \"30s\"}"
done
```

### 2. UC Volume への READ/WRITE VOLUME 権限

SP がサムネイル・動画ファイルを読み書きするには、Volume ごとに権限が必要 (テーブルの SELECT とは別)。

```bash
SP_ID="<app_sp_application_id>"
WH_ID="<warehouse_id>"

for STMT in \
  "GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.thumbnails TO \`${SP_ID}\`" \
  "GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.videos     TO \`${SP_ID}\`" \
  "GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.clips      TO \`${SP_ID}\`" \
  "GRANT WRITE VOLUME ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.clips      TO \`${SP_ID}\`"
do
  curl -s -X POST "https://fevm-classic-stable-ytcy.cloud.databricks.com/api/2.0/sql/statements" \
    -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
    -d "{\"warehouse_id\": \"${WH_ID}\", \"statement\": \"${STMT}\", \"wait_timeout\": \"30s\"}"
done
```

### 3. Model Serving エンドポイントへの CAN_QUERY 権限

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
| `403 Forbidden for url: .../vector-search/indexes/.../query` | SP が VS エンドポイントの CAN_MANAGE 権限なし、または UC (USE CATALOG / USE SCHEMA / SELECT) 権限なし | 手順 1 を実施 (CAN_MANAGE が必要、CAN_USE では不足) |
| `Insufficient permissions for UC entity ...<index_name>` | SP が VS インデックス UC オブジェクトの SELECT 権限なし (ソーステーブルの SELECT とは別) | 手順 1 の VS インデックス GRANT を実施 |
| `403 Forbidden for url: .../serving-endpoints/.../invocations` | SP が Model Serving endpoint の CAN_QUERY 権限なし | 手順 3 を実施 |
| サムネイル "No Thumbnail"、動画再生不可 | SP が UC Volume の READ VOLUME / WRITE VOLUME 権限なし (テーブルの SELECT とは別) | 手順 2 を実施 |
| `404 Not Found for url: .../vector-search/indexes/.../query` | Vector Search インデックスが未作成 | Step 5 (Vector Search Index 作成) を実施 |
| `os.path.exists("/Volumes/...")` が常に `False` | `data_security_mode: NONE` クラスタは UC Volume の FUSE マウントを提供しない | パイプラインに `job_cluster` (SINGLE_USER) を使用する |
| `Spark version ... does not support Table Access Control` | GPU ML ランタイムは `USER_ISOLATION` モード非対応 | `SINGLE_USER` + job_cluster で対処 |

---

### 問題 11: Vector Search TRIGGERED インデックスが作成直後に空のまま (自動同期されない)

**現象**  
`02_setup_vector_search.py` でインデックスを作成した直後に検索すると 0 件が返る。Delta Table にはデータが存在しており、インデックスのステータスは `ONLINE` になっているにもかかわらず、`data_array` が常に空。

**原因**  
Vector Search の `TRIGGERED` パイプライン型インデックスは、**作成時に自動同期しない**。インデックスを作成するだけでは Delta Table のデータが読み込まれず、明示的に `/sync` エンドポイントを呼び出すまでインデックスは空のまま。

> `CONTINUOUS` パイプライン型は作成後に自動で同期を開始するが、`TRIGGERED` 型は明示的なトリガーが必要。

**解決策**  
インデックス作成後 (新規・既存どちらも) に必ず `/sync` を呼び出す。新規作成後はインデックスが存在することを確認してから `/sync` を呼び出す。

```python
if index_name not in existing_indexes:
    # インデックス作成 API 呼び出し
    requests.post(f"{base_url}/indexes", ...)
    # 作成完了を待機 (存在確認)
    for i in range(30):
        r = requests.get(f"{base_url}/indexes/{index_name}", headers=headers)
        if r.status_code == 200:
            break
        time.sleep(5)

# 新規・既存に関わらず必ず同期トリガー (TRIGGERED は作成後に自動同期しない)
resp = requests.post(f"{base_url}/indexes/{index_name}/sync", headers=headers)
print(f"同期トリガー: {resp.status_code}")
```

---

### 問題 12: Databricks Apps の 60 秒プロキシタイムアウトで検索が失敗する

**現象**  
GPU エンドポイント (scale_to_zero) へのクエリに cold start が発生すると、検索に 60 秒以上かかる。この場合、フロントエンドが以下のエラーを表示する:

```
検索エラー:
```

(エラーメッセージが空) ブラウザコンソールには `net::ERR_INCOMPLETE_CHUNKED_ENCODING` または接続リセットが記録される。

**原因**  
Databricks Apps のリバースプロキシが **HTTP リクエストを 60 秒で強制切断する**。GPU コールドスタートに 2〜5 分かかることがあるため、同期 HTTP リクエストでは確実にタイムアウトする。また HTTP/2 では `statusText` が空文字になるため、フロントエンドのエラーメッセージが空になる。

**解決策**  
FastAPI の `BackgroundTasks` を使ったポーリング方式に変更する:

1. `POST /api/search` → バックグラウンドタスクを登録して即座に `{task_id}` を返す (< 1秒)
2. フロントエンドが `GET /api/search/result/{task_id}` を 2 秒間隔でポーリング
3. タスク完了時に結果を返す

```python
# Backend (FastAPI)
search_tasks: dict = {}

@app.post("/api/search")
async def search_videos(request: SearchRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    search_tasks[task_id] = {"status": "pending"}
    background_tasks.add_task(_run_cosmos_search, task_id, request.query, request.num_results)
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/search/result/{task_id}")
async def get_search_result(task_id: str):
    result = search_tasks.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result
```

```javascript
// Frontend: ポーリングループ
const { task_id } = await fetch("/api/search", {method:"POST",...}).then(r=>r.json());
for (let i = 0; i < 150; i++) {          // 最大 5 分
    await new Promise(r => setTimeout(r, 2000));
    const data = await fetch(`/api/search/result/${task_id}`).then(r=>r.json());
    if (data.status === "done") { setResults(data.results); return; }
    if (data.status === "error") { setError(data.error); return; }
}
```

> **補足**: GPU コールドスタートの進行状況をユーザーに伝えるため、ポーリング経過時間に応じて「接続中...」→「embedding計算中...」→「GPUコールドスタート中...」と段階的なメッセージを表示するとよい。

---

### 問題 13: Vector Search インデックスクエリに `CAN_USE` では 403、`CAN_MANAGE` が必要

**現象**  
`app.yaml` で VS エンドポイントを `permission: CAN_USE` で宣言し、SP に `CAN_USE` を手動付与しても、インデックスクエリで 403 が発生し続ける。

```
403 Client Error: Forbidden for url:
https://.../api/2.0/vector-search/indexes/.../multimodal_text_index/query
```

同じ操作を人間ユーザーのトークンで実行すると成功する。

**原因**  
Databricks Vector Search のインデックスクエリ (`/indexes/.../query`) には、VS エンドポイントへの **`CAN_MANAGE`** 権限が必要。`CAN_USE` ではクエリが拒否される (ドキュメントでは `CAN_USE` で十分とも読めるが、実際には不足する)。

さらに、Databricks SDK (`WorkspaceClient`) を使ったクエリは認証トークンの伝播方法が REST 直接呼び出しと異なるため、SDK 経由の方が App SP の認証に成功しやすい。

**解決策**

1. `app.yaml` の VS エンドポイントを `CAN_MANAGE` に変更:

```yaml
- name: vector-search-endpoint
  vector_search_endpoint:
    name: video-search-endpoint
    permission: CAN_MANAGE    # CAN_USE では 403 が発生する
```

2. VS インデックスクエリを REST 直接呼び出しから Databricks SDK に切り替える:

```python
# NG: REST 直接呼び出し (SP 認証で 403 が発生しやすい)
resp = requests.post(
    f"{DATABRICKS_HOST}/api/2.0/vector-search/indexes/{index_name}/query",
    headers=get_db_headers(),
    json={"columns": [...], "query_vector": embedding, "num_results": n},
)

# OK: Databricks SDK 経由
result = w.vector_search_indexes.query_index(
    index_name=index_name,
    columns=[...],
    query_vector=embedding,
    num_results=n,
)
rows = (result.result.data_array or []) if result.result else []
```

3. SP への `CAN_MANAGE` を手動付与:

```bash
ENDPOINT_ID=$(databricks --profile fevm-classic-stable-ytcy api get \
  "/api/2.0/vector-search/endpoints/video-search-endpoint" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

databricks --profile fevm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/vector-search-endpoints/${ENDPOINT_ID}" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_MANAGE"
  }]
}'
```

---

### 問題 14: Cosmos テキスト embedding で `got multiple values for keyword argument 'padding'`

**現象**  
Cosmos検索でクエリを送信すると以下のエラーが発生する:

```
400 Bad Request: Encountered an unexpected error while evaluating the model.
TypeError: got multiple values for keyword argument 'padding'
  at preprocessing_embed1.py, line 75, in __call__
      tokenized = self.tokenizer(...)
```

**原因**  
`CosmosVideoEncoder._embed_text` 内で `self.processor(text=[text], return_tensors="pt", padding=True)` を呼び出していたが、Cosmos-Embed1 のカスタムプロセッサ `preprocessing_embed1.py` は `__call__` 内部で `padding` キーワードを既にトークナイザーに渡している。そこへ呼び出し側からも `padding=True` を渡すため、**同じキーワード引数が二重に渡されて TypeError** が発生する。

```python
# NG: padding=True を渡すと preprocessing_embed1.py 内部の padding と競合
text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)

# OK: padding は processor 内部に任せる
text_inputs = self.processor(text=[text], return_tensors="pt")
```

**解決策**  
`notebooks/03b_deploy_cosmos_video_encoder.py` の `_embed_text` メソッドから `padding=True` を削除し、cosmos-video-encoder エンドポイントを再デプロイする。

```python
def _embed_text(self, text):
    import torch
    text_inputs = self.processor(text=[text], return_tensors="pt").to(self.device)  # padding=True を削除
    with torch.no_grad():
        text_emb = self.model.get_text_embeddings(**text_inputs)
    ...
```

再デプロイは `03b_deploy_cosmos_video_encoder.py` の Cell 8 以降を再実行すれば新バージョンが登録されてエンドポイントが自動更新される。

**CLI から runs/submit で再デプロイする場合:**

```bash
TOKEN=$(databricks auth token --profile fevm-classic-stable-ytcy \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -s -X POST "https://fevm-classic-stable-ytcy.cloud.databricks.com/api/2.1/jobs/runs/submit" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{
    "run_name": "cosmos-redeploy",
    "tasks": [{
      "task_key": "deploy",
      "notebook_task": {
        "notebook_path": "/Users/yusuke.tsuchiya@databricks.com/multimodal_video_search/notebooks/03b_deploy_cosmos_video_encoder"
      },
      "new_cluster": {
        "num_workers": 0,
        "spark_version": "15.4.x-gpu-ml-scala2.12",
        "node_type_id": "g4dn.xlarge",
        "aws_attributes": {"availability": "ON_DEMAND"}
      }
    }]
  }'
```

> **注意**: `notebook_path` は `/Repos/...` ではなく `/Users/<email>/...` を使う。Git Folder のノートブックは `/Users/<email>/...` 配下に配置されるため、`/Repos/...` パスは存在しない。

---

### 問題 15: Databricks SDK の `query_index()` で `Insufficient permissions for UC entity <index_name>`

**現象**  
`w.vector_search_indexes.query_index()` (Databricks SDK) でマルチモーダル検索を実行すると以下のエラーが発生する:

```
Insufficient permissions for UC entity
classic_stable_ytcy_catalog.multimodal_video_search.multimodal_text_index.
Config: ..., auth_type=oauth-m2m, client_id=<app_sp_id>
```

ソーステーブル (`multimodal_segments`) への SELECT は付与済み、VS エンドポイントへの CAN_MANAGE も付与済みにもかかわらず発生する。

**原因**  
Databricks Vector Search の Delta Sync インデックスは **Unity Catalog に独立したオブジェクトとして登録される**。SDK の `query_index()` はこの UC オブジェクトに対して権限チェックを行うため、ソーステーブルとは別に **インデックスオブジェクト自体への `SELECT`** が必要となる。

```
必要な権限の全体像:
  VS エンドポイント  →  CAN_MANAGE (CAN_USE では不足)
  UC カタログ        →  USE CATALOG
  UC スキーマ        →  USE SCHEMA
  ソーステーブル     →  SELECT (multimodal_segments, video_embeddings)
  VS インデックス    →  SELECT (← これが抜けると "Insufficient permissions for UC entity")
```

**解決策**  
3つの VS インデックスオブジェクトに `GRANT SELECT` を付与する:

```sql
GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings_index   TO `<app_sp_id>`;
GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_text_index    TO `<app_sp_id>`;
GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_image_index   TO `<app_sp_id>`;
```

VS インデックスは UC 上では `TABLE` と同じ GRANT 構文で権限付与できる (`GRANT SELECT ON TABLE <index_name>`)。

---

### 問題 16: サムネイル・動画が表示/再生できない (UC Volume への READ VOLUME 権限不足)

**現象**  
検索結果カードに "No Thumbnail" が表示される。動画ストリーミングやクリップ作成も失敗する。テーブルへの SELECT 権限は付与済みにもかかわらず発生する。

**原因**  
Unity Catalog の **Volume は Delta Table とは独立した UC オブジェクト**であり、`GRANT SELECT ON TABLE` の権限はVolumeには適用されない。SP がファイル API (`/api/2.0/fs/files/Volumes/...`) で Volume 内のファイルを読み書きするには、**Volume への `READ VOLUME` / `WRITE VOLUME`** が別途必要。

```
必要な権限の全体像 (Volume 関連):
  thumbnails Volume → READ VOLUME  (サムネイル表示)
  videos Volume     → READ VOLUME  (動画ストリーミング・クリップ切り出し)
  clips Volume      → READ VOLUME + WRITE VOLUME  (クリップDL・アップロード)
```

アプリ側で Volume アクセスに失敗しても例外は `404` として握り潰されるため、ブラウザには "No Thumbnail" や "動画が見つかりません" として表示される。

**解決策**  
各 Volume に権限を付与する:

```sql
GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.thumbnails TO `<app_sp_id>`;
GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.videos     TO `<app_sp_id>`;
GRANT READ VOLUME  ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.clips      TO `<app_sp_id>`;
GRANT WRITE VOLUME ON VOLUME classic_stable_ytcy_catalog.multimodal_video_search.clips      TO `<app_sp_id>`;
```

> **補足**: Volume 権限はアプリの再デプロイなしに即時反映される。
