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

### 問題 6: ノートブック実行結果の確認手段がなく問題の特定が困難

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

---

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

### 2. Vector Search エンドポイントへの CAN_USE 権限 + Unity Catalog 権限

SP が Vector Search インデックスをクエリするには、エンドポイントへの `CAN_USE` 権限に加え、Unity Catalog のカタログ・スキーマ・テーブルへのアクセス権が必要。`app.yaml` の `resources` 宣言だけでは自動付与されない。

```bash
# (a) VS エンドポイントへの CAN_USE
ENDPOINT_ID=$(databricks --profile fe-vm-classic-stable-ytcy api get \
  "/api/2.0/vector-search/endpoints/video-search-endpoint" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

databricks --profile fe-vm-classic-stable-ytcy api patch \
  "/api/2.0/permissions/vector-search-endpoints/${ENDPOINT_ID}" --json '{
  "access_control_list": [{
    "service_principal_name": "<app_sp_application_id>",
    "permission_level": "CAN_USE"
  }]
}'

# (b) Unity Catalog 権限 (SQL Warehouse 経由で実行)
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT USE CATALOG ON CATALOG classic_stable_ytcy_catalog TO `<app_sp_application_id>`"
}'
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT USE SCHEMA ON SCHEMA classic_stable_ytcy_catalog.multimodal_video_search TO `<app_sp_application_id>`"
}'
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.video_embeddings TO `<app_sp_application_id>`"
}'
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/sql/statements" --json '{
  "warehouse_id": "<warehouse_id>",
  "statement": "GRANT SELECT ON TABLE classic_stable_ytcy_catalog.multimodal_video_search.multimodal_segments TO `<app_sp_application_id>`"
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

### 4. GPU クラスタの data_security_mode

`SINGLE_USER` / `USER_ISOLATION` / `NONE` はそれぞれ制約があり、SP・Unity Catalog・GPU ML ランタイムの3つを同時に満たすには工夫が必要（詳細は「トラブルシューティング記録 問題 3」参照）。

推奨: **パイプライン実行にはジョブ定義の `job_cluster` を使用する。** `data_security_mode: SINGLE_USER` のまま `single_user_name` を指定しないと、実行者のアイデンティティが自動設定されるため UC Volume アクセスと多ユーザー実行が両立できる。

アプリからリアルタイムに呼び出す既存クラスタ (`0602-041404-xp8crh90`) は SP 専用として `SINGLE_USER` + `single_user_name: <sp_application_id>` で設定する。

```bash
# 既存クラスタを SP 専用 SINGLE_USER に変更する場合
databricks --profile fe-vm-classic-stable-ytcy api post "/api/2.0/clusters/edit" --json '{
  "cluster_id": "<cluster_id>",
  "data_security_mode": "SINGLE_USER",
  "single_user_name": "<app_sp_application_id>"
}'
```

### 権限設定が不足している場合のエラー

| エラーメッセージ | 原因 | 対処 |
|--------------|------|------|
| `Unable to access the notebook ... lacks the required permissions` | ノートブックディレクトリへの権限なし | 手順 1 を実施 |
| `403 Forbidden for url: .../vector-search/indexes/.../query` | SP が Vector Search エンドポイントの CAN_USE 権限なし、または Unity Catalog (USE CATALOG / USE SCHEMA / SELECT) 権限なし | 手順 2 を実施 |
| `job run-as ... lacks 'Attach' permissions on the underlying cluster` | GPU クラスタへの Attach 権限なし | 手順 3 を実施 |
| `Single-user check failed: user '...' attempted to run a command on single-user cluster` | クラスタが SINGLE_USER モードで作成されており SP が実行不可 | 手順 4 を実施: パイプラインには `job_cluster` を使用、またはクラスタの `single_user_name` を SP の application_id に変更 |
| `os.path.exists("/Volumes/...")` が常に `False` | `data_security_mode: NONE` クラスタは UC Volume の FUSE マウントを提供しない | `data_security_mode: SINGLE_USER` に変更し UC を有効化する |
| `Spark version ... does not support Table Access Control` | GPU ML ランタイム (`15.4.x-gpu-ml-scala2.12`) は `USER_ISOLATION` モード非対応 | `SINGLE_USER` + job_cluster で対処 |
| `404 Not Found for url: .../vector-search/indexes/.../query` | Vector Search インデックスが未作成 | Step 4 (Vector Search Index 作成) を実施 |
