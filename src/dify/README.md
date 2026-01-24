# Gemini Research Tools - Dify Plugin

Google Gemini API を活用した包括的なリサーチツールを Dify に提供するプラグインです。

## 機能

| ツール名              | 説明                                            | レイテンシ |
| --------------------- | ----------------------------------------------- | ---------- |
| `deep_research`       | Gemini Deep Research Agent による包括的リサーチ | 分単位     |
| `get_research_status` | リサーチタスクの状態確認                        | 秒単位     |
| `get_research_result` | 完了したリサーチの結果取得                      | 秒単位     |
| `quick_search`        | Google Search grounding による高速検索          | 秒単位     |
| `analyze_urls`        | 特定 URL の内容分析                             | 秒単位     |
| `search_and_analyze`  | Web 検索 + URL 分析のコンボ                     | 秒単位     |

## 必要条件

- Python 3.12 以上
- Dify 1.0.0 以上
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/app/apikey) で取得)

## インストール

### 方法 1: パッケージからインストール

1. プラグインをパッケージ化:

   ```bash
   cd src
   dify plugin package ./dify
   ```

2. 生成された `gemini-research.difypkg` を Dify にアップロード

### 方法 2: リモートデバッグ

1. `.env` ファイルを設定:

   ```bash
   cp .env.example .env
   # .env を編集してDifyプラグイン画面から取得した接続先、デバッグキーを設定
   ```

2. プラグインを起動:
   ```bash
   python -m main
   ```

## 使用方法

### Deep Research

長時間かかる包括的なリサーチを実行します。

```
# 非同期実行（推奨）
deep_research(
    query="量子コンピューティングの最新動向",
    wait_for_completion=False
)
# → interaction_id を返す

# 状態確認
get_research_status(interaction_id="...")

# 結果取得
get_research_result(interaction_id="...")
```

### Quick Search

高速な Web 検索を実行します。

```
quick_search(
    query="最新のAIニュース",
    language="ja"
)
```

### URL 分析

特定の URL からコンテンツを分析します。

```
analyze_urls(
    urls="https://example.com/article1, https://example.com/article2",
    query="これらの記事の主な違いを比較してください",
    language="ja"
)
```

### 検索と分析

Web 検索と URL 分析を組み合わせます。

```
search_and_analyze(
    query="最新のEV市場動向と、この記事の内容を比較してください",
    urls="https://example.com/ev-report-2024",
    language="ja"
)
```

## 設定

### Provider 設定

| 設定項目              | 説明                                  | 必須 |
| --------------------- | ------------------------------------- | ---- |
| `gemini_api_key`      | Gemini API キー                       | ✓    |
| `default_quick_model` | Quick Search 系ツールで使用するモデル | -    |

### モデル選択

- **Deep Research**: 専用エージェント（`deep-research-pro-preview-12-2025`）を使用
- **Quick Search 系**: `gemini-3-flash-preview`（高速）または `gemini-3-pro-preview`（高性能）

## ライセンス

MIT License

## 作者

atakamizawa
