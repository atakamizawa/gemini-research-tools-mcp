# Gemini Research Tools MCP

Gemini API の各種リサーチ機能（Deep Research、Quick Search、URL 分析）を統合し、複数のインターフェースから利用できるようにする MCP サーバーです。訳あってラップして使いたい時にサンプルとして使ってください。

## 🎯 概要

このプロジェクトは、Google Gemini API の 3 つの強力なリサーチ機能を統合して提供します：

| 機能              | 説明                                            | レイテンシ | ユースケース                               |
| ----------------- | ----------------------------------------------- | ---------- | ------------------------------------------ |
| **Deep Research** | Interactions API を使用した包括的な深層リサーチ | 分単位     | 詳細レポート、市場分析、競合調査           |
| **Quick Search**  | Google Search grounding による高速 Web 検索     | 秒単位     | 最新ニュース、ファクトチェック、簡単な質問 |
| **URL 分析**      | URL Context tool による特定 URL の分析・比較    | 秒単位     | 記事比較、ドキュメント要約、コード分析     |

### 提供インターフェース

- **MCP Server** - Cline 等の MCP クライアントからツールとして呼び出し
- **Web UI** - Streamlit ベースのシンプルなウェブインターフェース
- **ADK Tool** - Google ADK エージェントのツールとして利用
- **CLI** - コマンドラインからの実行

## 📋 前提条件

- Python 3.11+
- Gemini API Key ([Google AI Studio](https://aistudio.google.com/app/apikey) で取得)

## 🚀 インストール

```bash
# リポジトリをクローン
git clone https://github.com/atakamizawa/gemini-research-tools-mcp.git
cd gemini-research-tools-mcp

# 仮想環境を作成（推奨）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
.\venv\Scripts\activate  # Windows

# 依存パッケージをインストール
pip install -e ".[all]"

# 環境変数を設定
cp .env.example .env
# .env ファイルを編集して GEMINI_API_KEY を設定
```

## 🔧 使用方法

### 1. MCP Server

MCP クライアント（VS Code GitHub Copilot、Cline 等）からツールとして呼び出せます。

#### 起動方法

| 方法                   | コマンド                                                                  | 用途           |
| ---------------------- | ------------------------------------------------------------------------- | -------------- |
| ローカルクローン       | `grt-mcp`                                                                 | 開発・カスタマイズ |
| GitHub から直接（uvx） | `uvx --from git+https://github.com/atakamizawa/gemini-research-tools-mcp grt-mcp` | クローン不要で即利用 |

#### クライアント設定

<details>
<summary><b>VS Code (GitHub Copilot Chat)</b></summary>

**設定ファイル:** コマンドパレット → `MCP: Open User Configuration` または `.vscode/mcp.json`

```json
{
  "servers": {
    "gemini-research-tools": {
      "command": "grt-mcp",
      "env": { "GEMINI_API_KEY": "your-api-key" }
    }
  }
}
```

uvx を使う場合は `"command": "uvx"` + `"args": ["--from", "git+https://github.com/atakamizawa/gemini-research-tools-mcp", "grt-mcp"]` に変更。

</details>

<details>
<summary><b>Cline</b></summary>

**設定ファイル:**
- Mac/Linux: `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Windows: `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "gemini-research-tools": {
      "command": "grt-mcp",
      "env": { "GEMINI_API_KEY": "your-api-key" }
    }
  }
}
```

uvx を使う場合は `"command": "uvx"` + `"args": ["--from", "git+https://github.com/atakamizawa/gemini-research-tools-mcp", "grt-mcp"]` に変更。

</details>

> **💡 Note:** `.env` ファイルに `GEMINI_API_KEY` を設定している場合、`env` セクションでの指定は不要です。

#### 利用可能なツール

##### 🔬 Deep Research ツール（分単位、包括的なレポート）

Interactions API を使用して、複雑なトピックについて自律的にリサーチを行い、引用付きの詳細レポートを生成します。

| ツール名                | 説明                               | レイテンシ |
| ----------------------- | ---------------------------------- | ---------- |
| `deep_research`         | トピックについて深層リサーチを実行 | 分単位     |
| `get_research_status`   | リサーチの状態を確認               | 秒単位     |
| `get_research_result`   | 完了したリサーチの結果を取得       | 秒単位     |
| `ask_followup_question` | フォローアップ質問                 | 秒単位     |
| `stream_research`       | ストリーミングでリサーチを実行     | 分単位     |

##### 🔍 Quick Search ツール（秒単位、軽量検索）

Google Search grounding を使用して、リアルタイムの Web 情報を高速に取得します。

| ツール名             | 説明                                | レイテンシ |
| -------------------- | ----------------------------------- | ---------- |
| `quick_search`       | Google Search を使った高速 Web 検索 | 秒単位     |
| `analyze_urls`       | 特定 URL の内容を分析・比較         | 秒単位     |
| `search_and_analyze` | Web 検索 + URL 分析のコンボ         | 秒単位     |

#### ツールの使い分けガイド

| 用途                         | 推奨ツール           | 理由                   |
| ---------------------------- | -------------------- | ---------------------- |
| 最新ニュースの確認           | `quick_search`       | 高速、リアルタイム情報 |
| 簡単な質問への回答           | `quick_search`       | 秒単位で回答           |
| 特定記事の要約・比較         | `analyze_urls`       | URL を直接分析         |
| 検索結果と特定 URL の比較    | `search_and_analyze` | 両方の機能を組み合わせ |
| 包括的な市場分析             | `deep_research`      | 複数ソースを統合       |
| 詳細なレポート作成           | `deep_research`      | 引用付きの長文レポート |
| 競合分析・デューデリジェンス | `deep_research`      | 深い分析が必要         |

#### 使用例（Cline）

**Quick Search（高速）:**

```
User: "今日のAI関連ニュースを教えて"

Cline: quick_search ツールを使用します...
[数秒後]
検索結果: 本日のAI関連ニュースをお伝えします...
```

**URL 分析:**

```
User: "この2つの記事を比較して: https://example.com/article1 https://example.com/article2"

Cline: analyze_urls ツールを使用します...
[数秒後]
分析結果: 2つの記事の主な違いは...
```

**Deep Research（包括的）:**

```
User: "量子コンピュータの最新動向について詳しく調べて"

Cline: deep_research ツールを使用します...
[数分後]
リサーチ結果:

# 量子コンピュータの最新動向レポート

## エグゼクティブサマリー
...

## 主要な発見
...

## 引用
[1] https://...
[2] https://...
```

### 2. Web UI

```bash
# Streamlit UIを起動
streamlit run src/ui/app.py
```

ブラウザで `http://localhost:8501` を開きます。

**機能:**

- リサーチクエリの入力
- フォーマット指定
- リアルタイムストリーミング表示
- 状態確認
- フォローアップ質問

### 3. CLI

```bash
# Deep Research を実行
grt research "量子コンピュータの最新動向"

# ストリーミングで実行
grt research "AI trends in 2025" --stream

# フォーマット指定付き
grt research "EV batteries" -f "比較表を含めてください" -o report.md

# 状態確認
grt status <interaction_id>

# 結果取得
grt result <interaction_id>

# フォローアップ質問
grt followup <interaction_id> "主なリスクは何ですか？"
```

### 4. ADK Tool

```python
from google.adk.agents import Agent
from src.adk.tools import deep_research, get_research_status

agent = Agent(
    name="research_assistant",
    model="gemini-3-flash-preview",
    tools=[deep_research, get_research_status],
    instruction="You are a research assistant that can perform deep research."
)
```

## 📁 プロジェクト構造

```
gemini-research-tools-mcp/
├── src/
│   ├── core/           # コアライブラリ
│   │   ├── client.py   # DeepResearchClient, QuickSearchClient
│   │   └── models.py   # Pydanticモデル
│   ├── mcp/            # MCPサーバー
│   │   └── server.py   # FastMCP実装（8ツール提供）
│   ├── adk/            # ADKツール
│   │   └── tools.py    # ADKカスタムツール
│   ├── cli/            # CLIツール
│   │   └── main.py     # Typer CLI
│   └── ui/             # Web UI
│       └── app.py      # Streamlit
├── docs/               # ドキュメント
├── tests/              # テスト
├── pyproject.toml
├── requirements.txt
└── README.md
```

## ⚙️ 設定

### 環境変数

| 変数名           | 必須 | 説明            |
| ---------------- | ---- | --------------- |
| `GEMINI_API_KEY` | ✅   | Gemini API キー |

### モデル選択

| 機能                   | モデル選択 | 使用可能なモデル                                               |
| ---------------------- | ---------- | -------------------------------------------------------------- |
| **Deep Research**      | ❌ 不可    | エージェント `deep-research-pro-preview-12-2025` 固定          |
| **Quick Search**       | ✅ 可能    | `gemini-3-flash-preview`（デフォルト）, `gemini-3-pro-preview` |
| **URL 分析**           | ✅ 可能    | `gemini-3-flash-preview`（デフォルト）, `gemini-3-pro-preview` |
| **search_and_analyze** | ✅ 可能    | `gemini-3-flash-preview`（デフォルト）, `gemini-3-pro-preview` |

## 📝 API リファレンス

### DeepResearchClient

Interactions API を使用した深層リサーチ用クライアント。

```python
from src.core.client import DeepResearchClient

client = DeepResearchClient()

# リサーチを開始して完了まで待機
result = await client.research("Your query")
print(result.content)

# リサーチを開始（非同期）
interaction_id = await client.start_research("Your query")

# 状態確認
status = await client.get_status(interaction_id)

# 結果取得
result = await client.get_result(interaction_id)

# ストリーミング
async for event in client.stream_research("Your query"):
    print(event.content)

# フォローアップ
answer = await client.ask_followup(interaction_id, "Your question")
```

### QuickSearchClient

Google Search grounding と URL Context を使用した高速検索用クライアント。

```python
from src.core.client import QuickSearchClient

client = QuickSearchClient()

# 高速Web検索（Google Search grounding）
result = await client.quick_search("最新のAIニュース")
print(result.content)
for citation in result.citations:
    print(f"- {citation.title}: {citation.url}")

# URL分析（URL Context tool）
result = await client.analyze_urls(
    urls=["https://example.com/article1", "https://example.com/article2"],
    query="これらの記事の主な違いを比較してください"
)
print(result.content)

# 検索 + URL分析のコンボ
result = await client.search_and_analyze(
    query="最新のEV市場動向と、この記事の内容を比較してください",
    urls=["https://example.com/ev-report-2024"]
)
print(result.content)
```

## ⚠️ 注意事項

### Deep Research（Interactions API + Deep Research Agent）

- **実行時間**: 数分〜最大 60 分かかる場合があります
- **コスト**: 1 タスクあたり約$2〜$5（クエリの複雑さによる）
- **制限**: カスタム Function Calling や Remote MCP は非サポート
- **エージェント**: `deep-research-pro-preview-12-2025`（固定、モデル選択不可）
- **API**: `client.interactions.create(agent="deep-research-pro-preview-12-2025")`

### Quick Search（Google Search grounding）

- **実行時間**: 通常数秒で完了
- **コスト**: 標準の Gemini API 料金（トークンベース）
- **モデル**: `gemini-3-flash-preview`（デフォルト）または `gemini-3-pro-preview`
- **API**: `types.Tool(google_search=types.GoogleSearch())`

### URL 分析（URL Context tool）

- **実行時間**: 通常数秒で完了
- **モデル**: `gemini-3-flash-preview`（デフォルト）または `gemini-3-pro-preview`
- **API**: `types.Tool(url_context=types.UrlContext())`
- **制限**:
  - 最大 20 URL/リクエスト
  - 最大 34MB/URL
  - ペイウォール、YouTube 動画、Google Workspace ファイルは非対応

## 🔗 関連リンク

### Gemini API ドキュメント

- [Deep Research Agent](https://ai.google.dev/gemini-api/docs/deep-research) - Interactions API を使用した深層リサーチ
- [Interactions API](https://ai.google.dev/gemini-api/docs/interactions) - 長時間実行タスク用 API
- [Google Search Grounding](https://ai.google.dev/gemini-api/docs/grounding) - リアルタイム Web 検索
- [URL Context](https://ai.google.dev/gemini-api/docs/url-context) - 特定 URL の分析

### ツール

- [Google AI Studio](https://aistudio.google.com/) - API キー取得・テスト
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) - MCP の仕様

## 📄 ライセンス

MIT License
