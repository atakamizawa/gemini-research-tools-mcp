"""ADK Tools for Gemini Deep Research Agent.

These tools can be used with Google ADK (Agent Development Kit) to enable
LLM agents to perform deep research tasks and quick searches.

Usage with ADK:
    ```python
    from google.adk.agents import Agent
    from src.adk.tools import (
        deep_research,
        get_research_status,
        quick_search,
        analyze_urls,
    )

    agent = Agent(
        name="research_assistant",
        model="gemini-3-flash-preview",
        tools=[deep_research, get_research_status, quick_search, analyze_urls],
        instruction="You are a research assistant that can perform deep research and quick searches."
    )
    ```
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from src.core.client import DeepResearchClient, QuickSearchClient
from src.core.models import ResearchStatusEnum

# Load environment variables
load_dotenv()

# Lazy initialization of clients
_deep_research_client: Optional[DeepResearchClient] = None
_quick_search_client: Optional[QuickSearchClient] = None

# HTTP timeout settings (in seconds)
# Note: Gemini API requires minimum 10 seconds deadline
DEEP_RESEARCH_HTTP_TIMEOUT = 3600  # 60 minutes for long-running research
QUICK_SEARCH_HTTP_TIMEOUT = 300  # 5 minutes for quick search operations


def _get_deep_research_client() -> DeepResearchClient:
    """Get or create the DeepResearchClient instance."""
    global _deep_research_client
    if _deep_research_client is None:
        _deep_research_client = DeepResearchClient(http_timeout=DEEP_RESEARCH_HTTP_TIMEOUT)
    return _deep_research_client


def _get_quick_search_client() -> QuickSearchClient:
    """Get or create the QuickSearchClient instance."""
    global _quick_search_client
    if _quick_search_client is None:
        _quick_search_client = QuickSearchClient(http_timeout=QUICK_SEARCH_HTTP_TIMEOUT)
    return _quick_search_client


# Backward compatibility alias
def _get_client() -> DeepResearchClient:
    """Get or create the DeepResearchClient instance (backward compatibility)."""
    return _get_deep_research_client()


def _run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


# =============================================================================
# Deep Research Tools (Minutes to complete, comprehensive reports)
# =============================================================================


def deep_research(
    query: str,
    format_instructions: Optional[str] = None,
    timeout: int = 3600,
) -> dict:
    """Deep Research を実行する。

    Gemini Deep Research Agentを使用して、指定されたトピックについて
    詳細なリサーチを行い、引用付きのレポートを生成します。

    **注意**: この処理には数分〜数十分かかる場合があります。

    Args:
        query: リサーチクエリ（例: "量子コンピュータの最新動向"）
        format_instructions: 出力フォーマット指定（オプション）
            例: "技術レポート形式で、概要、主要な発見、詳細分析、結論のセクションを含めてください"
        timeout: タイムアウト秒数（デフォルト: 3600秒 = 1時間）

    Returns:
        リサーチ結果を含む辞書:
        - interaction_id: インタラクションID
        - status: "completed" | "failed"
        - content: リサーチ結果のテキスト
        - citations: 引用元のリスト
        - error: エラーメッセージ（失敗時）

    Example:
        >>> result = deep_research(
        ...     "電気自動車バッテリーの競争環境",
        ...     format_instructions="比較表を含めてください"
        ... )
        >>> print(result["content"])
    """
    client = _get_deep_research_client()

    async def _execute():
        try:
            result = await client.research(
                query=query,
                format_instructions=format_instructions,
                timeout=timeout,
            )
            return {
                "interaction_id": result.interaction_id,
                "status": result.status.value,
                "content": result.content,
                "citations": result.citations,
                "error": result.error,
            }
        except TimeoutError as e:
            return {
                "interaction_id": None,
                "status": "failed",
                "content": None,
                "citations": None,
                "error": str(e),
            }
        except Exception as e:
            return {
                "interaction_id": None,
                "status": "failed",
                "content": None,
                "citations": None,
                "error": f"Research failed: {str(e)}",
            }

    return _run_async(_execute())


def start_deep_research(
    query: str,
    format_instructions: Optional[str] = None,
) -> dict:
    """Deep Research を開始する（非同期）。

    リサーチを開始し、すぐにインタラクションIDを返します。
    結果は get_research_status と get_research_result で取得してください。

    Args:
        query: リサーチクエリ
        format_instructions: 出力フォーマット指定（オプション）

    Returns:
        開始情報を含む辞書:
        - interaction_id: インタラクションID
        - status: "in_progress"
        - message: 次のステップの説明

    Example:
        >>> result = start_deep_research("AI trends in 2025")
        >>> interaction_id = result["interaction_id"]
        >>> # 後で結果を確認
        >>> status = get_research_status(interaction_id)
    """
    client = _get_deep_research_client()

    async def _execute():
        try:
            interaction_id = await client.start_research(
                query=query,
                format_instructions=format_instructions,
            )
            return {
                "interaction_id": interaction_id,
                "status": "in_progress",
                "message": "Research started. Use get_research_status to check progress.",
            }
        except Exception as e:
            return {
                "interaction_id": None,
                "status": "failed",
                "error": f"Failed to start research: {str(e)}",
            }

    return _run_async(_execute())


def get_research_status(interaction_id: str) -> dict:
    """リサーチの状態を確認する。

    Args:
        interaction_id: リサーチのインタラクションID

    Returns:
        状態情報を含む辞書:
        - interaction_id: インタラクションID
        - status: "in_progress" | "completed" | "failed"
        - error: エラーメッセージ（失敗時）

    Example:
        >>> status = get_research_status("interactions/abc123")
        >>> if status["status"] == "completed":
        ...     result = get_research_result("interactions/abc123")
    """
    client = _get_deep_research_client()

    async def _execute():
        try:
            status = await client.get_status(interaction_id)
            return {
                "interaction_id": status.interaction_id,
                "status": status.status.value,
                "error": status.error,
            }
        except Exception as e:
            return {
                "interaction_id": interaction_id,
                "status": "failed",
                "error": f"Failed to get status: {str(e)}",
            }

    return _run_async(_execute())


def get_research_result(interaction_id: str) -> dict:
    """完了したリサーチの結果を取得する。

    Args:
        interaction_id: リサーチのインタラクションID

    Returns:
        結果を含む辞書:
        - interaction_id: インタラクションID
        - status: "completed" | "in_progress" | "failed"
        - content: リサーチ結果のテキスト
        - citations: 引用元のリスト
        - error: エラーメッセージ（失敗時）

    Example:
        >>> result = get_research_result("interactions/abc123")
        >>> print(result["content"])
    """
    client = _get_deep_research_client()

    async def _execute():
        try:
            result = await client.get_result(interaction_id)
            return {
                "interaction_id": result.interaction_id,
                "status": result.status.value,
                "content": result.content,
                "citations": result.citations,
                "error": result.error,
            }
        except Exception as e:
            return {
                "interaction_id": interaction_id,
                "status": "failed",
                "content": None,
                "citations": None,
                "error": f"Failed to get result: {str(e)}",
            }

    return _run_async(_execute())


def ask_followup_question(
    previous_interaction_id: str,
    question: str,
) -> dict:
    """完了したリサーチについてフォローアップ質問をする。

    Args:
        previous_interaction_id: 完了したリサーチのインタラクションID
        question: フォローアップ質問
            例: "レポートの2番目のポイントについて詳しく説明してください"

    Returns:
        回答を含む辞書:
        - previous_interaction_id: 元のリサーチのインタラクションID
        - question: 質問内容
        - answer: 回答
        - error: エラーメッセージ（失敗時）

    Example:
        >>> followup = ask_followup_question(
        ...     "interactions/abc123",
        ...     "レポートで言及されている主なリスクは何ですか？"
        ... )
        >>> print(followup["answer"])
    """
    client = _get_deep_research_client()

    async def _execute():
        try:
            answer = await client.ask_followup(
                previous_interaction_id=previous_interaction_id,
                question=question,
            )
            return {
                "previous_interaction_id": previous_interaction_id,
                "question": question,
                "answer": answer,
                "error": None,
            }
        except Exception as e:
            return {
                "previous_interaction_id": previous_interaction_id,
                "question": question,
                "answer": None,
                "error": f"Failed to get follow-up answer: {str(e)}",
            }

    return _run_async(_execute())


# =============================================================================
# Quick Search Tools (Seconds to complete, lightweight search)
# =============================================================================


def quick_search(
    query: str,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """Google Search groundingを使用した高速検索を実行する。

    Gemini の Google Search ツールを使用して、Web から情報を検索・統合します。
    deep_research と異なり、通常数秒で結果が返ります。

    **用途:**
    - 素早いファクトチェック
    - 最新ニュースやイベント
    - 現在の情報が必要な簡単な質問
    - 速度が深さより重要な場合

    **deep_research を使うべき場合:**
    - 包括的な分析
    - 複数のソースを使った詳細なレポート
    - 統合が必要な複雑なトピック

    Args:
        query: 検索クエリ（例: "最新のAIニュース"）
        model: 使用するモデル（デフォルト: gemini-3-flash-preview）
            選択肢: gemini-3-flash-preview, gemini-3-pro-preview
        language: 回答言語（デフォルト: ja）

    Returns:
        検索結果を含む辞書:
        - query: 元の検索クエリ
        - content: 生成された回答
        - citations: 引用元のリスト（url, title）
        - search_queries: モデルが使用した検索クエリ
        - model: 使用したモデル
        - error: エラーメッセージ（失敗時）

    Example:
        >>> result = quick_search("2024年のノーベル物理学賞")
        >>> print(result["content"])
        >>> for citation in result["citations"]:
        ...     print(f"- {citation['title']}: {citation['url']}")
    """
    client = _get_quick_search_client()

    async def _execute():
        try:
            result = await client.quick_search(
                query=query,
                model=model,
                language=language,
            )
            return {
                "query": result.query,
                "content": result.content,
                "citations": [{"url": c.url, "title": c.title} for c in result.citations],
                "search_queries": result.search_queries,
                "model": result.model,
                "error": result.error,
            }
        except Exception as e:
            return {
                "query": query,
                "content": None,
                "citations": [],
                "search_queries": [],
                "model": model,
                "error": f"Quick search failed: {str(e)}",
            }

    return _run_async(_execute())


def analyze_urls(
    urls: list[str],
    query: str,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """URL Context ツールを使用して特定のURLの内容を分析する。

    指定されたURLからコンテンツを取得・分析します。
    ドキュメントの比較、データ抽出、特定ページの要約、
    特定のWebコンテンツに関する質問への回答に便利です。

    **用途:**
    - 複数の記事やドキュメントの比較
    - 既知のURLからの特定情報の抽出
    - Webページの要約
    - コードリポジトリやドキュメントの分析

    **制限:**
    - リクエストあたり最大20 URL
    - URLあたり最大34MBのコンテンツ
    - 非対応: ペイウォールコンテンツ、YouTube動画、Google Workspaceファイル

    Args:
        urls: 分析するURLのリスト（最大20）
        query: 分析クエリまたは指示
        model: 使用するモデル（デフォルト: gemini-3-flash-preview）
            選択肢: gemini-3-flash-preview, gemini-3-pro-preview
        language: 回答言語（デフォルト: ja）

    Returns:
        分析結果を含む辞書:
        - urls: 分析されたURL
        - query: 分析クエリ
        - content: 生成された分析
        - url_metadata: 各URLのメタデータ（取得状況）
        - model: 使用したモデル
        - error: エラーメッセージ（失敗時）

    Example:
        >>> result = analyze_urls(
        ...     urls=["https://example.com/recipe1", "https://example.com/recipe2"],
        ...     query="これらのレシピの材料と調理時間を比較してください"
        ... )
        >>> print(result["content"])
    """
    client = _get_quick_search_client()

    async def _execute():
        try:
            result = await client.analyze_urls(
                urls=urls,
                query=query,
                model=model,
                language=language,
            )
            return {
                "urls": result.urls,
                "query": result.query,
                "content": result.content,
                "url_metadata": [
                    {"url": m.url, "status": m.status, "title": m.title}
                    for m in result.url_metadata
                ],
                "model": result.model,
                "error": result.error,
            }
        except Exception as e:
            return {
                "urls": urls,
                "query": query,
                "content": None,
                "url_metadata": [],
                "model": model,
                "error": f"URL analysis failed: {str(e)}",
            }

    return _run_async(_execute())


def search_and_analyze(
    query: str,
    urls: Optional[list[str]] = None,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """Web検索とオプションのURL分析を組み合わせて実行する。

    Google Search grounding と URL Context を組み合わせて、
    広範な検索結果と特定ページの深い分析の両方を提供します。

    **用途:**
    - 検索結果と特定の参考ドキュメントの比較
    - 権威あるソースに対する主張のファクトチェック
    - 一般的なリサーチと特定のURL分析の組み合わせ

    Args:
        query: 検索・分析クエリ
        urls: コンテキストに含めるURLのリスト（オプション）
        model: 使用するモデル（デフォルト: gemini-3-flash-preview）
            選択肢: gemini-3-flash-preview, gemini-3-pro-preview
        language: 回答言語（デフォルト: ja）

    Returns:
        結果を含む辞書:
        - query: 元のクエリ
        - content: 生成された回答
        - citations: 引用元のリスト
        - search_queries: モデルが使用した検索クエリ
        - model: 使用したモデル
        - error: エラーメッセージ（失敗時）

    Example:
        >>> result = search_and_analyze(
        ...     query="最新のEV市場動向と、この記事の内容を比較してください",
        ...     urls=["https://example.com/ev-report-2024"]
        ... )
        >>> print(result["content"])
    """
    client = _get_quick_search_client()

    async def _execute():
        try:
            result = await client.search_and_analyze(
                query=query,
                urls=urls,
                model=model,
                language=language,
            )
            return {
                "query": result.query,
                "content": result.content,
                "citations": [{"url": c.url, "title": c.title} for c in result.citations],
                "search_queries": result.search_queries,
                "model": result.model,
                "error": result.error,
            }
        except Exception as e:
            return {
                "query": query,
                "content": None,
                "citations": [],
                "search_queries": [],
                "model": model,
                "error": f"Search and analyze failed: {str(e)}",
            }

    return _run_async(_execute())
