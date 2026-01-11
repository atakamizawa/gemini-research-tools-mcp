"""Streamlit Web UI for Gemini Deep Research Agent.

Usage:
    streamlit run src/ui/app.py
"""

import asyncio
import os
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.client import DeepResearchClient, QuickSearchClient
from src.core.models import ResearchEventType, ResearchStatusEnum

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gemini ディープリサーチ",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    .thought-box {
        background-color: #1e3a5f;
        border-left: 4px solid #4fc3f7;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        color: #e0e0e0;
        font-size: 14px;
    }
    .result-box {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        color: #e0e0e0;
    }
    .citation-link {
        color: #4fc3f7;
        text-decoration: none;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
    }
    .status-in-progress {
        background-color: #fff3e0;
        color: #e65100;
    }
    .status-completed {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .status-failed {
        background-color: #ffebee;
        color: #c62828;
    }
    .history-card {
        margin-bottom: 10px;
        padding: 10px;
        background: #3d3d3d;
        border-radius: 8px;
        color: #e0e0e0;
    }
    .history-card small {
        color: #aaa;
    }
    .citation-box {
        background-color: #2d3748;
        border-left: 3px solid #4299e1;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .url-status-success {
        color: #48bb78;
    }
    .url-status-failed {
        color: #fc8181;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "current_research" not in st.session_state:
        st.session_state.current_research = None
    if "thoughts" not in st.session_state:
        st.session_state.thoughts = []
    if "accumulated_content" not in st.session_state:
        st.session_state.accumulated_content = ""
    if "quick_search_history" not in st.session_state:
        st.session_state.quick_search_history = []


# HTTP timeout settings (in seconds)
# Note: Gemini API requires minimum 10 seconds deadline
DEEP_RESEARCH_HTTP_TIMEOUT = 3600  # 60 minutes for long-running research
QUICK_SEARCH_HTTP_TIMEOUT = 300  # 5 minutes for quick search operations


def get_deep_research_client() -> DeepResearchClient:
    """Get or create the DeepResearchClient instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY 環境変数が設定されていません。")
        st.info(".env ファイルまたは環境変数に Gemini API キーを設定してください。")
        st.stop()
    return DeepResearchClient(api_key=api_key, http_timeout=DEEP_RESEARCH_HTTP_TIMEOUT)


def get_quick_search_client() -> QuickSearchClient:
    """Get or create the QuickSearchClient instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY 環境変数が設定されていません。")
        st.info(".env ファイルまたは環境変数に Gemini API キーを設定してください。")
        st.stop()
    return QuickSearchClient(api_key=api_key, http_timeout=QUICK_SEARCH_HTTP_TIMEOUT)


# Backward compatibility alias
def get_client() -> DeepResearchClient:
    """Get the DeepResearchClient instance (backward compatibility)."""
    return get_deep_research_client()


async def run_streaming_research(client: DeepResearchClient, query: str, format_instructions: str):
    """Run research with streaming updates."""
    st.session_state.thoughts = []
    st.session_state.accumulated_content = ""

    thought_container = st.container()
    content_container = st.empty()
    status_container = st.empty()

    try:
        async for event in client.stream_research(query, format_instructions or None):
            if event.event_type == ResearchEventType.START:
                st.session_state.current_research = {
                    "interaction_id": event.interaction_id,
                    "query": query,
                    "status": "in_progress",
                    "started_at": datetime.now().isoformat(),
                }
                status_container.info(f"🔬 リサーチを開始しました (ID: {event.interaction_id})")

            elif event.event_type == ResearchEventType.THOUGHT:
                if event.content:
                    st.session_state.thoughts.append(event.content)
                    with thought_container:
                        st.markdown(
                            f'<div class="thought-box">💭 {event.content}</div>',
                            unsafe_allow_html=True,
                        )

            elif event.event_type == ResearchEventType.TEXT_DELTA:
                if event.content:
                    st.session_state.accumulated_content += event.content
                    content_container.markdown(st.session_state.accumulated_content)

            elif event.event_type == ResearchEventType.COMPLETE:
                st.session_state.current_research["status"] = "completed"
                st.session_state.current_research["completed_at"] = datetime.now().isoformat()
                
                # Determine final content with fallback logic
                final_content = event.content or st.session_state.accumulated_content
                
                # If still no content, try to fetch it via API
                if not final_content and st.session_state.current_research.get("interaction_id"):
                    try:
                        result = await client.get_result(st.session_state.current_research["interaction_id"])
                        final_content = result.content
                    except Exception as e:
                        st.warning(f"⚠️ 結果の取得中にエラーが発生しました: {str(e)}")
                
                st.session_state.current_research["content"] = final_content
                
                # Display final result
                if final_content:
                    st.markdown("### 📄 リサーチレポート")
                    content_container.markdown(final_content)
                else:
                    st.warning("⚠️ レポート内容を取得できませんでした。「ステータス確認」タブからインタラクションIDを使って結果を取得してください。")
                
                st.session_state.research_history.append(st.session_state.current_research)
                status_container.success("✅ リサーチが完了しました！")

            elif event.event_type == ResearchEventType.ERROR:
                st.session_state.current_research["status"] = "failed"
                st.session_state.current_research["error"] = event.content
                status_container.error(f"❌ リサーチに失敗しました: {event.content}")

    except Exception as e:
        st.error(f"❌ リサーチ中にエラーが発生しました: {str(e)}")


async def run_polling_research(client: DeepResearchClient, query: str, format_instructions: str):
    """Run research with polling (non-streaming)."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.info("🚀 リサーチを開始しています...")
        interaction_id = await client.start_research(query, format_instructions or None)

        st.session_state.current_research = {
            "interaction_id": interaction_id,
            "query": query,
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
        }

        status_text.info(f"🔬 リサーチ実行中 (ID: {interaction_id})")

        # Poll for completion
        poll_count = 0
        max_polls = 360  # 1 hour with 10s intervals

        while poll_count < max_polls:
            status = await client.get_status(interaction_id)

            if status.status == ResearchStatusEnum.COMPLETED:
                result = await client.get_result(interaction_id)
                st.session_state.current_research["status"] = "completed"
                st.session_state.current_research["content"] = result.content
                st.session_state.current_research["citations"] = result.citations
                st.session_state.current_research["completed_at"] = datetime.now().isoformat()
                st.session_state.research_history.append(st.session_state.current_research)

                progress_bar.progress(100)
                status_text.success("✅ リサーチが完了しました！")

                # Display result
                st.markdown("### 📄 リサーチレポート")
                st.markdown(result.content)

                if result.citations:
                    st.markdown("### 📚 引用元")
                    for i, citation in enumerate(result.citations, 1):
                        st.markdown(f"{i}. [{citation}]({citation})")

                return

            elif status.status in (ResearchStatusEnum.FAILED, ResearchStatusEnum.CANCELLED):
                st.session_state.current_research["status"] = "failed"
                st.session_state.current_research["error"] = status.error
                status_text.error(f"❌ リサーチに失敗しました: {status.error}")
                return

            # Update progress (estimate based on typical research time)
            progress = min(95, (poll_count / max_polls) * 100)
            progress_bar.progress(int(progress))
            status_text.info(f"🔬 リサーチ実行中... (経過時間: {poll_count * 10}秒)")

            poll_count += 1
            await asyncio.sleep(10)

        status_text.warning("⏱️ リサーチがタイムアウトしました。手動でステータスを確認してください。")

    except Exception as e:
        st.error(f"❌ リサーチ中にエラーが発生しました: {str(e)}")


async def run_quick_search(client: QuickSearchClient, query: str, model: str, language: str):
    """Run quick search."""
    try:
        result = await client.quick_search(
            query=query,
            model=model,
            language=language,
        )
        return result
    except Exception as e:
        st.error(f"❌ 検索中にエラーが発生しました: {str(e)}")
        return None


async def run_url_analysis(client: QuickSearchClient, urls: list[str], query: str, model: str, language: str):
    """Run URL analysis."""
    try:
        result = await client.analyze_urls(
            urls=urls,
            query=query,
            model=model,
            language=language,
        )
        return result
    except Exception as e:
        st.error(f"❌ URL分析中にエラーが発生しました: {str(e)}")
        return None


async def run_search_and_analyze(client: QuickSearchClient, query: str, urls: list[str], model: str, language: str):
    """Run search and analyze."""
    try:
        result = await client.search_and_analyze(
            query=query,
            urls=urls if urls else None,
            model=model,
            language=language,
        )
        return result
    except Exception as e:
        st.error(f"❌ 検索+分析中にエラーが発生しました: {str(e)}")
        return None


def main():
    """Main application."""
    init_session_state()

    # Header
    st.title("🔬 Gemini ディープリサーチ")
    st.markdown(
        "Google の Gemini Deep Research Agent と Quick Search ツールを使用して"
        "包括的なリサーチや高速検索を実行します。"
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 設定")

        use_streaming = st.checkbox(
            "ストリーミングを有効化",
            value=True,
            help="リアルタイムの進捗状況と思考の要約を表示します（Deep Researchのみ）",
        )

        st.divider()

        st.header("📜 リサーチ履歴")
        if st.session_state.research_history:
            for i, research in enumerate(reversed(st.session_state.research_history[-5:])):
                status_class = f"status-{research['status'].replace('_', '-')}"
                status_label = {
                    "completed": "完了",
                    "in_progress": "実行中",
                    "failed": "失敗",
                }.get(research["status"], research["status"])
                st.markdown(
                    f"""
                    <div class="history-card">
                        <span class="status-badge {status_class}">{status_label}</span>
                        <p style="margin: 5px 0; font-size: 14px;">{research['query'][:50]}...</p>
                        <small>ID: {research['interaction_id'][:20]}...</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("リサーチ履歴はまだありません。")

    # Main content - 6 tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 新規リサーチ",
        "📊 ステータス確認",
        "❓ フォローアップ",
        "⚡ クイック検索",
        "🔗 URL分析",
        "🔍+🔗 検索+URL分析",
    ])

    # Tab 1: Deep Research
    with tab1:
        st.header("新規リサーチを開始")
        st.info("💡 Deep Researchは数分〜数十分かかる包括的なリサーチです。高速な検索には「クイック検索」タブをご利用ください。")

        query = st.text_area(
            "リサーチクエリ",
            placeholder="リサーチしたいトピックや質問を入力してください...\n\n例: 量子コンピューティングの最新動向とその応用可能性について",
            height=100,
            key="deep_research_query",
        )

        format_instructions = st.text_area(
            "フォーマット指示（任意）",
            placeholder="出力形式を指定してください...\n\n例: 技術レポート形式で、概要、主要な発見、詳細分析、結論のセクションを含めてください。",
            height=80,
            key="deep_research_format",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            start_button = st.button("🚀 リサーチ開始", type="primary", use_container_width=True)

        if start_button:
            if not query.strip():
                st.warning("リサーチクエリを入力してください。")
            else:
                client = get_deep_research_client()

                st.divider()
                st.header("📊 リサーチ進捗")

                if use_streaming:
                    asyncio.run(run_streaming_research(client, query, format_instructions))
                else:
                    asyncio.run(run_polling_research(client, query, format_instructions))

    # Tab 2: Status Check
    with tab2:
        st.header("リサーチステータスを確認")

        interaction_id = st.text_input(
            "インタラクション ID",
            placeholder="インタラクション ID を入力してください（例: interactions/abc123）",
        )

        if st.button("🔄 ステータスを確認"):
            if not interaction_id.strip():
                st.warning("インタラクション ID を入力してください。")
            else:
                client = get_deep_research_client()

                async def check_status():
                    status = await client.get_status(interaction_id)
                    return status

                status = asyncio.run(check_status())

                status_color = {
                    ResearchStatusEnum.IN_PROGRESS: "🟡",
                    ResearchStatusEnum.COMPLETED: "🟢",
                    ResearchStatusEnum.FAILED: "🔴",
                    ResearchStatusEnum.CANCELLED: "⚪",
                }

                status_label = {
                    ResearchStatusEnum.IN_PROGRESS: "実行中",
                    ResearchStatusEnum.COMPLETED: "完了",
                    ResearchStatusEnum.FAILED: "失敗",
                    ResearchStatusEnum.CANCELLED: "キャンセル",
                }

                st.markdown(f"**ステータス:** {status_color.get(status.status, '❓')} {status_label.get(status.status, status.status.value)}")

                if status.status == ResearchStatusEnum.COMPLETED:
                    if st.button("📄 結果を取得"):
                        async def get_result():
                            return await client.get_result(interaction_id)

                        result = asyncio.run(get_result())
                        st.markdown("### リサーチレポート")
                        st.markdown(result.content)

                        if result.citations:
                            st.markdown("### 引用元")
                            for citation in result.citations:
                                st.markdown(f"- [{citation}]({citation})")

                elif status.error:
                    st.error(f"エラー: {status.error}")

    # Tab 3: Follow-up
    with tab3:
        st.header("フォローアップ質問")

        prev_interaction_id = st.text_input(
            "前回のインタラクション ID",
            placeholder="完了したリサーチのインタラクション ID を入力してください",
            key="followup_id",
        )

        followup_question = st.text_area(
            "フォローアップ質問",
            placeholder="リサーチ結果について質問してください...\n\n例: レポートの2番目のポイントについて詳しく説明してください。",
            height=80,
        )

        if st.button("❓ 質問する"):
            if not prev_interaction_id.strip() or not followup_question.strip():
                st.warning("インタラクション ID と質問の両方を入力してください。")
            else:
                client = get_deep_research_client()

                async def ask_followup():
                    return await client.ask_followup(prev_interaction_id, followup_question)

                with st.spinner("回答を取得中..."):
                    answer = asyncio.run(ask_followup())

                st.markdown("### 回答")
                st.markdown(answer)

    # Tab 4: Quick Search
    with tab4:
        st.header("⚡ クイック検索")
        st.info("💡 Google Search groundingを使用した高速検索です。数秒で結果が返ります。")

        quick_query = st.text_area(
            "検索クエリ",
            placeholder="検索したい内容を入力してください...\n\n例: 2024年のノーベル物理学賞",
            height=80,
            key="quick_search_query",
        )

        col1, col2 = st.columns(2)
        with col1:
            quick_model = st.selectbox(
                "モデル",
                options=QuickSearchClient.SUPPORTED_MODELS,
                index=0,
                key="quick_search_model",
            )
        with col2:
            quick_language = st.selectbox(
                "言語",
                options=["ja", "en"],
                format_func=lambda x: "日本語" if x == "ja" else "English",
                index=0,
                key="quick_search_language",
            )

        if st.button("🔍 検索", type="primary", key="quick_search_button"):
            if not quick_query.strip():
                st.warning("検索クエリを入力してください。")
            else:
                client = get_quick_search_client()

                with st.spinner("検索中..."):
                    result = asyncio.run(run_quick_search(client, quick_query, quick_model, quick_language))

                if result:
                    if result.error:
                        st.error(f"❌ エラー: {result.error}")
                    else:
                        st.markdown("### 📄 検索結果")
                        st.markdown(result.content)

                        if result.citations:
                            st.markdown("### 📚 引用元")
                            for i, citation in enumerate(result.citations, 1):
                                title = citation.title or citation.url
                                st.markdown(
                                    f'<div class="citation-box">{i}. <a href="{citation.url}" target="_blank">{title}</a></div>',
                                    unsafe_allow_html=True,
                                )

                        if result.search_queries:
                            with st.expander("🔍 使用された検索クエリ"):
                                for sq in result.search_queries:
                                    st.markdown(f"- {sq}")

    # Tab 5: URL Analysis
    with tab5:
        st.header("🔗 URL分析")
        st.info("💡 指定したURLの内容を分析します。最大20個のURLを指定できます。")

        url_input = st.text_area(
            "分析するURL（1行に1つ）",
            placeholder="https://example.com/article1\nhttps://example.com/article2",
            height=120,
            key="url_analysis_urls",
        )

        url_query = st.text_area(
            "分析クエリ",
            placeholder="URLの内容についてどのような分析を行いたいですか？\n\n例: これらの記事の主な違いを比較してください",
            height=80,
            key="url_analysis_query",
        )

        col1, col2 = st.columns(2)
        with col1:
            url_model = st.selectbox(
                "モデル",
                options=QuickSearchClient.SUPPORTED_MODELS,
                index=0,
                key="url_analysis_model",
            )
        with col2:
            url_language = st.selectbox(
                "言語",
                options=["ja", "en"],
                format_func=lambda x: "日本語" if x == "ja" else "English",
                index=0,
                key="url_analysis_language",
            )

        if st.button("🔗 分析", type="primary", key="url_analysis_button"):
            urls = [u.strip() for u in url_input.strip().split("\n") if u.strip()]
            
            if not urls:
                st.warning("少なくとも1つのURLを入力してください。")
            elif len(urls) > 20:
                st.warning("URLは最大20個までです。")
            elif not url_query.strip():
                st.warning("分析クエリを入力してください。")
            else:
                client = get_quick_search_client()

                with st.spinner("URL分析中..."):
                    result = asyncio.run(run_url_analysis(client, urls, url_query, url_model, url_language))

                if result:
                    if result.error:
                        st.error(f"❌ エラー: {result.error}")
                    else:
                        st.markdown("### 📄 分析結果")
                        st.markdown(result.content)

                        if result.url_metadata:
                            st.markdown("### 🔗 URL取得状況")
                            for meta in result.url_metadata:
                                status_icon = "✅" if meta.status == "URL_RETRIEVAL_STATUS_SUCCESS" else "❌"
                                st.markdown(f"{status_icon} {meta.url}")

    # Tab 6: Search + URL Analysis
    with tab6:
        st.header("🔍+🔗 検索+URL分析")
        st.info("💡 Web検索と特定URLの分析を組み合わせます。検索結果と参考URLを比較・統合した分析が可能です。")

        combined_query = st.text_area(
            "検索・分析クエリ",
            placeholder="検索と分析を行いたい内容を入力してください...\n\n例: 最新のEV市場動向と、この記事の内容を比較してください",
            height=80,
            key="combined_query",
        )

        combined_urls = st.text_area(
            "参考URL（任意、1行に1つ）",
            placeholder="https://example.com/ev-report-2024\n（空欄の場合はWeb検索のみ）",
            height=80,
            key="combined_urls",
        )

        col1, col2 = st.columns(2)
        with col1:
            combined_model = st.selectbox(
                "モデル",
                options=QuickSearchClient.SUPPORTED_MODELS,
                index=0,
                key="combined_model",
            )
        with col2:
            combined_language = st.selectbox(
                "言語",
                options=["ja", "en"],
                format_func=lambda x: "日本語" if x == "ja" else "English",
                index=0,
                key="combined_language",
            )

        if st.button("🔍+🔗 検索+分析", type="primary", key="combined_button"):
            if not combined_query.strip():
                st.warning("検索・分析クエリを入力してください。")
            else:
                urls = [u.strip() for u in combined_urls.strip().split("\n") if u.strip()] if combined_urls.strip() else []
                
                if len(urls) > 20:
                    st.warning("URLは最大20個までです。")
                else:
                    client = get_quick_search_client()

                    with st.spinner("検索+分析中..."):
                        result = asyncio.run(run_search_and_analyze(client, combined_query, urls, combined_model, combined_language))

                    if result:
                        if result.error:
                            st.error(f"❌ エラー: {result.error}")
                        else:
                            st.markdown("### 📄 検索+分析結果")
                            st.markdown(result.content)

                            if result.citations:
                                st.markdown("### 📚 引用元")
                                for i, citation in enumerate(result.citations, 1):
                                    title = citation.title or citation.url
                                    st.markdown(
                                        f'<div class="citation-box">{i}. <a href="{citation.url}" target="_blank">{title}</a></div>',
                                        unsafe_allow_html=True,
                                    )

                            if result.search_queries:
                                with st.expander("🔍 使用された検索クエリ"):
                                    for sq in result.search_queries:
                                        st.markdown(f"- {sq}")


if __name__ == "__main__":
    main()
