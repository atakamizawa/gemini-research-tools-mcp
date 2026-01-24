"""Combined Search and URL Analysis tool."""

import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai
from google.genai import types

# Set up logging
logger = logging.getLogger(__name__)


class SearchAndAnalyzeTool(Tool):
    """Combined Search and URL Analysis tool.

    Combines Google Search grounding with URL Context for comprehensive analysis.
    Use for comparing search results with specific documents or fact-checking.
    """

    DEFAULT_MODEL = "gemini-3-flash-preview"
    SYSTEM_PROMPT_JA = "必ず日本語で回答してください。"

    def _get_client(self) -> genai.Client:
        """Get the Gemini client with credentials."""
        api_key = self.runtime.credentials.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        return genai.Client(api_key=api_key)

    def _get_model(self) -> str:
        """Get the model from credentials or use default."""
        return self.runtime.credentials.get("default_quick_model", self.DEFAULT_MODEL)

    def _build_prompt(self, query: str, language: str = "ja") -> str:
        """Build the prompt with language instruction."""
        if language == "ja":
            return f"{self.SYSTEM_PROMPT_JA}\n\n{query}"
        return query

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Search and analyze.

        Args:
            tool_parameters: Dictionary containing:
                - query: Search/analysis query
                - urls: Optional comma-separated URLs
                - language: Response language
        """
        query = tool_parameters.get("query", "")
        urls_str = tool_parameters.get("urls", "")
        language = tool_parameters.get("language", "ja")

        logger.info(f"SearchAndAnalyze invoked with query: {query[:100] if query else 'N/A'}...")
        logger.debug(f"URLs: {urls_str[:200] if urls_str else 'None'}")

        if not query:
            logger.warning("SearchAndAnalyze called without query parameter")
            yield self.create_json_message({"error": "query is required"})
            yield self.create_text_message("Error: query is required")
            return

        try:
            client = self._get_client()
            model = self._get_model()

            # Build prompt
            prompt = self._build_prompt(query, language)
            urls = []
            if urls_str:
                urls = [url.strip() for url in urls_str.split(",") if url.strip()]
                url_list = "\n".join(urls)
                prompt = f"{prompt}\n\n参考URL:\n{url_list}"

            # Configure both tools
            tools = [
                types.Tool(google_search=types.GoogleSearch()),
                types.Tool(url_context=types.UrlContext()),
            ]
            config = types.GenerateContentConfig(tools=tools)

            logger.debug(f"Sending request to Gemini API with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            logger.debug("Received response from Gemini API")

            # Extract content
            content = response.text if response.text else ""
            logger.info(f"SearchAndAnalyze completed, content length: {len(content)}")

            # Extract citations
            citations = []
            search_queries = []

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                        search_queries = list(metadata.web_search_queries)

                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web:
                                citations.append({
                                    "url": chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                    "title": chunk.web.title if hasattr(chunk.web, "title") else None,
                                })

            # 常に両方を返す（SharePointプラグインと同じパターン）
            yield self.create_json_message({
                "query": query,
                "urls": urls,
                "content": content,
                "citations": citations,
                "search_queries": search_queries,
                "model": model,
            })
            yield self.create_text_message(self._format_text_output(query, content, citations))

        except Exception as e:
            logger.error(f"SearchAndAnalyze failed with error: {str(e)}", exc_info=True)
            error_message = f"検索・分析中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "query": query,
                "urls": urls if 'urls' in locals() else [],
                "content": None,
                "citations": [],
                "search_queries": [],
                "model": self._get_model() if hasattr(self, '_get_model') else self.DEFAULT_MODEL,
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_text_output(self, query: str, content: str, citations: list) -> str:
        """Format the search and analysis results as readable text for LLM."""
        lines = [f"## 検索・分析結果: {query}", "", content, ""]
        
        if citations:
            lines.append("### 参照元:")
            for i, citation in enumerate(citations, 1):
                url = citation.get("url", "")
                title = citation.get("title", "")
                if title:
                    lines.append(f"{i}. [{title}]({url})")
                else:
                    lines.append(f"{i}. {url}")
        
        return "\n".join(lines)
