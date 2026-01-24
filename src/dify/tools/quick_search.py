"""Quick Search tool using Google Search grounding."""

import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai
from google.genai import types

# Set up logging
logger = logging.getLogger(__name__)


class QuickSearchTool(Tool):
    """Quick Search tool using Google Search grounding.

    Performs fast web searches using Gemini's Google Search tool.
    Responses typically return in seconds.
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
        """Execute quick search.

        Args:
            tool_parameters: Dictionary containing:
                - query: Search query
                - language: Response language
        """
        query = tool_parameters.get("query", "")
        language = tool_parameters.get("language", "ja")

        logger.info(f"QuickSearch invoked with query: {query}, language: {language}")

        if not query:
            logger.warning("QuickSearch called without query parameter")
            yield self.create_json_message({"error": "query is required"})
            yield self.create_text_message("Error: query is required")
            return

        try:
            client = self._get_client()
            model = self._get_model()
            prompt = self._build_prompt(query, language)
            logger.debug(f"Using model: {model}")

            # Configure Google Search tool
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[google_search_tool])

            logger.debug("Sending request to Gemini API...")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            logger.debug("Received response from Gemini API")

            # Extract content
            content = response.text if response.text else ""
            logger.info(f"QuickSearch completed, content length: {len(content)}")

            # Extract citations
            citations = []
            search_queries = []

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    # Extract search queries
                    if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                        search_queries = list(metadata.web_search_queries)

                    # Extract citations from grounding chunks
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
                "content": content,
                "citations": citations,
                "search_queries": search_queries,
                "model": model,
            })
            yield self.create_text_message(self._format_text_output(query, content, citations))

        except Exception as e:
            logger.error(f"QuickSearch failed with error: {str(e)}", exc_info=True)
            error_message = f"検索中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "query": query,
                "content": None,
                "citations": [],
                "search_queries": [],
                "model": self._get_model() if hasattr(self, '_get_model') else self.DEFAULT_MODEL,
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_text_output(self, query: str, content: str, citations: list) -> str:
        """Format the search results as readable text for LLM."""
        lines = [f"## 検索結果: {query}", "", content, ""]
        
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
