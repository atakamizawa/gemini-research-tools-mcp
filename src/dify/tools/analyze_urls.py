"""URL Analysis tool using URL Context."""

import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai
from google.genai import types

# Set up logging
logger = logging.getLogger(__name__)


class AnalyzeUrlsTool(Tool):
    """URL Analysis tool using URL Context.

    Fetches and analyzes content from specific URLs.
    Useful for comparing documents, extracting data, summarizing pages.
    """

    DEFAULT_MODEL = "gemini-3-flash-preview"
    SYSTEM_PROMPT_JA = "必ず日本語で回答してください。"
    MAX_URLS = 20

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
        """Analyze URLs.

        Args:
            tool_parameters: Dictionary containing:
                - urls: Comma-separated URLs
                - query: Analysis query
                - language: Response language
        """
        urls_str = tool_parameters.get("urls", "")
        query = tool_parameters.get("query", "")
        language = tool_parameters.get("language", "ja")

        logger.info(f"AnalyzeUrls invoked with query: {query[:100] if query else 'N/A'}...")
        logger.debug(f"URLs: {urls_str[:200]}...")

        if not urls_str:
            logger.warning("AnalyzeUrls called without urls parameter")
            yield self.create_json_message({"error": "urls is required"})
            yield self.create_text_message("Error: urls is required")
            return

        if not query:
            logger.warning("AnalyzeUrls called without query parameter")
            yield self.create_json_message({"error": "query is required"})
            yield self.create_text_message("Error: query is required")
            return

        # Parse URLs
        urls = [url.strip() for url in urls_str.split(",") if url.strip()]
        logger.info(f"Analyzing {len(urls)} URLs")

        if len(urls) > self.MAX_URLS:
            logger.warning(f"Too many URLs: {len(urls)} > {self.MAX_URLS}")
            yield self.create_json_message({"error": f"Maximum {self.MAX_URLS} URLs allowed"})
            yield self.create_text_message(f"Error: Maximum {self.MAX_URLS} URLs allowed per request")
            return

        try:
            client = self._get_client()
            model = self._get_model()

            # Build prompt with URLs
            url_list = "\n".join(urls)
            full_prompt = self._build_prompt(f"{query}\n\nURLs:\n{url_list}", language)

            # Configure URL Context tool
            url_context_tool = types.Tool(url_context=types.UrlContext())
            config = types.GenerateContentConfig(tools=[url_context_tool])

            logger.debug(f"Sending request to Gemini API with model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=config,
            )
            logger.debug("Received response from Gemini API")

            # Extract content
            content = response.text if response.text else ""
            logger.info(f"AnalyzeUrls completed, content length: {len(content)}")

            # Extract URL metadata
            url_metadata = []
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "url_context_metadata") and candidate.url_context_metadata:
                    metadata = candidate.url_context_metadata
                    if hasattr(metadata, "url_metadata") and metadata.url_metadata:
                        for url_meta in metadata.url_metadata:
                            url_metadata.append({
                                "url": url_meta.retrieved_url if hasattr(url_meta, "retrieved_url") else "",
                                "status": url_meta.url_retrieval_status if hasattr(url_meta, "url_retrieval_status") else "unknown",
                            })

            # 常に両方を返す（SharePointプラグインと同じパターン）
            yield self.create_json_message({
                "urls": urls,
                "query": query,
                "content": content,
                "url_metadata": url_metadata,
                "model": model,
            })
            yield self.create_text_message(self._format_text_output(query, urls, content, url_metadata))

        except Exception as e:
            logger.error(f"AnalyzeUrls failed with error: {str(e)}", exc_info=True)
            error_message = f"URL分析中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "urls": urls if 'urls' in locals() else [],
                "query": query,
                "content": None,
                "url_metadata": [],
                "model": self._get_model() if hasattr(self, '_get_model') else self.DEFAULT_MODEL,
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_text_output(self, query: str, urls: list, content: str, url_metadata: list) -> str:
        """Format the analysis results as readable text for LLM."""
        lines = [f"## URL分析結果: {query}", "", content, ""]
        
        if urls:
            lines.append("### 分析したURL:")
            for i, url in enumerate(urls, 1):
                # Find status for this URL
                status = "unknown"
                for meta in url_metadata:
                    if meta.get("url") == url:
                        status = meta.get("status", "unknown")
                        break
                lines.append(f"{i}. {url} (status: {status})")
        
        return "\n".join(lines)
