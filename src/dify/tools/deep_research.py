"""Deep Research tool using Gemini Interactions API."""

import logging
import time
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai

# Set up logging
logger = logging.getLogger(__name__)


class DeepResearchTool(Tool):
    """Deep Research tool using Gemini Interactions API.

    Performs comprehensive research on a topic using the Deep Research Agent.
    Research tasks typically take several minutes to complete.
    Best for complex topics requiring in-depth analysis with multiple sources.
    """

    AGENT_NAME = "deep-research-pro-preview-12-2025"
    DEFAULT_POLL_INTERVAL = 10  # seconds
    SYSTEM_PROMPT_JA = "必ず日本語で回答してください。推論過程や思考の要約も日本語で出力してください。"

    def _get_client(self) -> genai.Client:
        """Get the Gemini client with credentials."""
        api_key = self.runtime.credentials.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        return genai.Client(api_key=api_key)

    def _build_input(self, query: str, format_instructions: str | None = None) -> str:
        """Build the input prompt with optional format instructions."""
        parts = [self.SYSTEM_PROMPT_JA, query]
        if format_instructions:
            parts.append(format_instructions)
        return "\n\n".join(parts)

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Execute deep research.

        Args:
            tool_parameters: Dictionary containing:
                - query: Research topic
                - format_instructions: Optional formatting instructions
                - wait_for_completion: Whether to wait for completion
                - timeout: Maximum wait time in seconds
        """
        query = tool_parameters.get("query", "")
        format_instructions = tool_parameters.get("format_instructions")
        wait_for_completion = tool_parameters.get("wait_for_completion", False)
        timeout = tool_parameters.get("timeout", 3600)

        logger.info(f"DeepResearch invoked with query: {query[:100]}...")
        logger.debug(f"Parameters: wait_for_completion={wait_for_completion}, timeout={timeout}")

        if not query:
            logger.warning("DeepResearch called without query parameter")
            yield self.create_json_message({"error": "query is required"})
            yield self.create_text_message("Error: query is required")
            return

        try:
            client = self._get_client()
            input_text = self._build_input(query, format_instructions)

            # Start research
            logger.info(f"Starting deep research with agent: {self.AGENT_NAME}")
            interaction = client.interactions.create(
                input=input_text,
                agent=self.AGENT_NAME,
                background=True,
            )

            interaction_id = interaction.id
            logger.info(f"Research started with interaction_id: {interaction_id}")

            if not wait_for_completion:
                # Return immediately with interaction_id
                yield self.create_json_message({
                    "interaction_id": interaction_id,
                    "status": "in_progress",
                    "message": "Research started. Use get_research_status to check progress.",
                })
                yield self.create_text_message(
                    f"## リサーチ開始\n\n"
                    f"リサーチタスクを開始しました。\n"
                    f"- **Interaction ID**: `{interaction_id}`\n"
                    f"- **ステータス**: 進行中\n\n"
                    f"結果を取得するには `get_research_status` でステータスを確認し、"
                    f"完了後に `get_research_result` で結果を取得してください。"
                )
                return

            # Wait for completion
            start_time = time.time()
            while True:
                interaction = client.interactions.get(interaction_id)

                logger.debug(f"Polling interaction status: {interaction.status}")

                if interaction.status == "completed":
                    logger.info(f"Research completed for interaction_id: {interaction_id}")
                    # Extract content
                    content = None
                    citations = []

                    if interaction.outputs:
                        for output in reversed(interaction.outputs):
                            if hasattr(output, "text") and output.text:
                                content = output.text
                                break

                        # Extract citations
                        for output in interaction.outputs:
                            if hasattr(output, "grounding_metadata"):
                                metadata = output.grounding_metadata
                                if hasattr(metadata, "grounding_chunks"):
                                    for chunk in metadata.grounding_chunks:
                                        if hasattr(chunk, "web") and hasattr(chunk.web, "uri"):
                                            citations.append(chunk.web.uri)

                    # 常に両方を返す
                    yield self.create_json_message({
                        "interaction_id": interaction_id,
                        "status": "completed",
                        "content": content,
                        "citations": citations,
                    })
                    yield self.create_text_message(self._format_completed_output(query, content, citations))
                    return

                if interaction.status in ("failed", "cancelled"):
                    logger.error(f"Research failed with status: {interaction.status}")
                    error = getattr(interaction, "error", "Unknown error")
                    
                    yield self.create_json_message({
                        "interaction_id": interaction_id,
                        "status": interaction.status,
                        "error": str(error),
                    })
                    yield self.create_text_message(
                        f"## リサーチ失敗\n\n"
                        f"リサーチタスクが失敗しました。\n"
                        f"- **Interaction ID**: `{interaction_id}`\n"
                        f"- **ステータス**: {interaction.status}\n"
                        f"- **エラー**: {error}"
                    )
                    return

                # Check timeout
                elapsed = time.time() - start_time
                logger.debug(f"Elapsed time: {elapsed:.1f}s / {timeout}s")
                if elapsed >= timeout:
                    logger.warning(f"Research timeout after {elapsed:.1f}s")
                    
                    yield self.create_json_message({
                        "interaction_id": interaction_id,
                        "status": "timeout",
                        "error": f"Research did not complete within {timeout} seconds",
                    })
                    yield self.create_text_message(
                        f"## リサーチタイムアウト\n\n"
                        f"リサーチタスクが{timeout}秒以内に完了しませんでした。\n"
                        f"- **Interaction ID**: `{interaction_id}`\n\n"
                        f"`get_research_status` で後でステータスを確認してください。"
                    )
                    return

                time.sleep(self.DEFAULT_POLL_INTERVAL)

        except Exception as e:
            logger.error(f"DeepResearch failed with error: {str(e)}", exc_info=True)
            error_message = f"リサーチ開始中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "interaction_id": None,
                "status": "error",
                "content": None,
                "citations": [],
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_completed_output(self, query: str, content: str | None, citations: list) -> str:
        """Format the completed research results as readable text for LLM."""
        lines = [f"## Deep Research 結果: {query}", ""]
        
        if content:
            lines.append(content)
            lines.append("")
        else:
            lines.append("（コンテンツなし）")
            lines.append("")
        
        if citations:
            lines.append("### 参照元:")
            for i, url in enumerate(citations, 1):
                lines.append(f"{i}. {url}")
        
        return "\n".join(lines)
