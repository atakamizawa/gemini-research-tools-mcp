"""Tool to retrieve the result of a completed research task."""

import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai

# Set up logging
logger = logging.getLogger(__name__)


class GetResearchResultTool(Tool):
    """Tool to retrieve the result of a completed research task.
    
    Use this after get_research_status indicates the task is completed.
    """

    def _get_client(self) -> genai.Client:
        """Get the Gemini client with credentials."""
        api_key = self.runtime.credentials.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        return genai.Client(api_key=api_key)

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Get research result.

        Args:
            tool_parameters: Dictionary containing:
                - interaction_id: The interaction ID to retrieve
        """
        interaction_id = tool_parameters.get("interaction_id", "")

        logger.info(f"GetResearchResult invoked for interaction_id: {interaction_id}")

        if not interaction_id:
            logger.warning("GetResearchResult called without interaction_id parameter")
            yield self.create_json_message({"error": "interaction_id is required"})
            yield self.create_text_message("Error: interaction_id is required")
            return

        try:
            client = self._get_client()
            logger.debug(f"Fetching interaction result from API...")
            interaction = client.interactions.get(interaction_id)
            logger.debug(f"Interaction status: {interaction.status}")

            if interaction.status != "completed":
                logger.info(f"Research not completed yet, status: {interaction.status}")
                
                yield self.create_json_message({
                    "interaction_id": interaction_id,
                    "status": interaction.status,
                    "error": "Research is not yet completed",
                })
                yield self.create_text_message(
                    f"## リサーチ未完了\n\n"
                    f"リサーチはまだ完了していません。\n"
                    f"- **Interaction ID**: `{interaction_id}`\n"
                    f"- **現在のステータス**: {interaction.status}\n\n"
                    f"`get_research_status` でステータスを確認してください。"
                )
                return

            # Extract content
            logger.debug("Extracting content from interaction outputs...")
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

            logger.info(f"GetResearchResult completed, content length: {len(content) if content else 0}, citations: {len(citations)}")

            # 常に両方を返す
            yield self.create_json_message({
                "interaction_id": interaction_id,
                "status": "completed",
                "content": content,
                "citations": citations,
            })
            yield self.create_text_message(self._format_result_output(interaction_id, content, citations))

        except Exception as e:
            logger.error(f"GetResearchResult failed with error: {str(e)}", exc_info=True)
            error_message = f"結果取得中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "interaction_id": interaction_id,
                "status": "error",
                "content": None,
                "citations": [],
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_result_output(self, interaction_id: str, content: str | None, citations: list) -> str:
        """Format the research result as readable text for LLM."""
        lines = [
            "## リサーチ結果",
            "",
            f"**Interaction ID**: `{interaction_id}`",
            "",
        ]
        
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
