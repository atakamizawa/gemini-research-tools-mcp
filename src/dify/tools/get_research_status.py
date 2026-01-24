"""Tool to check the status of an ongoing research task."""

import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google import genai

# Set up logging
logger = logging.getLogger(__name__)


class GetResearchStatusTool(Tool):
    """Tool to check the status of an ongoing research task.
    
    Use this to check if a deep research task has completed.
    """

    def _get_client(self) -> genai.Client:
        """Get the Gemini client with credentials."""
        api_key = self.runtime.credentials.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
        return genai.Client(api_key=api_key)

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """Check research status.

        Args:
            tool_parameters: Dictionary containing:
                - interaction_id: The interaction ID to check
        """
        interaction_id = tool_parameters.get("interaction_id", "")

        logger.info(f"GetResearchStatus invoked for interaction_id: {interaction_id}")

        if not interaction_id:
            logger.warning("GetResearchStatus called without interaction_id parameter")
            yield self.create_json_message({"error": "interaction_id is required"})
            yield self.create_text_message("Error: interaction_id is required")
            return

        try:
            client = self._get_client()
            logger.debug(f"Fetching interaction status from API...")
            interaction = client.interactions.get(interaction_id)

            status_map = {
                "in_progress": "in_progress",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }

            status = status_map.get(interaction.status, "unknown")
            error = getattr(interaction, "error", None)
            logger.info(f"Research status for {interaction_id}: {status}")

            # 常に両方を返す
            yield self.create_json_message({
                "interaction_id": interaction_id,
                "status": status,
                "error": str(error) if error else None,
            })
            yield self.create_text_message(self._format_status_output(interaction_id, status, error))

        except Exception as e:
            logger.error(f"GetResearchStatus failed with error: {str(e)}", exc_info=True)
            error_message = f"ステータス確認中にエラーが発生しました: {str(e)}"
            
            yield self.create_json_message({
                "interaction_id": interaction_id,
                "status": "error",
                "error": str(e),
            })
            yield self.create_text_message(f"## エラー\n\n{error_message}")

    def _format_status_output(self, interaction_id: str, status: str, error: Any) -> str:
        """Format the status as readable text for LLM."""
        status_ja = {
            "in_progress": "進行中",
            "completed": "完了",
            "failed": "失敗",
            "cancelled": "キャンセル",
            "unknown": "不明",
        }
        
        lines = [
            "## リサーチステータス",
            "",
            f"- **Interaction ID**: `{interaction_id}`",
            f"- **ステータス**: {status_ja.get(status, status)}",
        ]
        
        if error:
            lines.append(f"- **エラー**: {error}")
        
        if status == "completed":
            lines.append("")
            lines.append("`get_research_result` で結果を取得できます。")
        elif status == "in_progress":
            lines.append("")
            lines.append("リサーチはまだ進行中です。しばらく待ってから再度確認してください。")
        
        return "\n".join(lines)
