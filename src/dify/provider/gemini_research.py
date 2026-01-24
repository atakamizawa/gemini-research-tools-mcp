"""Gemini Research Provider implementation."""

from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.quick_search import QuickSearchTool


class GeminiResearchProvider(ToolProvider):
    """Provider for Gemini Research Tools.

    Validates the Gemini API key by performing a simple quick search.
    """

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """Validate the Gemini API key.

        Args:
            credentials: Dictionary containing gemini_api_key

        Raises:
            ToolProviderCredentialValidationError: If validation fails
        """
        try:
            # Perform a simple search to validate the API key
            for _ in QuickSearchTool.from_credentials(credentials).invoke(
                tool_parameters={
                    "query": "test",
                    "language": "en",
                }
            ):
                pass
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e)) from e
