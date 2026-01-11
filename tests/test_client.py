"""Tests for DeepResearchClient."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.client import DeepResearchClient
from src.core.models import ResearchEventType, ResearchStatusEnum


class TestDeepResearchClientInit:
    """Tests for DeepResearchClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("src.core.client.genai.Client"):
            client = DeepResearchClient(api_key="test-api-key")
            assert client._api_key == "test-api-key"

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"}):
            with patch("src.core.client.genai.Client"):
                client = DeepResearchClient()
                assert client._api_key == "env-api-key"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GEMINI_API_KEY if it exists
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                DeepResearchClient()


class TestDeepResearchClientBuildInput:
    """Tests for _build_input method."""

    def test_build_input_query_only(self):
        """Test building input with query only."""
        with patch("src.core.client.genai.Client"):
            client = DeepResearchClient(api_key="test-key")
            result = client._build_input("Test query")
            assert result == "Test query"

    def test_build_input_with_format_instructions(self):
        """Test building input with format instructions."""
        with patch("src.core.client.genai.Client"):
            client = DeepResearchClient(api_key="test-key")
            result = client._build_input("Test query", "Include a table")
            assert result == "Test query\n\nInclude a table"


class TestDeepResearchClientStartResearch:
    """Tests for start_research method."""

    @pytest.mark.asyncio
    async def test_start_research(self):
        """Test starting a research task."""
        mock_interaction = MagicMock()
        mock_interaction.id = "interactions/test123"

        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            interaction_id = await client.start_research("Test query")

            assert interaction_id == "interactions/test123"
            mock_client.interactions.create.assert_called_once()


class TestDeepResearchClientGetStatus:
    """Tests for get_status method."""

    @pytest.mark.asyncio
    async def test_get_status_in_progress(self):
        """Test getting status of in-progress research."""
        mock_interaction = MagicMock()
        mock_interaction.status = "in_progress"
        mock_interaction.error = None

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            status = await client.get_status("interactions/test123")

            assert status.interaction_id == "interactions/test123"
            assert status.status == ResearchStatusEnum.IN_PROGRESS
            assert status.error is None

    @pytest.mark.asyncio
    async def test_get_status_completed(self):
        """Test getting status of completed research."""
        mock_interaction = MagicMock()
        mock_interaction.status = "completed"

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            status = await client.get_status("interactions/test123")

            assert status.status == ResearchStatusEnum.COMPLETED

    @pytest.mark.asyncio
    async def test_get_status_failed(self):
        """Test getting status of failed research."""
        mock_interaction = MagicMock()
        mock_interaction.status = "failed"
        mock_interaction.error = "Something went wrong"

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            status = await client.get_status("interactions/test123")

            assert status.status == ResearchStatusEnum.FAILED
            assert status.error == "Something went wrong"


class TestDeepResearchClientGetResult:
    """Tests for get_result method."""

    @pytest.mark.asyncio
    async def test_get_result_completed(self):
        """Test getting result of completed research."""
        mock_output = MagicMock()
        mock_output.text = "Research report content"

        mock_interaction = MagicMock()
        mock_interaction.status = "completed"
        mock_interaction.outputs = [mock_output]

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            result = await client.get_result("interactions/test123")

            assert result.interaction_id == "interactions/test123"
            assert result.status == ResearchStatusEnum.COMPLETED
            assert result.content == "Research report content"

    @pytest.mark.asyncio
    async def test_get_result_in_progress(self):
        """Test getting result of in-progress research."""
        mock_interaction = MagicMock()
        mock_interaction.status = "in_progress"
        mock_interaction.outputs = []

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            result = await client.get_result("interactions/test123")

            assert result.status == ResearchStatusEnum.IN_PROGRESS
            assert result.content is None


class TestDeepResearchClientPollUntilComplete:
    """Tests for poll_until_complete method."""

    @pytest.mark.asyncio
    async def test_poll_until_complete_immediate(self):
        """Test polling when research completes immediately."""
        mock_output = MagicMock()
        mock_output.text = "Research report"

        mock_interaction = MagicMock()
        mock_interaction.status = "completed"
        mock_interaction.outputs = [mock_output]

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            result = await client.poll_until_complete(
                "interactions/test123",
                poll_interval=1,
                timeout=10,
            )

            assert result.status == ResearchStatusEnum.COMPLETED
            assert result.content == "Research report"

    @pytest.mark.asyncio
    async def test_poll_until_complete_failed(self):
        """Test polling when research fails."""
        mock_interaction = MagicMock()
        mock_interaction.status = "failed"
        mock_interaction.error = "Research failed"
        mock_interaction.outputs = []

        mock_client = MagicMock()
        mock_client.interactions.get.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            result = await client.poll_until_complete(
                "interactions/test123",
                poll_interval=1,
                timeout=10,
            )

            assert result.status == ResearchStatusEnum.FAILED


class TestDeepResearchClientAskFollowup:
    """Tests for ask_followup method."""

    @pytest.mark.asyncio
    async def test_ask_followup(self):
        """Test asking a follow-up question."""
        mock_output = MagicMock()
        mock_output.text = "Follow-up answer"

        mock_interaction = MagicMock()
        mock_interaction.outputs = [mock_output]

        mock_client = MagicMock()
        mock_client.interactions.create.return_value = mock_interaction

        with patch("src.core.client.genai.Client", return_value=mock_client):
            client = DeepResearchClient(api_key="test-key")
            answer = await client.ask_followup(
                "interactions/test123",
                "What are the main risks?",
            )

            assert answer == "Follow-up answer"
            mock_client.interactions.create.assert_called_once()
