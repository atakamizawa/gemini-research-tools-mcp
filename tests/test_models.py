"""Tests for data models."""

import pytest

from src.core.models import (
    FollowupRequest,
    ResearchEvent,
    ResearchEventType,
    ResearchRequest,
    ResearchResult,
    ResearchStatus,
    ResearchStatusEnum,
)


class TestResearchStatusEnum:
    """Tests for ResearchStatusEnum."""

    def test_values(self):
        """Test enum values."""
        assert ResearchStatusEnum.IN_PROGRESS == "in_progress"
        assert ResearchStatusEnum.COMPLETED == "completed"
        assert ResearchStatusEnum.FAILED == "failed"
        assert ResearchStatusEnum.CANCELLED == "cancelled"


class TestResearchStatus:
    """Tests for ResearchStatus model."""

    def test_create_status(self):
        """Test creating a status object."""
        status = ResearchStatus(
            interaction_id="interactions/abc123",
            status=ResearchStatusEnum.IN_PROGRESS,
        )
        assert status.interaction_id == "interactions/abc123"
        assert status.status == ResearchStatusEnum.IN_PROGRESS
        assert status.error is None

    def test_create_status_with_error(self):
        """Test creating a status object with error."""
        status = ResearchStatus(
            interaction_id="interactions/abc123",
            status=ResearchStatusEnum.FAILED,
            error="Something went wrong",
        )
        assert status.status == ResearchStatusEnum.FAILED
        assert status.error == "Something went wrong"


class TestResearchResult:
    """Tests for ResearchResult model."""

    def test_create_result(self):
        """Test creating a result object."""
        result = ResearchResult(
            interaction_id="interactions/abc123",
            status=ResearchStatusEnum.COMPLETED,
            content="Research report content",
            citations=["https://example.com/1", "https://example.com/2"],
        )
        assert result.interaction_id == "interactions/abc123"
        assert result.status == ResearchStatusEnum.COMPLETED
        assert result.content == "Research report content"
        assert len(result.citations) == 2
        assert result.error is None

    def test_create_failed_result(self):
        """Test creating a failed result object."""
        result = ResearchResult(
            interaction_id="interactions/abc123",
            status=ResearchStatusEnum.FAILED,
            error="Research failed",
        )
        assert result.status == ResearchStatusEnum.FAILED
        assert result.content is None
        assert result.citations is None
        assert result.error == "Research failed"


class TestResearchEventType:
    """Tests for ResearchEventType."""

    def test_values(self):
        """Test enum values."""
        assert ResearchEventType.START == "start"
        assert ResearchEventType.THOUGHT == "thought"
        assert ResearchEventType.TEXT_DELTA == "text_delta"
        assert ResearchEventType.COMPLETE == "complete"
        assert ResearchEventType.ERROR == "error"


class TestResearchEvent:
    """Tests for ResearchEvent model."""

    def test_create_start_event(self):
        """Test creating a start event."""
        event = ResearchEvent(
            event_type=ResearchEventType.START,
            interaction_id="interactions/abc123",
        )
        assert event.event_type == ResearchEventType.START
        assert event.interaction_id == "interactions/abc123"
        assert event.content is None

    def test_create_text_delta_event(self):
        """Test creating a text delta event."""
        event = ResearchEvent(
            event_type=ResearchEventType.TEXT_DELTA,
            interaction_id="interactions/abc123",
            content="Some text content",
            event_id="event_001",
        )
        assert event.event_type == ResearchEventType.TEXT_DELTA
        assert event.content == "Some text content"
        assert event.event_id == "event_001"

    def test_create_thought_event(self):
        """Test creating a thought event."""
        event = ResearchEvent(
            event_type=ResearchEventType.THOUGHT,
            interaction_id="interactions/abc123",
            content="Thinking about the query...",
        )
        assert event.event_type == ResearchEventType.THOUGHT
        assert event.content == "Thinking about the query..."


class TestResearchRequest:
    """Tests for ResearchRequest model."""

    def test_create_request(self):
        """Test creating a request object."""
        request = ResearchRequest(
            query="Research quantum computing",
        )
        assert request.query == "Research quantum computing"
        assert request.format_instructions is None
        assert request.stream is False
        assert request.timeout == 3600
        assert request.poll_interval == 10

    def test_create_request_with_options(self):
        """Test creating a request with all options."""
        request = ResearchRequest(
            query="Research AI trends",
            format_instructions="Include a table",
            stream=True,
            timeout=1800,
            poll_interval=5,
        )
        assert request.query == "Research AI trends"
        assert request.format_instructions == "Include a table"
        assert request.stream is True
        assert request.timeout == 1800
        assert request.poll_interval == 5


class TestFollowupRequest:
    """Tests for FollowupRequest model."""

    def test_create_followup_request(self):
        """Test creating a followup request."""
        request = FollowupRequest(
            previous_interaction_id="interactions/abc123",
            question="What are the main risks?",
        )
        assert request.previous_interaction_id == "interactions/abc123"
        assert request.question == "What are the main risks?"
