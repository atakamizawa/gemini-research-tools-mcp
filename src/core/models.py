"""Data models for Gemini Deep Research Agent."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResearchStatusEnum(str, Enum):
    """Research task status."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchStatus(BaseModel):
    """Status of a research task."""

    interaction_id: str = Field(..., description="Unique identifier for the interaction")
    status: ResearchStatusEnum = Field(..., description="Current status of the research")
    error: Optional[str] = Field(None, description="Error message if failed")


class ResearchResult(BaseModel):
    """Result of a completed research task."""

    interaction_id: str = Field(..., description="Unique identifier for the interaction")
    status: ResearchStatusEnum = Field(..., description="Final status of the research")
    content: Optional[str] = Field(None, description="Research report content")
    citations: Optional[list[str]] = Field(None, description="List of citation URLs")
    error: Optional[str] = Field(None, description="Error message if failed")


class ResearchEventType(str, Enum):
    """Types of streaming events."""

    START = "start"
    THOUGHT = "thought"
    TEXT_DELTA = "text_delta"
    COMPLETE = "complete"
    ERROR = "error"


class ResearchEvent(BaseModel):
    """Streaming event from research task."""

    event_type: ResearchEventType = Field(..., description="Type of the event")
    interaction_id: Optional[str] = Field(None, description="Interaction ID (available after start)")
    content: Optional[str] = Field(None, description="Event content (text or thought)")
    event_id: Optional[str] = Field(None, description="Event ID for reconnection")


class ResearchRequest(BaseModel):
    """Request to start a research task."""

    query: str = Field(..., description="Research query/topic")
    format_instructions: Optional[str] = Field(
        None, description="Optional formatting instructions for the output"
    )
    stream: bool = Field(False, description="Whether to stream the response")
    timeout: int = Field(3600, description="Timeout in seconds (default: 1 hour)")
    poll_interval: int = Field(10, description="Polling interval in seconds")


class FollowupRequest(BaseModel):
    """Request for a follow-up question."""

    previous_interaction_id: str = Field(..., description="ID of the completed research")
    question: str = Field(..., description="Follow-up question")


# Quick Search Models (Google Search + URL Context)


class Citation(BaseModel):
    """Citation from grounding metadata."""

    url: str = Field(..., description="Source URL")
    title: Optional[str] = Field(None, description="Source title")


class GroundingSupport(BaseModel):
    """Grounding support linking text to sources."""

    text: str = Field(..., description="Text segment that is grounded")
    start_index: int = Field(..., description="Start index in the response")
    end_index: int = Field(..., description="End index in the response")
    citation_indices: list[int] = Field(default_factory=list, description="Indices of supporting citations")


class QuickSearchResult(BaseModel):
    """Result from quick search using Google Search grounding."""

    query: str = Field(..., description="Original search query")
    content: str = Field(..., description="Generated response content")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")
    grounding_supports: list[GroundingSupport] = Field(
        default_factory=list, description="Grounding supports linking text to citations"
    )
    search_queries: list[str] = Field(default_factory=list, description="Search queries used by the model")
    model: str = Field(..., description="Model used for generation")
    error: Optional[str] = Field(None, description="Error message if failed")


class UrlMetadata(BaseModel):
    """Metadata for a retrieved URL."""

    url: str = Field(..., description="Retrieved URL")
    status: str = Field(..., description="Retrieval status")
    title: Optional[str] = Field(None, description="Page title if available")


class UrlAnalysisResult(BaseModel):
    """Result from URL analysis using URL Context tool."""

    urls: list[str] = Field(..., description="URLs that were analyzed")
    query: str = Field(..., description="Analysis query/instruction")
    content: str = Field(..., description="Generated analysis content")
    url_metadata: list[UrlMetadata] = Field(default_factory=list, description="Metadata for each URL")
    model: str = Field(..., description="Model used for generation")
    error: Optional[str] = Field(None, description="Error message if failed")


class QuickSearchRequest(BaseModel):
    """Request for quick search."""

    query: str = Field(..., description="Search query")
    model: str = Field("gemini-3-flash-preview", description="Model to use")
    language: str = Field("ja", description="Response language")


class UrlAnalysisRequest(BaseModel):
    """Request for URL analysis."""

    urls: list[str] = Field(..., description="URLs to analyze (max 20)")
    query: str = Field(..., description="Analysis query/instruction")
    model: str = Field("gemini-3-flash-preview", description="Model to use")
    language: str = Field("ja", description="Response language")
