"""MCP Server implementation for Gemini Deep Research Agent.

This module provides an MCP server that exposes the Gemini Deep Research
Agent functionality as tools that can be called by MCP clients like Cline.

Usage:
    # Run as a module
    python -m src.mcp.server

    # Or use with uvx/npx
    uvx mcp run src.mcp.server
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.core.client import DeepResearchClient, QuickSearchClient
from src.core.models import ResearchEventType, ResearchStatusEnum

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("Gemini Research Tools")

# Lazy initialization of clients
_deep_research_client: Optional[DeepResearchClient] = None
_quick_search_client: Optional[QuickSearchClient] = None

# HTTP timeout settings (in seconds)
# Note: Gemini API requires minimum 10 seconds deadline
DEEP_RESEARCH_HTTP_TIMEOUT = 3600  # 60 minutes for long-running research
QUICK_SEARCH_HTTP_TIMEOUT = 300  # 5 minutes for quick search operations


def get_deep_research_client() -> DeepResearchClient:
    """Get or create the DeepResearchClient instance."""
    global _deep_research_client
    if _deep_research_client is None:
        _deep_research_client = DeepResearchClient(http_timeout=DEEP_RESEARCH_HTTP_TIMEOUT)
    return _deep_research_client


def get_quick_search_client() -> QuickSearchClient:
    """Get or create the QuickSearchClient instance."""
    global _quick_search_client
    if _quick_search_client is None:
        _quick_search_client = QuickSearchClient(http_timeout=QUICK_SEARCH_HTTP_TIMEOUT)
    return _quick_search_client


# Alias for backward compatibility
def get_client() -> DeepResearchClient:
    """Get or create the DeepResearchClient instance (backward compatibility)."""
    return get_deep_research_client()


# =============================================================================
# Deep Research Tools (Minutes to complete, comprehensive reports)
# =============================================================================


@mcp.tool()
async def deep_research(
    query: str,
    format_instructions: Optional[str] = None,
    wait_for_completion: bool = True,
    timeout: int = 3600,
) -> dict:
    """Execute deep research on a topic using Gemini Deep Research Agent.

    This tool performs comprehensive research on the given topic, searching
    the web and synthesizing information into a detailed report with citations.

    **Note**: Research tasks typically take several minutes to complete.
    If wait_for_completion is True, this tool will wait until the research
    is finished (up to the timeout).

    Args:
        query: The research query or topic to investigate.
            Example: "What are the latest developments in quantum computing?"
        format_instructions: Optional instructions for formatting the output.
            Example: "Format as a technical report with sections: Executive Summary,
            Key Findings, Detailed Analysis, and Conclusions."
        wait_for_completion: If True (default), wait for research to complete.
            If False, return immediately with the interaction_id for later polling.
        timeout: Maximum seconds to wait for completion (default: 3600 = 1 hour).

    Returns:
        A dictionary containing:
        - interaction_id: Unique ID for this research task
        - status: "completed", "in_progress", or "failed"
        - content: The research report (if completed)
        - citations: List of source URLs (if available)
        - error: Error message (if failed)

    Example:
        >>> result = await deep_research(
        ...     "Compare the top 5 electric vehicle manufacturers",
        ...     format_instructions="Include a comparison table"
        ... )
        >>> print(result["content"])
    """
    client = get_deep_research_client()

    try:
        if wait_for_completion:
            # Execute research and wait for completion
            result = await client.research(
                query=query,
                format_instructions=format_instructions,
                timeout=timeout,
            )
            return {
                "interaction_id": result.interaction_id,
                "status": result.status.value,
                "content": result.content,
                "citations": result.citations,
                "error": result.error,
            }
        else:
            # Start research and return immediately
            interaction_id = await client.start_research(
                query=query,
                format_instructions=format_instructions,
            )
            return {
                "interaction_id": interaction_id,
                "status": "in_progress",
                "content": None,
                "citations": None,
                "error": None,
                "message": "Research started. Use get_research_status to check progress.",
            }
    except TimeoutError as e:
        return {
            "interaction_id": None,
            "status": "failed",
            "content": None,
            "citations": None,
            "error": str(e),
        }
    except Exception as e:
        return {
            "interaction_id": None,
            "status": "failed",
            "content": None,
            "citations": None,
            "error": f"Research failed: {str(e)}",
        }


@mcp.tool()
async def get_research_status(interaction_id: str) -> dict:
    """Get the status of an ongoing research task.

    Use this tool to check if a research task started with
    deep_research(wait_for_completion=False) has completed.

    Args:
        interaction_id: The interaction ID returned from deep_research.

    Returns:
        A dictionary containing:
        - interaction_id: The interaction ID
        - status: "in_progress", "completed", or "failed"
        - error: Error message (if failed)

    Example:
        >>> status = await get_research_status("interactions/abc123")
        >>> if status["status"] == "completed":
        ...     result = await get_research_result("interactions/abc123")
    """
    client = get_deep_research_client()

    try:
        status = await client.get_status(interaction_id)
        return {
            "interaction_id": status.interaction_id,
            "status": status.status.value,
            "error": status.error,
        }
    except Exception as e:
        return {
            "interaction_id": interaction_id,
            "status": "failed",
            "error": f"Failed to get status: {str(e)}",
        }


@mcp.tool()
async def get_research_result(interaction_id: str) -> dict:
    """Get the result of a completed research task.

    Use this tool to retrieve the full research report after
    get_research_status indicates the task is completed.

    Args:
        interaction_id: The interaction ID returned from deep_research.

    Returns:
        A dictionary containing:
        - interaction_id: The interaction ID
        - status: "completed", "in_progress", or "failed"
        - content: The research report (if completed)
        - citations: List of source URLs (if available)
        - error: Error message (if failed or not completed)

    Example:
        >>> result = await get_research_result("interactions/abc123")
        >>> print(result["content"])
    """
    client = get_deep_research_client()

    try:
        result = await client.get_result(interaction_id)
        return {
            "interaction_id": result.interaction_id,
            "status": result.status.value,
            "content": result.content,
            "citations": result.citations,
            "error": result.error,
        }
    except Exception as e:
        return {
            "interaction_id": interaction_id,
            "status": "failed",
            "content": None,
            "citations": None,
            "error": f"Failed to get result: {str(e)}",
        }


@mcp.tool()
async def ask_followup_question(
    previous_interaction_id: str,
    question: str,
) -> dict:
    """Ask a follow-up question about a completed research.

    Use this tool to get clarification, elaboration, or additional
    information based on a previously completed research task.

    Args:
        previous_interaction_id: The interaction ID of the completed research.
        question: The follow-up question to ask.
            Example: "Can you elaborate on the second point in the report?"

    Returns:
        A dictionary containing:
        - previous_interaction_id: The original research interaction ID
        - question: The follow-up question asked
        - answer: The response to the follow-up question
        - error: Error message (if failed)

    Example:
        >>> followup = await ask_followup_question(
        ...     "interactions/abc123",
        ...     "What are the main risks mentioned in the report?"
        ... )
        >>> print(followup["answer"])
    """
    client = get_deep_research_client()

    try:
        answer = await client.ask_followup(
            previous_interaction_id=previous_interaction_id,
            question=question,
        )
        return {
            "previous_interaction_id": previous_interaction_id,
            "question": question,
            "answer": answer,
            "error": None,
        }
    except Exception as e:
        return {
            "previous_interaction_id": previous_interaction_id,
            "question": question,
            "answer": None,
            "error": f"Failed to get follow-up answer: {str(e)}",
        }


@mcp.tool()
async def stream_research(
    query: str,
    format_instructions: Optional[str] = None,
) -> dict:
    """Execute deep research with streaming progress updates.

    This tool performs research while providing real-time updates
    on the agent's thinking process and intermediate results.

    **Note**: This returns the accumulated result after streaming completes.
    For true streaming in supported clients, use the streaming protocol.

    Args:
        query: The research query or topic to investigate.
        format_instructions: Optional instructions for formatting the output.

    Returns:
        A dictionary containing:
        - interaction_id: Unique ID for this research task
        - status: "completed" or "failed"
        - content: The accumulated research report
        - thoughts: List of thinking summaries during research
        - error: Error message (if failed)
    """
    client = get_deep_research_client()

    try:
        interaction_id = None
        content_parts = []
        thoughts = []

        async for event in client.stream_research(query, format_instructions):
            if event.event_type == ResearchEventType.START:
                interaction_id = event.interaction_id
            elif event.event_type == ResearchEventType.TEXT_DELTA:
                if event.content:
                    content_parts.append(event.content)
            elif event.event_type == ResearchEventType.THOUGHT:
                if event.content:
                    thoughts.append(event.content)
            elif event.event_type == ResearchEventType.ERROR:
                return {
                    "interaction_id": interaction_id,
                    "status": "failed",
                    "content": None,
                    "thoughts": thoughts,
                    "error": event.content,
                }

        return {
            "interaction_id": interaction_id,
            "status": "completed",
            "content": "".join(content_parts),
            "thoughts": thoughts,
            "error": None,
        }
    except Exception as e:
        return {
            "interaction_id": None,
            "status": "failed",
            "content": None,
            "thoughts": [],
            "error": f"Streaming research failed: {str(e)}",
        }


# =============================================================================
# Quick Search Tools (Seconds to complete, lightweight search)
# =============================================================================


@mcp.tool()
async def quick_search(
    query: str,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """Perform a quick web search using Google Search grounding.

    This tool uses Gemini's Google Search tool to find and synthesize
    information from the web. Unlike deep_research, responses typically
    return in seconds, making it ideal for quick lookups and real-time
    information needs.

    **Use this for:**
    - Quick fact-checking
    - Latest news and events
    - Simple questions requiring current information
    - When speed is more important than depth

    **Use deep_research instead for:**
    - Comprehensive analysis
    - Detailed reports with multiple sources
    - Complex topics requiring synthesis

    Args:
        query: The search query.
            Example: "最新のAIニュース" or "Who won the 2024 Nobel Prize in Physics?"
        model: Model to use (default: gemini-3-flash-preview).
            Options: gemini-3-pro-preview, gemini-3-flash-preview
        language: Response language (default: ja for Japanese).
            Use "en" for English responses.

    Returns:
        A dictionary containing:
        - query: The original search query
        - content: The generated response
        - citations: List of source citations with URLs and titles
        - search_queries: Search queries used by the model
        - model: Model used for generation
        - error: Error message (if failed)

    Example:
        >>> result = await quick_search("2024年のノーベル物理学賞")
        >>> print(result["content"])
        >>> for citation in result["citations"]:
        ...     print(f"- {citation['title']}: {citation['url']}")
    """
    client = get_quick_search_client()

    try:
        result = await client.quick_search(
            query=query,
            model=model,
            language=language,
        )
        return {
            "query": result.query,
            "content": result.content,
            "citations": [{"url": c.url, "title": c.title} for c in result.citations],
            "search_queries": result.search_queries,
            "model": result.model,
            "error": result.error,
        }
    except Exception as e:
        return {
            "query": query,
            "content": None,
            "citations": [],
            "search_queries": [],
            "model": model,
            "error": f"Quick search failed: {str(e)}",
        }


@mcp.tool()
async def analyze_urls(
    urls: list[str],
    query: str,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """Analyze content from specific URLs using URL Context tool.

    This tool fetches and analyzes content from the provided URLs.
    Useful for comparing documents, extracting data, summarizing
    specific pages, or answering questions about particular web content.

    **Use this for:**
    - Comparing multiple articles or documents
    - Extracting specific information from known URLs
    - Summarizing web pages
    - Analyzing code repositories or documentation

    **Limitations:**
    - Maximum 20 URLs per request
    - Maximum 34MB content per URL
    - Does not support: paywalled content, YouTube videos, Google Workspace files

    Args:
        urls: List of URLs to analyze (max 20).
            Example: ["https://example.com/article1", "https://example.com/article2"]
        query: Analysis query or instruction.
            Example: "これらの記事の主な違いを比較してください"
        model: Model to use (default: gemini-3-flash-preview).
            Options: gemini-3-pro-preview, gemini-3-flash-preview
        language: Response language (default: ja for Japanese).

    Returns:
        A dictionary containing:
        - urls: The URLs that were analyzed
        - query: The analysis query
        - content: The generated analysis
        - url_metadata: Metadata for each URL (retrieval status)
        - model: Model used for generation
        - error: Error message (if failed)

    Example:
        >>> result = await analyze_urls(
        ...     urls=["https://example.com/recipe1", "https://example.com/recipe2"],
        ...     query="これらのレシピの材料と調理時間を比較してください"
        ... )
        >>> print(result["content"])
    """
    client = get_quick_search_client()

    try:
        result = await client.analyze_urls(
            urls=urls,
            query=query,
            model=model,
            language=language,
        )
        return {
            "urls": result.urls,
            "query": result.query,
            "content": result.content,
            "url_metadata": [
                {"url": m.url, "status": m.status, "title": m.title}
                for m in result.url_metadata
            ],
            "model": result.model,
            "error": result.error,
        }
    except Exception as e:
        return {
            "urls": urls,
            "query": query,
            "content": None,
            "url_metadata": [],
            "model": model,
            "error": f"URL analysis failed: {str(e)}",
        }


@mcp.tool()
async def search_and_analyze(
    query: str,
    urls: Optional[list[str]] = None,
    model: str = "gemini-3-flash-preview",
    language: str = "ja",
) -> dict:
    """Perform web search with optional URL context for deeper analysis.

    This tool combines Google Search grounding with URL Context to provide
    both broad search results and deep analysis of specific pages. Use this
    when you need to compare web search results with specific documents.

    **Use this for:**
    - Comparing search results with specific reference documents
    - Fact-checking claims against authoritative sources
    - Combining general research with specific URL analysis

    Args:
        query: The search/analysis query.
            Example: "最新のEV市場動向と、この記事の内容を比較してください"
        urls: Optional list of URLs to include for context.
            Example: ["https://example.com/ev-report-2024"]
        model: Model to use (default: gemini-3-flash-preview).
            Options: gemini-3-pro-preview, gemini-3-flash-preview
        language: Response language (default: ja for Japanese).

    Returns:
        A dictionary containing:
        - query: The original query
        - content: The generated response
        - citations: List of source citations
        - search_queries: Search queries used by the model
        - model: Model used for generation
        - error: Error message (if failed)

    Example:
        >>> result = await search_and_analyze(
        ...     query="最新のEV市場動向と、この記事の内容を比較してください",
        ...     urls=["https://example.com/ev-report-2024"]
        ... )
        >>> print(result["content"])
    """
    client = get_quick_search_client()

    try:
        result = await client.search_and_analyze(
            query=query,
            urls=urls,
            model=model,
            language=language,
        )
        return {
            "query": result.query,
            "content": result.content,
            "citations": [{"url": c.url, "title": c.title} for c in result.citations],
            "search_queries": result.search_queries,
            "model": result.model,
            "error": result.error,
        }
    except Exception as e:
        return {
            "query": query,
            "content": None,
            "citations": [],
            "search_queries": [],
            "model": model,
            "error": f"Search and analyze failed: {str(e)}",
        }


def main():
    """Entry point for the MCP server.
    
    This function is called when running the server via:
    - `grt-mcp` command (after pip install)
    - `uvx --from git+https://github.com/atakamizawa/gemini-research-tools-mcp grt-mcp`
    - `python -m src.mcp.server`
    """
    mcp.run()


# Entry point for running the server
if __name__ == "__main__":
    main()
