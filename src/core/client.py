"""Client for Gemini Deep Research Agent and Quick Search Tools."""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from typing import Optional

from google import genai
from google.genai import types

from .models import (
    Citation,
    GroundingSupport,
    QuickSearchResult,
    ResearchEvent,
    ResearchEventType,
    ResearchResult,
    ResearchStatus,
    ResearchStatusEnum,
    UrlAnalysisResult,
    UrlMetadata,
)


class DeepResearchClient:
    """Client for interacting with Gemini Deep Research Agent.

    This client wraps the Google GenAI Interactions API to provide
    a simplified interface for deep research tasks.

    Example:
        ```python
        client = DeepResearchClient()

        # Start research and wait for completion
        result = await client.research("History of quantum computing")
        print(result.content)

        # Or stream the research progress
        async for event in client.stream_research("AI trends in 2025"):
            if event.event_type == ResearchEventType.TEXT_DELTA:
                print(event.content, end="")
        ```
    """

    AGENT_NAME = "deep-research-pro-preview-12-2025"
    DEFAULT_POLL_INTERVAL = 10  # seconds
    DEFAULT_TIMEOUT = 3600  # 1 hour

    # HTTP timeout for Deep Research (60 minutes for long-running tasks)
    DEFAULT_HTTP_TIMEOUT = 3600  # 60 minutes

    def __init__(self, api_key: Optional[str] = None, http_timeout: Optional[int] = None):
        """Initialize the client.

        Args:
            api_key: Gemini API key. If not provided, uses GEMINI_API_KEY env var.
            http_timeout: HTTP request timeout in seconds (currently unused, SDK uses defaults).
        """
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Use SDK defaults for HTTP options to avoid timeout issues
        self._client = genai.Client(api_key=self._api_key)

    # 日本語回答を指示するシステムプロンプト
    SYSTEM_PROMPT_JA = "必ず日本語で回答してください。推論過程や思考の要約も日本語で出力してください。"

    def _build_input(self, query: str, format_instructions: Optional[str] = None) -> str:
        """Build the input prompt with optional format instructions."""
        parts = [self.SYSTEM_PROMPT_JA, query]
        if format_instructions:
            parts.append(format_instructions)
        return "\n\n".join(parts)

    async def start_research(
        self,
        query: str,
        format_instructions: Optional[str] = None,
    ) -> str:
        """Start a research task asynchronously.

        Args:
            query: The research query/topic.
            format_instructions: Optional formatting instructions for the output.

        Returns:
            The interaction ID for tracking the research.

        Example:
            ```python
            interaction_id = await client.start_research(
                "Compare EV battery technologies",
                format_instructions="Include a comparison table"
            )
            ```
        """
        input_text = self._build_input(query, format_instructions)

        # Run in executor since the SDK is synchronous
        loop = asyncio.get_event_loop()
        interaction = await loop.run_in_executor(
            None,
            lambda: self._client.interactions.create(
                input=input_text,
                agent=self.AGENT_NAME,
                background=True,
            ),
        )

        return interaction.id

    async def get_status(self, interaction_id: str) -> ResearchStatus:
        """Get the status of a research task.

        Args:
            interaction_id: The interaction ID from start_research.

        Returns:
            ResearchStatus with current status information.
        """
        loop = asyncio.get_event_loop()
        interaction = await loop.run_in_executor(
            None,
            lambda: self._client.interactions.get(interaction_id),
        )

        status_map = {
            "in_progress": ResearchStatusEnum.IN_PROGRESS,
            "completed": ResearchStatusEnum.COMPLETED,
            "failed": ResearchStatusEnum.FAILED,
            "cancelled": ResearchStatusEnum.CANCELLED,
        }

        status = status_map.get(interaction.status, ResearchStatusEnum.IN_PROGRESS)
        error = getattr(interaction, "error", None)

        return ResearchStatus(
            interaction_id=interaction_id,
            status=status,
            error=str(error) if error else None,
        )

    async def get_result(self, interaction_id: str) -> ResearchResult:
        """Get the result of a completed research task.

        Args:
            interaction_id: The interaction ID from start_research.

        Returns:
            ResearchResult with the research content and citations.
        """
        loop = asyncio.get_event_loop()
        interaction = await loop.run_in_executor(
            None,
            lambda: self._client.interactions.get(interaction_id),
        )

        status_map = {
            "in_progress": ResearchStatusEnum.IN_PROGRESS,
            "completed": ResearchStatusEnum.COMPLETED,
            "failed": ResearchStatusEnum.FAILED,
            "cancelled": ResearchStatusEnum.CANCELLED,
        }

        status = status_map.get(interaction.status, ResearchStatusEnum.IN_PROGRESS)

        content = None
        citations = None

        if status == ResearchStatusEnum.COMPLETED and interaction.outputs:
            # Get the last text output
            for output in reversed(interaction.outputs):
                if hasattr(output, "text") and output.text:
                    content = output.text
                    break

            # Extract citations if available
            citations = self._extract_citations(interaction)

        error = getattr(interaction, "error", None)

        return ResearchResult(
            interaction_id=interaction_id,
            status=status,
            content=content,
            citations=citations,
            error=str(error) if error else None,
        )

    def _extract_citations(self, interaction) -> Optional[list[str]]:
        """Extract citation URLs from the interaction outputs."""
        citations = []
        for output in interaction.outputs:
            # Check for grounding metadata or citations
            if hasattr(output, "grounding_metadata"):
                metadata = output.grounding_metadata
                if hasattr(metadata, "grounding_chunks"):
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web") and hasattr(chunk.web, "uri"):
                            citations.append(chunk.web.uri)
        return citations if citations else None

    async def poll_until_complete(
        self,
        interaction_id: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ResearchResult:
        """Poll until the research is complete.

        Args:
            interaction_id: The interaction ID from start_research.
            poll_interval: Seconds between status checks (default: 10).
            timeout: Maximum seconds to wait (default: 3600).

        Returns:
            ResearchResult with the final result.

        Raises:
            TimeoutError: If the research doesn't complete within timeout.
        """
        start_time = time.time()

        while True:
            status = await self.get_status(interaction_id)

            if status.status == ResearchStatusEnum.COMPLETED:
                return await self.get_result(interaction_id)

            if status.status in (ResearchStatusEnum.FAILED, ResearchStatusEnum.CANCELLED):
                return ResearchResult(
                    interaction_id=interaction_id,
                    status=status.status,
                    error=status.error,
                )

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Research did not complete within {timeout} seconds. "
                    f"Interaction ID: {interaction_id}"
                )

            await asyncio.sleep(poll_interval)

    async def research(
        self,
        query: str,
        format_instructions: Optional[str] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> ResearchResult:
        """Execute research and wait for completion.

        This is a convenience method that combines start_research and poll_until_complete.

        Args:
            query: The research query/topic.
            format_instructions: Optional formatting instructions.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            ResearchResult with the research content.
        """
        interaction_id = await self.start_research(query, format_instructions)
        return await self.poll_until_complete(interaction_id, poll_interval, timeout)

    async def stream_research(
        self,
        query: str,
        format_instructions: Optional[str] = None,
    ) -> AsyncGenerator[ResearchEvent, None]:
        """Stream research progress in real-time.

        Args:
            query: The research query/topic.
            format_instructions: Optional formatting instructions.

        Yields:
            ResearchEvent objects with progress updates.

        Example:
            ```python
            async for event in client.stream_research("AI trends"):
                if event.event_type == ResearchEventType.THOUGHT:
                    print(f"Thinking: {event.content}")
                elif event.event_type == ResearchEventType.TEXT_DELTA:
                    print(event.content, end="")
            ```
        """
        input_text = self._build_input(query, format_instructions)

        # Create streaming request
        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(
            None,
            lambda: self._client.interactions.create(
                input=input_text,
                agent=self.AGENT_NAME,
                background=True,
                stream=True,
                agent_config={
                    "type": "deep-research",
                    "thinking_summaries": "auto",
                },
            ),
        )

        interaction_id = None
        last_event_id = None

        # Process stream events
        for chunk in stream:
            # Capture interaction ID from start event
            if chunk.event_type == "interaction.start":
                interaction_id = chunk.interaction.id
                yield ResearchEvent(
                    event_type=ResearchEventType.START,
                    interaction_id=interaction_id,
                )

            # Track event ID for potential reconnection
            if hasattr(chunk, "event_id") and chunk.event_id:
                last_event_id = chunk.event_id

            # Handle content deltas
            if chunk.event_type == "content.delta":
                if hasattr(chunk, "delta"):
                    delta = chunk.delta
                    if delta.type == "text":
                        yield ResearchEvent(
                            event_type=ResearchEventType.TEXT_DELTA,
                            interaction_id=interaction_id,
                            content=delta.text,
                            event_id=last_event_id,
                        )
                    elif delta.type == "thought_summary":
                        thought_text = (
                            delta.content.text
                            if hasattr(delta, "content") and hasattr(delta.content, "text")
                            else str(delta)
                        )
                        yield ResearchEvent(
                            event_type=ResearchEventType.THOUGHT,
                            interaction_id=interaction_id,
                            content=thought_text,
                            event_id=last_event_id,
                        )

            # Handle completion
            elif chunk.event_type == "interaction.complete":
                # Get final result from the completed interaction
                final_content = None
                if hasattr(chunk, "interaction") and chunk.interaction:
                    if hasattr(chunk.interaction, "outputs") and chunk.interaction.outputs:
                        for output in reversed(chunk.interaction.outputs):
                            if hasattr(output, "text") and output.text:
                                final_content = output.text
                                break
                
                # If final_content is not available from the event, fetch it via API
                if final_content is None and interaction_id:
                    try:
                        result = await self.get_result(interaction_id)
                        if result.content:
                            final_content = result.content
                    except Exception:
                        pass  # Will be handled by the UI layer
                
                yield ResearchEvent(
                    event_type=ResearchEventType.COMPLETE,
                    interaction_id=interaction_id,
                    content=final_content,
                    event_id=last_event_id,
                )

            # Handle errors
            elif chunk.event_type == "error":
                error_msg = getattr(chunk, "error", "Unknown error")
                yield ResearchEvent(
                    event_type=ResearchEventType.ERROR,
                    interaction_id=interaction_id,
                    content=str(error_msg),
                    event_id=last_event_id,
                )

    async def ask_followup(
        self,
        previous_interaction_id: str,
        question: str,
    ) -> str:
        """Ask a follow-up question about a completed research.

        Args:
            previous_interaction_id: The interaction ID of the completed research.
            question: The follow-up question.

        Returns:
            The answer to the follow-up question.
        """
        loop = asyncio.get_event_loop()
        interaction = await loop.run_in_executor(
            None,
            lambda: self._client.interactions.create(
                input=question,
                agent=self.AGENT_NAME,
                previous_interaction_id=previous_interaction_id,
            ),
        )

        # Get the text response
        if interaction.outputs:
            for output in reversed(interaction.outputs):
                if hasattr(output, "text") and output.text:
                    return output.text

        return ""


class QuickSearchClient:
    """Client for quick search using Google Search and URL Context tools.

    This client provides fast, lightweight search capabilities using
    Gemini models with Google Search grounding and URL Context tools.
    Unlike DeepResearchClient, responses are returned in seconds.

    Example:
        ```python
        client = QuickSearchClient()

        # Quick web search
        result = await client.quick_search("最新のAIニュース")
        print(result.content)
        for citation in result.citations:
            print(f"- {citation.title}: {citation.url}")

        # Analyze specific URLs
        result = await client.analyze_urls(
            urls=["https://example.com/article1", "https://example.com/article2"],
            query="これらの記事の主な違いを比較してください"
        )
        print(result.content)
        ```
    """

    DEFAULT_MODEL = "gemini-3-flash-preview"
    SUPPORTED_MODELS = [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    ]

    # HTTP timeout for Quick Search (5 minutes - Google Search grounding can take time)
    # Note: Minimum allowed deadline by Gemini API is 10 seconds
    DEFAULT_HTTP_TIMEOUT = 300  # 5 minutes

    def __init__(self, api_key: Optional[str] = None, http_timeout: Optional[int] = None):
        """Initialize the client.

        Args:
            api_key: Gemini API key. If not provided, uses GEMINI_API_KEY env var.
            http_timeout: HTTP request timeout in seconds (currently unused, SDK uses defaults).
        """
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Use SDK defaults for HTTP options to avoid timeout issues
        self._client = genai.Client(api_key=self._api_key)

    # 日本語回答を指示するシステムプロンプト
    SYSTEM_PROMPT_JA = "必ず日本語で回答してください。"

    def _build_prompt(self, query: str, language: str = "ja") -> str:
        """Build the prompt with language instruction."""
        if language == "ja":
            return f"{self.SYSTEM_PROMPT_JA}\n\n{query}"
        return query

    async def quick_search(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        language: str = "ja",
    ) -> QuickSearchResult:
        """Perform a quick web search using Google Search grounding.

        This method uses Gemini's Google Search tool to find and synthesize
        information from the web. Responses typically return in seconds.

        Args:
            query: The search query.
            model: Model to use (default: gemini-3-flash-preview).
                Options: gemini-3-pro-preview, gemini-3-flash-preview
            language: Response language (default: ja for Japanese).

        Returns:
            QuickSearchResult with content, citations, and grounding info.

        Example:
            ```python
            result = await client.quick_search("2024年のノーベル物理学賞")
            print(result.content)
            ```
        """
        prompt = self._build_prompt(query, language)

        # Configure Google Search tool
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[google_search_tool])

        try:
            # Run in executor since the SDK is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                ),
            )

            # Extract content
            content = response.text if response.text else ""

            # Extract grounding metadata
            citations = []
            grounding_supports = []
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
                                citations.append(
                                    Citation(
                                        url=chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                        title=chunk.web.title if hasattr(chunk.web, "title") else None,
                                    )
                                )

                    # Extract grounding supports
                    if hasattr(metadata, "grounding_supports") and metadata.grounding_supports:
                        for support in metadata.grounding_supports:
                            if hasattr(support, "segment") and support.segment:
                                segment = support.segment
                                grounding_supports.append(
                                    GroundingSupport(
                                        text=segment.text if hasattr(segment, "text") else "",
                                        start_index=segment.start_index if hasattr(segment, "start_index") else 0,
                                        end_index=segment.end_index if hasattr(segment, "end_index") else 0,
                                        citation_indices=list(support.grounding_chunk_indices)
                                        if hasattr(support, "grounding_chunk_indices")
                                        else [],
                                    )
                                )

            return QuickSearchResult(
                query=query,
                content=content,
                citations=citations,
                grounding_supports=grounding_supports,
                search_queries=search_queries,
                model=model,
            )

        except Exception as e:
            return QuickSearchResult(
                query=query,
                content="",
                citations=[],
                grounding_supports=[],
                search_queries=[],
                model=model,
                error=str(e),
            )

    async def analyze_urls(
        self,
        urls: list[str],
        query: str,
        model: str = DEFAULT_MODEL,
        language: str = "ja",
    ) -> UrlAnalysisResult:
        """Analyze content from specific URLs.

        This method uses Gemini's URL Context tool to fetch and analyze
        content from the provided URLs. Useful for comparing documents,
        extracting data, or summarizing specific pages.

        Args:
            urls: List of URLs to analyze (max 20).
            query: Analysis query or instruction.
            model: Model to use (default: gemini-3-flash-preview).
                Options: gemini-3-pro-preview, gemini-3-flash-preview
            language: Response language (default: ja for Japanese).

        Returns:
            UrlAnalysisResult with analysis content and URL metadata.

        Example:
            ```python
            result = await client.analyze_urls(
                urls=["https://example.com/recipe1", "https://example.com/recipe2"],
                query="これらのレシピの材料と調理時間を比較してください"
            )
            print(result.content)
            ```
        """
        if len(urls) > 20:
            return UrlAnalysisResult(
                urls=urls,
                query=query,
                content="",
                url_metadata=[],
                model=model,
                error="Maximum 20 URLs allowed per request",
            )

        # Build prompt with URLs
        url_list = "\n".join(urls)
        full_prompt = self._build_prompt(f"{query}\n\nURLs:\n{url_list}", language)

        # Configure URL Context tool
        url_context_tool = types.Tool(url_context=types.UrlContext())
        config = types.GenerateContentConfig(tools=[url_context_tool])

        try:
            # Run in executor since the SDK is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=config,
                ),
            )

            # Extract content
            content = response.text if response.text else ""

            # Extract URL metadata
            url_metadata = []
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "url_context_metadata") and candidate.url_context_metadata:
                    metadata = candidate.url_context_metadata
                    if hasattr(metadata, "url_metadata") and metadata.url_metadata:
                        for url_meta in metadata.url_metadata:
                            url_metadata.append(
                                UrlMetadata(
                                    url=url_meta.retrieved_url if hasattr(url_meta, "retrieved_url") else "",
                                    status=url_meta.url_retrieval_status
                                    if hasattr(url_meta, "url_retrieval_status")
                                    else "unknown",
                                    title=None,  # Title not provided in URL context metadata
                                )
                            )

            return UrlAnalysisResult(
                urls=urls,
                query=query,
                content=content,
                url_metadata=url_metadata,
                model=model,
            )

        except Exception as e:
            return UrlAnalysisResult(
                urls=urls,
                query=query,
                content="",
                url_metadata=[],
                model=model,
                error=str(e),
            )

    async def search_and_analyze(
        self,
        query: str,
        urls: Optional[list[str]] = None,
        model: str = DEFAULT_MODEL,
        language: str = "ja",
    ) -> QuickSearchResult:
        """Perform search with optional URL context for deeper analysis.

        This method combines Google Search grounding with URL Context
        to provide both broad search results and deep analysis of
        specific pages.

        Args:
            query: The search/analysis query.
            urls: Optional list of URLs to include for context.
            model: Model to use (default: gemini-3-flash-preview).
                Options: gemini-3-pro-preview, gemini-3-flash-preview
            language: Response language (default: ja for Japanese).

        Returns:
            QuickSearchResult with combined search and URL analysis.

        Example:
            ```python
            result = await client.search_and_analyze(
                query="最新のEV市場動向と、この記事の内容を比較してください",
                urls=["https://example.com/ev-report-2024"]
            )
            print(result.content)
            ```
        """
        # Build prompt
        prompt = self._build_prompt(query, language)
        if urls:
            url_list = "\n".join(urls)
            prompt = f"{prompt}\n\n参考URL:\n{url_list}"

        # Configure both tools
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(url_context=types.UrlContext()),
        ]
        config = types.GenerateContentConfig(tools=tools)

        try:
            # Run in executor since the SDK is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                ),
            )

            # Extract content
            content = response.text if response.text else ""

            # Extract grounding metadata (same as quick_search)
            citations = []
            grounding_supports = []
            search_queries = []

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                    metadata = candidate.grounding_metadata

                    if hasattr(metadata, "web_search_queries") and metadata.web_search_queries:
                        search_queries = list(metadata.web_search_queries)

                    if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web:
                                citations.append(
                                    Citation(
                                        url=chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                        title=chunk.web.title if hasattr(chunk.web, "title") else None,
                                    )
                                )

                    if hasattr(metadata, "grounding_supports") and metadata.grounding_supports:
                        for support in metadata.grounding_supports:
                            if hasattr(support, "segment") and support.segment:
                                segment = support.segment
                                grounding_supports.append(
                                    GroundingSupport(
                                        text=segment.text if hasattr(segment, "text") else "",
                                        start_index=segment.start_index if hasattr(segment, "start_index") else 0,
                                        end_index=segment.end_index if hasattr(segment, "end_index") else 0,
                                        citation_indices=list(support.grounding_chunk_indices)
                                        if hasattr(support, "grounding_chunk_indices")
                                        else [],
                                    )
                                )

            return QuickSearchResult(
                query=query,
                content=content,
                citations=citations,
                grounding_supports=grounding_supports,
                search_queries=search_queries,
                model=model,
            )

        except Exception as e:
            return QuickSearchResult(
                query=query,
                content="",
                citations=[],
                grounding_supports=[],
                search_queries=[],
                model=model,
                error=str(e),
            )
