"""CLI for Gemini Deep Research Agent.

Usage:
    # Execute deep research
    deep-research research "Your research query"

    # Execute research with streaming
    deep-research research "Your research query" --stream

    # Check status
    deep-research status <interaction_id>

    # Get result
    deep-research result <interaction_id>

    # Ask follow-up question
    deep-research followup <interaction_id> "Your question"

    # Quick search (fast, seconds)
    deep-research quick-search "Your search query"

    # Analyze URLs
    deep-research analyze-urls "Your query" --url https://example.com/page1 --url https://example.com/page2

    # Search and analyze URLs together
    deep-research search-analyze "Your query" --url https://example.com/reference
"""

import asyncio
import os
import sys
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.client import DeepResearchClient, QuickSearchClient
from src.core.models import ResearchEventType, ResearchStatusEnum

# Load environment variables
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="deep-research",
    help="CLI for Gemini Deep Research Agent and Quick Search Tools",
    add_completion=False,
)
console = Console()


# HTTP timeout settings (in seconds)
# Note: Gemini API requires minimum 10 seconds deadline
DEEP_RESEARCH_HTTP_TIMEOUT = 3600  # 60 minutes for long-running research
QUICK_SEARCH_HTTP_TIMEOUT = 300  # 5 minutes for quick search operations


def get_deep_research_client() -> DeepResearchClient:
    """Get the DeepResearchClient instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] GEMINI_API_KEY environment variable is not set.",
            style="bold",
        )
        console.print("Please set your Gemini API key in the .env file or environment variables.")
        raise typer.Exit(1)
    return DeepResearchClient(api_key=api_key, http_timeout=DEEP_RESEARCH_HTTP_TIMEOUT)


def get_quick_search_client() -> QuickSearchClient:
    """Get the QuickSearchClient instance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(
            "[red]Error:[/red] GEMINI_API_KEY environment variable is not set.",
            style="bold",
        )
        console.print("Please set your Gemini API key in the .env file or environment variables.")
        raise typer.Exit(1)
    return QuickSearchClient(api_key=api_key, http_timeout=QUICK_SEARCH_HTTP_TIMEOUT)


# Backward compatibility alias
def get_client() -> DeepResearchClient:
    """Get the DeepResearchClient instance (backward compatibility)."""
    return get_deep_research_client()


# =============================================================================
# Deep Research Commands
# =============================================================================


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query or topic"),
    stream: bool = typer.Option(
        False, "--stream", "-s", help="Stream output with real-time progress"
    ),
    format_instructions: Optional[str] = typer.Option(
        None, "--format", "-f", help="Format instructions for the output"
    ),
    timeout: int = typer.Option(3600, "--timeout", "-t", help="Timeout in seconds"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save result to file"
    ),
):
    """Execute deep research on a topic.

    This command performs comprehensive research using Gemini Deep Research Agent.
    Research typically takes several minutes to complete.

    Examples:
        deep-research research "History of quantum computing"
        deep-research research "AI trends" --stream
        deep-research research "EV batteries" -f "Include comparison table" -o report.md
    """
    client = get_deep_research_client()

    if stream:
        _run_streaming_research(client, query, format_instructions, output)
    else:
        _run_polling_research(client, query, format_instructions, timeout, output)


def _run_streaming_research(
    client: DeepResearchClient,
    query: str,
    format_instructions: Optional[str],
    output: Optional[str],
):
    """Run research with streaming output."""
    console.print(Panel(f"[bold]Research Query:[/bold] {query}", title="🔬 Deep Research"))

    accumulated_content = []
    thoughts = []

    async def stream():
        nonlocal accumulated_content, thoughts

        async for event in client.stream_research(query, format_instructions):
            if event.event_type == ResearchEventType.START:
                console.print(f"\n[dim]Interaction ID: {event.interaction_id}[/dim]")
                console.print("[yellow]Research started...[/yellow]\n")

            elif event.event_type == ResearchEventType.THOUGHT:
                if event.content:
                    thoughts.append(event.content)
                    console.print(f"[dim]💭 {event.content}[/dim]")

            elif event.event_type == ResearchEventType.TEXT_DELTA:
                if event.content:
                    accumulated_content.append(event.content)
                    console.print(event.content, end="")

            elif event.event_type == ResearchEventType.COMPLETE:
                console.print("\n")
                console.print("[green]✓ Research completed![/green]")

            elif event.event_type == ResearchEventType.ERROR:
                console.print(f"\n[red]✗ Error: {event.content}[/red]")

    try:
        asyncio.run(stream())

        # Save to file if requested
        if output and accumulated_content:
            content = "".join(accumulated_content)
            with open(output, "w") as f:
                f.write(content)
            console.print(f"\n[dim]Result saved to: {output}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _run_polling_research(
    client: DeepResearchClient,
    query: str,
    format_instructions: Optional[str],
    timeout: int,
    output: Optional[str],
):
    """Run research with polling."""
    console.print(Panel(f"[bold]Research Query:[/bold] {query}", title="🔬 Deep Research"))

    async def execute():
        # Start research
        interaction_id = await client.start_research(query, format_instructions)
        console.print(f"[dim]Interaction ID: {interaction_id}[/dim]\n")

        # Poll with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Researching...", total=None)

            try:
                result = await client.poll_until_complete(
                    interaction_id,
                    poll_interval=10,
                    timeout=timeout,
                )

                progress.update(task, description="[green]Complete![/green]")

                return result

            except TimeoutError:
                progress.update(task, description="[yellow]Timed out[/yellow]")
                console.print(
                    f"\n[yellow]Research timed out after {timeout} seconds.[/yellow]"
                )
                console.print(f"Check status with: deep-research status {interaction_id}")
                return None

    try:
        result = asyncio.run(execute())

        if result and result.status == ResearchStatusEnum.COMPLETED:
            console.print("\n[green]✓ Research completed![/green]\n")

            # Display result
            if result.content:
                console.print(Markdown(result.content))

                # Save to file if requested
                if output:
                    with open(output, "w") as f:
                        f.write(result.content)
                    console.print(f"\n[dim]Result saved to: {output}[/dim]")

            # Display citations
            if result.citations:
                console.print("\n[bold]Citations:[/bold]")
                for i, citation in enumerate(result.citations, 1):
                    console.print(f"  {i}. {citation}")

        elif result and result.status == ResearchStatusEnum.FAILED:
            console.print(f"\n[red]✗ Research failed: {result.error}[/red]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    interaction_id: str = typer.Argument(..., help="Interaction ID to check"),
):
    """Check the status of a research task.

    Example:
        deep-research status interactions/abc123
    """
    client = get_deep_research_client()

    async def check():
        return await client.get_status(interaction_id)

    try:
        result = asyncio.run(check())

        # Create status table
        table = Table(title="Research Status")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        status_style = {
            ResearchStatusEnum.IN_PROGRESS: "[yellow]in_progress[/yellow]",
            ResearchStatusEnum.COMPLETED: "[green]completed[/green]",
            ResearchStatusEnum.FAILED: "[red]failed[/red]",
            ResearchStatusEnum.CANCELLED: "[dim]cancelled[/dim]",
        }

        table.add_row("Interaction ID", interaction_id)
        table.add_row("Status", status_style.get(result.status, str(result.status)))

        if result.error:
            table.add_row("Error", f"[red]{result.error}[/red]")

        console.print(table)

        if result.status == ResearchStatusEnum.COMPLETED:
            console.print("\n[dim]Get full result with: deep-research result " + interaction_id + "[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def result(
    interaction_id: str = typer.Argument(..., help="Interaction ID to get result for"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save result to file"
    ),
):
    """Get the result of a completed research task.

    Example:
        deep-research result interactions/abc123
        deep-research result interactions/abc123 -o report.md
    """
    client = get_deep_research_client()

    async def get():
        return await client.get_result(interaction_id)

    try:
        result = asyncio.run(get())

        if result.status == ResearchStatusEnum.COMPLETED:
            console.print("[green]✓ Research completed[/green]\n")

            if result.content:
                console.print(Markdown(result.content))

                if output:
                    with open(output, "w") as f:
                        f.write(result.content)
                    console.print(f"\n[dim]Result saved to: {output}[/dim]")

            if result.citations:
                console.print("\n[bold]Citations:[/bold]")
                for i, citation in enumerate(result.citations, 1):
                    console.print(f"  {i}. {citation}")

        elif result.status == ResearchStatusEnum.IN_PROGRESS:
            console.print("[yellow]Research is still in progress.[/yellow]")
            console.print("Check status with: deep-research status " + interaction_id)

        elif result.status == ResearchStatusEnum.FAILED:
            console.print(f"[red]Research failed: {result.error}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def followup(
    interaction_id: str = typer.Argument(..., help="Interaction ID of completed research"),
    question: str = typer.Argument(..., help="Follow-up question to ask"),
):
    """Ask a follow-up question about a completed research.

    Example:
        deep-research followup interactions/abc123 "What are the main risks?"
    """
    client = get_deep_research_client()

    async def ask():
        return await client.ask_followup(interaction_id, question)

    try:
        console.print(Panel(f"[bold]Question:[/bold] {question}", title="❓ Follow-up"))

        with console.status("Getting answer..."):
            answer = asyncio.run(ask())

        console.print("\n[bold]Answer:[/bold]\n")
        console.print(Markdown(answer))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


# =============================================================================
# Quick Search Commands
# =============================================================================


@app.command("quick-search")
def quick_search(
    query: str = typer.Argument(..., help="Search query"),
    model: str = typer.Option(
        "gemini-3-flash-preview", "--model", "-m", help="Model to use (gemini-3-flash-preview or gemini-3-pro-preview)"
    ),
    language: str = typer.Option(
        "ja", "--language", "-l", help="Response language (ja/en)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save result to file"
    ),
):
    """Perform a quick web search using Google Search grounding.

    This command uses Gemini's Google Search tool for fast searches.
    Results typically return in seconds.

    Examples:
        deep-research quick-search "2024年のノーベル物理学賞"
        deep-research quick-search "latest AI news" -l en
        deep-research quick-search "EV market trends" -m gemini-3-pro-preview -o result.md
    """
    client = get_quick_search_client()

    console.print(Panel(f"[bold]Search Query:[/bold] {query}", title="⚡ Quick Search"))

    async def search():
        return await client.quick_search(query=query, model=model, language=language)

    try:
        with console.status("Searching..."):
            result = asyncio.run(search())

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

        console.print("\n[green]✓ Search completed![/green]\n")

        # Display result
        if result.content:
            console.print(Markdown(result.content))

            if output:
                with open(output, "w") as f:
                    f.write(result.content)
                console.print(f"\n[dim]Result saved to: {output}[/dim]")

        # Display citations
        if result.citations:
            console.print("\n[bold]Citations:[/bold]")
            for i, citation in enumerate(result.citations, 1):
                title = citation.title or citation.url
                console.print(f"  {i}. {title}")
                console.print(f"     [dim]{citation.url}[/dim]")

        # Display search queries used
        if result.search_queries:
            console.print("\n[dim]Search queries used:[/dim]")
            for sq in result.search_queries:
                console.print(f"  [dim]- {sq}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("analyze-urls")
def analyze_urls(
    query: str = typer.Argument(..., help="Analysis query or instruction"),
    url: list[str] = typer.Option(
        ..., "--url", "-u", help="URL to analyze (can be specified multiple times, max 20)"
    ),
    model: str = typer.Option(
        "gemini-3-flash-preview", "--model", "-m", help="Model to use (gemini-3-flash-preview or gemini-3-pro-preview)"
    ),
    language: str = typer.Option(
        "ja", "--language", "-l", help="Response language (ja/en)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save result to file"
    ),
):
    """Analyze content from specific URLs.

    This command uses Gemini's URL Context tool to fetch and analyze
    content from the provided URLs.

    Examples:
        deep-research analyze-urls "この記事を要約してください" -u https://example.com/article
        deep-research analyze-urls "Compare these articles" -u https://example.com/a1 -u https://example.com/a2
    """
    if len(url) > 20:
        console.print("[red]Error: Maximum 20 URLs allowed per request.[/red]")
        raise typer.Exit(1)

    client = get_quick_search_client()

    console.print(Panel(f"[bold]Query:[/bold] {query}\n[bold]URLs:[/bold] {len(url)} URL(s)", title="🔗 URL Analysis"))

    async def analyze():
        return await client.analyze_urls(urls=url, query=query, model=model, language=language)

    try:
        with console.status("Analyzing URLs..."):
            result = asyncio.run(analyze())

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

        console.print("\n[green]✓ Analysis completed![/green]\n")

        # Display result
        if result.content:
            console.print(Markdown(result.content))

            if output:
                with open(output, "w") as f:
                    f.write(result.content)
                console.print(f"\n[dim]Result saved to: {output}[/dim]")

        # Display URL metadata
        if result.url_metadata:
            console.print("\n[bold]URL Retrieval Status:[/bold]")
            for meta in result.url_metadata:
                status_icon = "✓" if meta.status == "URL_RETRIEVAL_STATUS_SUCCESS" else "✗"
                status_color = "green" if meta.status == "URL_RETRIEVAL_STATUS_SUCCESS" else "red"
                console.print(f"  [{status_color}]{status_icon}[/{status_color}] {meta.url}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("search-analyze")
def search_analyze(
    query: str = typer.Argument(..., help="Search and analysis query"),
    url: Optional[list[str]] = typer.Option(
        None, "--url", "-u", help="Reference URL (optional, can be specified multiple times)"
    ),
    model: str = typer.Option(
        "gemini-3-flash-preview", "--model", "-m", help="Model to use (gemini-3-flash-preview or gemini-3-pro-preview)"
    ),
    language: str = typer.Option(
        "ja", "--language", "-l", help="Response language (ja/en)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save result to file"
    ),
):
    """Perform web search with optional URL context for deeper analysis.

    This command combines Google Search grounding with URL Context
    to provide both broad search results and deep analysis of specific pages.

    Examples:
        deep-research search-analyze "最新のEV市場動向"
        deep-research search-analyze "Compare with this report" -u https://example.com/ev-report
    """
    urls = url or []
    if len(urls) > 20:
        console.print("[red]Error: Maximum 20 URLs allowed per request.[/red]")
        raise typer.Exit(1)

    client = get_quick_search_client()

    url_info = f" + {len(urls)} URL(s)" if urls else ""
    console.print(Panel(f"[bold]Query:[/bold] {query}{url_info}", title="🔍+🔗 Search + Analyze"))

    async def search_and_analyze():
        return await client.search_and_analyze(
            query=query, urls=urls if urls else None, model=model, language=language
        )

    try:
        with console.status("Searching and analyzing..."):
            result = asyncio.run(search_and_analyze())

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            raise typer.Exit(1)

        console.print("\n[green]✓ Search and analysis completed![/green]\n")

        # Display result
        if result.content:
            console.print(Markdown(result.content))

            if output:
                with open(output, "w") as f:
                    f.write(result.content)
                console.print(f"\n[dim]Result saved to: {output}[/dim]")

        # Display citations
        if result.citations:
            console.print("\n[bold]Citations:[/bold]")
            for i, citation in enumerate(result.citations, 1):
                title = citation.title or citation.url
                console.print(f"  {i}. {title}")
                console.print(f"     [dim]{citation.url}[/dim]")

        # Display search queries used
        if result.search_queries:
            console.print("\n[dim]Search queries used:[/dim]")
            for sq in result.search_queries:
                console.print(f"  [dim]- {sq}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Gemini Deep Research CLI[/bold]")
    console.print("Version: 0.1.0")
    console.print("Agent: deep-research-pro-preview-12-2025")
    console.print("\n[bold]Available Commands:[/bold]")
    console.print("  [cyan]research[/cyan]       - Deep research (minutes)")
    console.print("  [cyan]quick-search[/cyan]   - Quick web search (seconds)")
    console.print("  [cyan]analyze-urls[/cyan]   - Analyze specific URLs")
    console.print("  [cyan]search-analyze[/cyan] - Search + URL analysis")
    console.print("  [cyan]status[/cyan]         - Check research status")
    console.print("  [cyan]result[/cyan]         - Get research result")
    console.print("  [cyan]followup[/cyan]       - Ask follow-up question")


if __name__ == "__main__":
    app()
