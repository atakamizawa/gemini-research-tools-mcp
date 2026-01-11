"""MCP Server for Gemini Deep Research Agent."""

# Note: server module is imported lazily to avoid circular import issues
# when running as `python -m src.mcp.server`

__all__ = ["mcp"]


def __getattr__(name: str):
    """Lazy import for mcp server."""
    if name == "mcp":
        from .server import mcp
        return mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
