#!/usr/bin/env python3
"""A tiny stdio MCP server for tests: add, echo and a failing tool."""
from mcp.server.mcpserver import MCPServer

app = MCPServer("fake")


@app.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@app.tool()
def echo(text: str) -> str:
    """Echo text back."""
    return text


@app.tool()
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


if __name__ == "__main__":
    app.run()
