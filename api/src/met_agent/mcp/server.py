"""Serve identical registry schemas and validation over official MCP stdio and SSE transports."""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

import uvicorn
from mcp.server import Server, ServerRequestContext
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send

from met_agent.config import load_settings
from met_agent.observability.logging import configure_logging
from met_agent.runtime import Runtime
from met_agent.tools.models import ToolResult
from met_agent.tools.registry import ToolRegistry


class WorkspaceMCP:
    """Forward raw arguments to the registry so transports cannot coerce invalid tool inputs."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.server: Server[None] = Server(
            "Met Collection Agent",
            version="0.1.0",
            instructions="Independent factual collection tools. No artistic interpretation.",
            on_list_tools=self._list,
            on_call_tool=self._call,
        )

    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name=tool.name,
                description=tool.description,
                input_schema=tool.arguments.model_json_schema(),
                output_schema=ToolResult.model_json_schema(),
                annotations=ToolAnnotations(
                    read_only_hint=True, destructive_hint=False, open_world_hint=True
                ),
            )
            for tool in self.registry.tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        result = await self.registry.execute(name, json.dumps(arguments))
        return CallToolResult(
            content=[TextContent(type="text", text=result.model_dump_json())],
            structured_content=result.model_dump(mode="json"),
            is_error=result.error is not None,
        )

    async def _list(
        self, context: ServerRequestContext[None], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        return ListToolsResult(tools=await self.list_tools())

    async def _call(
        self, context: ServerRequestContext[None], params: CallToolRequestParams
    ) -> CallToolResult:
        return await self.call_tool(params.name, params.arguments or {})

    async def stdio(self) -> None:
        async with stdio_server() as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())

    def sse_app(self) -> Starlette:
        transport = SseServerTransport(
            "/messages/",
            max_request_body_size=65536,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=["127.0.0.1:*", "localhost:*"],
                allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
            ),
        )
        return Starlette(
            routes=[
                Route("/sse", endpoint=SSEConnection(self.server, transport)),
                Mount("/messages/", app=transport.handle_post_message),
            ]
        )


class SSEConnection:
    def __init__(self, server: Server[None], transport: SseServerTransport) -> None:
        self.server, self.transport = server, transport

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with self.transport.connect_sse(scope, receive, send) as (read, write):
            await self.server.run(read, write, self.server.create_initialization_options())


def create_server(registry: ToolRegistry) -> WorkspaceMCP:
    return WorkspaceMCP(registry)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    settings = load_settings()
    configure_logging(settings.log_level, stream=sys.stderr)
    owner = Runtime(settings)
    try:
        server = create_server(owner.tools)
        if args.transport == "sse":
            uvicorn.run(server.sse_app(), host="127.0.0.1", port=args.port, access_log=False)
        else:
            asyncio.run(server.stdio())
    finally:
        owner.close()


if __name__ == "__main__":
    main()
