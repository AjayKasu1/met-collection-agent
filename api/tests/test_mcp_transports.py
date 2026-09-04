"""Verify real MCP handshakes and dispatch through stdio and loopback SSE."""

import asyncio
import socket
import sys
from pathlib import Path

import pytest
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Implementation
from test_agent_workspace import registry_with_calls

from met_agent.mcp.server import create_server


async def exercise(session: ClientSession) -> None:
    await session.initialize()
    listed = await session.list_tools()
    assert any(tool.name == "handoff" for tool in listed.tools)
    result = await session.call_tool("handoff", {"reason": "account"})
    assert isinstance(result, CallToolResult) and not result.is_error
    assert result.structured_content["output"]["suggested_contact"] == "info@metmuseum.org"
    for args in ({"object_id": True}, {"object_id": "1"}, {"object_id": 1, "extra": True}):
        bad = await session.call_tool("get_object", args)
        assert isinstance(bad, CallToolResult) and bad.is_error
        assert bad.structured_content["error"]["code"] == "invalid_arguments"


def test_stdio_protocol_and_clean_shutdown(tmp_path: Path) -> None:
    async def run() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "met_agent.mcp.server"],
            cwd=tmp_path,
            env={
                "GEMINI_API_KEY": "synthetic-test",
                "LLM_MODEL": "test/main",
                "LLM_MODEL_LITE": "test/lite",
                "DATA_DIR": str(tmp_path / "data"),
            },
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(
                read,
                write,
                read_timeout_seconds=15,
                client_info=Implementation(name="protocol-test", version="1"),
            ) as session,
        ):
            await exercise(session)
            assert len((await session.list_tools()).tools) == 5

    asyncio.run(run())


@pytest.mark.integration
def test_sse_protocol_preserves_validation() -> None:
    async def run() -> None:
        workspace = create_server(registry_with_calls([]))
        server = uvicorn.Server(
            uvicorn.Config(workspace.sse_app(), log_level="error", access_log=False, lifespan="off")
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        if task.done():
                            await task
                        await asyncio.sleep(0.01)
                port = listener.getsockname()[1]
                async with (
                    sse_client(f"http://127.0.0.1:{port}/sse") as (read, write),
                    ClientSession(read, write, read_timeout_seconds=10) as session,
                ):
                    await exercise(session)
            finally:
                server.should_exit = True
                await task

    asyncio.run(run())
