# MCP tools

The official `mcp` Python SDK serves the same five tools and Pydantic schemas as chat: `search_collection`, `get_object`, `search_visitor_info`, `find_similar_objects`, and `handoff`. Raw arguments go directly to the shared registry, so numeric strings, booleans used as IDs, unknown fields, and out-of-range values are rejected consistently. Results contain `output` or `error` plus server-derived citation evidence. A tool error also sets the MCP `isError` flag.

## Standard input/output

Run from the repository root after `make setup` and local `.env` configuration:

```sh
uv run --project api --locked met-agent-mcp
```

The process reserves stdout for JSON-RPC and logs to stderr. A client starts the process and performs the MCP handshake. For a client with a working-directory setting, select this repository as that directory and use the command above.

For Claude Desktop, a local configuration using the project's conventional checkout directory is:

```json
{
  "mcpServers": {
    "met-collection-agent": {
      "command": "/bin/sh",
      "args": [
        "-c",
        "cd \"$HOME/Documents/met-collection-agent\" && exec uv run --project api --locked met-agent-mcp"
      ]
    }
  }
}
```

Adjust the checkout directory and ensure the client process can find `uv`. Keep provider keys in the local `.env`; do not copy them into a shared client configuration. Gemini CLI, Cursor, and other MCP clients can launch the same stdio command.

## Server-sent events

```sh
uv run --project api --locked met-agent-mcp --transport sse --port 8001
```

Connect an SSE-capable MCP client to `http://127.0.0.1:8001/sse`. The SDK advertises its `/messages/` endpoint for subsequent JSON-RPC requests. The server binds to loopback and validates Host/Origin against loopback names. It does not expose an authenticated public MCP service. Stdio and SSE protocol tests perform real handshakes, tool listing, valid calls, and invalid-input rejection.

## Use and limits

`get_object` provides current gallery information through a five-minute cache. Search gallery filters describe the index capture. `find_similar_objects` searches only the Met's published image vectors, excludes the source, and explicitly reports missing image coverage. Search queries should be English rewrites for the English reranker; returned evidence can support answers in other languages.

MCP exposes tools, not the chat agent's policy loop. The host client is responsible for its own tool-call budget, answer language, citation verification, and non-interpretive behavior. The `handoff` result suggests a contact and sends nothing. MCP clients should treat it as terminal in their own conversation flow.

Local embeddings are the default. Text queries must match the index's provider, model, and dimensions. Model assets are downloaded only when a text search first requires them. Listing tools, requesting a contact, and image-vector similarity do not invoke an LLM.
