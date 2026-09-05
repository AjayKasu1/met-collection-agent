# Web client

This Next.js client presents the verified FastAPI agent as an accessible collection guide. The browser never receives provider credentials or talks directly to a model. A same-origin route translates the API's verified server-sent events into the AI SDK UI message protocol.

## Local development

Use Node.js 20 and pnpm 10.34.5. Start the API from the repository root, then run the client in a second terminal:

```sh
make dev
make web-install
make web-dev
```

`NEXT_PUBLIC_API_URL` selects the API origin and defaults to `http://localhost:8000`. It is public build configuration, so it must never contain credentials. The backend must allow the web origin through `CORS_ORIGINS` when they run on different origins.

## Response boundary

`POST /api/chat` sends only the latest user message, session UUID, and language hint to the API. It buffers no unverified model draft of its own. Text received from the API is compared with the final `AgentAnswer`; a mismatch terminates the stream without attaching citations or provenance. Runtime validation rejects malformed sessions, answer fields, citation identities, and non-HTTP source URLs.

The final typed data part contains citations and a bounded projection of the session audit. The interface renders collection objects through the same-origin object proxy and displays visitor citations as source links. Provider errors are mapped to fixed public messages so upstream response bodies cannot reach the browser.

## Quality gates

```sh
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm build:worker
```

`pnpm check` runs lint, strict TypeScript, tests, and the Cloudflare Workers build. Tests cover SSE framing across chunk boundaries, malformed streams, runtime answer validation, unsafe URLs, provider error redaction, tool-order projection, verified text equality, and the provenance display.

## Cloudflare Workers

OpenNext produces `.open-next/worker.js` and static assets according to `wrangler.jsonc`. `pnpm preview` runs the generated Worker locally. `pnpm deploy` performs a live deployment and belongs to the deployment phase; it requires an authenticated Wrangler session or scoped Cloudflare CI credentials.
