"""Own shared service resources; construction is deferred until the first workspace request."""

import asyncio
from uuid import uuid4

import httpx
from pydantic import SecretStr

from met_agent.agent.events import AuditStore, EventStore, PostgresEventStore
from met_agent.agent.loop import Agent
from met_agent.agent.models import AgentAnswer, ChatRequest
from met_agent.config import Settings
from met_agent.ingestion.commands import qdrant_client
from met_agent.llm.chat import LiteLLMChat, ModelError
from met_agent.observability.tracing import Telemetry
from met_agent.retrieval.service import SearchService
from met_agent.tools.get_object import LiveObjectClient
from met_agent.tools.registry import create_registry


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        secrets = [
            value.get_secret_value()
            for name in type(settings).model_fields
            if isinstance(value := getattr(settings, name), SecretStr)
        ]
        self.events: AuditStore = (
            PostgresEventStore(
                settings.audit_database_url.get_secret_value(),
                secrets=secrets,
                retention_days=settings.audit_retention_days,
                pool_min_size=settings.audit_pool_min_size,
                pool_max_size=settings.audit_pool_max_size,
            )
            if settings.audit_database_url
            else EventStore(settings.data_dir / "sessions.sqlite3", secrets=secrets)
        )
        self.telemetry = (
            Telemetry(settings, self.events) if settings.optional_services.langfuse else None
        )
        self.http = httpx.Client(timeout=30, follow_redirects=False)
        self.qdrant = qdrant_client(settings)
        self.search = SearchService(self.qdrant, settings)
        self.live = LiveObjectClient(self.http, str(settings.met_api_base))
        self.tools = create_registry(settings, self.search, self.live)
        self.tools.telemetry = self.telemetry
        self.agent: Agent | None = None

    async def chat(self, request: ChatRequest) -> AgentAnswer:
        if self.agent is None:
            self.agent = Agent(
                LiteLLMChat(
                    self.settings, callback=self.telemetry.callback if self.telemetry else None
                ),
                self.tools,
                self.events,
            )
        request = request.model_copy(update={"session_id": request.session_id or uuid4()})
        try:
            async with asyncio.timeout(self.settings.chat_deadline_seconds):
                if self.telemetry:
                    return await self.telemetry.run(self.agent.run, request)
                return await self.agent.run(request)
        except TimeoutError:
            if request.session_id is not None:
                events = self.events.read(request.session_id, limit=500)
                self.events.append(
                    request.session_id,
                    events[-1].turn_id if events else uuid4(),
                    "turn_error",
                    {"code": "verification_unavailable", "reason": "interactive_deadline"},
                )
            raise ModelError(
                "verification_unavailable",
                "A verified answer was not ready in time. Please try again shortly.",
            ) from None

    def readiness(self) -> dict[str, bool]:
        """Probe required durable dependencies without loading embedding models."""
        checks: dict[str, bool] = {}
        try:
            checks["audit_store"] = self.events.ready()
        except Exception:
            checks["audit_store"] = False
        try:
            self.qdrant.get_collections()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False
        return checks

    def close(self) -> None:
        self.http.close()
        self.qdrant.close()
        self.events.close()
        if self.telemetry:
            self.telemetry.close()
