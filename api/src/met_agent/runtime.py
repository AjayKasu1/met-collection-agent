"""Own shared service resources; construction is deferred until the first workspace request."""

from uuid import uuid4

import httpx
from pydantic import SecretStr

from met_agent.agent.events import EventStore
from met_agent.agent.loop import Agent
from met_agent.agent.models import AgentAnswer, ChatRequest
from met_agent.config import Settings
from met_agent.ingestion.commands import qdrant_client
from met_agent.llm.chat import LiteLLMChat
from met_agent.observability.tracing import Telemetry
from met_agent.retrieval.service import SearchService
from met_agent.tools.get_object import LiveObjectClient
from met_agent.tools.registry import create_registry


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.events = EventStore(
            settings.data_dir / "sessions.sqlite3",
            secrets=[
                value.get_secret_value()
                for name in type(settings).model_fields
                if isinstance(value := getattr(settings, name), SecretStr)
            ],
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
        if self.telemetry:
            return await self.telemetry.run(self.agent.run, request)
        return await self.agent.run(request)

    def close(self) -> None:
        self.http.close()
        self.qdrant.close()
        if self.telemetry:
            self.telemetry.close()
