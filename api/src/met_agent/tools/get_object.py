"""Fetch authoritative Met object metadata with a bounded five-minute cache and safe errors."""

import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock

import httpx

from met_agent.ingestion.http import USER_AGENT, SourceError
from met_agent.ingestion.visitors import validate_visitor_url
from met_agent.tools.models import Evidence, GetObjectArguments, LiveObject


class ObjectNotFound(SourceError):
    """Distinguish confirmed HTTP 404 evidence from an unavailable upstream service."""

    def __init__(self, message: str, *, evidence: Evidence | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


class LiveObjectClient:
    def __init__(
        self, http: httpx.Client, base_url: str, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.http, self.base_url, self.clock = http, base_url.rstrip("/"), clock
        self._cache: OrderedDict[int, tuple[float, LiveObject]] = OrderedDict()
        self._lock = Lock()

    def get(self, arguments: GetObjectArguments) -> LiveObject:
        object_id = arguments.object_id
        with self._lock:
            existing = self._cache.get(object_id)
            if existing and self.clock() - existing[0] < 300:
                self._cache.move_to_end(object_id)
                return existing[1].model_copy(deep=True)
            response = self.http.get(
                f"{self.base_url}/objects/{object_id}", headers={"User-Agent": USER_AGENT}
            )
            if response.status_code == 404:
                raise ObjectNotFound(
                    "Object not found in the Met API",
                    evidence=Evidence(
                        key=f"lookup:{object_id}",
                        source_url=f"{self.base_url}/objects/{object_id}",
                        kind="lookup_status",
                        text=(
                            f"Met API lookup for Object ID {object_id}: not found (HTTP 404).\n"
                            "No object record was returned for this requested ID.\n"
                            f"Fetched at: {datetime.now(UTC).isoformat()}"
                        ),
                    ),
                )
            if response.status_code != 200:
                raise SourceError(f"Met object service returned HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or data.get("objectID") != object_id:
                raise SourceError("Met object response identity is invalid")
            source = validate_visitor_url(str(data.get("objectURL", "")))
            fields = {
                "title": "title",
                "artist": "artistDisplayName",
                "culture": "culture",
                "medium": "medium",
                "object_date": "objectDate",
                "department": "department",
                "gallery_number": "GalleryNumber",
                "primary_image": "primaryImage",
                "primary_image_small": "primaryImageSmall",
            }
            values = {key: str(data.get(source_key) or "") for key, source_key in fields.items()}
            lines = [f"Object ID: {object_id}"] + [
                f"{key}: {value}" for key, value in values.items() if value
            ]
            lines.append(
                "Current display status: "
                + ("on view" if values["gallery_number"] else "not currently on view")
            )
            obj = LiveObject(
                object_id=object_id,
                **values,
                is_on_view=bool(values["gallery_number"]),
                is_public_domain=data.get("isPublicDomain") is True,
                source_url=source,
                fetched_at=datetime.now(UTC).isoformat(),
                text="\n".join(lines),
            )
            self._cache[object_id] = (self.clock(), obj)
            self._cache.move_to_end(object_id)
            while len(self._cache) > 512:
                self._cache.popitem(last=False)
            return obj.model_copy(deep=True)
