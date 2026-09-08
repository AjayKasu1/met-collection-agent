"""Resolve bounded indoor routes against The Met's official interactive map."""

import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Lock
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from met_agent.ingestion.http import USER_AGENT, SourceError
from met_agent.tools.models import GetDirectionsArguments, WayfindingResult

MAP_PROJECT = "the_met"
FLOOR_ONE_ID = 2
MAP_CENTER = (40.779448, -73.963517)
REQUEST_TIMEOUT_SECONDS = 5.0


class _LocalizedText(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lang: str
    text: str | None


class _Label(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: list[_LocalizedText]


class _Floor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    floor: str
    name: list[_LocalizedText]
    short_name: str


class _Center(BaseModel):
    model_config = ConfigDict(extra="ignore")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class _Location(BaseModel):
    model_config = ConfigDict(extra="ignore")
    center: _Center
    floor: _Floor


class _MapFeature(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    is_temporarily_closed: bool
    label: _Label
    location: _Location


class _SearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: list[_MapFeature]


class _RouteMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total_time: float = Field(alias="totalTime", ge=0, le=120)
    total_length: float = Field(alias="totalLength", ge=0, le=10)


class _RouteResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    route_metadata: list[_RouteMetadata] = Field(alias="routeMetadata", min_length=1, max_length=10)


def _english(values: list[_LocalizedText]) -> list[str]:
    return [item.text.strip() for item in values if item.lang == "en-GB" and item.text]


class WayfindingClient:
    """Fetch official map features and routes with a bounded five-minute cache."""

    def __init__(
        self,
        http: httpx.Client,
        api_base: str,
        map_base: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.http = http
        self.api_base = api_base.rstrip("/")
        self.map_base = map_base.rstrip("/")
        self.clock = clock
        self._cache: OrderedDict[str, tuple[float, WayfindingResult]] = OrderedDict()
        self._lock = Lock()

    def get(self, arguments: GetDirectionsArguments) -> WayfindingResult:
        key = arguments.destination_gallery
        with self._lock:
            existing = self._cache.get(key)
            if existing and self.clock() - existing[0] < 300:
                self._cache.move_to_end(key)
                return existing[1].model_copy(deep=True)

        origin = self._feature(
            "The Great Hall", lambda item: "The Great Hall" in _english(item.label.name)
        )
        destination = self._feature(
            arguments.destination_gallery,
            lambda item: arguments.destination_gallery in _english(item.label.name),
        )
        if (
            origin.location.floor.id != FLOOR_ONE_ID
            or destination.location.floor.id != FLOOR_ONE_ID
        ):
            raise SourceError("Wayfinding endpoints are not on the expected floor")
        if destination.is_temporarily_closed:
            raise SourceError("The destination is marked temporarily closed")

        body: dict[str, object] = {
            "from": {"lmId": origin.id},
            "to": {"lmId": destination.id},
            "options": {},
            "project": MAP_PROJECT,
        }
        response = self.http.post(
            f"{self.api_base}/v2/route",
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        route = self._validated(response, _RouteResponse)
        metadata = route.route_metadata[0]
        metres = round(metadata.total_length * 1000)
        feet = round(metadata.total_length * 3280.839895)
        minutes = max(1, round(metadata.total_time))
        route_url = (
            f"{self.map_base}/navigate/{quote(origin.id, safe='')}/"
            f"{quote(destination.id, safe='')}?floor=1&lang=en-GB"
        )
        fetched_at = datetime.now(UTC).isoformat()
        text = "\n".join(
            [
                "The Met Interactive Map route",
                "Start: The Great Hall",
                f"Destination: Gallery {arguments.destination_gallery}",
                "Floor: Floor 1",
                f"Estimated walking time: {minutes} minutes",
                f"Distance: {metres} metres ({feet} feet)",
                "Destination status: not marked temporarily closed",
                f"Fetched at: {fetched_at}",
            ]
        )
        result = WayfindingResult(
            origin="The Great Hall",
            destination=f"Gallery {arguments.destination_gallery}",
            floor="Floor 1",
            distance_metres=metres,
            distance_feet=feet,
            duration_minutes=minutes,
            source_url=route_url,
            fetched_at=fetched_at,
            text=text,
        )
        with self._lock:
            self._cache[key] = (self.clock(), result)
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return result.model_copy(deep=True)

    def _feature(self, query: str, matches: Callable[[_MapFeature], bool]) -> _MapFeature:
        latitude, longitude = MAP_CENTER
        response = self.http.get(
            f"{self.api_base}/v1/maps/{MAP_PROJECT}/search",
            headers={"User-Agent": USER_AGENT},
            params={
                "query": query,
                "latitude": latitude,
                "longitude": longitude,
                "floor_id": FLOOR_ONE_ID,
                "limit": 30,
                "lang": "en-GB",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        search = self._validated(response, _SearchResponse)
        exact = [item for item in search.data if matches(item)]
        if len(exact) != 1:
            raise SourceError("The map search did not resolve one exact feature")
        return exact[0]

    @staticmethod
    def _validated[Model: BaseModel](response: httpx.Response, model: type[Model]) -> Model:
        if response.status_code != 200:
            raise SourceError(f"The Met map service returned HTTP {response.status_code}")
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError):
            raise SourceError("The Met map response is invalid") from None
