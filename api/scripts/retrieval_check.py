"""Measure startup preparation and two visitor searches without generation API calls."""

import argparse
import json
import time
from contextlib import closing
from pathlib import Path

from met_agent.config import Settings, load_settings
from met_agent.ingestion.commands import qdrant_client
from met_agent.observability.logging import configure_logging
from met_agent.retrieval.service import SearchService


def run(settings: Settings) -> None:
    if settings.embedding_provider != "local":
        raise ValueError("This check requires local embeddings to avoid provider API calls")
    configure_logging("INFO")
    with closing(qdrant_client(settings)) as client:
        search = SearchService(client, settings)
        started = time.monotonic()
        search.store(settings.qdrant_collection)
        search.store(settings.qdrant_visitor_collection)
        search.warmup()
        startup_ms = (time.monotonic() - started) * 1000
        timings: list[float] = []
        results: list[list[str]] = []
        for _ in range(2):
            started = time.monotonic()
            matches = search.search(
                settings.qdrant_visitor_collection,
                "What should families know before visiting?",
            )
            timings.append(round((time.monotonic() - started) * 1000, 2))
            results.append([str(point.id) for point, _ in matches])
        if not results[0] or results[0] != results[1]:
            raise RuntimeError("Visitor search returned empty or inconsistent results")
        print(
            json.dumps(
                {
                    "startup_ms": round(startup_ms, 2),
                    "visitor_search_ms": timings,
                    "matched_chunks": len(results[0]),
                    "same_results": True,
                    "generation_calls": 0,
                    "models_offline": settings.models_offline,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    configured = load_settings()
    overrides: dict[str, object] = {"models_offline": True}
    if args.cache_dir:
        overrides["model_cache_dir"] = args.cache_dir
    try:
        run(configured.model_copy(update=overrides))
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
