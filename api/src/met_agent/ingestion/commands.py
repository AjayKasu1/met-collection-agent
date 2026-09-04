"""Compose source preparation and vector ingestion behind explicit command-line operations."""

import argparse
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import httpx
import structlog
import yaml
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient

from met_agent.config import ConfigurationError, Settings, load_settings
from met_agent.ingestion.collection import CSV_URL, enrich_object, image_inventory, select_objects
from met_agent.ingestion.http import SourceClient, SourceError, download_csv
from met_agent.ingestion.models import VisitorChunk
from met_agent.ingestion.storage import atomic_write, load_objects, save_chunks, save_objects
from met_agent.ingestion.verify import read_golden, verification_markdown, verify_golden
from met_agent.ingestion.visitors import VisitorCrawler, chunk_markdown, html_to_markdown
from met_agent.llm.router import create_embedding_router
from met_agent.observability.logging import configure_logging
from met_agent.retrieval.embeddings import BM25Embedder, EmbeddingError, GeminiEmbedder
from met_agent.retrieval.qdrant_store import HybridStore, IndexCompatibilityError

logger = structlog.get_logger(__name__)


def qdrant_client(settings: Settings) -> QdrantClient:
    """Pass credentials explicitly instead of relying on SDK environment discovery."""
    return QdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        timeout=60,
    )


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, help="Output directory; defaults to DATA_DIR")
    return parser


def ingest_collection(argv: Sequence[str] | None = None) -> int:
    """Prepare or index a bounded collection sample; keep pilots in a separate directory/index."""
    parser = _base_parser("Prepare and ingest public-domain Met collection records")
    parser.add_argument("--limit", type=int, help="Maximum records, overriding INGEST_MAX_OBJECTS")
    parser.add_argument(
        "--enrich-live",
        action="store_true",
        help="Refresh every selected record through the live Met API",
    )
    parser.add_argument("--collection", help="Qdrant collection, overriding QDRANT_COLLECTION")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Download and prepare without LLM calls"
    )
    parser.add_argument(
        "--reuse-prepared", action="store_true", help="Index existing objects.parquet"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh the cached CSV and image inventory"
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    output: Path = args.data_dir or settings.data_dir
    limit: int = args.limit or settings.ingest_max_objects
    if limit <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Object limit must be positive")
    if args.reuse_prepared:
        objects = load_objects(output / "objects.parquet")[:limit]
    else:
        cache = settings.data_dir / "sources"
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            csv_path = download_csv(client, CSV_URL, cache / "MetObjects.csv", refresh=args.refresh)
            source = SourceClient(client, interval=1)
            image_ids = image_inventory(
                source,
                str(settings.met_api_base),
                cache / "image_ids.json",
                refresh=args.refresh,
            )
            selected = select_objects(csv_path, limit=limit, image_ids=image_ids)
            objects = selected
            if args.enrich_live:
                objects = []
                for index, obj in enumerate(selected, 1):
                    objects.append(enrich_object(obj, source, str(settings.met_api_base)))
                    if index % 20 == 0 or index == len(selected):
                        logger.info("objects_prepared", count=index, total=len(selected))
        save_objects(output / "objects.parquet", objects)
    if not objects:
        raise SourceError("Prepared collection is empty")
    logger.info(
        "collection_prepared", objects=len(objects), images=sum(obj.has_image for obj in objects)
    )
    if not args.prepare_only:
        dense = GeminiEmbedder(create_embedding_router(settings))
        sparse = BM25Embedder(settings.data_dir / "models")
        with closing(qdrant_client(settings)) as client:
            store = HybridStore(
                client, args.collection or settings.qdrant_collection, settings.embedding_model
            )
            count = store.ingest(
                [obj.document() for obj in objects],
                dense,
                sparse,
                batch_size=settings.embedding_batch_size,
            )
        logger.info("collection_indexed", objects=count)
    return 0


class _VisitorPage(BaseModel):
    label: str
    url: str


class _VisitorSources(BaseModel):
    pages: list[_VisitorPage]


def ingest_visitor_info(argv: Sequence[str] | None = None) -> int:
    """Fetch the explicit visitor source list, persist provenance, and replace stale page chunks."""
    parser = _base_parser("Ingest curated public visitor-information pages")
    parser.add_argument("--sources", type=Path, default=Path("api/data_sources/visitor_pages.yaml"))
    parser.add_argument(
        "--collection", help="Override the visitor collection for an isolated pilot"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    sources = _VisitorSources.model_validate(yaml.safe_load(args.sources.read_text()))
    output: Path = args.data_dir or settings.data_dir
    chunks: list[VisitorChunk] = []
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        crawler = VisitorCrawler(SourceClient(client, interval=1))
        for page in sources.pages:
            url, html = crawler.fetch(page.url)
            title, markdown = html_to_markdown(html)
            fetched_at = datetime.now(UTC)
            chunks.extend(
                chunk_markdown(
                    markdown,
                    source_url=url,
                    page_title=title or page.label,
                    fetched_at=fetched_at,
                )
            )
            logger.info("visitor_page_prepared", source_url=url)
    if not chunks:
        raise SourceError("No visitor chunks were produced")
    save_chunks(output / "visitor_chunks.jsonl", chunks)
    if not args.prepare_only:
        dense = GeminiEmbedder(create_embedding_router(settings))
        sparse = BM25Embedder(settings.data_dir / "models")
        with closing(qdrant_client(settings)) as client:
            store = HybridStore(
                client,
                args.collection or settings.qdrant_visitor_collection,
                settings.embedding_model,
            )
            store.ingest(
                [chunk.document() for chunk in chunks],
                dense,
                sparse,
                batch_size=settings.embedding_batch_size,
            )
            for url in {chunk.source_url for chunk in chunks}:
                store.remove_stale_page_chunks(
                    url, [c.point_id for c in chunks if c.source_url == url]
                )
    logger.info("visitor_ingestion_completed", pages=len(sources.pages), chunks=len(chunks))
    return 0


def verify(argv: Sequence[str] | None = None) -> int:
    """Print live object existence, selected-sample membership, and manual-review rows."""
    parser = _base_parser("Verify the golden set against the live Met API and prepared data")
    parser.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    args = parser.parse_args(argv)
    settings = load_settings()
    output: Path = args.data_dir or settings.data_dir
    objects = load_objects(output / "objects.parquet")
    rows = read_golden(args.golden)
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        results = verify_golden(
            rows,
            {obj.object_id for obj in objects},
            SourceClient(client, interval=1),
            str(settings.met_api_base),
        )
    report = verification_markdown(rows, results)
    atomic_write(output / "verify_golden.md", report.encode())
    print(report)
    return 1 if any(item.exists is not True for item in results) else 0


def run_command(command: Callable[[], int]) -> int:
    """Present controlled errors without printing SDK exception bodies or credentials."""
    try:
        return command()
    except (ConfigurationError, SourceError, EmbeddingError, IndexCompatibilityError) as error:
        print(str(error))
    except (ValidationError, ValueError, OSError) as error:
        print(f"Ingestion failed validation or local I/O ({type(error).__name__})")
    except Exception as error:
        print(f"Ingestion operation failed ({type(error).__name__}); no secret values were logged")
    return 1
