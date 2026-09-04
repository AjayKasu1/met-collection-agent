"""Compose source preparation and vector ingestion behind explicit command-line operations."""

import argparse
import math
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
from met_agent.ingestion.collection import (
    CSV_URL,
    enrich_object,
    image_inventory,
    reuse_selection,
    select_objects,
)
from met_agent.ingestion.http import SourceClient, SourceError, download_csv
from met_agent.ingestion.models import VisitorChunk
from met_agent.ingestion.saved_pages import VisitorPageContent, load_saved_pages
from met_agent.ingestion.storage import (
    atomic_write,
    load_objects,
    save_chunks,
    save_objects,
    sha256_file,
    write_json,
)
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


def _batch_delay(value: str) -> float:
    """Validate pacing before any preparation or provider calls."""
    delay = float(value)
    if not math.isfinite(delay) or delay < 0:
        raise argparse.ArgumentTypeError("Batch delay must be finite and nonnegative")
    return delay


def _add_batch_pacing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-delay-seconds",
        type=_batch_delay,
        default=0,
        help="Wait this many seconds between embedding batches; tune to provider quota",
    )


def ingest_collection(argv: Sequence[str] | None = None) -> int:
    """Prepare or index a bounded collection sample; keep pilots in a separate directory/index."""
    parser = _base_parser("Prepare and ingest public-domain Met collection records")
    _add_batch_pacing(parser)
    parser.add_argument("--limit", type=int, help="Maximum records, overriding INGEST_MAX_OBJECTS")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("evals/golden.jsonl"),
        help="Reserve every eligible expected object ID from this evaluation file",
    )
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
    required_ids = frozenset(
        object_id for row in read_golden(args.golden) for object_id in row.expected_object_ids
    )
    if args.reuse_prepared:
        selection = reuse_selection(output, limit=limit, must_include_ids=required_ids)
        objects = selection.objects
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
            selection = select_objects(
                csv_path, limit=limit, image_ids=image_ids, must_include_ids=required_ids
            )
            objects = selection.objects
            if args.enrich_live:
                objects = []
                for index, obj in enumerate(selection.objects, 1):
                    objects.append(enrich_object(obj, source, str(settings.met_api_base)))
                    if index % 20 == 0 or index == len(selection.objects):
                        logger.info("objects_prepared", count=index, total=len(selection.objects))
        save_objects(output / "objects.parquet", objects)
        selection.report.artifact_sha256 = sha256_file(output / "objects.parquet")
        write_json(output / "selection.json", selection.report.model_dump(mode="json"))
    if not objects:
        raise SourceError("Prepared collection is empty")
    for object_id, reason in sorted(selection.report.excluded_ids.items()):
        logger.warning("golden_object_excluded", object_id=object_id, reason=reason)
    logger.info(
        "golden_coverage", included=len(selection.report.included_ids), required=len(required_ids)
    )
    logger.info(
        "collection_prepared", objects=len(objects), images=sum(obj.has_image for obj in objects)
    )
    if not args.prepare_only:
        dense = GeminiEmbedder(
            create_embedding_router(settings), output_dimensionality=settings.embedding_dimensions
        )
        sparse = BM25Embedder(settings.data_dir / "models")
        with closing(qdrant_client(settings)) as client:
            store = HybridStore(
                client,
                args.collection or settings.qdrant_collection,
                settings.embedding_model,
                settings.embedding_dimensions,
            )
            count = store.ingest(
                [obj.document() for obj in objects],
                dense,
                sparse,
                batch_size=settings.embedding_batch_size,
                batch_delay_seconds=args.batch_delay_seconds,
            )
        logger.info("collection_indexed", objects=count)
    return 0


class _VisitorPage(BaseModel):
    label: str
    url: str


class _VisitorSources(BaseModel):
    pages: list[_VisitorPage]


def ingest_visitor_info(argv: Sequence[str] | None = None) -> int:
    """Prepare live or saved visitor pages, then replace stale chunks after successful indexing."""
    parser = _base_parser("Ingest curated public visitor-information pages")
    _add_batch_pacing(parser)
    parser.add_argument(
        "--sources",
        type=Path,
        help="Source YAML; defaults to --html-dir/sources.yaml or the curated live source list",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        nargs="?",
        const=Path("data/visitor_pages"),
        help="Read saved HTML without visitor HTTP requests; defaults to data/visitor_pages",
    )
    parser.add_argument(
        "--collection", help="Override the visitor collection for an isolated pilot"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level)
    output: Path = args.data_dir or settings.data_dir
    chunks: list[VisitorChunk] = []
    if args.html_dir is not None:
        pages = load_saved_pages(args.html_dir, manifest=args.sources)
    else:
        sources_path = args.sources or Path("api/data_sources/visitor_pages.yaml")
        sources = _VisitorSources.model_validate(yaml.safe_load(sources_path.read_text()))
        pages = []
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            crawler = VisitorCrawler(SourceClient(client, interval=1))
            for page in sources.pages:
                url, html = crawler.fetch(page.url)
                pages.append(
                    VisitorPageContent(
                        label=page.label,
                        url=url,
                        html=html,
                        fetched_at=datetime.now(UTC),
                    )
                )
    for content in pages:
        title, markdown = html_to_markdown(content.html)
        chunks.extend(
            chunk_markdown(
                markdown,
                source_url=content.url,
                page_title=title or content.label,
                fetched_at=content.fetched_at,
            )
        )
        logger.info(
            "visitor_page_prepared", source_url=content.url, saved_html=args.html_dir is not None
        )
    if not chunks:
        raise SourceError("No visitor chunks were produced")
    save_chunks(output / "visitor_chunks.jsonl", chunks)
    if not args.prepare_only:
        dense = GeminiEmbedder(
            create_embedding_router(settings), output_dimensionality=settings.embedding_dimensions
        )
        sparse = BM25Embedder(settings.data_dir / "models")
        with closing(qdrant_client(settings)) as client:
            store = HybridStore(
                client,
                args.collection or settings.qdrant_visitor_collection,
                settings.embedding_model,
                settings.embedding_dimensions,
            )
            store.ingest(
                [chunk.document() for chunk in chunks],
                dense,
                sparse,
                batch_size=settings.embedding_batch_size,
                batch_delay_seconds=args.batch_delay_seconds,
            )
            for url in {chunk.source_url for chunk in chunks}:
                store.remove_stale_page_chunks(
                    url, [c.point_id for c in chunks if c.source_url == url]
                )
    logger.info("visitor_ingestion_completed", pages=len(pages), chunks=len(chunks))
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
        print(f"Ingestion operation failed ({type(error).__name__})")
    return 1


def _snapshot_http(settings: Settings) -> httpx.Client:
    """Use explicit Qdrant authentication for streaming snapshot endpoints."""
    headers = (
        {"api-key": settings.qdrant_api_key.get_secret_value()} if settings.qdrant_api_key else {}
    )
    return httpx.Client(
        base_url=str(settings.qdrant_url).rstrip("/") + "/", headers=headers, timeout=600
    )


def publish_index(argv: Sequence[str] | None = None) -> int:
    """Export a validated bundle; upload only when publication is explicitly requested."""
    from met_agent.ingestion.artifacts import IndexKind, export_bundle, publish_bundle

    parser = _base_parser("Export or publish verified Qdrant snapshots and source artifacts")
    parser.add_argument("--collection", help="Override the source collection")
    parser.add_argument("--visitor-collection", help="Override the source visitor collection")
    parser.add_argument("--include-visitor", action="store_true")
    parser.add_argument(
        "--upload", action="store_true", help="Publish to the existing HF_DATASET_REPO"
    )
    args = parser.parse_args(argv)
    settings = load_settings()
    output: Path = args.data_dir or settings.data_dir
    if args.upload and not (settings.hf_dataset_repo and settings.hf_token):
        raise ConfigurationError("Publication requires HF_DATASET_REPO and HF_TOKEN")
    collections: dict[IndexKind, str] = {
        "collection": args.collection or settings.qdrant_collection
    }
    if args.include_visitor:
        collections["visitor"] = args.visitor_collection or settings.qdrant_visitor_collection
    with closing(qdrant_client(settings)) as client, _snapshot_http(settings) as http:
        export_bundle(
            client,
            http,
            output,
            output / "bundle",
            collections,
            settings.embedding_model,
            settings.embedding_dimensions,
        )
    print("Verified snapshot bundle exported")
    if args.upload:
        if settings.hf_dataset_repo is None or settings.hf_token is None:
            raise ConfigurationError("Publication settings are incomplete")
        revision = publish_bundle(
            output / "bundle", settings.hf_dataset_repo, settings.hf_token.get_secret_value()
        )
        print(f"Published dataset revision: {revision}")
    return 0


def seed(argv: Sequence[str] | None = None) -> int:
    """Download a pinned dataset bundle and restore absent collections without LLM calls."""
    from met_agent.ingestion.artifacts import download_bundle, restore_bundle

    parser = _base_parser("Restore a published index without recomputing embeddings")
    parser.add_argument(
        "--revision", default="main", help="HF commit, tag, or branch resolved once"
    )
    parser.add_argument("--bundle", type=Path, help="Restore an already downloaded local bundle")
    parser.add_argument("--collection", help="Override the target collection")
    parser.add_argument("--visitor-collection", help="Override the target visitor collection")
    args = parser.parse_args(argv)
    settings = load_settings()
    output: Path = args.data_dir or settings.data_dir
    bundle: Path = args.bundle or output / "bundle"
    if args.bundle is None:
        if settings.hf_dataset_repo is None:
            raise ConfigurationError("Seed requires HF_DATASET_REPO or --bundle")
        download_bundle(
            settings.hf_dataset_repo,
            args.revision,
            bundle,
            settings.hf_token.get_secret_value() if settings.hf_token else False,
        )
    with closing(qdrant_client(settings)) as client, _snapshot_http(settings) as http:
        manifest = restore_bundle(
            client,
            http,
            bundle,
            {
                "collection": args.collection or settings.qdrant_collection,
                "visitor": args.visitor_collection or settings.qdrant_visitor_collection,
            },
            settings.embedding_model,
            settings.embedding_dimensions,
        )
    for name in manifest.files:
        if not name.endswith(".snapshot"):
            atomic_write(output / name, (bundle / name).read_bytes())
    print(f"Restored {len(manifest.indexes)} index(es) without embedding calls")
    return 0
