"""Populate a deployment cache at build time without credentials or provider calls."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from met_agent.retrieval.embeddings import BM25Embedder
from met_agent.retrieval.local_embeddings import (
    LOCAL_DIMENSIONS,
    LOCAL_FILES,
    LOCAL_MODEL,
    LOCAL_WEIGHTS_REPO,
    LOCAL_WEIGHTS_REVISION,
    LocalEmbedder,
)
from met_agent.retrieval.rerank import MODEL, MODEL_FILES, REVISION, LocalReranker


def prepare(cache: Path) -> None:
    """Download pinned ONNX assets without allocating the dense inference session."""
    cache.mkdir(parents=True, exist_ok=True)
    for repo, revision, files in (
        (LOCAL_WEIGHTS_REPO, LOCAL_WEIGHTS_REVISION, list(LOCAL_FILES)),
        (MODEL, REVISION, list(MODEL_FILES)),
    ):
        snapshot_download(
            repo,
            revision=revision,
            token=False,
            cache_dir=str(cache / "hf"),
            allow_patterns=files,
            max_workers=2,
        )
    # FastEmbed's locked version owns the BM25 stopword/tokenization assets.
    BM25Embedder(cache).embed(["Museum visitor information"])
    BM25Embedder(cache, local_files_only=True).embed(["Museum visitor information"])


def verify(cache: Path) -> None:
    """Exercise the actual image assets with network access disabled by the caller."""
    query = "Museum visitor information"
    dense = LocalEmbedder(LOCAL_MODEL, LOCAL_DIMENSIONS, cache, local_files_only=True)
    dense.embed([query], purpose="query")
    BM25Embedder(cache, local_files_only=True).embed([query])
    LocalReranker(cache, local_files_only=True).score(query, [query])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.cache_dir)
        print("Bundled local models passed offline inference", flush=True)
    else:
        prepare(args.cache_dir)
        verify(args.cache_dir)
        print("Bundled local models prepared and verified", flush=True)
