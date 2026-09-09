"""Select one explicit embedding provider; never fall back across incompatible vector spaces."""

from met_agent.config import Settings
from met_agent.llm.router import create_embedding_router
from met_agent.retrieval.embeddings import GeminiEmbedder, IdentifiedEmbedder
from met_agent.retrieval.local_embeddings import LocalEmbedder


def create_embedder(settings: Settings) -> IdentifiedEmbedder:
    if settings.embedding_provider == "local":
        return LocalEmbedder(
            settings.embedding_model,
            settings.embedding_dimensions,
            settings.model_cache_dir or settings.data_dir / "models",
            threads=settings.embedding_threads,
            local_files_only=settings.models_offline,
        )
    return GeminiEmbedder(
        create_embedding_router(settings),
        output_dimensionality=settings.embedding_dimensions,
        model_name=settings.embedding_model,
    )
