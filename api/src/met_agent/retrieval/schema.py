"""Describe embedding spaces independently of any provider SDK."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EmbeddingProvider = Literal["local", "gemini"]


class EmbeddingIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: EmbeddingProvider
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


class ImageVectorSpec(BaseModel):
    """Immutable provenance for one published image space, never a mixture of models."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: str = Field(pattern=r"^metmuseum/[a-zA-Z0-9_-]+$")
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
