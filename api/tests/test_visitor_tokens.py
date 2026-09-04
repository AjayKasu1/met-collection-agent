"""Check pinned tokenizer counting and model-sized, lossless visitor windows offline."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from met_agent.ingestion import visitor_tokens
from met_agent.ingestion.http import SourceError
from met_agent.ingestion.visitors import chunk_markdown
from met_agent.retrieval.embeddings import EmbeddingError
from met_agent.retrieval.local_embeddings import LOCAL_WEIGHTS_REPO, LOCAL_WEIGHTS_REVISION


@pytest.mark.parametrize("max_tokens,length", [(512, 700), (41, 80)])
def test_model_windows_cover_unicode_text_even_when_overlap_exceeds_window(
    max_tokens: int, length: int
) -> None:
    text = "".join(chr(0x4E00 + index) for index in range(length))

    def count(value: str) -> int:
        return len(value.encode("utf-8")) + 2

    chunks = chunk_markdown(
        text,
        source_url="https://www.metmuseum.org/visit",
        page_title="Visitor details",
        fetched_at=datetime(2026, 9, 4, tzinfo=UTC),
        token_count=count,
        max_model_tokens=max_tokens,
    )
    assert len(chunks) > 1
    assert all(count(chunk.text) <= max_tokens and chunk.text in text for chunk in chunks)
    assert {char for chunk in chunks for char in chunk.text} == set(text)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_impossible_token_budget_fails_instead_of_losing_text() -> None:
    with pytest.raises(SourceError, match="token limit"):
        chunk_markdown(
            "Text",
            source_url="https://www.metmuseum.org/visit",
            page_title="Visit",
            fetched_at=datetime.now(UTC),
            token_count=lambda _: 513,
        )


def test_counter_uses_full_pinned_passages_without_padding_or_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[S]": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[S] $A [S]", special_tokens=[("[S]", 1)])
    tokenizer.enable_truncation(max_length=4)
    tokenizer.enable_padding(length=20, pad_id=0, pad_token="[UNK]")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    calls = []

    def download(repo: str, filename: str, **kwargs: Any) -> str:
        assert repo == LOCAL_WEIGHTS_REPO and filename == "tokenizer.json"
        assert kwargs["revision"] == LOCAL_WEIGHTS_REVISION and kwargs["token"] is False
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("Not cached")
        return str(path)

    monkeypatch.setattr(visitor_tokens, "hf_hub_download", download)
    count = visitor_tokens.local_passage_counter(tmp_path)
    assert len(calls) == 2
    assert count("hello world") == 6  # Two prefix tokens, two words, two special tokens.
    assert count(" ".join(["word"] * 600)) == 604

    monkeypatch.setattr(visitor_tokens, "hf_hub_download", lambda *a, **kw: str(path))
    assert visitor_tokens.local_passage_counter(tmp_path)("hello") == 5

    monkeypatch.setattr(visitor_tokens, "hf_hub_download", lambda *a, **kw: "missing")
    with pytest.raises(EmbeddingError, match="tokenizer initialization"):
        visitor_tokens.local_passage_counter(tmp_path)
