"""Protect private inputs while allowing publishable runtime prompts and configuration examples."""

import pytest

from scripts import check_repository


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.production",
        "api/.env",
        "BUILD_PROMPT.md",
        "golden.jsonl",
        "data/objects.parquet",
        "api/.venv/pyvenv.cfg",
        "runtime.log",
        "qdrant.snapshot",
        "credentials.json",
        "service-account-prod.json",
        ".local/local.md",
        "private.key",
    ],
)
def test_private_paths_are_rejected(path: str) -> None:
    assert check_repository.is_private_path(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "evals/golden.jsonl",
        "api/prompts/system_v1.md",
        "api/data_sources/pages.yaml",
    ],
)
def test_public_artifacts_are_allowed(path: str) -> None:
    assert not check_repository.is_private_path(path)


@pytest.mark.parametrize(
    "credential",
    [b"ghp_" + b"a" * 36, b"AIza" + b"b" * 35, b"gsk_" + b"c" * 40, b"hf_" + b"d" * 30],
)
def test_recognizable_provider_keys_are_detected(credential: bytes) -> None:
    assert check_repository.has_credential(b'key="' + credential + b'"')
    assert not check_repository.has_credential(b'key="unit-test-secret-do-not-use"')


def test_forbidden_paths_are_never_opened(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def git_without_file_reads(*arguments: str) -> bytes:
        assert arguments == ("ls-files", "-z")
        return b".env\0BUILD_PROMPT.md\0"

    monkeypatch.setattr(check_repository, "_git", git_without_file_reads)
    assert check_repository.main() == 1
    assert "Private or generated file" in capsys.readouterr().out


def test_scanner_uses_index_content_and_never_prints_a_match(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = b"ghp_" + b"z" * 36

    def staged_git(*arguments: str) -> bytes:
        if arguments == ("ls-files", "-z"):
            return b"README.md\0"
        assert arguments == ("show", ":README.md")
        return token

    monkeypatch.setattr(check_repository, "_git", staged_git)
    assert check_repository.main() == 1
    output = capsys.readouterr().out
    assert "README.md" in output
    assert token.decode() not in output
