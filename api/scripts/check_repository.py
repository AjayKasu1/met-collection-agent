"""Reject private artifacts and recognizable credentials in the staged Git snapshot.

Read file paths first, never inspect forbidden files, and never print matching
credential values. This is a focused backstop, not a universal secret detector.
"""

import re
import subprocess
from pathlib import PurePosixPath

_PRIVATE_DIRECTORIES = frozenset({"node_modules", "__pycache__"})
_PUBLISHABLE_HIDDEN_DIRECTORIES = frozenset({".devcontainer", ".github"})
_CREDENTIAL_PATTERNS = (
    re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bgsk_[A-Za-z0-9]{40,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def is_private_path(path: str) -> bool:
    """Allow versioned runtime prompts and .env.example, but reject build instructions."""
    file = PurePosixPath(path)
    name = file.name.lower()
    return (
        ((name == ".env" or name.startswith(".env.")) and name != ".env.example")
        or (file.parent == PurePosixPath(".") and "prompt" in name)
        or path == "golden.jsonl"
        or file.parts[0] == "data"
        or bool(_PRIVATE_DIRECTORIES.intersection(file.parts))
        or any(
            part.startswith(".") and part not in _PUBLISHABLE_HIDDEN_DIRECTORIES
            for part in file.parts[:-1]
        )
        or file.suffix.lower() in {".pem", ".key", ".log", ".snapshot", ".sqlite", ".db"}
        or (name.startswith(("credentials", "service-account")) and name.endswith(".json"))
    )


def has_credential(content: bytes) -> bool:
    """Recognize common provider credentials without returning the matched value."""
    return any(pattern.search(content) for pattern in _CREDENTIAL_PATTERNS)


def _git(*arguments: str) -> bytes:
    # Arguments are separate argv entries and are never evaluated by a shell.
    return subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        check=True,
        capture_output=True,
    ).stdout


def main() -> int:
    """Audit staged content so unstaged edits cannot hide a credential in a commit."""
    paths = _git("ls-files", "-z").decode().split("\0")
    problems: list[str] = []
    for path in filter(None, paths):
        if is_private_path(path):
            problems.append(f"Private or generated file: {path}")
            continue
        content = _git("show", f":{path}")
        if has_credential(content):
            problems.append(f"Recognizable credential in staged file: {path}")
    if problems:
        print("Repository check failed:\n" + "\n".join(problems))
        return 1
    print(
        "Repository check passed: no forbidden paths or recognizable credentials in the Git index."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
