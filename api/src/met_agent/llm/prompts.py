"""Load only versioned, packaged prompt files and expose their content hashes for audit."""

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import Literal

PromptName = Literal[
    "system_v1",
    "system_v2",
    "tools_v1",
    "intent_v1",
    "intent_v2",
    "intent_v3",
    "intent_v4",
    "grounding_v1",
    "evaluation_v1",
    "citations_v1",
]


def load_prompt(name: PromptName) -> str:
    if name not in {
        "system_v1",
        "system_v2",
        "tools_v1",
        "intent_v1",
        "intent_v2",
        "intent_v3",
        "intent_v4",
        "grounding_v1",
        "evaluation_v1",
        "citations_v1",
    }:
        raise ValueError("Unknown prompt version")
    resource = files("met_agent").joinpath("prompts", name + ".md")
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    # Editable installs expose src directly; wheels include these resources in the package.
    return (Path(__file__).resolve().parents[3] / "prompts" / (name + ".md")).read_text(
        encoding="utf-8"
    )


def prompt_hash(name: PromptName) -> str:
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()
