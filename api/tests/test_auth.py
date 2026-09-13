"""Test session ownership signing, constant-time verification, and security guarantees."""

from uuid import uuid4

from met_agent.agent.auth import get_session_secret, sign_session_token, verify_session_token
from met_agent.config import Settings


def test_session_token_signing_and_verification() -> None:
    session_id = uuid4()
    secret = "test-secret-key-32-bytes-long"
    token = sign_session_token(session_id, secret)

    assert isinstance(token, str) and len(token) == 64  # SHA-256 hex
    assert verify_session_token(session_id, token, secret) is True

    # Tampered token fails
    tampered = "a" + token[1:]
    assert verify_session_token(session_id, tampered, secret) is False

    # Different session ID fails with same token
    other_session_id = uuid4()
    assert verify_session_token(other_session_id, token, secret) is False

    # Different secret fails
    assert verify_session_token(session_id, token, "different-secret") is False

    # Missing or empty tokens fail
    assert verify_session_token(session_id, None, secret) is False
    assert verify_session_token(session_id, "", secret) is False
    assert verify_session_token(session_id, token, "") is False


def test_get_session_secret_resolution(settings: Settings) -> None:
    # 1. Fallback when neither session_secret nor edge_origin_token is set
    settings_default = settings.model_copy(
        update={"session_secret": None, "edge_origin_token": None}
    )
    assert get_session_secret(settings_default) == "met-agent-local-session-secret-key"

    # 2. Uses edge_origin_token if present and session_secret is None
    settings_edge = settings.model_copy(
        update={"session_secret": None, "edge_origin_token": "edge-secret-value"}
    )
    assert get_session_secret(settings_edge) == "edge-secret-value"

    # 3. Explicit session_secret takes precedence over edge_origin_token
    settings_explicit = settings.model_copy(
        update={
            "session_secret": "explicit-session-secret",
            "edge_origin_token": "edge-secret-value",
        }
    )
    assert get_session_secret(settings_explicit) == "explicit-session-secret"
