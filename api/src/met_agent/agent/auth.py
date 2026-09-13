"""Session token signing and verification for session ownership."""

import hashlib
import hmac
from uuid import UUID

from met_agent.config import Settings


def sign_session_token(session_id: UUID, secret: str) -> str:
    """Produce a deterministic HMAC-SHA256 signature for session ownership."""
    return hmac.new(secret.encode("utf-8"), session_id.bytes, hashlib.sha256).hexdigest()


def verify_session_token(session_id: UUID, token: str | None, secret: str) -> bool:
    """Verify session token in constant time."""
    if not token or not secret:
        return False
    expected = sign_session_token(session_id, secret)
    return hmac.compare_digest(expected, token)


def get_session_secret(settings: Settings) -> str:
    """Resolve session secret from configuration or fallback safely in dev/test."""
    if settings.session_secret is not None:
        val = settings.session_secret
        return val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)
    if settings.edge_origin_token is not None:
        val = settings.edge_origin_token
        return val.get_secret_value() if hasattr(val, "get_secret_value") else str(val)
    return "met-agent-local-session-secret-key"
