"""Caller identity for the demo: HMAC-signed user tokens.

A token is "<user_id>.<signature>", signed with a server secret. It proves the
caller did not just type someone else's user id. Real deployments replace this with
OIDC or SAML from the company identity provider (Okta, Entra ID) and read groups
from the verified token claims or a directory lookup, never from the request body.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

DEV_SECRET = "dev-only-not-a-secret"


def secret_from_env() -> str:
    return os.environ.get("SECURERAG_SECRET", DEV_SECRET)


def _sig(user_id: str, secret: str) -> str:
    mac = hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def sign(user_id: str, secret: str) -> str:
    return f"{user_id}.{_sig(user_id, secret)}"


def verify(token: str, secret: str) -> str | None:
    """Return the user id if the signature is valid, else None."""
    user_id, dot, sig = (token or "").rpartition(".")
    if not dot or not user_id:
        return None
    return user_id if hmac.compare_digest(sig, _sig(user_id, secret)) else None
