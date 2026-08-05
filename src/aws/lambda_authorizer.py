"""Lambda authorizer HTTP API : valide le header x-api-key contre le secret configuré."""
from __future__ import annotations

import os

API_KEY = os.environ.get("API_KEY", "")


def handler(event: dict, context) -> dict:
    """Point d'entrée Lambda authorizer (format simple response, payload v2.0)."""
    provided_key = event.get("headers", {}).get("x-api-key", "")
    is_authorized = bool(API_KEY) and provided_key == API_KEY
    return {"isAuthorized": is_authorized}
