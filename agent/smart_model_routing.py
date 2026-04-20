"""Minimal smart model routing shim used for tests.

This module provides a small `resolve_turn_route` implementation so unit
tests that import `agent.smart_model_routing` can run even if a full
implementation is not present in this branch.

Behavior: preserve `credential_pool` from the `primary` dict into
the returned `runtime` mapping. Tests only assert on that field.
"""
from typing import Any, Dict, Optional


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal routing decision that preserves credential_pool."""
    credential_pool = None
    if isinstance(primary, dict):
        credential_pool = primary.get("credential_pool")

    runtime = {"credential_pool": credential_pool}
    return {"runtime": runtime}
