import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

with tempfile.TemporaryDirectory() as td:
    os.environ["HERMES_HOME"] = td

    from hermes_cli.config import load_config, save_config
    from hermes_cli.auth_commands import auth_add_command
    from agent.credential_pool import load_pool
    from agent.models.vertex_ai import _parse_project_and_region_from_base_url

    cfg = load_config()
    cfg["credential_pool_strategies"] = {"vertex": "round_robin"}
    save_config(cfg)

    def add_vertex(label: str, key: str, base: str, project: str, region: str):
        answers = iter([base, project, region])
        with patch("builtins.input", lambda prompt="": next(answers)):
            auth_add_command(
                SimpleNamespace(
                    provider="vertex",
                    auth_type="api_key",
                    label=label,
                    api_key=key,
                    portal_url=None,
                    inference_url=None,
                    client_id=None,
                    scope=None,
                    no_browser=False,
                    timeout=None,
                    insecure=False,
                    ca_bundle=None,
                    min_key_ttl_seconds=None,
                )
            )

    add_vertex("k1", "dummy-key-1", "", "proj-alpha", "us-central1")
    add_vertex("k2", "dummy-key-2", "", "proj-beta", "asia-east1")

    pool = load_pool("vertex")
    entries = pool.entries()
    assert len(entries) == 2, f"expected 2 entries, got {len(entries)}"

    p1, r1 = _parse_project_and_region_from_base_url(entries[0].base_url)
    p2, r2 = _parse_project_and_region_from_base_url(entries[1].base_url)

    assert p1 == "proj-alpha" and r1 == "us-central1", (p1, r1, entries[0].base_url)
    assert p2 == "proj-beta" and r2 == "asia-east1", (p2, r2, entries[1].base_url)

    first = pool.select()
    second = pool.select()
    assert first is not None and second is not None and first.id != second.id, (
        getattr(first, "id", None),
        getattr(second, "id", None),
    )

    rotated = pool.mark_exhausted_and_rotate(status_code=429)
    assert rotated is not None and rotated.id != second.id, (
        getattr(rotated, "id", None),
        getattr(second, "id", None),
    )

    print("SMOKE_OK")
    print("entry1:", entries[0].base_url)
    print("entry2:", entries[1].base_url)
    print("select1:", first.id)
    print("select2:", second.id)
    print("after429:", rotated.id)
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

with tempfile.TemporaryDirectory() as td:
    os.environ["HERMES_HOME"] = td

    from hermes_cli.config import load_config, save_config
    from hermes_cli.auth_commands import auth_add_command
    from agent.credential_pool import load_pool
    from agent.models.vertex_ai import _parse_project_and_region_from_base_url

    cfg = load_config()
    cfg["credential_pool_strategies"] = {"vertex": "round_robin"}
    save_config(cfg)

    def add_vertex(label: str, key: str, base: str, project: str, region: str):
        answers = iter([base, project, region])
        with patch("builtins.input", lambda prompt="": next(answers)):
            auth_add_command(
                SimpleNamespace(
                    provider="vertex",
                    auth_type="api_key",
                    label=label,
                    api_key=key,
                    portal_url=None,
                    inference_url=None,
                    client_id=None,
                    scope=None,
                    no_browser=False,
                    timeout=None,
                    insecure=False,
                    ca_bundle=None,
                    min_key_ttl_seconds=None,
                )
            )

    add_vertex("k1", "dummy-key-1", "", "proj-alpha", "us-central1")
    add_vertex("k2", "dummy-key-2", "", "proj-beta", "asia-east1")

    pool = load_pool("vertex")
    entries = pool.entries()
    assert len(entries) == 2, f"expected 2 entries, got {len(entries)}"

    p1, r1 = _parse_project_and_region_from_base_url(entries[0].base_url)
    p2, r2 = _parse_project_and_region_from_base_url(entries[1].base_url)

    assert p1 == "proj-alpha" and r1 == "us-central1", (p1, r1, entries[0].base_url)
    assert p2 == "proj-beta" and r2 == "asia-east1", (p2, r2, entries[1].base_url)

    first = pool.select()
    second = pool.select()
    assert first is not None and second is not None and first.id != second.id, (
        getattr(first, "id", None),
        getattr(second, "id", None),
    )

    rotated = pool.mark_exhausted_and_rotate(status_code=429)
    assert rotated is not None and rotated.id != second.id, (
        getattr(rotated, "id", None),
        getattr(second, "id", None),
    )

    print("SMOKE_OK")
    print("entry1:", entries[0].base_url)
    print("entry2:", entries[1].base_url)
    print("select1:", first.id)
    print("select2:", second.id)
    print("after429:", rotated.id)
