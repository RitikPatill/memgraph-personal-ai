"""Session-wide test configuration.

Permanently injects a stub `sentence_transformers` module into sys.modules
before any test runs so that the real package (broken by a NumPy ABI mismatch
in this environment) is never imported.  The stub is present in sys.modules
from the very first line, so patch.dict(sys.modules, ...) calls in tests
always see it in their snapshot and never remove it on teardown.

Also provides an autouse fixture that clears stale package-attribute
references after each test. unittest.mock.patch() resolves targets via
_importer(), which traverses package *attributes* (getattr) rather than
sys.modules directly. When patch.dict(sys.modules) reverts sys.modules at
teardown, it removes the module from the registry but NOT the attribute set
on the parent package object. On the next test, _importer() then finds the
stale module via the attribute chain and patches it, while ``import
memgraph.api.main`` creates a fresh one — causing the mock to be silently
skipped. Deleting the attribute forces a clean import on the next test.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_global_st_stub() -> MagicMock:
    stub = MagicMock()
    # encode returns one float32 vector per input text
    stub.SentenceTransformer.return_value.encode.side_effect = (
        lambda texts, **kw: np.array(
            [[0.1] * 384] * max(len(texts), 1), dtype=np.float32
        )
    )
    return stub


# Inject once, before any test or import triggers sentence_transformers
sys.modules.setdefault("sentence_transformers", _make_global_st_stub())


def _clear_stale_api_refs() -> None:
    """Remove memgraph.api (and its .main child) from the memgraph package object."""
    if "memgraph" not in sys.modules:
        return
    memgraph_pkg = sys.modules["memgraph"]
    api_mod = getattr(memgraph_pkg, "api", None)
    if api_mod is not None:
        try:
            delattr(api_mod, "main")
        except AttributeError:
            pass
        try:
            delattr(memgraph_pkg, "api")
        except AttributeError:
            pass


@pytest.fixture(autouse=True)
def _isolate_api_module():
    """Ensure each test gets a fresh memgraph.api.main import."""
    _clear_stale_api_refs()
    yield
    _clear_stale_api_refs()
