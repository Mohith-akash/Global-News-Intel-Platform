"""Unit tests for the headline polish guardrail (etl/headline_polish_job.py)."""

import importlib.util
import sys
import types
from pathlib import Path

# Load the module file directly with stubs for its heavy runtime deps
# (dagster/duckdb/requests/dotenv aren't needed to test validate_polished).
for name in ("duckdb", "requests"):
    sys.modules.setdefault(name, types.ModuleType(name))

dagster_stub = types.ModuleType("dagster")
for attr in ("asset", "Output", "Definitions", "ScheduleDefinition", "define_asset_job", "AssetExecutionContext"):
    setattr(dagster_stub, attr, lambda *a, **k: (lambda f: f) if attr == "asset" else None)
dagster_stub.asset = lambda *a, **k: (lambda f: f)
dagster_stub.define_asset_job = lambda *a, **k: None
dagster_stub.ScheduleDefinition = lambda *a, **k: None
dagster_stub.Definitions = lambda *a, **k: None
sys.modules.setdefault("dagster", dagster_stub)

dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_stub)

_spec = importlib.util.spec_from_file_location(
    "headline_polish_job", Path(__file__).resolve().parent.parent / "etl" / "headline_polish_job.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

validate_polished = _mod.validate_polished


class TestValidatePolished:
    def test_accepts_simple_recasing(self):
        original = "raf flying aces paint sky red white blue"
        assert validate_polished(original, "RAF Flying Aces Paint Sky Red White Blue") is not None

    def test_accepts_deglued_tokens(self):
        original = "caln township celebrates america250 at spackman davis farm"
        result = validate_polished(original, "Caln Township Celebrates America 250 at Spackman Davis Farm")
        assert result is not None

    def test_accepts_deduplicated_words(self):
        original = "korea Korea Ukraine foreign ministers talks Korea"
        assert validate_polished(original, "Korea Ukraine Foreign Ministers Talks") is not None

    def test_rejects_none_and_empty(self):
        assert validate_polished("some original headline", None) is None
        assert validate_polished("some original headline", "") is None
        assert validate_polished("", "Polished Headline Here") is None

    def test_rejects_invented_words(self):
        original = "germany announces new climate policy"
        assert validate_polished(original, "Germany Announces Radical New Climate Policy") is None

    def test_rejects_invented_digits(self):
        original = "germany announces new climate policy"
        assert validate_polished(original, "Germany Announces 2030 Climate Policy") is None

    def test_rejects_too_short_output(self):
        assert validate_polished("germany announces new climate policy", "Germany") is None

    def test_rejects_bloated_output(self):
        original = "short headline text here"
        bloated = "Short Headline Text Here " * 5
        assert validate_polished(original, bloated) is None

    def test_allows_small_words_regardless(self):
        # words of <= 2 chars ("of", "in", "a") are not hallucination-checked
        original = "chancellor visits flood region emergency"
        assert validate_polished(original, "Chancellor Visits Flood Region in Emergency") is not None
