import json
import sys
import types
from pathlib import Path

import pytest

# Stub optional provider SDKs so imports in agents_orchestrator do not fail during tests.
anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.RateLimitError = type("RateLimitError", (Exception,), {})
anthropic_stub.APIConnectionError = type("APIConnectionError", (Exception,), {})
anthropic_stub.APIStatusError = type("APIStatusError", (Exception,), {})
anthropic_stub.APIResponseValidationError = type("APIResponseValidationError", (Exception,), {})
anthropic_stub.Anthropic = type("Anthropic", (), {})
anthropic_stub.AnthropicBedrock = type("AnthropicBedrock", (), {})
anthropic_stub.AnthropicVertex = type("AnthropicVertex", (), {})
anthropic_stub.__getattr__ = lambda name: type(name, (Exception,), {})
sys.modules["anthropic"] = anthropic_stub

openai_stub = types.ModuleType("openai")
openai_stub.AsyncOpenAI = type("AsyncOpenAI", (), {})
openai_stub.OpenAI = type("OpenAI", (), {})
openai_stub.RateLimitError = type("RateLimitError", (Exception,), {})
openai_stub.APITimeoutError = type("APITimeoutError", (Exception,), {})
openai_stub.InternalServerError = type("InternalServerError", (Exception,), {})
openai_stub.APIError = type("APIError", (Exception,), {})
openai_stub.__getattr__ = lambda name: type(name, (Exception,), {})
sys.modules["openai"] = openai_stub

sys.modules["dataclasses_json"] = types.SimpleNamespace(DataClassJsonMixin=type("DataClassJsonMixin", (), {}))
agents_stub = types.ModuleType("agents")
agents_stub.Agent = type("Agent", (), {})
agents_stub.Runner = type("Runner", (), {"run": staticmethod(lambda *args, **kwargs: None)})
agents_stub.ModelSettings = type("ModelSettings", (), {})


def _function_tool(func=None, **_kwargs):
    if func is None:
        return lambda f: f
    return func


agents_stub.function_tool = _function_tool
sys.modules["agents"] = agents_stub

import agents_orchestrator as ao  # noqa: E402


def _write_claim_graph(base: Path, missing_support: bool):
    claims = [
        {"claim_id": "thesis", "claim_text": "root claim", "parent_id": None, "support": ["E1"], "status": "complete"},
        {
            "claim_id": "c1",
            "claim_text": "important claim",
            "parent_id": "thesis",
            "support": [] if missing_support else ["E1"],
            "status": "unlinked" if missing_support else "complete",
        },
    ]
    (base / "claim_graph.json").write_text(json.dumps(claims))


def _write_hypothesis_trace(base: Path, include_support: bool):
    experiments = [
        {
            "id": "E1",
            "description": "exp1",
            "sim_runs": [{"baseline": "g"}] if include_support else [],
            "metrics": ["m1"] if include_support else [],
            "figures": [],
        }
    ]
    trace = {"hypotheses": [{"id": "H1", "name": "hyp", "experiments": experiments}]}
    (base / "hypothesis_trace.json").write_text(json.dumps(trace))


def test_claim_consistency_flags_missing(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _write_claim_graph(exp_dir, missing_support=True)
    _write_hypothesis_trace(exp_dir, include_support=False)

    result = ao.check_claim_consistency()
    assert result["overall_status"] == "not_ready_for_publication"
    missing = [c for c in result["claims"] if c["support_status"] == "missing"]
    assert missing, "Expected at least one missing claim"


def test_claim_consistency_passes_when_supported(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _write_claim_graph(exp_dir, missing_support=False)
    _write_hypothesis_trace(exp_dir, include_support=True)

    result = ao.check_claim_consistency()
    assert result["overall_status"] != "not_ready_for_publication"
    for claim in result["claims"]:
        assert claim["support_status"] in {"ok", "weak"}


def test_run_writeup_blocks_on_missing_claim_support(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))
    monkeypatch.setenv("AISC_ENFORCE_CLAIM_CONSISTENCY", "true")

    _write_claim_graph(exp_dir, missing_support=True)
    _write_hypothesis_trace(exp_dir, include_support=False)

    # Avoid calling heavy writeup; stub perform_writeup return True
    monkeypatch.setattr(ao, "perform_writeup", lambda *args, **kwargs: True)

    with pytest.raises(RuntimeError):
        ao.run_writeup_task()

