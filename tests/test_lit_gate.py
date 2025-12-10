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


def _seed_lit_artifacts(exp_dir: Path, ready: bool = True):
    record = {
        "region": "test",
        "axon_length": 1,
        "branch_order": 1,
        "node_degree": 2,
        "transport_rate": 0.1,
        "mitophagy_rate": 0.01,
        "atp_diffusion_time": 1.0,
        "calcium_energy_cost": 0.5,
    }
    if not ready:
        record.pop("branch_order", None)
    (exp_dir / "lit_summary.json").write_text(json.dumps([record]))

    ver_rows = [
        {"ref_id": "p1", "title": "paper one", "found": ready, "match_score": 0.9},
        {"ref_id": "p2", "title": "paper two", "found": ready, "match_score": 0.8},
    ]
    (exp_dir / "lit_reference_verification.json").write_text(json.dumps(ver_rows))


def test_check_lit_ready_logs_and_marks_ready(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _seed_lit_artifacts(exp_dir, ready=True)

    result = ao.check_lit_ready()
    assert result["status"] == "ready"

    knowledge_path = tmp_path / "project_knowledge.md"
    assert knowledge_path.exists()
    assert "Status=READY" in knowledge_path.read_text()

    provenance_path = exp_dir / "provenance_summary.md"
    assert provenance_path.exists()
    assert "Lit gate: READY" in provenance_path.read_text()


def test_modeling_blocked_when_lit_not_ready(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _seed_lit_artifacts(exp_dir, ready=False)

    class DummyRunBioTool:
        def use_tool(self, **kwargs):
            return {"output_json": "dummy.json"}

    monkeypatch.setattr(ao, "RunBiologicalModelTool", DummyRunBioTool)

    with pytest.raises(RuntimeError):
        ao.run_biological_model(model_key="dummy_model")

    # Skip flag should bypass the gate and allow execution.
    res = ao.run_biological_model(model_key="dummy_model", skip_lit_gate=True)
    assert res["output_json"] == "dummy.json"
