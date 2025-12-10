import csv
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


def _write_spec_and_params(exp_dir: Path, model_key: str, missing: bool = True):
    models_dir = exp_dir / "models"
    params_dir = exp_dir / "parameters"
    models_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "model_key": model_key,
        "description": "dummy",
        "parameters": {"alpha": 1.0, "beta": 2.0},
    }
    (models_dir / f"{model_key}_spec.yaml").write_text(json.dumps(spec))

    fieldnames = ["param_name", "value", "units", "source_type", "lit_claim_id", "reference_id", "notes"]
    with (params_dir / f"{model_key}_param_sources.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "param_name": "alpha",
                "value": 1.0,
                "units": "unitless",
                "source_type": "free_hyperparameter" if missing else "lit_value",
                "lit_claim_id": "",
                "reference_id": "",
                "notes": "",
            }
        )
        if not missing:
            writer.writerow(
                {
                    "param_name": "beta",
                    "value": 2.0,
                    "units": "unitless",
                    "source_type": "fit_to_data",
                    "lit_claim_id": "",
                    "reference_id": "",
                    "notes": "",
                }
            )


def test_model_provenance_blocks_when_enforced(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _write_spec_and_params(exp_dir, "dummy_model", missing=True)

    class DummyRunBioTool:
        def use_tool(self, **kwargs):
            return {"output_json": "dummy.json"}

    monkeypatch.setattr(ao, "RunBiologicalModelTool", DummyRunBioTool)

    with pytest.raises(RuntimeError):
        ao.run_biological_model(model_key="dummy_model", enforce_param_provenance=True, skip_lit_gate=True)


def test_model_provenance_logs_failure_pattern_when_not_enforced(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _write_spec_and_params(exp_dir, "dummy_model", missing=True)

    class DummyRunBioTool:
        def use_tool(self, **kwargs):
            return {"output_json": "dummy.json"}

    monkeypatch.setattr(ao, "RunBiologicalModelTool", DummyRunBioTool)

    res = ao.run_biological_model(model_key="dummy_model", enforce_param_provenance=False, skip_lit_gate=True)
    assert res["output_json"] == "dummy.json"

    knowledge_path = tmp_path / "project_knowledge.md"
    assert knowledge_path.exists()
    content = knowledge_path.read_text()
    assert "[FAILURE_PATTERN]" in content


def test_model_provenance_passes_when_complete(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    _write_spec_and_params(exp_dir, "dummy_model", missing=False)

    class DummyRunBioTool:
        def use_tool(self, **kwargs):
            return {"output_json": "dummy.json"}

    monkeypatch.setattr(ao, "RunBiologicalModelTool", DummyRunBioTool)

    res = ao.run_biological_model(model_key="dummy_model", enforce_param_provenance=True, skip_lit_gate=True)
    assert res["output_json"] == "dummy.json"

    provenance_path = exp_dir / "provenance_summary.md"
    assert provenance_path.exists()
    assert "dummy_model: READY" in provenance_path.read_text()
