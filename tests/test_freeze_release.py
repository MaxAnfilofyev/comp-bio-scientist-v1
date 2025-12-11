import json
import sys
import types

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
sys.modules["pymupdf"] = types.ModuleType("pymupdf")
sys.modules["fitz"] = sys.modules["pymupdf"]
sys.modules["pymupdf4llm"] = types.ModuleType("pymupdf4llm")
omegaconf_stub = types.ModuleType("omegaconf")
omegaconf_stub.OmegaConf = type("OmegaConf", (), {})
omegaconf_stub.DictConfig = dict
omegaconf_stub.ListConfig = list
omegaconf_errors = types.ModuleType("omegaconf.errors")
DummyOmegaConfError = type("OmegaConfBaseException", (Exception,), {})
omegaconf_errors.OmegaConfBaseException = DummyOmegaConfError
omegaconf_errors.ConfigAttributeError = DummyOmegaConfError
omegaconf_errors.ConfigKeyError = DummyOmegaConfError
sys.modules["omegaconf.errors"] = omegaconf_errors
omegaconf_stub.errors = omegaconf_errors
sys.modules["omegaconf"] = omegaconf_stub
coolname_stub = types.ModuleType("coolname")
coolname_stub.generate = lambda: ["stub"]
sys.modules["coolname"] = coolname_stub
shutup_stub = types.ModuleType("shutup")
shutup_stub.mute_warnings = lambda *args, **kwargs: None
sys.modules["shutup"] = shutup_stub
igraph_stub = types.ModuleType("igraph")
igraph_stub.Graph = type("Graph", (), {})
sys.modules["igraph"] = igraph_stub
humanize_stub = types.ModuleType("humanize")
humanize_stub.naturaldelta = lambda value, *_, **__: str(value)
sys.modules["humanize"] = humanize_stub
black_stub = types.ModuleType("black")
black_stub.format_str = lambda src, mode=None: src
black_stub.FileMode = type("FileMode", (), {})
sys.modules["black"] = black_stub
funcy_stub = types.ModuleType("funcy")
funcy_stub.notnone = lambda x: x is not None
funcy_stub.select_values = lambda predicate, mapping: {k: v for k, v in mapping.items() if predicate(v)}
sys.modules["funcy"] = funcy_stub

import agents_orchestrator as ao  # noqa: E402
from ai_scientist.utils import manifest as manifest_utils  # noqa: E402


@pytest.mark.parametrize("include_large_artifacts", [False, True])
def test_freeze_release_creates_bundle(tmp_path, monkeypatch, include_large_artifacts):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    (tmp_path / "project_knowledge.md").write_text("knowledge")
    sim_path = exp_dir / "sim.json"
    sim_path.write_text("{}")
    hypothesis_trace = {
        "hypotheses": [
            {"id": "H1", "experiments": [{"id": "E1", "sim_runs": [str(sim_path)], "figures": [], "metrics": []}]}
        ]
    }
    (exp_dir / "hypothesis_trace.json").write_text(json.dumps(hypothesis_trace))
    (exp_dir / "claim_graph.json").write_text("{}")
    (exp_dir / "provenance_summary.md").write_text("prov")
    manifest_utils.append_or_update(
        {"path": str(sim_path), "kind": "data", "created_by": "test", "status": "ok"}, base_folder=exp_dir
    )

    res = ao.freeze_release(tag="test_release", include_large_artifacts=include_large_artifacts)
    release_dir = exp_dir / "releases" / "test_release"

    assert release_dir.exists()
    code_archive = release_dir / "code_release.zip"
    env_manifest = release_dir / "env_manifest.json"
    release_manifest = release_dir / "release_manifest.json"
    assert code_archive.exists()
    assert env_manifest.exists()
    assert release_manifest.exists()

    manifest_data = json.loads(release_manifest.read_text())
    release_paths = [p["path"] for p in manifest_data.get("files", [])]
    assert any("code_release.zip" in p for p in release_paths)
    assert any("env_manifest.json" in p for p in release_paths)
    # Registered in manifest for lookups by kind.
    env_list = ao.list_artifacts_by_kind("env_manifest")
    assert str(env_manifest) in env_list.get("paths", [])
    assert res["git"].get("dirty") in (True, False)
