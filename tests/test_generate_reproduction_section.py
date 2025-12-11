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


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("AISC_ACTIVE_ROLE", "tester")
    yield


def test_generate_reproduction_section(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiment_results"
    exp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISC_BASE_FOLDER", str(tmp_path))
    monkeypatch.setenv("AISC_EXP_RESULTS", str(exp_dir))

    tag = "test_release"
    release_root = exp_dir / "releases" / tag
    release_root.mkdir(parents=True, exist_ok=True)

    env_manifest = release_root / "env_manifest.json"
    env_manifest.write_text(json.dumps({"python_version": "3.11.4"}))

    fig_path = release_root / "figures" / "figure1.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.write_text("fake")
    fig_checksum = ao._safe_sha256(fig_path)

    files = [
        {"path": ao._relative_to_release(env_manifest, release_root), "checksum": ao._safe_sha256(env_manifest)},
        {"path": ao._relative_to_release(fig_path, release_root), "checksum": fig_checksum},
    ]
    release_manifest = {
        "tag": tag,
        "git": {"commit": "abc123", "dirty": False},
        "env_manifest_path": ao._relative_to_release(env_manifest, release_root),
        "files": files,
    }
    (release_root / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2))

    res = ao.generate_reproduction_section(tag=tag)
    assert "methods_section_md" in res and "supplementary_protocol_md" in res
    assert "code_release.zip" not in res["methods_section_md"] or len(res["methods_section_md"].split()) <= 400

    methods_path = release_root / "reproduction_methods.md"
    protocol_path = release_root / "reproduction_protocol.md"
    assert methods_path.exists()
    assert protocol_path.exists()

    manifest_entries = ao.manifest_utils.load_entries(base_folder=exp_dir, limit=None)
    kinds = {e.get("kind") for e in manifest_entries}
    assert "repro_methods_md" in kinds
    assert "repro_protocol_md" in kinds
