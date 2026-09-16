import json
import os

from fastapi.testclient import TestClient
from app.main import app
from app.shadow import (
    ShadowConfig,
    ShadowDeployment,
    ShadowDispatcher,
    ShadowLogStore,
    ShadowPredictor,
    ShadowResult,
    _MAX_SHADOW_TOKENS,
    _redact_for_log,
    compare_outputs,
)

client = TestClient(app)


class _FakePredictor:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []
        self._loaded = result is not None and exc is None

    @property
    def is_loaded(self):
        return self._loaded

    def predict(self, prompt, params):
        self.calls.append((prompt, params))
        if self.exc is not None:
            raise self.exc
        return self.result


def _config(tmp_path, **overrides):
    kwargs = dict(
        enabled=True,
        model_name="candidate-model",
        model_path="",
        log_path=str(tmp_path / "shadow_log.jsonl"),
        queue_maxsize=100,
        status_entries=50,
    )
    kwargs.update(overrides)
    return ShadowConfig(**kwargs)


def _deployment(tmp_path, predictor, config=None):
    cfg = config or _config(tmp_path)
    log_store = ShadowLogStore(cfg.log_path)
    dispatcher = ShadowDispatcher(cfg, predictor, log_store)
    return ShadowDeployment(config=cfg, predictor=predictor, log_store=log_store, dispatcher=dispatcher)


class _FakeInputs(dict):
    def __init__(self, ids):
        super().__init__(input_ids=_FakeTensor(ids))

    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (1, len(rows[0]))


class _FakeTokenizer:
    chat_template = "<|im_start|>"
    pad_token = "<pad>"
    eos_token_id = 50256

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "system: assistant: user text"

    def __call__(self, texts, return_tensors="pt"):
        return _FakeInputs([[1, 2, 3, 4, 5]])

    def decode(self, tokens, skip_special_tokens=True):
        return "hello shadow world"


class _FakeModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5, 6, 7, 8, 9]]


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_compare_outputs():
    assert compare_outputs("Hello World", "hello world") == "identical"
    assert compare_outputs("  alpha beta  ", "alpha  beta") == "identical"
    assert compare_outputs("hello", "goodbye") == "differing"
    assert compare_outputs(None, "hello") == "error"


def test_redact_for_log_empty():
    assert _redact_for_log(None) == ""
    assert _redact_for_log("") == ""


def test_redact_for_log_short():
    result = _redact_for_log("hello world")
    # Only a hash is returned; no raw content is preserved
    assert result != "hello world"
    assert len(result) == 8  # _PROMPT_HASH_PREFIX


def test_redact_for_log_long_text():
    long = "A" * 200
    result = _redact_for_log(long)
    assert result == _redact_for_log("A" * 200)
    assert len(result) == 8


def test_redact_for_log_strips_pii():
    result = _redact_for_log("My SSN is 123-45-6789")
    assert "123-45-6789" not in result


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_validates_negative_queue_max(monkeypatch):
    monkeypatch.setenv("SHADOW_QUEUE_MAX", "not-a-number")
    cfg = ShadowConfig.from_env()
    assert cfg.queue_maxsize == 100  # falls back to default

    monkeypatch.setenv("SHADOW_QUEUE_MAX", "-5")
    cfg = ShadowConfig.from_env()
    assert cfg.queue_maxsize == 1  # clamped to min_value


def test_config_validates_negative_status_entries(monkeypatch):
    monkeypatch.setenv("SHADOW_STATUS_ENTRIES", "abc")
    cfg = ShadowConfig.from_env()
    assert cfg.status_entries == 50


# ---------------------------------------------------------------------------
# Disabled shadow
# ---------------------------------------------------------------------------


def test_disabled_capture_noop(tmp_path):
    cfg = ShadowConfig(
        enabled=False,
        model_name="",
        model_path="",
        log_path=str(tmp_path / "shadow_log.jsonl"),
    )
    predictor = _FakePredictor(result=ShadowResult("x", "p", "t", 1, 2, 3.0))
    dep = _deployment(tmp_path, predictor, config=cfg)
    dep.start()

    ok = dep.capture(prompt="p", params={}, active_text="a", active_tokens=1, request_id="r1", active_model="active")

    assert ok is False
    assert dep.dispatcher.stats.skipped == 1
    assert predictor.calls == []
    assert not (tmp_path / "shadow_log.jsonl").exists()

    status = dep.status()
    assert status["status"] == "disabled"
    assert status["enabled"] is False
    assert status["processed"] == 0
    assert status["recent_entries"] == []


# ---------------------------------------------------------------------------
# End-to-end (fake predictor)
# ---------------------------------------------------------------------------


def test_shadow_end_to_end(tmp_path):
    predictor = _FakePredictor(
        result=ShadowResult(
            model_name="candidate-model",
            prompt="p",
            shadow_text="hello shadow world",
            input_tokens=5,
            output_tokens=4,
            latency_ms=3.2,
        )
    )
    dep = _deployment(tmp_path, predictor)
    dep.start()

    ok = dep.capture(
        prompt="p",
        params={"temperature": 0.7},
        active_text="hello shadow world",
        active_tokens=9,
        request_id="r1",
        active_model="active-model",
    )

    assert ok is True
    assert dep.dispatcher.queue_size() == 1
    dep.flush(timeout=5)

    assert predictor.calls == [("p", {"temperature": 0.7})]
    assert dep.dispatcher.stats.queued == 1
    assert dep.dispatcher.stats.completed == 1
    assert dep.dispatcher.stats.identical == 1

    log_path = tmp_path / "shadow_log.jsonl"
    assert log_path.exists()
    entries = dep.log_store.recent(50)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["request_id"] == "r1"
    assert entry["agreement"] == "identical"
    assert entry["active"]["model"] == "active-model"
    assert entry["active"]["tokens"] == 9
    # PII: raw text must not appear in on-disk log
    assert "text" not in entry.get("active", {})
    assert "text" not in entry.get("shadow", {})
    assert "prompt" not in entry or entry["prompt"] != "p"

    status = dep.status()
    assert status["status"] == "ok"
    assert status["enabled"] is True
    assert status["processed"] == 1
    assert status["agreement_identical"] == 1
    # recent_entries are redacted for security
    assert status["recent_entries"] == []

    dep.stop()


# ---------------------------------------------------------------------------
# Predictor internals
# ---------------------------------------------------------------------------


def test_shadow_predictor_run(tmp_path):
    config = ShadowConfig(enabled=True, model_name="fake", model_path="", log_path=str(tmp_path / "l.jsonl"))
    predictor = ShadowPredictor(config)
    predictor._tokenizer = _FakeTokenizer()
    predictor._model = _FakeModel()
    predictor._torch = None

    result = predictor.predict("hello", {"max_new_tokens": 10})

    assert result.prompt == "hello"
    assert result.shadow_text == "hello shadow world"
    assert result.input_tokens == 5
    assert result.output_tokens == 4
    assert result.latency_ms >= 0
    assert predictor.is_loaded


def test_predictor_caps_max_new_tokens(tmp_path):
    config = ShadowConfig(enabled=True, model_name="fake", model_path="", log_path=str(tmp_path / "l.jsonl"))
    predictor = ShadowPredictor(config)
    predictor._tokenizer = _FakeTokenizer()
    predictor._model = _FakeModel()
    predictor._torch = None

    # FakeModel.generate records kwargs for inspection
    captured_kwargs = {}

    class _SpyModel:
        def generate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

    predictor._model = _SpyModel()

    predictor.predict("test", {"max_new_tokens": 99999})

    assert captured_kwargs["max_new_tokens"] == _MAX_SHADOW_TOKENS


def test_predictor_rejects_out_of_project_path(tmp_path):
    config = ShadowConfig(
        enabled=True,
        model_name="",
        model_path="/tmp/evil/malicious-model",
        log_path=str(tmp_path / "l.jsonl"),
    )
    predictor = ShadowPredictor(config)
    import pytest

    with pytest.raises(RuntimeError, match="outside the project root"):
        predictor._load()


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_shadow_failure_isolated(tmp_path):
    predictor = _FakePredictor(exc=RuntimeError("shadow boom"))
    dep = _deployment(tmp_path, predictor)
    dep.start()

    ok = dep.capture(prompt="p", params={}, active_text="a", active_tokens=1, request_id="r2", active_model="active")

    assert ok is True
    dep.flush(timeout=5)

    assert dep.dispatcher.stats.failed == 1
    assert dep.dispatcher.stats.completed == 1

    entries = dep.log_store.recent(50)
    assert len(entries) == 1
    assert entries[0]["agreement"] == "error"
    # PII: error text and prompt are redacted
    assert "text" not in entries[0].get("shadow", {})

    status = dep.status()
    assert status["failed"] == 1
    assert status["processed"] == 1

    dep.stop()


# ---------------------------------------------------------------------------
# Queue overflow
# ---------------------------------------------------------------------------


def test_queue_full_drops(tmp_path):
    config = _config(tmp_path, queue_maxsize=1)
    predictor = _FakePredictor(result=ShadowResult("candidate-model", "p", "t", 1, 2, 3.0))
    dep = _deployment(tmp_path, predictor, config=config)

    ok1 = dep.capture(prompt="p1", params={}, active_text="a", active_tokens=1, request_id="r3", active_model="active")
    ok2 = dep.capture(prompt="p2", params={}, active_text="a", active_tokens=1, request_id="r4", active_model="active")

    assert ok1 is True
    assert ok2 is False
    assert dep.dispatcher.stats.queued == 1
    assert dep.dispatcher.stats.dropped == 1
    dep.stop()


# ---------------------------------------------------------------------------
# API: /shadow-status endpoint
# ---------------------------------------------------------------------------


def test_shadow_status_endpoint_disabled():
    response = client.get("/shadow-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "disabled"
    assert body["enabled"] is False
    assert body["recent_entries"] == []
    assert "processed" in body


# ---------------------------------------------------------------------------
# API: /predict triggers capture
# ---------------------------------------------------------------------------


def test_predict_triggers_capture_without_model(monkeypatch):
    calls = []

    class _FakeDeployment:
        def capture(self, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr("app.main.SHADOW", _FakeDeployment())
    monkeypatch.setattr("app.main.get_model", lambda: (_FakeTokenizer(), _FakeModel()))

    response = client.post("/predict", json={"prompt": "hi", "max_new_tokens": 10})

    assert response.status_code == 200
    assert response.json()["generated_text"] == "hello shadow world"
    assert len(calls) == 1
    payload = calls[0]
    assert payload["prompt"] == "hi"
    assert payload["params"]["max_new_tokens"] == 10
    assert payload["active_model"] and isinstance(payload["active_model"], str)
    assert isinstance(payload["request_id"], str)
    assert payload["active_tokens"] == 4
    assert "active_text" in payload
