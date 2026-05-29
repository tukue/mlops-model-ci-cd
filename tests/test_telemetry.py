import os

os.environ["SKIP_MODEL_LOAD_ON_STARTUP"] = "1"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://noop:4318"

import pytest

pytest.importorskip("opentelemetry")
pytest.importorskip("openlit")

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Import the module (this runs module-level code but NOT startup events)
import app.main as main

# Patch telemetry init functions so startup event is a no-op
with patch.object(main, "setup_telemetry"):
    # Now create TestClient — this triggers startup_event which calls patched setup_telemetry
    client = TestClient(main.app)


class DummyInputIds:
    shape = (1, 2)


class DummyInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class DummyTokenizer:
    chat_template = None
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = "<eos>"

    def __call__(self, texts, return_tensors):
        return DummyInputs(input_ids=DummyInputIds())

    def decode(self, tokens, skip_special_tokens=True):
        return "hello from test"


class DummyModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]


@pytest.fixture
def mock_tracer():
    span_mock = MagicMock()
    tracer_mock = MagicMock()
    tracer_mock.start_as_current_span.return_value.__enter__.return_value = span_mock
    with patch.object(main, "TRACER", tracer_mock):
        yield tracer_mock, span_mock


def test_genai_span_attributes_set(mock_tracer):
    tracer_mock, span_mock = mock_tracer
    with patch.object(main, "get_model", return_value=(DummyTokenizer(), DummyModel())):
        response = client.post(
            "/predict",
            json={"prompt": "Hello", "max_new_tokens": 10},
        )

    assert response.status_code == 200

    tracer_mock.start_as_current_span.assert_called_once_with("chat")
    span_mock.set_attribute.assert_any_call("gen_ai.operation.name", "chat")
    span_mock.set_attribute.assert_any_call("gen_ai.provider.name", "huggingface")
    span_mock.set_attribute.assert_any_call("gen_ai.request.model", main.MODEL_NAME)
    span_mock.set_attribute.assert_any_call("gen_ai.request.max_tokens", 10)
    span_mock.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 2)
    span_mock.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 3)
    span_mock.set_attribute.assert_any_call("gen_ai.response.model", main.MODEL_NAME)
    span_mock.set_attribute.assert_any_call("gen_ai.response.finish_reasons", ["stop"])


def test_genai_span_not_created_on_validation_error():
    response = client.post("/predict", json={"max_new_tokens": 10})
    assert response.status_code == 422
