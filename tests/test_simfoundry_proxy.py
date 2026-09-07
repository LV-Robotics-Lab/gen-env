from types import SimpleNamespace

import pytest

from self_improving.sim_adapters.simfoundry.runtime.simfoundry_proxy import adapt_models_class


@pytest.fixture
def sdk_models():
    class Models:
        def __init__(self):
            self.calls = []
            self.response = SimpleNamespace(finish_reason="MAX_TOKENS", text="incomplete")
            self.error = None

        def generate_content(self, **kwargs):
            self.calls.append(("nonstream", kwargs))
            if self.error:
                raise self.error
            return self.response

        def generate_content_stream(self, **kwargs):
            self.calls.append(("stream", kwargs))
            yield self.response

    return Models


@pytest.mark.parametrize("finish_reason", ["STOP", "MAX_TOKENS", "SAFETY"])
def test_text_transport_preserves_complete_response_metadata(sdk_models, finish_reason):
    adapt_models_class(sdk_models)
    sdk = sdk_models()
    sdk.response.finish_reason = finish_reason
    contents = [object()]
    config = SimpleNamespace(response_modalities=["TEXT"])
    result = list(sdk.generate_content_stream(model="gemini", contents=contents, config=config))
    assert result == [sdk.response]
    assert result[0].finish_reason == finish_reason
    assert sdk.calls == [("nonstream", {"model": "gemini", "contents": contents, "config": config})]


@pytest.mark.parametrize("config", [
    {"response_modalities": ["TEXT", "IMAGE"]},
    {"responseModalities": ["IMAGE"]},
    SimpleNamespace(response_modalities=["IMAGE"]),
])
def test_image_generation_keeps_original_stream(sdk_models, config):
    adapt_models_class(sdk_models)
    adapt_models_class(sdk_models)
    sdk = sdk_models()
    result = list(sdk.generate_content_stream(model="image", contents=[], config=config))
    assert result == [sdk.response]
    assert [call[0] for call in sdk.calls] == ["stream"]


def test_remote_failure_is_not_converted_into_success(sdk_models):
    adapt_models_class(sdk_models)
    sdk = sdk_models()
    sdk.error = RuntimeError("quota exhausted")
    with pytest.raises(RuntimeError, match="quota exhausted"):
        list(sdk.generate_content_stream(model="text", contents=[]))
