"""Use native nonstream text responses for proxies with incompatible SSE metadata."""

from collections.abc import Mapping
from functools import wraps


def adapt_models_class(models_class):
    """Preserve SDK response objects and finish reasons; only change text transport."""
    original = models_class.generate_content_stream
    if getattr(original, "_simfoundry_nonstream_text", False):
        return

    @wraps(original)
    def generate_content_stream(self, *, model, contents, config=None):
        modalities = (
            config.get("response_modalities", config.get("responseModalities", []))
            if isinstance(config, Mapping)
            else getattr(config, "response_modalities", [])
        )
        if any(str(value).upper().split(".")[-1] == "IMAGE" for value in modalities or []):
            yield from original(self, model=model, contents=contents, config=config)
        else:
            # Yield the complete SDK response. Do not erase safety, truncation, or usage metadata.
            yield self.generate_content(model=model, contents=contents, config=config)

    generate_content_stream._simfoundry_nonstream_text = True
    models_class.generate_content_stream = generate_content_stream


def install():
    from google.genai.models import Models

    adapt_models_class(Models)
