"""Opt-in process initialization for the SimFoundry proxy runtime only."""

import importlib.util
import os

if os.environ.get("SIMFOUNDRY_GEMINI_NONSTREAM_TEXT") == "1":
    try:
        available = importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        available = False
    # Geometry-only environments need no Google client. Main VLM environments have it installed.
    if available:
        from simfoundry_proxy import install

        install()
