"""Native Gemini transport and capability probes, with secret-free error reports."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from scene_gen.llm_provider import load_llm_provider_config
from self_improving.sim_adapters.genesis.vision_request import NoRedirect, SelectionError

BASE_URL = "https://api2.aigcbest.top"
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-3-pro-image"


class NativeClient:
    def __init__(self, config_path):
        self.config = load_llm_provider_config(config_path)

    def safe_dict(self):
        import hashlib
        return dict(model=TEXT_MODEL, api_mode="gemini_native", timeout_s=180,
                    endpoint_sha256=hashlib.sha256(BASE_URL.encode()).hexdigest())

    def generate(self, parts, *, model=TEXT_MODEL, image=False, system=None):
        payload = {"contents": [{"role": "user", "parts": parts}]}
        payload["generationConfig"] = (
            {"responseModalities": ["TEXT", "IMAGE"]} if image else {}
        )
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        request = urllib.request.Request(
            f"{BASE_URL}/v1beta/models/{model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.config.api_key},
            method="POST",
        )
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=180) as response:
                body = response.read(32 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise SelectionError(f"gemini_http_{exc.code}") from None
        except Exception:
            raise SelectionError("gemini_transport_error") from None
        if len(body) > 32 * 1024 * 1024:
            raise SelectionError("gemini_response_too_large")
        try:
            envelope = json.loads(body)
            candidate = envelope["candidates"][0]
            if candidate.get("finishReason") != "STOP" or envelope.get("promptFeedback", {}).get(
                "blockReason"
            ):
                raise SelectionError("gemini_incomplete_or_blocked")
            return candidate["content"]["parts"]
        except SelectionError:
            raise
        except (ValueError, KeyError, IndexError, TypeError):
            raise SelectionError("gemini_invalid_response") from None

    def __call__(self, messages):
        parts, system = [], []
        for message in messages:
            content = message["content"]
            if message["role"] == "system":
                system.append(content)
                continue
            if isinstance(content, str):
                parts.append({"text": content})
                continue
            for item in content:
                if item["type"] == "text":
                    parts.append({"text": item["text"]})
                else:
                    url = item["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise SelectionError("gemini_requires_inline_image")
                    header, data = url.split(",", 1)
                    parts.append({"inlineData": {"mimeType": header[5:].split(";")[0],
                                                 "data": data}})
        response = self.generate(parts, system="\n".join(system))
        text = "".join(p.get("text", "") for p in response if not p.get("thought"))
        if not text:
            raise SelectionError("gemini_missing_text")
        return text


def probe(config_path, image_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    client = NativeClient(config_path)
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        source.thumbnail((512, 512))
        buffer = io.BytesIO()
        source.save(buffer, "PNG")
    inline = {"inlineData": {"mimeType": "image/png",
                             "data": base64.b64encode(buffer.getvalue()).decode()}}
    results = []
    for name, model, parts, is_image in [
        ("text", TEXT_MODEL, [{"text": "Reply with OK."}], False),
        ("vision", TEXT_MODEL, [{"text": "Describe the main object."}, inline], False),
        ("image_edit", IMAGE_MODEL, [{"text": "Isolate the main object on a white background, "
                                     "preserving its shape and color. Return an edited image."},
                                    inline], True),
    ]:
        started = time.perf_counter()
        row = dict(capability=name, model=model, status="failed")
        try:
            response = client.generate(parts, model=model, image=is_image)
            if is_image:
                images = [p.get("inlineData", p.get("inline_data")) for p in response
                          if "inlineData" in p or "inline_data" in p]
                if not images:
                    raise SelectionError("gemini_missing_image")
                decoded = base64.b64decode(images[0]["data"], validate=True)
                with Image.open(io.BytesIO(decoded)) as im:
                    im.load()
                    im.save(out / "image_edit.png")
                    row["image_size"] = list(im.size)
            elif not any(p.get("text") and not p.get("thought") for p in response):
                raise SelectionError("gemini_missing_text")
            row["status"] = "passed"
        except SelectionError as exc:
            row["error"] = str(exc)
        except Exception:
            row["error"] = "gemini_invalid_image"
        row["elapsed_s"] = time.perf_counter() - started
        results.append(row)
        report = dict(status="passed" if all(r["status"] == "passed" for r in results)
                      and len(results) == 3 else "failed", endpoint=BASE_URL, probes=results)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.config, args.image, args.output_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
