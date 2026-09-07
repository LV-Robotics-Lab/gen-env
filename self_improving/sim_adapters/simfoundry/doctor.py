"""Report local installation facts without booting Isaac Sim or exposing credentials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
UPSTREAM = WORKSPACE / "external/SimFoundry"
RUNTIME = WORKSPACE / ".cache/simfoundry/miniforge"


def inspect_environment(name: str) -> dict:
    python = RUNTIME / "envs" / name / "bin/python"
    if not python.is_file():
        return {"python_exists": False, "modules": {}}
    # find_spec on top-level packages avoids OmniGibson's import-time simulator startup.
    probe = (
        "import importlib.util,json; "
        "names=['simfoundry','torch','omegaconf','omnigibson']; "
        "print(json.dumps({n:(s.origin if (s:=importlib.util.find_spec(n)) "
        "else None) for n in names}))"
    )
    result = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, timeout=30,
        cwd=WORKSPACE / ".cache/simfoundry",
    )
    return {
        "python_exists": True,
        "probe_exit_code": result.returncode,
        "modules": json.loads(result.stdout) if result.returncode == 0 else {},
    }


def main() -> None:
    revision = subprocess.run(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    log_dir = WORKSPACE / "logs/simfoundry"
    attempt = "install-recovery" if (log_dir / "install-recovery.log").exists() else "install-core"
    exit_file = log_dir / f"{attempt}.exit"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hf_token_path = Path(os.environ.get("HF_TOKEN_PATH", hf_home / "token"))
    report = {
        "schema_version": "simfoundry.local_preflight.v1",
        "upstream_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "tools": {n: shutil.which(n) for n in ("mamba", "ffmpeg", "git-lfs", "nvidia-smi")},
        "free_disk_gib": round(shutil.disk_usage(WORKSPACE).free / 2**30, 1),
        "environments": {
            n: inspect_environment(n) for n in ("simfoundry", "hunyuan", "any6d", "da3")
        },
        "credentials": {
            "hf_token_present": bool(os.environ.get("HF_TOKEN")) or hf_token_path.is_file(),
            "gemini_key_in_environment": bool(os.environ.get("GEMINI_API_KEY")),
            "upstream_key_file_exists": (UPSTREAM / "api_keys.txt").is_file(),
            "gated_model_access": "not_checked",
            "gemini_service_access": "not_checked",
        },
        "installer_exit_code": int(exit_file.read_text()) if exit_file.exists() else None,
        "installer_log": str(log_dir / f"{attempt}.log"),
        "verification": "Presence only; not proof of a working GPU runtime or reconstruction.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
