"""Hash-bound media stage checkpoints; partial outputs are never reusable."""
from pathlib import Path

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official


def snapshot(root):
    root = Path(root)
    return [official.fingerprint(p, root) for p in sorted(root.rglob("*")) if p.is_file()]


def scoped_config(name, config):
    return ({k: v for k, v in config.items() if k != 'physics_implementation_sha256'}
            if name in ('objects', 'scene', 'reconstruction') or name.startswith('sf_') else config)


def save(task, name, directory, config, result=None):
    path = task.root / "checkpoints" / f"{name}.json"
    official.write_json(path, dict(config=scoped_config(name, config), directory=str(directory),
                                   files=snapshot(directory), result=result))


def read(task, name, directory, config):
    path = task.root / "checkpoints" / f"{name}.json"
    if not path.exists():
        return None
    report = lib.read_json(path)
    if (scoped_config(name, report["config"]) != scoped_config(name, config)
            or report["directory"] != str(directory)):
        raise ValueError(f"{name} checkpoint configuration mismatch; use a new task")
    official.verify_files(directory, report["files"])
    if snapshot(directory) != report["files"]:
        raise ValueError(f"{name} checkpoint file set changed; use a new task")
    return report


def invalidate(task, first):
    """Archive invalidated outputs inside the owned task, then reset downstream stages."""
    import shutil
    import time
    names = ["objects", "scene", "physics", "final_render"]
    affected = names[names.index(first):]
    archive = task.root / "attempt_history" / str(time.time_ns())
    archive.mkdir(parents=True)
    for name in affected:
        stage = task.stage(name)
        if stage.exists():
            shutil.move(str(stage), str(archive / stage.name))
        stage.mkdir()
        task.report["stages"][name] = "not_run"
        checkpoint = task.root / "checkpoints" / f"{name}.json"
        if checkpoint.exists():
            shutil.move(str(checkpoint), str(archive / checkpoint.name))
    if first == "objects":
        checkpoint = task.root / "checkpoints/reconstruction.json"
        if checkpoint.exists():
            shutil.move(str(checkpoint), str(archive / "reconstruction.json"))
    for key in ("physics", "physics_status", "render_status", "failure_phase", "error"):
        task.report.pop(key, None)
    task.report.update(status="running")
    task.seal()


def reusable(task, name, directory, config):
    try:
        return read(task, name, directory, config)
    except (OSError, ValueError, KeyError):
        invalidate(task, name)
        return None


def save_upstream(task, scene, config):
    for info in sorted(Path(scene).glob("*/stage_info.json")):
        try:
            successful = lib.read_json(info).get("success") is True
        except (ValueError, OSError):
            successful = False
        if successful:
            save(task, "sf_" + info.parent.name, info.parent, config)


def reusable_upstream(task, scene, config):
    """Only enable upstream skip-successful after binding every success marker's outputs."""
    markers = sorted(Path(scene).glob("*/stage_info.json"))
    successful = [p for p in markers if lib.read_json(p).get("success") is True]
    if not successful:
        return False
    for info in successful:
        if read(task, "sf_" + info.parent.name, info.parent, config) is None:
            raise ValueError("unbound upstream success marker")
    return True
