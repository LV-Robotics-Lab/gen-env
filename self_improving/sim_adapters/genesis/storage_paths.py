"""Resolve known storage migrations without rewriting hash-bound evidence."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO_ROOT / ".cache/genesis"

# Only these repository-owned resource roots moved. Never relocate arbitrary paths.
RESOURCE_MOVES = {
    "genesis_assets": "assets/genesis",
    "asset_selection": "data/genesis_history/asset_selection",
    "scene_planning_acceptance_20260906":
        "data/genesis_history/scene_planning_acceptance_20260906",
}



def evidence_path(value):
    """Normalize only our verified legacy root alias, never arbitrary dependency symlinks."""
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    old, current = REPO_ROOT / "ouput", REPO_ROOT / "output"
    if (
        path.is_absolute()
        and ".." not in path.parts
        and path.is_relative_to(old)
        and old.is_symlink()
        and current.is_dir()
        and not current.is_symlink()
        and old.resolve() == current
    ):
        return str(current / path.relative_to(old))
    if path.is_absolute() and ".." not in path.parts:
        for old_name in ("ouput", "output"):
            legacy = REPO_ROOT / old_name
            if not path.is_relative_to(legacy):
                continue
            relative = path.relative_to(legacy)
            if not relative.parts or relative.parts[0] not in RESOURCE_MOVES:
                continue
            original = legacy / relative.parts[0]
            current = REPO_ROOT / RESOURCE_MOVES[relative.parts[0]]
            # Existing old roots (including dangling links) are not silently redirected.
            if (not original.exists() and not original.is_symlink() and current.is_dir()
                    and not any(p.is_symlink() for p in (current, *current.parents))):
                return str(current.joinpath(*relative.parts[1:]))
    return str(value)


def local_path(value):
    """Map a read/load path; leave payload strings and dependency checks unchanged."""
    return Path(evidence_path(Path(value).absolute()))


def same_evidence_path(left, right):
    return evidence_path(left) == evidence_path(right)
