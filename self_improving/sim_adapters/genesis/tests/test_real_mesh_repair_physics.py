"""Opt-in real non-convex mesh controls for the text-repair physics entrance.

The existing fixtures for this entrance are generated primitive boxes, which never lose
their contact set. That is the regime in which the old acceptance limits looked reachable.
These controls run the assets production actually loads -- real meshes with convexify
disabled -- so the positive case has to clear the criteria in the regime that produced the
contact-detection dropout, and the attacks still have to fail.

    GENESIS_MESH_REPAIR_REAL=1 pytest test_real_mesh_repair_physics.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_physics as physics

pytestmark = pytest.mark.skipif(
    os.environ.get("GENESIS_MESH_REPAIR_REAL") != "1",
    reason="opt-in real non-convex mesh repair physics",
)

REPO = Path(__file__).resolve().parents[4]
INDEX = Path(os.environ.get("GENESIS_MESH_REPAIR_INDEX",
                            str(REPO / "assets/genesis/clip_readme_subset/index.json")))
# Table plus a cup, the trio the native entrance already calibrates against. Both are real
# non-convex sources, so `convexify=False` in repair_physics.simulate keeps them non-convex.
BOUND = [("table", "dex_table_d3996872", "table"), ("cup", "cup_2", "cup")]


def bindings():
    index, _ = clip.load_index(INDEX)
    document = dict(request="mesh contact control", objects=[], relations=[])
    bound = {}
    for name, asset_id, category in BOUND:
        asset = next(a for a in index["assets"] if a["asset_id"] == asset_id)
        root = Path(asset["source_root"])
        official.verify_files(root, asset["source_files"])
        document["objects"].append(
            dict(object_id=name, category=category, description=name)
        )
        bound[name] = dict(
            asset_id=asset_id,
            source_root=str(root),
            model_entrypoint=str(official.safe_file(root, asset["entrypoint"])),
            source_files=asset["source_files"],
        )
    return document, bound


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """Prepare once: asset preparation is the slow half and none of the attacks change it."""
    root = Path(os.environ.get("GENESIS_MESH_REPAIR_OUTPUT",
                               str(tmp_path_factory.mktemp("mesh_repair")))).resolve()
    out = root / "02_scene"
    out.mkdir(parents=True, exist_ok=True)
    document, bound = bindings()
    assets = prep.prepare(document, bound, out, ["table"], {}, lambda: None)
    # The support parent is assigned by the geometry graph, exactly as production does it.
    geo.graph(document, assets)
    top = assets["table"]["surface"]["z_m"]
    poses = {
        "table": dict(position=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0]),
        "cup": dict(position=[0, 0, top], orientation_wxyz=[1, 0, 0, 0]),
    }
    assert assets["table"]["fixed"] and not assets["cup"]["fixed"]
    assert assets["cup"]["support"] == "table"
    return root, assets, poses


def run(root, assets, poses, name):
    out = root / name
    out.mkdir(parents=True, exist_ok=False)
    data = physics.frozen_input(assets, poses, [], 0)
    clip.write_json(out / "physics_input.json", data)
    result = physics.simulate(data, out, lambda: None)
    clip.write_json(out / "validation_result.json", result)
    return data, result


def test_real_mesh_rest_clears_every_criterion(scene):
    """The positive control the primitive fixtures cannot provide.

    A pass here means the limits are reachable under real mesh contact, which is the claim
    the old instantaneous-speed limit could not support at any step size.
    """
    root, assets, poses = scene
    data, result = run(root, assets, poses, "03_physics_rest")
    assert result["passed"], result["failures"]
    cup = result["objects"]["cup"]
    # The criteria that used to be unreachable, with the margin they were met by.
    assert cup["overspeed_run_steps"] < data["settings"]["speed_run_steps"]
    assert cup["contact_dropout_fraction"] <= data["settings"]["contact_dropout_max"]
    assert cup["support_fraction"] >= data["settings"]["support_fraction"]
    assert cup["window_drift_rate_mps"] <= data["settings"]["drift_rate_mps"]
    # And the reading that made the old limit unsatisfiable, recorded rather than acted on.
    floor = abs(data["settings"]["gravity"][2]) * data["settings"]["dt"]
    assert cup["effective_velocity_limit_mps"] > floor
    rows = [json.loads(s) for s in (root / "03_physics_rest/trace.jsonl").read_text().splitlines()]
    assert len(rows) == data["settings"]["steps"] + 1


def test_real_mesh_body_off_its_support_is_rejected(scene):
    """Negative control in the same regime: nothing about mesh contact excuses this."""
    root, assets, poses = scene
    moved = {n: dict(p, position=list(p["position"])) for n, p in poses.items()}
    extent = max(np.ptp(np.asarray(assets["table"]["hull"]), axis=0)[:2])
    moved["cup"]["position"][0] += extent
    _, result = run(root, assets, moved, "03_physics_off_support")
    assert not result["passed"]
    assert result["failures"]["cup"]


def test_real_mesh_dropped_body_is_rejected(scene):
    """Dropped onto its support, it settles well before the window and is caught on impact.

    Penetration is deliberately checked over the whole run rather than the terminal window,
    so a body that arrives hard cannot be excused by having come to rest afterwards.
    """
    root, assets, poses = scene
    dropped = {n: dict(p, position=list(p["position"])) for n, p in poses.items()}
    dropped["cup"]["position"][2] += 0.5
    _, result = run(root, assets, dropped, "03_physics_dropped")
    assert not result["passed"]
    assert "penetration" in result["failures"]["cup"]


def test_real_mesh_body_still_in_flight_is_rejected(scene):
    """Released high enough to still be falling through the window: motion, not rest.

    This is the case the criteria have to catch on their own terms -- sustained speed, lost
    contact and absent support -- rather than on a contact event that never happens.
    """
    root, assets, poses = scene
    data = physics.frozen_input(assets, poses, [], 0)
    window_end_s = data["settings"]["steps"] * data["settings"]["dt"]
    flying = {n: dict(p, position=list(p["position"])) for n, p in poses.items()}
    # Still above the table at the last sample: half g t^2 with room to spare.
    flying["cup"]["position"][2] += 9.81 * window_end_s**2
    _, result = run(root, assets, flying, "03_physics_in_flight")
    assert not result["passed"]
    failures = result["failures"]["cup"]
    assert "stable_velocity" in failures
    assert "contact_continuity" in failures and "support_preserved" in failures
    cup = result["objects"]["cup"]
    assert cup["touching_samples"] == 0
    assert cup["overspeed_run_steps"] >= data["settings"]["speed_run_steps"]


def test_the_shipped_stiffness_is_the_one_every_geom_actually_used(scene):
    """Layer 0, where it actually bites: these assets author solref="0.001 1" per geom.

    The solver-level constraint_timeconst reaches only geoms that carry none of their own,
    so raising it would have left these meshes on the stiffness floor while the evidence
    recorded the softer value. The audit reads each geom back after the override.
    """
    root, assets, poses = scene
    data, _ = run(root, assets, poses, "03_physics_stiffness")
    report = json.loads((root / "03_physics_stiffness/asset_physics_report.json").read_text())
    assert report["rigid_options"]["constraint_timeconst"] == pytest.approx(
        data["settings"]["constraint_timeconst"]
    )
    audit = report["contact_parameter_audit"]
    assert audit, "every geom, ground included, must be audited"
    floor = 2 * data["settings"]["dt"]
    authored = 0
    for geom in audit:
        assert geom["sol_params_after"][:2] == pytest.approx(data["settings"]["contact_solref"])
        assert geom["sol_params_after"][0] >= floor
        assert geom["friction_after"] == geom["friction_before"]
        authored += geom["sol_params_before"][0] < floor
    # And the reason the override is not optional: the sources really do ship a value the
    # solver would have had to clamp.
    assert authored, "expected at least one geom authoring a sub-floor time constant"


def test_geometry_declarations_are_measurable_from_the_mesh(scene):
    """Layer 2: every declared number must be re-derivable from the geometry it describes."""
    _, assets, _ = scene
    for name, asset in assets.items():
        hull = np.asarray(asset["hull"])
        assert asset["diagonal_m"] == pytest.approx(float(np.linalg.norm(np.ptp(hull, axis=0))))
        assert asset["radius_m"] == pytest.approx(asset["diagonal_m"] / 2)
        assert np.allclose(asset["bbox_size_m"], np.ptp(hull, axis=0))
        for part in asset["collision_hulls"]:
            assert geo.bounds(np.asarray(part)) is not None
