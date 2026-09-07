"""Opt-in Genesis replay using measured, generated rigid fixture assets."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import repair_video as video

pytestmark = pytest.mark.skipif(
    os.environ.get("GENESIS_TEXT_REPAIR_REAL") != "1", reason="opt-in real text repair physics"
)


def fixture_data(root):
    assets, poses = {}, {}
    for name, size, parent, z, fixed in [
        ("table", [1.2, 0.8, 0.7], "ground", 0, True),
        ("carrier", [0.3, 0.3, 0.1], "table", 0.7, False),
        ("child", [0.06, 0.06, 0.1], "carrier", 0.8, False),
    ]:
        out = root / "02_scene" / name
        out.mkdir(parents=True)
        mesh = trimesh.creation.box(size)
        mesh.apply_translation([0, 0, size[2] / 2])
        visual = out / "source.glb"
        gltf_mesh = mesh.copy()
        gltf_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        gltf_mesh.export(visual)
        mass = float(mesh.volume * 600)
        com = mesh.center_mass
        inertia = mesh.moment_inertia * 600
        model = prep.write_collision_model([mesh], out, mass, com, inertia, fixed=fixed)
        np.savez_compressed(out / "visual_geometry.npz", vertices=mesh.vertices, faces=mesh.faces)
        d = float(np.linalg.norm(size))
        assets[name] = dict(
            category="table" if fixed else "measured_fixture",
            scale=1,
            bbox_size_m=size,
            diagonal_m=d,
            radius_m=d / 2,
            hull=mesh.vertices.tolist(),
            collision_hulls=[mesh.vertices.tolist()],
            surface=builder.support_surface(mesh.vertices, mesh.faces),
            native_collision=False,
            physics_file=str(model),
            model_entrypoint=str(visual),
            geometry_file=str(out / "visual_geometry.npz"),
            anchor_m=[0, 0, 0],
            mass_kg=mass,
            com_local_m=com.tolist(),
            inertia_local_kg_m2=inertia.tolist(),
            fixed=fixed,
            support=parent,
            natural_up=[0, 0, 1],
            tip_limit_deg=15,
            margin_m=max(0.01, 0.02 * d),
            buffer_m=max(0.002, 0.005 * d),
        )
        poses[name] = dict(position=[0, 0, z], orientation_wxyz=[1, 0, 0, 0])
    return physics.frozen_input(assets, poses, [], 0)


@pytest.mark.parametrize(
    "variant",
    [
        "stack",
        "wrong_support",
        "falling",
        "initial_penetration",
        "fixed_suspended",
        "no_dofs",
        "collision_off",
        "wrong_mass",
        "wrong_inertia",
    ],
)
def test_real_fixture(tmp_path, variant):
    base = Path(os.environ.get("GENESIS_TEXT_REPAIR_OUTPUT", str(tmp_path))).resolve()
    root = base / variant
    root.mkdir(parents=True, exist_ok=False)
    data = fixture_data(root)
    if variant == "falling":
        data["poses"]["child"]["position"][2] = 50
    if variant == "wrong_support":
        data["poses"]["child"]["position"][0] = 0.45
    if variant == "initial_penetration":
        data["poses"]["child"]["position"][2] -= 0.03
    if variant == "fixed_suspended":
        data["poses"]["table"]["position"][2] = 0.1
    if variant == "no_dofs":
        p = Path(data["assets"]["carrier"]["physics_file"])
        p.write_text(p.read_text().replace("<freejoint />", ""))
    if variant == "collision_off":
        p = Path(data["assets"]["carrier"]["physics_file"])
        p.write_text(p.read_text().replace("<geom ", '<geom contype="0" conaffinity="0" '))
    if variant == "wrong_mass":
        data["assets"]["carrier"]["mass_kg"] *= 2
    if variant == "wrong_inertia":
        data["assets"]["carrier"]["inertia_local_kg_m2"][0][0] *= 2
    out = root / "03_physics"
    out.mkdir()
    clip.write_json(out / "physics_input.json", data)
    if variant in {"fixed_suspended", "no_dofs", "collision_off", "wrong_mass", "wrong_inertia"}:
        with pytest.raises(ValueError):
            physics.simulate(data, out, lambda: None)
        return
    result = physics.simulate(data, out, lambda: None)
    clip.write_json(out / "validation_result.json", result)
    assert result["passed"] is (variant == "stack")
    rows = [json.loads(s) for s in (out / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == (1 if variant == "initial_penetration" else 1501)
    if variant == "falling":
        assert "stable_velocity" in result["failures"]["child"]
    if variant == "stack":
        assert result["objects"]["carrier"]["support_fraction"] >= 0.95
        assert result["objects"]["child"]["support_fraction"] >= 0.95
        video.render(data, out / "trace.jsonl", out / "video", lambda: None, physics_passed=True)
        video.render(
            data,
            out / "trace.jsonl",
            root / "04_final_render",
            lambda: None,
            final=True,
            physics_passed=True,
        )
    elif variant == "wrong_support":
        assert "support_preserved" in result["failures"]["child"]
        video.render(data, out / "trace.jsonl", out / "video", lambda: None, physics_passed=False)
        assert not (root / "04_final_render").exists()


def test_real_subtree_repair(tmp_path):
    from self_improving.sim_adapters.genesis import construct_asset_scene as entry
    from self_improving.sim_adapters.genesis import repair_geometry as geometry

    root = (
        Path(os.environ.get("GENESIS_TEXT_REPAIR_OUTPUT", str(tmp_path))).resolve()
        / "subtree_repair"
    )
    root.mkdir(parents=True, exist_ok=False)
    data = fixture_data(root)
    for n in ("carrier", "child"):
        data["poses"][n]["position"][0] = 0.5
    out = root / "03_physics"
    out.mkdir()
    result, attempts, poses, passing = entry.repair_loop(
        data["assets"],
        data["poses"],
        dict(order=geometry.topology(data["assets"]), relations=[]),
        [],
        0,
        out,
        lambda: None,
        simulator=physics.simulate,
        recorder=video.render,
    )
    assert result["passed"] and passing and len(attempts) == 2
    assert not attempts[0]["passed"] and attempts[1]["passed"]
    assert poses["table"] == data["poses"]["table"]
    relative = geometry.inverse([poses["child"]["position"]], poses["carrier"])[0]
    assert np.allclose(relative, [0, 0, 0.1], atol=1e-6)
    assert result["objects"]["carrier"]["support_fraction"] >= 0.95
