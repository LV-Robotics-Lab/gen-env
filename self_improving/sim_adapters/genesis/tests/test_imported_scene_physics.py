"""Prevent contact-free, floating-stack and partial-trace false positives."""

import copy

import pytest

from self_improving.sim_adapters.genesis import asset_physics
from self_improving.sim_adapters.genesis import validate_imported_scene as physics


def fixture():
    state = dict(
        position=[0, 0, 0],
        orientation_wxyz=[1, 0, 0, 0],
        velocity=[0, 0, 0],
        angular_velocity=[0, 0, 0],
    )
    layout = dict(
        objects=[
            dict(
                object_id=n,
                category="box",
                fixed=False,
                translation_m=[0, 0, 0],
                orientation_wxyz=[1, 0, 0, 0],
                source_velocity_mps=[0, 0, 0],
                source_angular_velocity_radps=[0, 0, 0],
            )
            for n in ("lower", "upper")
        ]
    )
    rows = []
    for i in range(1001):
        cs = []
        for a, b, force in [("ground", "lower", 2.0), ("lower", "upper", 1.0)]:
            cs.append(
                dict(
                    a=a,
                    b=b,
                    geom_a=0,
                    geom_b=1,
                    link_a=0,
                    link_b=1,
                    position=[0, 0, 0],
                    normal=[0, 0, 1],
                    penetration=0,
                    force_a=None if i == 0 else [0, 0, -force],
                    force_b=None if i == 0 else [0, 0, force],
                )
            )
        rows.append(
            dict(
                step=i,
                time_s=i * 0.004,
                contact_phase="initial_detection" if i == 0 else "solved_step",
                objects={n: copy.deepcopy(state) for n in ("lower", "upper", "ground")},
                contacts=cs,
            )
        )
    return rows, layout, asset_physics.settings("baseline")


def test_grounded_stack_is_stable_but_not_semantic_acceptance():
    result = physics.evaluate(*fixture())
    assert result["stability_status"] == "passed"
    assert result["physics_status"] == "incomplete"
    assert result["exit_code"] == 3
    assert {v["target"] for v in result["observed_support_contacts"]} == {"lower", "ground"}


@pytest.mark.parametrize("floating", [False, True])
def test_still_screenshots_or_floating_contacts_cannot_establish_support(floating):
    rows, layout, cfg = fixture()
    for row in rows:
        row["contacts"] = row["contacts"][1:] if floating else []
    result = physics.evaluate(rows, layout, cfg)
    assert result["physics_status"] == "failed"
    for obj in result["objects"].values():
        check = next(c for c in obj["checks"] if c["name"] == "support_fraction")
        assert check["observed"] == 0 and not check["passed"]


def test_intermittent_ground_breaks_support_of_entire_stack():
    rows, layout, cfg = fixture()
    for row in rows[-125::2]:
        row["contacts"] = row["contacts"][1:]
    result = physics.evaluate(rows, layout, cfg)
    assert all(not obj["passed"] for obj in result["objects"].values())


@pytest.mark.parametrize("step", [0, 10])
def test_early_or_initial_penetration_is_never_hidden(step):
    rows, layout, cfg = fixture()
    rows[step]["contacts"][0]["penetration"] = 0.00101
    result = physics.evaluate(rows, layout, cfg)
    assert not result["objects"]["lower"]["passed"]
    assert result["objects"]["upper"]["passed"]


def test_fixed_nested_body_rejected():
    rows, layout, cfg = fixture()
    layout["objects"][1]["fixed"] = True
    with pytest.raises(ValueError, match="dynamic"):
        physics.evaluate(rows, layout, cfg)


@pytest.mark.parametrize(
    "change,match",
    [
        ("truncated", "incomplete"),
        ("skip", "sequential"),
        ("pose", "initial pose"),
        ("velocity", "initial velocity"),
        ("force", "inconsistent"),
    ],
)
def test_invalid_evidence_rejected(change, match):
    rows, layout, cfg = fixture()
    if change == "truncated":
        rows.pop()
    elif change == "skip":
        rows[2]["step"] = 3
    elif change == "pose":
        rows[0]["objects"]["lower"]["position"][0] = 0.01
    elif change == "velocity":
        rows[0]["objects"]["lower"]["velocity"][0] = 0.01
    else:
        rows[2]["contacts"][0]["force_a"][2] = 99
    with pytest.raises(ValueError, match=match):
        physics.evaluate(rows, layout, cfg)


def test_support_sdf_override_is_explicit_and_bounded():
    assert physics.support_sdf() == {}
    assert physics.support_sdf(.0015, 384) == dict(sdf_cell_size=.0015, sdf_max_res=384)
    for cell, res in [(None, 384), (.0015, None), (float('nan'), 128),
                      (.0001, 384), (.0015, 1000), (True, 128), (.0015, 128.5)]:
        with pytest.raises(ValueError, match='support SDF'):
            physics.support_sdf(cell, res)
