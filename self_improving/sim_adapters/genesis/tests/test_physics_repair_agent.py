"""The guards around the search, exercised without Genesis or the network.

The loop's value is entirely in what it refuses. A search that retunes until something
passes would launder a scene defect into an acceptance, so these tests are written from the
refusals outwards: the model's reply is stubbed and the physics runs are stubbed, leaving
only the decision logic under test.
"""

import json

import pytest

from self_improving.sim_adapters.genesis import physics_criteria as criteria
from self_improving.sim_adapters.genesis import physics_repair_agent as agent
from self_improving.sim_adapters.genesis import repair_numerics as numerics


def reply(**fields):
    return json.dumps(fields)


def check(name, passed, observed=0.0, limit=1.0):
    return dict(name=name, observed=observed, limit=limit, passed=passed,
                category=criteria.category(name))


def outcome(*, exit_code, failed=(), budget=0.2):
    """A stubbed physics result shaped like validate_single_asset's report."""
    names = ["drift_rate_mps", "penetration_m", "contact_dropout_max", "support_fraction"]
    checks = [check(n, n not in failed) for n in names]
    return dict(
        exit_code=exit_code,
        physics_status="passed" if exit_code == 0 else "failed",
        checks=checks,
        numerics_profile="stub",
        numerics_origin="default",
        penetration_budget_used=budget,
        numerics_saturated=budget > criteria.SATURATED_BUDGET,
        failure_categories=criteria.classify(failed),
        numerics_tunable=criteria.tunable(failed),
    )


@pytest.fixture
def runs(monkeypatch):
    """Queue physics outcomes; record the numerics each attempt was given."""
    queued, seen = [], []

    def runner(output_dir, numerics):
        seen.append(numerics)
        return queued.pop(0)

    return queued, seen, runner


# --------------------------------------------------------------------------------------
# The gate: a scene defect must end the search, not start a hunt for a kinder solve.
# --------------------------------------------------------------------------------------


def test_scene_truth_failure_stops_without_ever_asking_the_model(tmp_path, runs):
    """A body that actually moved is a scene defect; retuning would only hide it."""
    queued, seen, runner = runs
    queued.append(outcome(exit_code=2, failed=["drift_rate_mps"]))

    def transport(system, user):  # pragma: no cover - must never be reached
        raise AssertionError("model consulted about a scene-truth failure")

    report = agent.repair(tmp_path / "r", runner, budget=4, transport=transport)
    assert report["status"] == "scene_defect"
    assert report["failure_categories"]["scene_truth"] == ["drift_rate_mps"]
    assert len(report["attempts"]) == 1
    assert seen == [None], "no retune may be attempted after a scene-truth failure"


def test_mixed_failure_is_still_a_scene_defect(tmp_path, runs):
    """Penetration failing alongside real motion does not license tuning the motion away."""
    queued, _, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m", "support_fraction"]))
    report = agent.repair(tmp_path / "r", runner, budget=4,
                          transport=lambda s, u: reply(action="stop"))
    assert report["status"] == "scene_defect"
    assert report["failure_categories"]["scene_truth"] == ["support_fraction"]


# --------------------------------------------------------------------------------------
# The search, when it is legitimate.
# --------------------------------------------------------------------------------------


def test_numerics_failure_retunes_and_accepts(tmp_path, runs):
    queued, seen, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"], budget=0.7))
    queued.append(outcome(exit_code=0))
    report = agent.repair(tmp_path / "r", runner, budget=4,
        transport=lambda s, u: reply(action="adjust_numerics", dt=0.004,
                                     constraint_timeconst=0.02, rationale="stiffer"),
    )
    assert report["status"] == "passed"
    assert seen[0] is None and seen[1]["constraint_timeconst"] == 0.02
    # The second run is told which configuration it is, so the evidence is reproducible.
    assert seen[1]["numerics_profile"] == "dt4ms_tau20ms_authored_v1"


def test_model_sees_failures_history_and_never_the_thresholds(tmp_path, runs):
    queued, _, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"], budget=0.7))
    queued.append(outcome(exit_code=0))
    captured = {}

    def transport(system, user):
        captured["packet"] = json.loads(user)
        return reply(action="adjust_numerics", dt=0.004, constraint_timeconst=0.02)

    agent.repair(tmp_path / "r", runner, budget=4, transport=transport)
    packet = captured["packet"]
    assert [f["name"] for f in packet["failed"]] == ["penetration_m"]
    assert packet["failed"][0]["category"] == criteria.NUMERICS_COUPLED
    assert packet["numerics_saturated"] is True
    assert packet["stability_floor_timeconst"] == pytest.approx(0.008)
    # Only the criteria that failed and their observed values travel; the packet carries no
    # way to name a threshold as an adjustable quantity.
    assert "penetration_m" not in packet.get("current", {})
    assert set(packet["current"]) == {"dt", "constraint_timeconst"}


# --------------------------------------------------------------------------------------
# Refusals. Each one ends the run rather than being repaired into something usable.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,fragment",
    [
        (reply(action="adjust_numerics", dt=0.004, constraint_timeconst=0.02,
               penetration_m=0.05), "acceptance thresholds"),
        (reply(action="adjust_numerics", dt=0.004, constraint_timeconst=0.002),
         "stability floor"),
        (reply(action="adjust_numerics", dt=0.02, constraint_timeconst=0.05),
         "characterised range"),
        (reply(action="relax_limits"), "unknown action"),
        ("not json at all", "not valid JSON"),
    ],
)
def test_invalid_proposals_are_refused(tmp_path, runs, body, fragment):
    queued, seen, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"]))
    report = agent.repair(tmp_path / "r", runner, budget=4,
                          transport=lambda s, u: body)
    assert report["status"] == "rejected"
    assert fragment in report["reason"]
    assert seen == [None], "a refused proposal must never reach physics"


def test_repeated_configuration_is_refused(tmp_path, runs):
    """Without this the loop can spend its whole budget re-running one configuration."""
    queued, _, runner = runs
    for _ in range(3):
        queued.append(outcome(exit_code=2, failed=["penetration_m"]))
    report = agent.repair(tmp_path / "r", runner, budget=4,
        transport=lambda s, u: reply(action="adjust_numerics", dt=0.004,
                                     constraint_timeconst=0.02),
    )
    assert report["status"] == "rejected"
    assert "already tried" in report["reason"]


def test_model_may_stop(tmp_path, runs):
    queued, _, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"]))
    report = agent.repair(tmp_path / "r", runner, budget=4,
                          transport=lambda s, u: reply(action="stop", rationale="no room"))
    assert report["status"] == "stopped"
    assert "no room" in report["reason"]


def test_budget_exhaustion_is_not_a_pass(tmp_path, runs):
    queued, _, runner = runs
    for _ in range(3):
        queued.append(outcome(exit_code=2, failed=["penetration_m"]))
    taus = iter([0.02, 0.015, 0.012])
    report = agent.repair(tmp_path / "r", runner, budget=3,
        transport=lambda s, u: reply(action="adjust_numerics", dt=0.004,
                                     constraint_timeconst=next(taus)),
    )
    assert report["status"] == "exhausted"
    assert len(report["attempts"]) == 3
    assert "accepted" not in report


def test_model_transport_failure_is_reported_not_swallowed(tmp_path, runs):
    queued, _, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"]))

    def transport(system, user):
        raise OSError("connection reset")

    report = agent.repair(tmp_path / "r", runner, budget=4, transport=transport)
    assert report["status"] == "model_unavailable"
    assert "connection reset" in report["error"]


def test_attempts_are_recorded_including_refusals(tmp_path, runs):
    queued, _, runner = runs
    queued.append(outcome(exit_code=2, failed=["penetration_m"]))
    agent.repair(tmp_path / "r", runner, budget=4,
                 transport=lambda s, u: reply(action="relax_limits"))
    saved = json.loads((tmp_path / "r/attempts.json").read_text())
    assert saved["status"] == "rejected"
    # The reply is kept even though it was refused: it is the record of how the model behaved.
    assert "relax_limits" in saved["attempts"][0]["model_reply"]
    assert "unknown action" in saved["attempts"][0]["proposal_note"]


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def test_unregistered_proposal_is_marked_and_registered_one_is_not():
    adhoc = numerics.synthesize(0.004, 0.017)
    known = numerics.synthesize(0.004, 0.02)
    assert adhoc["numerics_origin"] == "agent_proposed"
    assert known["numerics_origin"] == "registered"
    assert known["numerics_profile"] == "dt4ms_tau20ms_authored_v1"
    assert agent.override(adhoc)["numerics_digest"] == adhoc["numerics_digest"]


def test_agent_tuned_evidence_is_refused_by_name(tmp_path, monkeypatch):
    """The provenance gate must fire on its own, not ride on a neighbouring check.

    Reached directly because no run can demonstrate it end to end: an agent-tuned pass also
    changes the settings and step count, so the generic checks would mask which rule acted.
    """
    from self_improving.sim_adapters.genesis import validate_single_asset as single

    directory = tmp_path / "evidence"
    directory.mkdir()
    keys = ("model_entrypoint", "source_files", "source_root")
    frozen = dict(
        binding={k: k for k in keys}, settings={},
        clearance_m=single.DROP_CLEARANCE_M,
        numerics_origin="agent_proposed", numerics_profile="adhoc_deadbeef",
    )
    (directory / "physics_input.json").write_text(json.dumps(frozen))
    report = dict(exit_code=0, physics_status="passed", steps_executed=1000,
                  simulation_executed=True, artifacts=[], physics_input_sha256="x")
    (directory / "physics_result.json").write_text(json.dumps(report))
    monkeypatch.setattr(single.official, "verify_files", lambda *a, **k: None)
    monkeypatch.setattr(single.official, "safe_file",
                        lambda d, p: directory / "physics_result.json")
    monkeypatch.setattr(single.standard.library, "sha256", lambda p: "x")
    binding = dict(physics_status="passed", physics_evidence=dict(path="physics_result.json"),
                   **{k: k for k in keys})
    with pytest.raises(ValueError, match="agent-proposed"):
        single.verify_evidence(directory, binding)


# --------------------------------------------------------------------------------------
# Generalisation: the scene entrance reports per body, and aggregating must not soften.
# --------------------------------------------------------------------------------------


def body(*failed, budget=0.2):
    names = ["drift_rate_mps", "penetration_m", "contact_dropout_max", "support_fraction"]
    checks = [check(n, n not in failed) for n in names]
    return dict(checks=checks, passed=not failed, penetration_budget_used=budget,
                numerics_saturated=budget > criteria.SATURATED_BUDGET)


def scene_result(exit_code, bodies):
    return dict(exit_code=exit_code, physics_status="passed" if not exit_code else "failed",
                objects=bodies)


def test_scene_outcome_is_normalised_across_bodies():
    view = agent.normalise(scene_result(2, dict(
        cup=body("penetration_m", budget=0.8), table=body())))
    assert view["failed"] == ["penetration_m"]
    assert view["tunable"] is True
    # The worst body sets the budget; averaging would hide the one that is saturated.
    assert view["budget"] == 0.8 and view["saturated"] is True
    assert {c["object_id"] for c in view["checks"]} == {"cup", "table"}


def test_one_moving_body_makes_the_whole_scene_untunable():
    """Retuning to fix another body's penetration would carry this body's motion with it."""
    view = agent.normalise(scene_result(2, dict(
        cup=body("penetration_m"), pen=body("drift_rate_mps"))))
    assert sorted(view["failed"]) == ["drift_rate_mps", "penetration_m"]
    assert view["tunable"] is False


def test_scene_runner_drives_the_imported_entrance(tmp_path, monkeypatch):
    from self_improving.sim_adapters.genesis import validate_imported_scene as scene

    seen = {}

    def fake_run(package, output_dir, *, profile="baseline", numerics=None):
        seen.update(package=package, profile=profile, numerics=numerics)
        return scene_result(0, dict(iter_0=body()))

    monkeypatch.setattr(scene, "run", fake_run)
    runner = agent.scene_runner("pkg", profile="half_dt")
    report = agent.repair(tmp_path / "r", runner, budget=2,
                          transport=lambda s, u: reply(action="stop"))
    assert report["status"] == "passed"
    assert seen["package"] == "pkg" and seen["profile"] == "half_dt"
