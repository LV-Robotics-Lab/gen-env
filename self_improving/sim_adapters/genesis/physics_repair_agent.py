"""Bounded search for a numerical configuration that lets a sound scene be validated.

Physics acceptance can fail for two unrelated reasons, and the difference decides whether
searching is legitimate at all. A body that slid, tipped, or left its declared support is a
scene defect: retuning the solver would only hide it, so this loop refuses to search and
reports the defect. A body that sat still while the contact solve let it sink too deep is a
numerical result, and a different configuration is a fair next step. `physics_criteria`
draws that line from measured sensitivity, and `tunable()` is the gate here.

The model proposes; it never decides. Its reply is a candidate that must survive
deterministic validation -- key allowlist, registry-characterised bounds, the Genesis
stability floor, `satisfiable()`, and a no-repeat rule -- before any physics runs on it. A
rejected proposal is recorded with its reason and fed back, never quietly repaired into
something usable. Acceptance thresholds appear nowhere in the schema the model answers in,
so no reply it can produce moves the bar it is being measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import physics_criteria as criteria
from self_improving.sim_adapters.genesis import repair_numerics as numerics
from self_improving.sim_adapters.genesis import validate_single_asset as single
from self_improving.sim_adapters.genesis.physics_math import stiffness_floor

SCHEMA = "genenv.physics_repair_agent.v1"
MAX_RESPONSE_BYTES = 64 * 1024
# Answers the model may give. "stop" is deliberately available: a search that cannot say
# "this will not work" would keep proposing until its budget ran out.
ACTIONS = ("adjust_numerics", "stop")

SYSTEM_PROMPT = """You tune the numerical configuration of a Genesis rigid-body simulation \
so that a physically sound scene can be validated. You never change acceptance thresholds \
and you are never asked to: they are fixed and are not in your output schema.

Two facts about this simulator decide almost every case.

1. Contact force is produced by letting bodies overlap. A resting body must sink until the \
contact spring pushes back with its weight, so penetration depth is set by \
constraint_timeconst, not by whether the scene is correct. Measured on one unchanged \
resting scene at dt=0.004: timeconst 0.008 -> 0.034 mm of resting penetration, 0.020 -> \
0.175 mm, 0.050 -> 0.697 mm. Depth grows steeply with timeconst (roughly its 1.65 power). \
A smaller timeconst is a stiffer contact and a shallower overlap.

2. Genesis silently raises any constraint_timeconst below 2*dt to that floor, and the \
solve is least stable there. Never propose a timeconst below 2*dt; if you need one, \
propose a smaller dt as well.

Contact dropout moves the other way: too stiff a solve can make a resting body lose its \
contact set for single steps. If contact_dropout_max failed, go softer (larger timeconst). \
If penetration failed, go stiffer (smaller timeconst), or reduce dt so a smaller timeconst \
becomes legal.

You are given the failed criteria, how much of the penetration budget the solve spends at \
rest, the configuration used, and every configuration already tried with its result. Do not \
repeat a configuration. If the remaining room cannot plausibly work, answer with \
action "stop" and say why.

Reply with JSON only:
{"diagnosis": "<one sentence on what the numbers indicate>",
 "action": "adjust_numerics" | "stop",
 "dt": <seconds, omit when action is stop>,
 "constraint_timeconst": <seconds, omit when action is stop>,
 "rationale": "<why this configuration should behave differently>"}"""


def normalise(result):
    """One shape from either entrance: flat checks, or per-object checks for a scene.

    The scene entrance reports each body separately. Aggregating is not a formality -- a
    scene is tunable only when *every* failing body failed on a numerics-coupled criterion.
    One body that genuinely moved makes the whole scene a defect, however many others merely
    sank too deep, because retuning would hide that body's motion along with the rest.
    """
    if "checks" in result:
        return dict(
            checks=result["checks"],
            failed=[c["name"] for c in result["checks"] if not c["passed"]],
            tunable=bool(result.get("numerics_tunable")),
            budget=result.get("penetration_budget_used"),
            saturated=result.get("numerics_saturated"),
        )
    bodies = result.get("objects") or {}
    checks = [dict(c, object_id=name) for name, v in bodies.items() for c in v["checks"]]
    failed = [c["name"] for c in checks if not c["passed"]]
    budgets = [v.get("penetration_budget_used") for v in bodies.values()
               if v.get("penetration_budget_used") is not None]
    return dict(
        checks=checks,
        failed=failed,
        # criteria.tunable() over the union: any scene-truth failure anywhere disqualifies.
        tunable=criteria.tunable(failed),
        budget=max(budgets, default=None),
        saturated=any(v.get("numerics_saturated") for v in bodies.values()),
    )


def single_asset_runner(*, package=None, binding=None, at_rest=False):
    """Drive the single-body drop/at-rest entrance."""
    def run(output_dir, numerics):
        return single.run(output_dir, package=package, binding=binding,
                          at_rest=at_rest, numerics=numerics)
    return run


def scene_runner(scene_package, *, profile="baseline"):
    """Drive the imported-scene entrance, including scenes supported by the ground plane."""
    from self_improving.sim_adapters.genesis import validate_imported_scene as scene

    def run(output_dir, numerics):
        return scene.run(scene_package, output_dir, profile=profile, numerics=numerics)
    return run


def evidence(result, cfg, history):
    """What the model is shown: measurements and history, never the thresholds to move."""
    view = normalise(result)
    return dict(
        failed=[
            dict(name=c["name"], observed=c["observed"], limit=c["limit"],
                 category=c["category"], **({"object_id": c["object_id"]}
                                            if "object_id" in c else {}))
            for c in view["checks"]
            if not c["passed"]
        ],
        penetration_budget_used=view["budget"],
        numerics_saturated=view["saturated"],
        current=dict(dt=cfg["dt"], constraint_timeconst=cfg["constraint_timeconst"]),
        stability_floor_timeconst=stiffness_floor(cfg),
        dt_bounds=list(numerics.DT_BOUNDS),
        timeconst_max=numerics.TIMECONST_MAX,
        attempts=history,
    )


def ask(config, packet, *, transport=None):
    """One JSON-mode completion. No retry: a malformed reply is evidence, not a hiccup."""
    if transport is not None:
        return transport(SYSTEM_PROMPT, json.dumps(packet, ensure_ascii=False))
    payload = dict(
        model=config.model,
        messages=[
            dict(role="system", content=SYSTEM_PROMPT),
            dict(role="user", content=json.dumps(packet, ensure_ascii=False)),
        ],
        response_format={"type": "json_object"},
        max_tokens=600,
    )
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    endpoint = config.endpoint
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("model response exceeds the size limit")
    envelope = json.loads(body)
    return envelope["choices"][0]["message"]["content"]


def proposal(reply, *, tried):
    """Validate a reply into a usable configuration, or say exactly why it is refused.

    Returns (config or None, note). Every refusal is returned rather than raised so the
    reason can be recorded and shown to the model on the next turn.
    """
    try:
        data = json.loads(reply)
    except (TypeError, ValueError):
        return None, "reply is not valid JSON"
    if not isinstance(data, dict):
        return None, "reply is not a JSON object"
    action = data.get("action")
    if action not in ACTIONS:
        return None, f"unknown action {action!r}; expected one of {list(ACTIONS)}"
    if action == "stop":
        return None, "model stopped: " + str(data.get("rationale", ""))[:300]
    # An acceptance threshold named anywhere in the reply is refused outright rather than
    # ignored: silently dropping it would let the model believe the bar had moved.
    forbidden = sorted(set(data) & set(single.asset_physics.DEFAULTS) - set(numerics.SOLVER_KEYS))
    if forbidden:
        return None, f"reply may not set acceptance thresholds: {forbidden}"
    try:
        cfg = numerics.synthesize(data.get("dt"), data.get("constraint_timeconst"))
    except (TypeError, ValueError) as exc:
        return None, f"invalid numerics: {exc}"
    if cfg["constraint_timeconst"] < 2 * cfg["dt"]:
        return None, (
            f"constraint_timeconst {cfg['constraint_timeconst']} is below the Genesis "
            f"stability floor 2*dt = {2 * cfg['dt']} and would be silently clamped"
        )
    try:
        single.asset_physics.settings("baseline", numerics.as_override(cfg))
    except ValueError as exc:
        return None, f"configuration is unsatisfiable: {exc}"
    key = (cfg["dt"], cfg["constraint_timeconst"])
    if key in tried:
        return None, f"configuration dt={key[0]} timeconst={key[1]} was already tried"
    return cfg, "accepted"


def override(cfg):
    """Solver keys plus the provenance the run records."""
    return numerics.as_override(cfg) | {
        k: cfg[k] for k in ("numerics_profile", "numerics_origin", "numerics_digest")
    }


def repair(output_dir, runner, *, budget=4, vlm_config="configs/llm.yaml", profile=None,
           transport=None):
    """Validate, and while only numerics-coupled criteria fail, search for a better solve.

    `runner(output_dir, numerics) -> report` is any physics entrance; see
    single_asset_runner and scene_runner. The loop reads only the normalised view, so a new
    entrance joins by reporting categorised checks, not by changing the search.
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = None
    if transport is None:
        from scene_gen.llm_provider import load_llm_provider_config

        config = load_llm_provider_config(vlm_config, profile=profile)
    # Release mode, package identity and profile belong to the runner and are recorded in
    # each attempt's own frozen physics_input.json; the loop stays agnostic about them.
    report = dict(schema_version=SCHEMA, status="running", attempts=[],
                  model=None if config is None else config.model)
    tried, cfg, history = set(), None, []
    for index in range(budget):
        numeric = None if cfg is None else override(cfg)
        result = runner(out / f"attempt_{index:02d}", numeric)
        view = normalise(result)
        attempt = dict(
            index=index,
            numerics_profile=(cfg or {}).get("numerics_profile", "default"),
            numerics_origin=(cfg or {}).get("numerics_origin", "default"),
            dt=None if cfg is None else cfg["dt"],
            constraint_timeconst=None if cfg is None else cfg["constraint_timeconst"],
            physics_status=result.get("physics_status"),
            exit_code=result["exit_code"],
            failed=view["failed"],
            penetration_budget_used=view["budget"],
        )
        report["attempts"].append(attempt)
        if result["exit_code"] == 1:
            report.update(status="error", error=result.get("error"))
            break
        if result["exit_code"] == 0:
            report.update(status="passed", accepted=attempt)
            break
        if not view["tunable"]:
            # A scene-truth failure, or no failure a configuration change could reach.
            report.update(status="scene_defect",
                          failure_categories=criteria.classify(view["failed"]))
            break
        if index == budget - 1:
            report.update(status="exhausted")
            break
        history.append(dict(attempt, evidence_only=True))
        packet = evidence(result, single.asset_physics.settings(
            "baseline", None if cfg is None else numerics.as_override(cfg)), history)
        try:
            reply = ask(config, packet, transport=transport)
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            report.update(status="model_unavailable", error=f"{type(exc).__name__}: {exc}")
            break
        candidate, note = proposal(reply, tried=tried)
        attempt.update(model_reply=reply[:2000], proposal_note=note)
        history[-1].update(proposal_note=note)
        if candidate is None:
            report.update(status="stopped" if note.startswith("model stopped") else "rejected",
                          reason=note)
            break
        tried.add((candidate["dt"], candidate["constraint_timeconst"]))
        cfg = candidate
    (out / "attempts.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--package", type=Path)
    inputs.add_argument("--binding", type=Path)
    inputs.add_argument("--scene-package", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--at-rest", action="store_true")
    parser.add_argument("--scene-profile", choices=("baseline", "half_dt"), default="baseline")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--vlm-config", default="configs/llm.yaml")
    parser.add_argument("--profile", default=None)
    args = vars(parser.parse_args())
    out = args.pop("output_dir")
    if args.get("scene_package"):
        runner = scene_runner(args.pop("scene_package"), profile=args.pop("scene_profile"))
        args.pop("package"), args.pop("binding"), args.pop("at_rest")
    else:
        args.pop("scene_package"), args.pop("scene_profile")
        runner = single_asset_runner(package=args.pop("package"), binding=args.pop("binding"),
                                     at_rest=args.pop("at_rest"))
    report = repair(out, runner, **args)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
