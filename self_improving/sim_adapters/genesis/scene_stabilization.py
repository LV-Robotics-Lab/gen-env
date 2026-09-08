"""Intervened pose settling. Candidate generation, never free-dynamics acceptance."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_physics as evidence
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
from self_improving.sim_adapters.genesis import scene_physics_graph as graph
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis import validate_imported_scene as free
from self_improving.sim_adapters.genesis.physics_math import angle
from self_improving.sim_adapters.genesis.validate_asset_scene import contacts

VERSION = 'genenv.scene_stabilization.v1'
SETTINGS = dict(dt=.004, max_steps=2000, consecutive_steps=50, position_delta_m=.00001,
                orientation_delta_rad=.001, max_translation_m=.02, max_rotation_deg=5.)


def discriminating(cfg):
    """Reject a convergence delta that free fall on its own would satisfy.

    Velocity is zeroed after every sample, so an unsupported body still creeps
    0.5*g*dt^2 per step -- 0.0785 mm at the baseline step. A delta at or above that
    cannot tell a body resting on its support from one falling through empty air.
    """
    creep = .5*abs(cfg['gravity'][2])*SETTINGS['dt']**2
    if SETTINGS['position_delta_m'] >= creep:
        raise ValueError(
            f"stabilization delta {SETTINGS['position_delta_m']} m cannot distinguish rest "
            f'from free fall: zeroed-velocity creep is {creep:.6f} m per step')
    return creep


def assess(rows, layout, geometry):
    """Replay pre-intervention samples; never mistake zeroed velocity for convergence."""
    cfg = evidence.settings('baseline')
    discriminating(cfg)
    dynamic = [o['object_id'] for o in layout['objects'] if not o['fixed']]
    count = 0
    failure = None
    for i, row in enumerate(rows):
        free.validate_scene_row(row, i, geometry, SETTINGS['dt'])
        if row.get('zeroed_after_sample') != (dynamic if i else []):
            raise ValueError('missing or invalid intervention evidence')
        for n in dynamic:
            state, initial = row['objects'][n], rows[0]['objects'][n]
            if (np.linalg.norm(np.array(state['position'])-initial['position'])
                    > SETTINGS['max_translation_m']
                    or angle(state['orientation_wxyz'], initial['orientation_wxyz'])
                    > SETTINGS['max_rotation_deg']):
                failure = 'pose_budget_exceeded'
        if any(c['penetration'] > cfg['penetration_m'] for c in row['contacts']):
            failure = 'collision_penetration'
        if failure:
            break
        if i:
            # Standing still is only evidence of support when something is holding the
            # body up. Without this an airborne body creeps below any small delta and
            # reports converged for as long as it is allowed to fall.
            supported = {n for n in dynamic
                         if any(n in (c['a'], c['b']) for c in row['contacts'])}
            small = all(
                n in supported
                and np.linalg.norm(np.array(row['objects'][n]['position'])
                               - rows[i-1]['objects'][n]['position'])
                <= SETTINGS['position_delta_m']
                and np.radians(angle(row['objects'][n]['orientation_wxyz'],
                                     rows[i-1]['objects'][n]['orientation_wxyz']))
                <= SETTINGS['orientation_delta_rad'] for n in dynamic)
            count = count+1 if small else 0
    ready = count >= SETTINGS['consecutive_steps'] and failure is None
    if ready:
        checks = graph.geometric_checks(rows[-1], layout, geometry, cfg)
        if not all(c['passed'] for c in checks):
            failure, ready = 'candidate_graph_constraints', False
    return dict(status='candidate_ready' if ready else 'failed' if failure else 'running',
                reason=failure, consecutive_steps=count, physics_status='not_run')


def run(root, out, sdf):
    import genesis as gs

    root, out = Path(root), Path(out)
    layout = imported.verify(root)
    graph.topology(layout)
    geometry = graph.geometry(root, layout)
    cfg = evidence.settings('baseline')
    out.mkdir(parents=True, exist_ok=False)
    frozen = dict(version=VERSION, settings=SETTINGS, layout=layout, geometry=geometry,
                  scene_manifest_sha256=standard.library.sha256(root/'manifest.json'), sdf=sdf)
    official.write_json(out/'input.json', frozen)
    rows = []
    initialized = False
    try:
        gs.init(backend=gs.cpu, precision='32', seed=cfg['seed'], logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(show_viewer=False,
            sim_options=gs.options.SimOptions(dt=cfg['dt'], substeps=1,
                                             gravity=tuple(cfg['gravity'])),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton, iterations=cfg['iterations'],
                ls_iterations=cfg['ls_iterations'], tolerance=cfg['tolerance'],
                constraint_timeconst=cfg['constraint_timeconst'], use_hibernation=False))
        entities, owners, links, loaded = {}, {}, {}, {}
        for obj in layout['objects']:
            n = obj['object_id']
            _, entry, physics = standard.verify_package(root/obj['standard_package'])
            morph = standard.morph(gs, entry, fixed=obj['fixed'])
            morph.pos, morph.quat = tuple(obj['translation_m']), tuple(obj['orientation_wxyz'])
            entities[n] = scene.add_entity(morph, material=gs.materials.Rigid(
                friction=physics['friction'], **(sdf if obj['fixed'] else {})))
        scene.build()
        for obj in layout['objects']:
            n = obj['object_id']
            _, entry, _ = standard.verify_package(root/obj['standard_package'])
            entity = entities[n]
            loaded[n] = standard.audit(entity, standard.inspect(entry), collision=not obj['fixed'])
            if bool(entity.base_link.is_fixed) != obj['fixed']:
                raise ValueError('loaded fixed/dynamic property mismatch')
            for link in entity.links:
                for geom in link.geoms:
                    owners[geom.idx], links[geom.idx] = n, link.idx
        official.write_json(out/'loaded_scene.json', loaded)
        dynamic = [o['object_id'] for o in layout['objects'] if not o['fixed']]
        scene.rigid_solver.detect_collision()
        verdict = None
        with (out/'trace.jsonl').open('x') as stream:
            for step in range(SETTINGS['max_steps']+1):
                if step:
                    scene.step()
                row = dict(step=step, time_s=step*SETTINGS['dt'],
                    contact_phase='solved_step' if step else 'initial_detection',
                    objects={n: dict(position=standard.array(e.get_pos()).reshape(3).tolist(),
                                    orientation_wxyz=standard.array(e.get_quat()).reshape(4).tolist(),
                                    velocity=standard.array(e.get_vel()).reshape(3).tolist(),
                                    angular_velocity=standard.array(e.get_ang()).reshape(3).tolist())
                             for n, e in entities.items()},
                    contacts=contacts(scene.rigid_solver.collider.get_contacts(to_torch=False),
                                      owners, links, initial=step == 0),
                    zeroed_after_sample=dynamic if step else [])
                if step:
                    for n in dynamic:
                        entities[n].zero_all_dofs_velocity()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                verdict = assess(rows, layout, geometry)
                if verdict['status'] != 'running':
                    break
        if verdict['status'] == 'running':
            verdict.update(status='failed', reason='step_budget_exhausted')
        if standard.library.sha256(root/'manifest.json') != frozen['scene_manifest_sha256']:
            raise ValueError('stabilization source changed')
        imported.verify(root)
        verdict.update(steps=len(rows)-1, version=VERSION, settings=SETTINGS)
        official.write_json(out/'final_state.json', rows[-1])
    except Exception as exc:
        verdict = dict(status='error', reason=f'{type(exc).__name__}: {exc}',
                       physics_status='not_run', steps=max(0, len(rows)-1))
    finally:
        if initialized:
            gs.destroy()
    official.write_json(out/'report.json', verdict)
    return verdict


def verify(directory):
    directory = Path(directory)
    frozen = standard.library.read_json(directory/'input.json')
    if frozen['settings'] != SETTINGS or frozen['version'] != VERSION:
        raise ValueError('stabilization configuration changed')
    rows = [json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
    report = standard.library.read_json(directory/'report.json')
    result = assess(rows, frozen['layout'], frozen['geometry'])
    if report['status'] != 'candidate_ready' or result['status'] != 'candidate_ready':
        raise ValueError('stabilization not converged')
    if len(rows)-1 > SETTINGS['max_steps'] or len(rows)-1 != report['steps']:
        raise ValueError('invalid stabilization step count')
    if rows[-1] != standard.library.read_json(directory/'final_state.json'):
        raise ValueError('stabilization final state mismatch')
    return report
