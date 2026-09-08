"""Real Genesis box-fixture smoke replay; not media-scene production acceptance."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import position_solver as solver
from self_improving.sim_adapters.genesis.physics_math import angle, rotation
from self_improving.sim_adapters.genesis.validate_asset_scene import contacts
from self_improving.sim_adapters.genesis.validate_imported_scene import validate_scene_row


def run(source, out, dt):
    import genesis as gs

    document = json.loads(source.read_text())
    proposal = solver.solve(document)
    assert proposal['status'] == 'proposal_ready'
    out.mkdir(parents=True, exist_ok=False)
    cfg = dict(dt=dt, steps=round(4/dt), gravity=[0, 0, -9.81], seed=0,
               speed_mps=.01, angular_speed_radps=.05, penetration_m=.001,
               support_fraction=.8, support_force_n=1e-6, window_s=.5,
               translation_m=.001, rotation_deg=.5, use_hibernation=False,
               iterations=50, ls_iterations=50, tolerance=1e-8, constraint_timeconst=.001)
    (out/'input.json').write_text(json.dumps(document, indent=2)+'\n')
    (out/'proposal.json').write_text(json.dumps(proposal, indent=2)+'\n')
    (out/'settings.json').write_text(json.dumps(cfg, indent=2)+'\n')
    gs.init(backend=gs.cpu, precision='32', seed=0, logging_level=logging.WARNING)
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=dt, substeps=1,
                                                      gravity=tuple(cfg['gravity'])),
                     rigid_options=gs.options.RigidOptions(
                         constraint_solver=gs.constraint_solver.Newton,
                         iterations=50, ls_iterations=50, tolerance=1e-8,
                         constraint_timeconst=.001, use_hibernation=False))
    entities, owners, links = {}, {}, {}
    for obj in document['objects']:
        n = obj['object_id']
        pose = proposal['poses'][n]
        v = np.array(obj['geometry_vertices_m'])
        assert np.allclose(v.min(0), -v.max(0))
        entities[n] = scene.add_entity(gs.morphs.Box(
            size=tuple(np.ptp(v, axis=0)), pos=tuple(pose['translation_m']),
            quat=tuple(pose['orientation_wxyz']), fixed=obj['fixed']),
            material=gs.materials.Rigid(friction=.8, rho=500))
    scene.build()
    for name, entity in entities.items():
        for link in entity.links:
            for geom in link.geoms:
                owners[geom.idx], links[geom.idx] = name, link.idx
    rows = []
    scene.rigid_solver.detect_collision()
    with (out/'trace.jsonl').open('w') as f:
        for step in range(cfg['steps']+1):
            if step:
                scene.step()
            states = {n: {key: np.asarray(fn().cpu()).reshape(-1).tolist() for key, fn in (
                ('position', e.get_pos), ('orientation_wxyz', e.get_quat),
                ('velocity', e.get_vel), ('angular_velocity', e.get_ang))}
                for n, e in entities.items()}
            row = dict(step=step, time_s=step*dt, objects=states,
                       contact_phase='solved_step' if step else 'initial_detection',
                       contacts=contacts(scene.rigid_solver.collider.get_contacts(to_torch=False),
                                         owners, links, initial=step == 0))
            validate_scene_row(row, step, entities, dt)
            rows.append(row)
            f.write(json.dumps(row)+'\n')
    window = [r for r in rows if r['time_s'] >= 3.5]
    metrics = {}
    for rel in document['relations']:
        n, target = rel['source'], rel['target']
        states = [r['objects'][n] for r in window]
        supported, covered = [], []
        patch = next(s for s in document['surfaces'] if s['surface_id'] == rel['surface_id'])
        poly, normals = solver.polygon(patch['polygon_xy_m'])
        vertices = np.array(next(o for o in document['objects'] if o['object_id'] == n)[
            'geometry_vertices_m'])
        for row in window:
            force = 0
            for c in row['contacts']:
                if {c['a'], c['b']} == {n, target}:
                    force += c['force_a' if c['a'] == n else 'force_b'][2]
            supported.append(force > cfg['support_force_n'])
            a, b = row['objects'][n], row['objects'][target]
            local = (vertices @ rotation(a['orientation_wxyz']).T + a['position'] -
                     np.array(b['position'])) @ rotation(b['orientation_wxyz'])
            covered.append(all(np.min((local[:, :2]-p) @ normal) >= -1e-8
                               for p, normal in zip(poly, normals, strict=True)))
        m = dict(max_speed_mps=max(np.linalg.norm(s['velocity']) for s in states),
                 max_angular_speed_radps=max(np.linalg.norm(s['angular_velocity']) for s in states),
                 support_fraction=float(np.mean(supported)), full_footprint_covered=all(covered),
                 translation_m=max(np.linalg.norm(np.array(s['position'])-states[0]['position'])
                                   for s in states),
                 rotation_deg=max(angle(s['orientation_wxyz'], states[0]['orientation_wxyz'])
                                  for s in states))
        m['passed'] = (m['max_speed_mps'] <= .01 and m['max_angular_speed_radps'] <= .05
                       and m['support_fraction'] >= .8 and m['full_footprint_covered']
                       and m['translation_m'] <= .001 and m['rotation_deg'] <= .5)
        metrics[n] = m
    penetration = max((c['penetration'] for r in rows for c in r['contacts']), default=0)
    report = dict(fixture=source.stem, settings=cfg, input_sha256=solver.digest(document),
                  metrics=metrics, max_penetration_m=penetration,
                  scope='real Genesis primitive fixture smoke; not imported media acceptance',
                  passed=all(m['passed'] for m in metrics.values()) and penetration <= .001)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    gs.destroy()
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, choices=[.004, .002], default=.004)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.input, args.output, args.dt) else 2)
