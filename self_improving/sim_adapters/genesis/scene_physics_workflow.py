"""Internal scene-graph preparation and dual free replay behind existing entrances."""
from __future__ import annotations

import copy
import fcntl
import shutil
import time
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
from self_improving.sim_adapters.genesis import position_solver as solver
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis import scene_physics_graph as graph
from self_improving.sim_adapters.genesis import scene_stabilization as settle
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis import validate_imported_scene as free
from self_improving.sim_adapters.genesis.physics_math import angle, rotation

VERSION = 'genenv.scene_physics_workflow.v1'


def derive(source, destination, poses, operation):
    """Preserve source bytes; make a new, fully bound pose-only scene package."""
    source, destination = Path(source), Path(destination)
    original = imported.verify(source)
    shutil.copytree(source, destination)
    layout = copy.deepcopy(original)
    geometry = lib.read_json(destination/'native_geometry.json')
    if set(poses) != {o['object_id'] for o in layout['objects']}:
        raise ValueError('candidate changed object set')
    for obj in layout['objects']:
        n = obj['object_id']
        pose = poses[n]
        obj['translation_m'] = pose.get('translation_m', pose.get('position'))
        obj['orientation_wxyz'] = pose['orientation_wxyz']
        if obj['fixed']:
            prior = next(o for o in original['objects'] if o['object_id'] == n)
            if (np.linalg.norm(np.array(obj['translation_m'])-prior['translation_m']) > 1e-6):
                raise ValueError('candidate moved fixed root')
            obj['translation_m'] = prior['translation_m']
            obj['orientation_wxyz'] = prior['orientation_wxyz']
        obj['source_velocity_mps'] = [0, 0, 0]
        obj['source_angular_velocity_radps'] = [0, 0, 0]
        _, entry, _ = standard.verify_package(destination/obj['standard_package'])
        vertices = standard.inspect(entry)['visual']
        bounds = official.bounds(vertices @ rotation(obj['orientation_wxyz']).T
                                 + obj['translation_m']).tolist()
        obj['world_visual_bounds_m'] = bounds
        geometry[n]['world_bounds'] = bounds
    layout.update(layout_method=operation, native_geometry_sha256=clip.digest(geometry),
                  physics_status='not_run', physics_steps=0)
    official.write_json(destination/'scene_layout.json', layout)
    official.write_json(destination/'native_geometry.json', geometry)
    official.write_json(destination/'physics_derivation.json', dict(operation=operation,
                        source_manifest_sha256=lib.sha256(source/'manifest.json')))
    manifest = lib.read_json(destination/'manifest.json')
    manifest['files'] = [official.fingerprint(p, destination)
                         for p in sorted(destination.rglob('*'))
                         if p.is_file() and p != destination/'manifest.json']
    official.write_json(destination/'manifest.json', manifest)
    imported.verify(destination)
    return destination


def sdf_config(source, cell=None, resolution=None):
    if cell is not None or resolution is not None:
        return free.support_sdf(cell, resolution)
    layout = imported.verify(source)
    value = layout.get('support_sdf', {})
    # Existing explicitly recorded mouse derivation remains an opt-in configuration.
    path = Path(source)/'derivation.json'
    manifest = lib.read_json(Path(source)/'manifest.json')
    if not value and 'derivation.json' in {f['path'] for f in manifest['files']}:
        value = lib.read_json(path).get('support_sdf', {})
    return free.support_sdf(value.get('sdf_cell_size'), value.get('sdf_max_res'))


def planning_margins(steps=4):
    """Planning margins to try, from the full settling slack down to the red line."""
    return list(np.linspace(spatial.PLANNING_MARGIN, spatial.MARGIN, steps))


def stage_config(name, config):
    if not config.get('scoped_checkpoints'):
        return config
    shared = dict(version=config['version'], source=config['source'],
                  scene_manifest_sha256=config['scene_manifest_sha256'],
                  workflow_code=config['code']['scene_physics_workflow.py'])
    if name == 'position':
        shared.update(settings=config['position_settings'], code={k: v for k, v in
            config['code'].items() if k in ('position_solver.py', 'scene_physics_graph.py',
                                            'standard_urdf.py', 'import_simfoundry_scene.py')})
    else:
        shared.update(sdf=config['sdf'], genesis_commit=config['genesis_commit'],
                      code={k: v for k, v in config['code'].items() if k != 'position_solver.py'},
                      settings=config['stabilization_settings'] if name == 'stabilization'
                      else config['free_profiles'][name])
    return shared


def root_dependency(config):
    return (clip.digest(dict(source=config['source'],
                            manifest=config['scene_manifest_sha256']))
            if config.get('scoped_checkpoints') else clip.digest(config))


def run(scene_package, output_dir, *, resume=False, support_sdf_cell_size=None,
        support_sdf_max_res=None):
    source, out = Path(scene_package).resolve(), Path(output_dir).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('source scene and physics output must be separate')
    if out.exists() and not resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('physics workflow already running') from None
        return _run(source, out, support_sdf_cell_size, support_sdf_max_res)


def _run(source, out, cell, resolution):
    phase = 'graph'
    report = dict(schema_version=VERSION, physics_status='failed', status='error',
                  exit_code=1, stages={}, attempts=[], scene_package=str(source))
    try:
        layout = imported.verify(source)
        if lib.read_json(source/'scene_graph.json')['edges'] != layout['relations']:
            raise ValueError('scene graph and layout relations differ')
        graph.topology(layout)
        report['support_surfaces'] = graph.verify_support(source, layout)
        sdf = sdf_config(source, cell, resolution)
        code = {p.name: lib.sha256(p) for p in [Path(__file__), Path(solver.__file__),
                Path(graph.__file__), Path(settle.__file__), Path(free.__file__),
                Path(standard.__file__), Path(imported.__file__)]}
        config = dict(version=VERSION, scoped_checkpoints=True,
                      scene_manifest_sha256=lib.sha256(source/'manifest.json'),
                      source=str(source), code=code, sdf=sdf,
                      genesis_commit=official.GENESIS_COMMIT,
                      position_settings=solver.DEFAULTS | dict(
                          margin_m=spatial.PLANNING_MARGIN,
                          fallback_margins_m=planning_margins()),
                      stabilization_settings=settle.SETTINGS,
                      free_profiles={p: free.evidence.settings(p) for p in ('baseline', 'half_dt')})
        official.write_json(out/'workflow_input.json', config)
        key = root_dependency(config)

        def stage(name, dependency, action, verify):
            nonlocal phase
            phase = name
            directory, checkpoint = out/name, out/(name+'.checkpoint.json')
            stage_key = clip.digest(dict(config=clip.digest(stage_config(name, config)),
                                         dependency=dependency, stage=name))
            reusable = False
            if checkpoint.exists() and directory.is_dir():
                try:
                    saved = lib.read_json(checkpoint)
                    if saved['key'] != stage_key:
                        raise ValueError('stage dependency changed')
                    official.verify_files(directory, saved['files'])
                    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                              if p.is_file()}
                    if actual != {f['path'] for f in saved['files']}:
                        raise ValueError('stage file set mismatch')
                    verify(directory)
                    reusable = True
                except (OSError, ValueError, KeyError, TypeError):
                    reusable = False
            if not reusable:
                if directory.exists():
                    history = out/'history'/str(time.time_ns())
                    history.mkdir(parents=True)
                    shutil.move(str(directory), history/name)
                    if checkpoint.exists():
                        shutil.move(str(checkpoint), history/checkpoint.name)
                action(directory)
                verify(directory)
                saved = dict(key=stage_key, files=[official.fingerprint(p, directory)
                             for p in sorted(directory.rglob('*')) if p.is_file()])
                official.write_json(checkpoint, saved)
            report['stages'][name] = dict(status='passed', reused=reusable)
            return clip.digest(saved)

        def prepare(directory):
            directory.mkdir()
            document = solver.from_scene(source)
            # Aim for the full settling slack, but a small support surface must not become
            # unplaceable because of it. Give the slack up gradually, never below the
            # acceptance margin the validator will hold the result to.
            attempted = []
            for margin in planning_margins():
                proposal = solver.solve(document, margin_m=margin)
                attempted.append(dict(margin_m=margin, status=proposal['status']))
                if proposal['status'] == 'proposal_ready':
                    break
            official.write_json(directory/'input.json', document)
            official.write_json(directory/'proposal.json',
                                dict(proposal, attempted_margins=attempted))
            if proposal['status'] != 'proposal_ready':
                raise ValueError(
                    'position_search_exhausted down to the acceptance margin '
                    f'{spatial.MARGIN} m: the support surface cannot hold these objects')
            derive(source, directory/'scene', proposal['poses'], 'position_solver')

        position = stage('position', key, prepare, lambda d: imported.verify(d/'scene'))
        candidate = out/'position/scene'

        def stabilize(directory):
            result = settle.run(candidate, directory, sdf)
            if result['status'] != 'candidate_ready':
                raise ValueError('stabilization: '+str(result.get('reason')))
            poses = lib.read_json(directory/'final_state.json')['objects']
            derive(candidate, directory/'scene', poses, 'intervened_stabilization_candidate')

        def verify_settle(directory):
            settle.verify(directory)
            imported.verify(directory/'scene')

        settled = stage('stabilization', position, stabilize, verify_settle)
        candidate = out/'stabilization/scene'
        report['scene_package'] = str(candidate)
        passed = True
        for profile in ('baseline', 'half_dt'):
            def replay(directory, profile=profile):
                result = free.run(candidate, directory, profile=profile,
                    support_sdf_cell_size=sdf.get('sdf_cell_size'),
                    support_sdf_max_res=sdf.get('sdf_max_res'))
                if result['status'] != 'complete':
                    raise ValueError('free replay execution: '+str(result.get('error')))
            stage(profile, settled, replay, free.verify_evidence)
            result = free.verify_evidence(out/profile)
            passed &= result['physics_status'] == 'passed'
            report['stages'][profile]['status'] = result['physics_status']
            report['attempts'].append(dict(profile=profile, result_path=str(out/profile/
                'physics_result.json'), physics_status=result['physics_status']))
        divergence = agreement(out)
        report['profile_agreement'] = divergence
        if lib.sha256(source/'manifest.json') != config['scene_manifest_sha256']:
            raise ValueError('source scene changed during workflow')
        imported.verify(source)
        # A verdict that only holds at one step size is not a verdict. Both profiles
        # passing while they disagree on where the bodies ended up means the acceptance
        # window happened to close before the divergence showed, so report it as
        # inconclusive rather than turning a step-size artefact into a pass.
        status = 'passed' if passed else 'failed'
        if passed and divergence['status'] == 'compared' and not divergence['agreed']:
            status = 'inconclusive'
        report.update(status='complete', physics_status=status,
                      exit_code=0 if status == 'passed' else 2,
                      physics_result=str(out/'half_dt/physics_result.json'))
    except Exception as exc:
        report['stages'][phase] = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
        report.update(error=f'{type(exc).__name__}: {exc}', failed_stage=phase)
    official.write_json(out/'workflow_report.json', report)
    return report


# How far the two step sizes may land apart before the shared verdict stops meaning
# anything. Both budgets are the free replay's own per-profile settled budgets: two runs
# that each call a body settled to within these numbers must also agree with each other
# to within them, or one of the two is measuring something the other never saw.
AGREEMENT_POSITION_M = .01
AGREEMENT_ROTATION_DEG = 5.


def agreement(out):
    """Measure how far the two step sizes disagree on the final pose, and gate on it.

    Both profiles must still pass on their own, so this never widens acceptance. What it
    adds is the case neither profile can see alone: a body that looks nearly settled at
    the coarse step and is plainly rolling at the fine one produces two passes and a large
    divergence, and that combination is inconclusive rather than accepted.
    """
    states = {}
    for p in ('baseline', 'half_dt'):
        path = Path(out)/p/'final_state.json'
        if not path.is_file():
            # Reporting only: say so rather than failing an otherwise valid workflow.
            return dict(status='unavailable', reason=f'{p}/final_state.json missing')
        states[p] = lib.read_json(path)['objects']
    shared = sorted(set(states['baseline']) & set(states['half_dt']))
    objects = {}
    for n in shared:
        a, b = states['baseline'][n], states['half_dt'][n]
        objects[n] = dict(
            position_delta_m=float(np.linalg.norm(np.array(a['position'])-b['position'])),
            rotation_delta_deg=angle(a['orientation_wxyz'], b['orientation_wxyz']))
    position = max((o['position_delta_m'] for o in objects.values()), default=0.0)
    degrees = max((o['rotation_delta_deg'] for o in objects.values()), default=0.0)
    return dict(status='compared', objects=objects,
                maximum_position_delta_m=position,
                maximum_rotation_delta_deg=degrees,
                position_limit_m=AGREEMENT_POSITION_M,
                rotation_limit_deg=AGREEMENT_ROTATION_DEG,
                disagreeing_objects=sorted(
                    n for n, o in objects.items()
                    if o['position_delta_m'] > AGREEMENT_POSITION_M
                    or o['rotation_delta_deg'] > AGREEMENT_ROTATION_DEG),
                agreed=bool(position <= AGREEMENT_POSITION_M
                            and degrees <= AGREEMENT_ROTATION_DEG))


def verify(directory):
    directory = Path(directory)
    report = lib.read_json(directory/'workflow_report.json')
    config = lib.read_json(directory/'workflow_input.json')
    if report['schema_version'] != VERSION or report['physics_status'] != 'passed':
        raise ValueError('workflow has not passed both free replays')
    source = Path(config['source'])
    imported.verify(source)
    if lib.sha256(source/'manifest.json') != config['scene_manifest_sha256']:
        raise ValueError('workflow source changed')
    dependency = root_dependency(config)
    for name in ('position', 'stabilization', 'baseline', 'half_dt'):
        saved = lib.read_json(directory/(name+'.checkpoint.json'))
        expected_dependency = (clip.digest(lib.read_json(directory/'stabilization.checkpoint.json'))
                               if name == 'half_dt' else dependency)
        expected_key = clip.digest(dict(config=clip.digest(stage_config(name, config)),
                                       dependency=expected_dependency, stage=name))
        if saved['key'] != expected_key:
            raise ValueError('workflow dependency binding mismatch')
        official.verify_files(directory/name, saved['files'])
        if {p.relative_to(directory/name).as_posix() for p in (directory/name).rglob('*')
                if p.is_file()} != {f['path'] for f in saved['files']}:
            raise ValueError('workflow stage file set changed')
        if name in ('baseline', 'half_dt'):
            if free.verify_evidence(directory/name)['physics_status'] != 'passed':
                raise ValueError('free replay not passed')
        dependency = clip.digest(saved)
    settle.verify(directory/'stabilization')
    for name in ('baseline', 'half_dt'):
        frozen = lib.read_json(directory/name/'physics_input.json')
        if frozen['scene_manifest_sha256'] != lib.sha256(
                directory/'stabilization/scene/manifest.json'):
            raise ValueError('free replays do not share stabilized candidate')
    # Recomputed from the stored final states, so a stale or edited report cannot claim
    # a pass the two step sizes never actually agreed on.
    divergence = agreement(directory)
    if divergence != report.get('profile_agreement'):
        raise ValueError('profile agreement differs from stored evidence')
    if divergence['status'] != 'compared' or not divergence['agreed']:
        raise ValueError('free replays disagree across step sizes')
    return report
