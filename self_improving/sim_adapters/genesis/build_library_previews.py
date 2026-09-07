"""Render independent non-robot assets with native Genesis, six views and zero physics steps."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official


def usd_audit(source, source_root, source_records):
    """Reject unresolved/external dependencies and unrenderable authored materials."""
    from genesis.utils.usd import UsdContext
    from pxr import Sdf, UsdGeom

    context = UsdContext(str(source))
    if Path(context.stage_file).resolve() != source.resolve():
        raise ValueError('USD parser cache cannot replace authenticated source')
    # This workflow never bakes or rewrites source assets, even on a machine with Omniverse.
    context._need_bake = False
    stage = context.stage
    if stage.GetCompositionErrors():
        raise ValueError('USD composition has unresolved references')
    paths = {source.relative_to(source_root).as_posix()}
    for layer in stage.GetUsedLayers():
        if layer.anonymous:
            continue
        path = Path(layer.realPath).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError('USD layer outside source inventory')
        relative = path.relative_to(source_root).as_posix()
        if relative not in source_records:
            raise ValueError('USD layer missing from source inventory')
        paths.add(relative)
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get()
            values = [value] if isinstance(value, Sdf.AssetPath) else []
            if attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray and value is not None:
                values = list(value)
            for asset in values:
                if not asset.path:
                    continue
                path = Path(asset.resolvedPath).resolve() if asset.resolvedPath else None
                if path is None or not path.is_file() or not path.is_relative_to(source_root):
                    raise ValueError(f'USD asset dependency unresolved or external: {asset.path}')
                relative = path.relative_to(source_root).as_posix()
                if relative not in source_records:
                    raise ValueError('USD dependency missing from source inventory')
                paths.add(relative)
    context.find_all_materials()
    if any(not material for material, _ in context._material_properties.values()):
        raise ValueError('USD authored material requires unavailable baking; no gray fallback')
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise ValueError('USD contains no mesh geometry')
    return context, sorted(paths)


def worker(job_path):
    job = library.read_json(job_path)
    output, source_root = Path(job['output']), Path(job['source_root'])
    candidate, source_records = job['candidate'], job['source_records']
    destination = output / 'previews' / candidate['asset_id']
    destination.mkdir(parents=True, exist_ok=False)
    result_path = destination / 'preview_result.json'
    result = dict(status='failed', entrypoint=candidate['entrypoint'],
                  source_root=str(source_root), source_files=[], views=[],
                  material_status='not_evaluated', physics_steps=0,
                  physics_status='not_evaluated', renderer='Genesis CPU Rasterizer',
                  genesis_commit=official.GENESIS_COMMIT, resolution=[512, 512],
                  scale=1.0, view_convention='four_zup_azimuths_plus_top_bottom',
                  geometry_status='native_parser_visual_geometry',
                  mode='asset_preview_zero_physics_steps')
    started = time.perf_counter()
    initialized = False
    try:
        if candidate['discovery_error']:
            raise ValueError(candidate['discovery_error'])
        source = official.safe_file(source_root, candidate['entrypoint'])
        dependencies = candidate['dependency_paths']
        os.environ['GS_HEADLESS'] = '1'
        os.environ['PYGLET_HEADLESS'] = '1'
        import genesis as gs

        actual_commit = subprocess.check_output(
            ['git', '-C', str(Path(gs.__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
            text=True).strip()
        if actual_commit != official.GENESIS_COMMIT:
            raise ValueError('Genesis revision mismatch')
        # An accidental physics step is a hard error, including during preview setup.
        def forbidden_step(*args, **kwargs):
            raise RuntimeError('physics step forbidden in retrieval previews')
        gs.Scene.step = forbidden_step
        gs.init(backend=gs.cpu, seed=0, logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer(),
                         vis_options=gs.options.VisOptions(
                             background_color=(1, 1, 1), ambient_light=(.4, .4, .4),
                             lights=official.LIGHTS, shadow=False, segmentation_level='entity'))
        options = dict(file=str(source), scale=1.0, convexify=False, decimate=False,
                       watertighten=None, collision=False)
        suffix = source.suffix.lower()
        if job.get('standard_urdf'):
            options.update(align=False, recompute_inertia=False, merge_fixed_links=False)
        if suffix in library.USD_FORMATS:
            context, dependencies = usd_audit(source, source_root, source_records)
            entities = scene.add_stage(gs.morphs.USD(**options, usd_ctx=context, fixed=True),
                                       material=gs.materials.Rigid(), vis_mode='visual')
        else:
            morph = (gs.morphs.URDF(**options, fixed=True) if suffix == '.urdf' else
                     gs.morphs.MJCF(**options) if suffix == '.xml' else
                     gs.morphs.Mesh(**options, fixed=True))
            entities = [scene.add_entity(morph, material=gs.materials.Rigid(), vis_mode='visual')]
        result['source_files'] = [source_records[p] for p in dependencies]
        official.verify_files(source_root, result['source_files'])
        result['material_status'] = 'passed'
        camera = scene.add_camera(res=(512, 512), GUI=False, pos=(3, 0, 2),
                                  lookat=(0, 0, 0), fov=35)
        scene.build()
        if job.get('standard_urdf'):
            from self_improving.sim_adapters.genesis import standard_urdf
            result['geometry_audit'] = standard_urdf.audit(
                entities[0], standard_urdf.inspect(source), collision=False)
        parts = [g.get_vverts().detach().cpu().numpy().reshape(-1, 3)
                 for entity in entities for link in entity.links for g in link.vgeoms]
        box = official.bounds(np.concatenate(parts))
        # Flat cloth and planes still have valid geometry; pad only the camera fitting box.
        frame_box = box.copy()
        span = max(float(np.max(box[1] - box[0])), 1e-6)
        for axis in range(3):
            if frame_box[1, axis] - frame_box[0, axis] < span * 1e-6:
                frame_box[:, axis] += [-span * 1e-6, span * 1e-6]
        result.update(loaded_bounds=box.tolist(), visual_parts=len(parts),
                      entity_count=len(entities), camera_fit_bounds=frame_box.tolist())
        ids = {int(i) for i, key in scene.visualizer.segmentation_idx_dict.items()
               if int(key) in {int(e.idx) for e in entities}}
        if not ids:
            raise ValueError('missing preview segmentation IDs')
        sheets = []
        for view in official.camera_views(frame_box):
            camera.set_pose(pos=view['pos'], lookat=view['lookat'], up=view['up'])
            # Camera near/far and FOV are not changed by set_pose.
            camera._near, camera._far = view['near'], view['far']
            native_camera = camera._rasterizer._camera_nodes[camera.uid].camera
            native_camera.znear, native_camera.zfar = view['near'], view['far']
            rgb, _, segmentation, _ = camera.render(rgb=True, segmentation=True)
            mask = np.isin(np.asarray(segmentation).squeeze(), list(ids)).astype(np.int32)
            visibility = official.check_visibility(rgb, mask, 1)
            path = destination / f"view_{view['name']}.png"
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
            result['views'].append(dict(camera=view, visibility=visibility,
                                        image=official.fingerprint(path, output)))
            sheets.append((view['name'], path))
        official.make_sheet(sheets, destination / 'contact_sheet.png', 3)
        result['contact_sheet'] = official.fingerprint(destination / 'contact_sheet.png', output)
        official.verify_files(source_root, result['source_files'])
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1500])
    finally:
        result['elapsed_s'] = time.perf_counter() - started
        official.write_json(result_path, result)
        if initialized:
            gs.destroy()
    return result


def build(manifest_path, output_dir, *, workers=2, timeout_s=180, only=None):
    source_root, source_records = library.verify_download(manifest_path)
    output = Path(output_dir).resolve()
    if (output.is_relative_to(source_root) or source_root.is_relative_to(output)):
        raise ValueError('preview output must be separate from downloaded source package')
    if output.exists():
        raise FileExistsError('preview output directory exists')
    if any((p / marker).is_file() for p in output.parents
           for marker in ('asset_index.json', 'download_manifest.json')):
        raise ValueError('preview output inside sealed asset package')
    output.mkdir(parents=True, exist_ok=False)
    candidates, excluded = library.discover(source_root, source_records)
    if only:
        # Debug builds are explicitly partial and cannot pass the full preview-index verifier.
        candidates = [c for c in candidates if c['asset_id'] in only]
    reference = dict(path=str(Path(manifest_path).resolve()), sha256=library.sha256(manifest_path))
    for candidate in candidates:
        official.write_json(output / 'jobs' / f"{candidate['asset_id']}.json",
                            dict(output=str(output), source_root=str(source_root),
                                 candidate=candidate, source_records=source_records))
    started = time.perf_counter()
    assets = []

    def run(candidate):
        job = output / 'jobs' / f"{candidate['asset_id']}.json"
        logs = output / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / f"{candidate['asset_id']}.log").open('w') as stream:
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        '--worker', str(job)], stdout=stream, stderr=stream,
                                       timeout=timeout_s, check=False,
                                       env=dict(os.environ, OMP_NUM_THREADS='2'))
                error = f'worker_exit_{child.returncode}' if child.returncode else None
            except subprocess.TimeoutExpired:
                error = 'preview_timeout'
        preview_relative = f"previews/{candidate['asset_id']}/preview_result.json"
        path = output / preview_relative
        preview = library.read_json(path) if path.exists() else {}
        passed = error is None and preview.get('status') == 'passed'
        record = dict(candidate, source_root=str(source_root), source_inventory=reference,
                      source_files=preview.get('source_files', []), preview_result=preview_relative,
                      status='preview_passed' if passed else 'failed',
                      physics_status='not_evaluated', error=preview.get('error') or error)
        record_path = f"assets/{candidate['asset_id']}.json"
        official.write_json(output / record_path, record)
        return dict(asset_id=candidate['asset_id'], record=record_path, status=record['status'])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, c) for c in candidates]
        for future in as_completed(futures):
            item = future.result()
            assets.append(item)
            progress = f"[{len(assets)}/{len(candidates)}] {item['asset_id']} {item['status']}"
            print(progress, flush=True)
    library.verify_download(manifest_path)
    if library.sha256(manifest_path) != reference['sha256']:
        raise ValueError('source inventory changed during preview build')
    status = 'passed' if all(a['status'] == 'preview_passed' for a in assets) else 'partial'
    report = dict(status=status,
                  discovered_assets=len(candidates), preview_passed=sum(
                      a['status'] == 'preview_passed' for a in assets),
                  preview_failed=sum(a['status'] != 'preview_passed' for a in assets),
                  excluded_entries=len(excluded), elapsed_s=time.perf_counter() - started,
                  source_inventory=reference, physics_status='not_evaluated')
    official.write_json(output / 'build_report.json', report)
    files = [official.fingerprint(p, output) for p in sorted(output.rglob('*')) if p.is_file()]
    index = dict(schema_version=library.SCHEMA, source_inventory=reference,
                 assets=sorted(assets, key=lambda a: a['asset_id']), excluded=excluded,
                 status=report['status'], files=files)
    official.write_json(output / 'asset_index.json', index)
    if not only:
        library.verify_preview_index(output / 'asset_index.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--timeout-s', type=float, default=180)
    parser.add_argument('--only', nargs='+', help='Explicit debug subset; not a production index')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return 0 if worker(args.worker)['status'] == 'passed' else 1
    if not args.download_manifest or not args.output_dir:
        parser.error('--download-manifest and --output-dir are required')
    result = build(args.download_manifest, args.output_dir, workers=args.workers,
                   timeout_s=args.timeout_s, only=args.only)
    print(json.dumps(result))
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
