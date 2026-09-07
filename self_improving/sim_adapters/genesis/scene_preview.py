"""Compatibility entrypoint for complete extraction/selection; legacy preview helpers below."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scene_gen.parser import COLOR_TERMS, MATERIAL_TERMS
from scene_gen.schema import RelationType, SceneSpecError
from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.storage_paths import local_path, same_evidence_path
from self_improving.sim_adapters.genesis.task_output import STAGES

SCHEMA = 'genenv.genesis_scene_preview.v1'
# Canonical category labels, not a translation/model call or extra extraction prompt.
LABELS = {'apple': '苹果', 'cup': '杯子', 'mug': '马克杯', 'bowl': '碗',
          'donut': '甜甜圈', 'plate': '盘子', 'bottle': '瓶子', 'box': '盒子'}
RENDER_FORMATS = {'.xml', '.urdf', '.glb', '.gltf', '.obj', '.stl'}


def object_query(obj):
    def label(value, terms):
        choices = terms.get(value, (value,))
        return next((v for v in choices if any('\u4e00' <= c <= '\u9fff' for c in v)), value)
    parts = [label(obj.color, COLOR_TERMS) if obj.color else '',
             label(obj.material, MATERIAL_TERMS) if obj.material else '',
             LABELS.get(obj.category, obj.category)]
    return ''.join(parts)


def check_scope(spec):
    if any(r.relation != RelationType.ON_TABLE for r in spec.relations):
        raise SceneSpecError('preview currently supports independent on_table objects only')
    if any(o.articulation is not None or o.region != 'center' for o in spec.objects):
        raise SceneSpecError('preview does not implement articulation or table-region constraints')


def row_layout(spec, bindings):
    """A measured, deterministic visual arrangement; never a contact/stability verdict."""
    check_scope(spec)
    if [b['object_id'] for b in bindings] != [o.object_id for o in spec.objects]:
        raise ValueError('preview object binding set/order mismatch')
    boxes = [np.asarray(b['local_visual_bounds_m'], dtype=float) for b in bindings]
    for box in boxes:
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] <= box[0]):
            raise ValueError('invalid measured visual bounds')
    gap, edge = .06, .025
    widths = [float(b[1, 0] - b[0, 0]) for b in boxes]
    width = sum(widths) + gap * (len(boxes) - 1)
    ws = spec.workspace
    xmin, xmax = ws.x_bounds_m
    ymin, ymax = ws.y_bounds_m
    depth = max(float(b[1, 1] - b[0, 1]) for b in boxes)
    # Stay above the robot keepout strip; all full visual footprints remain on the tabletop.
    row_ymin = max(ymin + edge, ws.robot_keepout_y_m[1] + edge)
    if width > xmax - xmin - 2 * edge or depth > ymax - row_ymin - edge:
        raise ValueError('selected assets do not fit table at original scale')
    center_y = (row_ymin + ymax - edge) / 2
    cursor = (xmin + xmax - width) / 2
    objects = []
    for binding, box, size in zip(bindings, boxes, widths, strict=True):
        center_x = cursor + size / 2
        pos = np.array([center_x - box[:, 0].mean(), center_y - box[:, 1].mean(),
                        ws.table_height_m - box[0, 2]])
        objects.append(dict(binding, translation_m=pos.tolist(), orientation_wxyz=[1, 0, 0, 0],
                            scale=1.0, world_visual_bounds_m=(box + pos).tolist(),
                            support='table', intended_dynamic=True))
        cursor += size + gap
    return dict(schema_version=SCHEMA, source_scene_spec_sha256=spec.digest(),
                layout_method='measured_bounds_row_in_object_order',
                layout_meaning='初始视觉摆放；未验证接触、碰撞、稳定性或物理可用性。',
                frame=spec.frame.model_dump(mode='json'), minimum_visual_gap_m=gap,
                table=dict(size_m=[xmax-xmin, ymax-ymin, .04],
                           position_m=[(xmin+xmax)/2, (ymin+ymax)/2, ws.table_height_m-.02],
                           top_z_m=ws.table_height_m, representation='context_box'),
                objects=objects, physics_status='not_evaluated', physics_steps=0)


def trusted_binding(spec, obj, selection_dir, clip_path):
    """Re-derive paths and geometry from the verified index, never from model text."""
    if obj not in spec.objects:
        raise ValueError('selected object absent from parsed scene')
    index, _ = clip.load_index(clip_path)
    binding = library.read_json(selection_dir / 'selected_asset.json')
    report = library.read_json(selection_dir / 'run_report.json')
    official.verify_files(selection_dir, report['files'])
    if report['status'] != 'selected' or binding['query'] != object_query(obj):
        raise ValueError('selection query/status binding mismatch')
    if binding['clip_index_sha256'] != library.sha256(clip_path):
        raise ValueError('selection index hash mismatch')
    asset = next(a for a in index['assets'] if a['asset_id'] == binding['asset_id'])
    preview_root = Path(index['official_index']['path']).parent
    record = library.read_json(official.safe_file(preview_root, asset['record']))
    preview = library.read_json(official.safe_file(preview_root, record['preview_result']))
    source_root = Path(asset.get('source_root', preview_root))
    source = official.safe_file(source_root, asset['entrypoint'])
    if (not same_evidence_path(str(source), binding['model_entrypoint'])
            or binding['source_files'] != asset['source_files']):
        raise ValueError('selection source binding mismatch')
    if source.suffix.lower() not in RENDER_FORMATS:
        raise ValueError('selected format not supported by multi-object preview')
    if 'loaded_bounds' in preview:
        bounds = preview['loaded_bounds']
    else:
        bounds = preview['geometry']['loaded_bounds_m']
    return dict(object_id=obj.object_id, query=object_query(obj), asset_id=asset['asset_id'],
                selection_file=str((selection_dir/'selected_asset.json').resolve()),
                selection_sha256=library.sha256(selection_dir/'selected_asset.json'),
                source_root=str(source_root), source_files=asset['source_files'],
                model_entrypoint=str(source), model_format=source.suffix.lstrip('.'),
                official_index=index['official_index'], clip_index_sha256=library.sha256(clip_path),
                local_visual_bounds_m=bounds, visible_differences=binding['visible_differences'])


def render(layout, output):
    """Native Genesis scene with original materials, rendered before any physics step."""
    os.environ['GS_HEADLESS'] = '1'
    os.environ['PYGLET_HEADLESS'] = '1'
    import genesis as gs

    commit = subprocess.check_output(
        ['git', '-C', str(Path(gs.__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
        text=True).strip()
    if commit != official.GENESIS_COMMIT:
        raise ValueError('Genesis revision mismatch')
    original_step = gs.Scene.step

    def forbidden_step(*args, **kwargs):
        raise RuntimeError('physics step forbidden in initial scene preview')
    gs.Scene.step = forbidden_step
    initialized = False
    destination = output / STAGES['scene']
    destination.mkdir(exist_ok=True)
    result = dict(status='error', schema_version=SCHEMA, genesis_commit=commit,
                  renderer='Genesis CPU Rasterizer', physics_steps=0,
                  physics_status='not_evaluated', views=[], geometry=[])
    try:
        gs.init(backend=gs.cpu, seed=0, logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer(),
                         vis_options=gs.options.VisOptions(
                             background_color=(1, 1, 1), ambient_light=(.4, .4, .4),
                             lights=official.LIGHTS, shadow=False, segmentation_level='entity'))
        table = layout['table']
        scene.add_entity(gs.morphs.Box(size=table['size_m'], pos=table['position_m'],
                                       fixed=True, collision=False),
                         surface=gs.surfaces.Default(color=(.78, .78, .75)))
        entities = {}
        for obj in layout['objects']:
            official.verify_files(Path(obj['source_root']), obj['source_files'])
            if library.sha256(obj['selection_file']) != obj['selection_sha256']:
                raise ValueError('selected asset changed before render')
            selected = library.read_json(obj['selection_file'])
            if (selected['model_entrypoint'] != obj['model_entrypoint']
                    or selected['asset_id'] != obj['asset_id']
                    or selected['source_files'] != obj['source_files']):
                raise ValueError('render source differs from selected asset')
            options = dict(file=str(local_path(obj['model_entrypoint'])),
                           pos=obj['translation_m'], scale=1.0,
                           convexify=False, decimate=False, watertighten=None, collision=False)
            suffix = Path(obj['model_entrypoint']).suffix.lower()
            morph = (gs.morphs.MJCF(**options) if suffix == '.xml' else
                     gs.morphs.URDF(**options, fixed=True) if suffix == '.urdf' else
                     gs.morphs.Mesh(**options, fixed=True))
            entities[obj['object_id']] = scene.add_entity(
                morph, material=gs.materials.Rigid(), vis_mode='visual')
        camera = scene.add_camera(res=(1280, 960), pos=(0, -1, 1.7),
                                  lookat=(0, 0, .8), fov=35, GUI=False)
        scene.build()
        for obj in layout['objects']:
            e = entities[obj['object_id']]
            points = np.concatenate([g.get_vverts().detach().cpu().numpy().reshape(-1, 3)
                                     for link in e.links for g in link.vgeoms])
            actual = official.bounds(points)
            error = float(np.max(np.abs(actual - np.asarray(obj['world_visual_bounds_m']))))
            if error > 1e-5:
                raise ValueError('loaded scene geometry differs from measured preview placement')
            result['geometry'].append(dict(object_id=obj['object_id'], bounds_m=actual.tolist(),
                                           max_bounds_error_m=error))
        all_boxes = np.array([o['world_visual_bounds_m'] for o in layout['objects']])
        center = (all_boxes[:, 0].min(axis=0) + all_boxes[:, 1].max(axis=0)) / 2
        # Include the table edges for scene context, looking from negative y so x-right is visible.
        radius = float(np.linalg.norm(np.asarray(table['size_m'])[:2]) / 2)
        distance = radius / np.sin(np.deg2rad(35/2)) * 1.15
        directions = [('overview', [0, -1, .85], [0, 0, 1]),
                      ('top', [0, 0, 1], [0, 1, 0]),
                      ('side', [1, -.7, .8], [0, 0, 1])]
        counts = {o['object_id']: [] for o in layout['objects']}
        for name, direction, up in directions:
            direction = np.asarray(direction, dtype=float)
            pos = center + distance * direction / np.linalg.norm(direction)
            camera.set_pose(pos=pos.tolist(), lookat=center.tolist(), up=up)
            rgb, _, segmentation, _ = camera.render(rgb=True, segmentation=True)
            segmentation = np.asarray(segmentation).squeeze()
            visible = {}
            for object_id, entity in entities.items():
                ids = [int(i) for i, key in scene.visualizer.segmentation_idx_dict.items()
                       if int(key) == int(entity.idx)]
                pixels = int(np.isin(segmentation, ids).sum())
                counts[object_id].append(pixels)
                visible[object_id] = pixels
            path = destination / f'{name}.png'
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
            result['views'].append(dict(name=name, image=official.fingerprint(path, output),
                                        camera=dict(pos=pos.tolist(),
                                                    lookat=center.tolist(), up=up),
                                        visible_pixels=visible))
        if any(min(v) < 64 for v in counts.values()):
            raise ValueError('an object is not clearly visible in every scene view')
        result['status'] = 'preview_passed'
    finally:
        official.write_json(destination/'render_report.json', result)
        try:
            if initialized:
                gs.destroy()
        finally:
            gs.Scene.step = original_step
    return result


# Legacy geometry helpers above remain available to offline regressions. Public natural-language
# execution builds a planned initial scene after selection; --stop-after assets disables it.
from self_improving.sim_adapters.genesis.extract_assets import main, run  # noqa: E402,F401,I001


if __name__ == '__main__':
    raise SystemExit(main())
