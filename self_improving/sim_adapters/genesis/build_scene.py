"""Build an initial visual scene from existing asset bindings with LLM planning; no physics."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scene_gen.llm_provider import load_llm_provider_config
from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.extract_assets import verified_binding
from self_improving.sim_adapters.genesis.scene_planning import CACHE_DIR, ScenePlanner
from self_improving.sim_adapters.genesis.storage_paths import local_path
from self_improving.sim_adapters.genesis.task_output import TaskOutput

SCHEMA = spatial.SCHEMA
GAP, MARGIN = .06, .02
SETTLING_SLACK = .015
PLANNING_MARGIN = MARGIN+SETTLING_SLACK
FORMATS = {'.xml', '.urdf', '.glb', '.gltf', '.obj', '.stl'}


def polygon_area(points):
    if len(points) < 3:
        return 0.
    points = np.asarray(points)
    return abs(float(np.sum(points[:, 0]*np.roll(points[:, 1], -1)
                            - points[:, 1]*np.roll(points[:, 0], -1))))/2


def clip_polygon(subject, boundary):
    """Convex polygon intersection, both boundaries counter-clockwise."""
    result = [np.asarray(v, float) for v in subject]
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0), strict=True):
        source, result = result, []
        if not source:
            break
        edge = b-a
        def side(p):
            return edge[0]*(p[1]-a[1])-edge[1]*(p[0]-a[0])
        for p, q in zip(source, source[1:]+source[:1], strict=True):
            dp, dq = side(p), side(q)
            if dp >= -1e-12:
                result.append(p)
            if (dp >= 0) != (dq >= 0):
                result.append(p+(q-p)*(dp/(dp-dq)))
    return np.asarray(result)


def plane_surface(vertices, faces, top):
    """Prove convex coverage at one measured height."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    triangles = vertices[faces]
    horizontal = np.max(np.abs(triangles[:, :, 2]-top), axis=1) < 1e-5
    patches = []
    seen = set()
    for triangle in triangles[horizontal, :, :2]:
        area = polygon_area(triangle)
        if area < 1e-10:
            continue
        key = tuple(sorted(map(tuple, np.round(triangle, 8))))
        if key in seen:
            continue
        seen.add(key)
        if np.linalg.det(np.stack([triangle[1]-triangle[0], triangle[2]-triangle[0]])) < 0:
            triangle = triangle[::-1]
        patches.append(triangle)
    if not patches or len(patches) > 2000:
        raise ValueError('no bounded horizontal top surface')
    points = np.concatenate(patches)
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    area = sum(polygon_area(t) for t in patches)
    tolerance = max(1e-9, polygon_area(polygon)*1e-5)
    # Holes, disconnected patches and overlapping triangles must not masquerade as a solid top.
    if abs(area-polygon_area(polygon)) > tolerance:
        raise ValueError('top surface is not a solid convex patch')
    for i, a in enumerate(patches):
        for b in patches[:i]:
            if polygon_area(clip_polygon(a, b)) > tolerance:
                raise ValueError('overlapping top triangles cannot establish support coverage')
    return dict(z_m=float(top), polygon_xy_m=polygon.tolist(), area_m2=area,
                triangles_xy_m=np.asarray(patches).tolist(),
                method='level_top_triangles_verified_convex_coverage',
                meaning='visual surface only; collision and physical contact unverified')


def support_surface(vertices, faces):
    """Find a solid plane beneath shallow trim; never descend into a deep container."""
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    triangles = vertices[faces]
    flat = np.ptp(triangles[:, :, 2], axis=1) < 1e-5
    top = vertices[:, 2].max()
    levels = sorted(set(triangles[flat, :, 2].mean(axis=1)), reverse=True)
    max_recess = min(.005, float(np.ptp(vertices[:, 2]))*.01)
    failures = []
    for level in levels:
        if top-level > max_recess:
            break
        try:
            surface = plane_surface(vertices, faces, level)
        except ValueError as exc:
            failures.append(dict(z_m=float(level), reason=str(exc)))
            continue
        above = triangles[np.max(triangles[:, :, 2], axis=1) > level+1e-5, :, :2]
        surface.update(highest_vertex_z_m=float(top), recess_m=float(top-level),
                       maximum_recess_m=max_recess, skipped_planes=failures,
                       above_triangles_xy_m=[t.tolist() for t in above if polygon_area(t)>1e-10])
        return surface
    reason = failures[0]['reason'] if failures else 'no bounded horizontal top surface'
    raise ValueError(reason)


def fits_surface(surface, rectangle, margin=MARGIN):
    polygon = np.asarray(surface['polygon_xy_m'])
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = b-a
        for p in rectangle:
            signed = (edge[0]*(p[1]-a[1])-edge[1]*(p[0]-a[0]))/np.linalg.norm(edge)
            if signed < margin-1e-8:
                return False
    for triangle in surface.get('above_triangles_xy_m', []):
        if polygon_area(clip_polygon(triangle, np.asarray(rectangle))) > 1e-9:
            return False
    return True


def support_graph(document):
    ids = [o['object_id'] for o in document['objects']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate asset object ID')
    parents = {}
    for row in document['relations']:
        if row['relation'] != 'on':
            raise ValueError('initial layout currently supports explicit on relations only')
        source, target = row['source'], row['target']
        if source not in ids or target not in ids or source == target or source in parents:
            raise ValueError('invalid or multiple support targets')
        parents[source] = target
    if any(target in parents for target in parents.values()):
        raise ValueError('initial layout currently supports one support level only')
    return parents


def layout_objects(document, bindings, geometry):
    parents = support_graph(document)
    ids = [o['object_id'] for o in document['objects']]
    if set(ids) != set(bindings) or set(ids) != set(geometry):
        raise ValueError('geometry/binding object set mismatch')
    boxes = {name: np.asarray(geometry[name]['bounds'], float) for name in ids}
    if any(b.shape != (2, 3) or not np.isfinite(b).all() or not (b[1] > b[0]).all()
           for b in boxes.values()):
        raise ValueError('invalid native visual bounds')
    roots = [name for name in ids if name not in parents]
    widths = {name: b[1, 0]-b[0, 0] for name, b in boxes.items()}
    cursor = -(sum(widths[n] for n in roots)+GAP*(len(roots)-1))/2
    poses, surfaces = {}, {}
    for name in roots:
        box = boxes[name]
        poses[name] = np.array([cursor-box[0, 0], -box[:, 1].mean(), -box[0, 2]])
        cursor += widths[name]+GAP
        children = [n for n in ids if parents.get(n) == name]
        if not children:
            continue
        surface = geometry[name]['surface']
        polygon = np.asarray(surface['polygon_xy_m'])
        center = (polygon.min(axis=0)+polygon.max(axis=0))/2
        total_width = sum(widths[n] for n in children)+GAP*(len(children)-1)
        child_x = center[0]-total_width/2
        placements = []
        for child in children:
            box = boxes[child]
            translation = np.array([child_x-box[0, 0], center[1]-box[:, 1].mean(),
                                    surface['z_m']-box[0, 2]])
            placed = box+translation
            rectangle = np.array([[placed[0, 0], placed[0, 1]], [placed[1, 0], placed[0, 1]],
                                  [placed[1, 0], placed[1, 1]], [placed[0, 0], placed[1, 1]]])
            if not fits_surface(surface, rectangle, margin=PLANNING_MARGIN):
                raise ValueError('full object footprint does not fit measured support surface')
            poses[child] = poses[name]+translation
            placements.append(dict(object_id=child, footprint_target_xy_m=rectangle.tolist(),
                                   margin_m=PLANNING_MARGIN, coverage='passed'))
            child_x += widths[child]+GAP
        surfaces[name] = dict(surface, world_z_m=surface['z_m']+poses[name][2],
                              child_placements=placements)
    return dict(schema_version=SCHEMA, objects=[dict(bindings[n], object_id=n,
                translation_m=poses[n].tolist(), scale=1.0,
                local_visual_bounds_m=boxes[n].tolist(),
                world_visual_bounds_m=(boxes[n]+poses[n]).tolist(),
                support=parents.get(n, 'ground'), orientation_policy='preserve_native_orientation')
                for n in ids], support_surfaces=surfaces, relations=document['relations'],
                environment=dict(ground='genesis_builtin_plane', z_m=0),
                minimum_visual_gap_m=GAP, physics_steps=0, physics_status='not_run',
                meaning='initial visual placement; not physical stability/contact evidence')


def entity_mesh(entity):
    vertices, faces, offset = [], [], 0
    for link in entity.links:
        for geom in link.vgeoms:
            points = geom.get_vverts().detach().cpu().numpy().reshape(-1, 3)
            vertices.append(points)
            faces.append(np.asarray(geom.init_vfaces)+offset)
            offset += len(points)
    return np.concatenate(vertices), np.concatenate(faces)


def measure_geometry(entities, expected_bounds, parents):
    """Measure native visual meshes before any planning or image rendering."""
    geometry, base_positions = {}, {}
    for name, entity in entities.items():
        vertices, faces = entity_mesh(entity)
        bounds = np.asarray(official.bounds(vertices))
        if np.max(np.abs(bounds-np.asarray(expected_bounds[name]))) > 1e-5:
            raise ValueError('native asset bounds differ from verified preview index')
        base_positions[name] = entity.get_pos().detach().cpu().numpy().copy()
        geometry[name] = dict(bounds=bounds.tolist(), vertex_count=len(vertices),
                              triangle_count=len(faces),
                              mesh_sha256=hashlib.sha256(
                                  vertices.tobytes()+faces.tobytes()).hexdigest())
        if name in parents.values():
            try:
                geometry[name]['surface'] = support_surface(vertices, faces)
            except ValueError as exc:
                geometry[name]['support_error'] = str(exc)
    return geometry, base_positions


def render_scene(document, bindings, expected_bounds, output, check_inputs, *, plan_layout):
    os.environ['GS_HEADLESS'] = '1'
    os.environ['PYGLET_HEADLESS'] = '1'
    import genesis as gs

    commit = subprocess.check_output(['git', '-C', str(Path(gs.__file__).resolve().parents[1]),
                                      'rev-parse', 'HEAD'], text=True).strip()
    if commit != official.GENESIS_COMMIT:
        raise ValueError('Genesis revision mismatch')
    original_step = gs.Scene.step
    def no_step(*args, **kwargs):
        raise RuntimeError('physics step forbidden during scene construction')
    gs.Scene.step = no_step
    initialized = False
    report = dict(status='error', physics_steps=0, physics_status='not_run',
                  renderer='Genesis CPU Rasterizer', genesis_commit=commit, views=[], geometry=[])
    try:
        gs.init(backend=gs.cpu, seed=0, logging_level=logging.WARNING)
        initialized = True
        scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer(),
                         vis_options=gs.options.VisOptions(background_color=(1, 1, 1),
                         ambient_light=(.4, .4, .4), lights=official.LIGHTS, shadow=False,
                         segmentation_level='entity'))
        scene.add_entity(gs.morphs.Plane(collision=False),
                         surface=gs.surfaces.Default(color=(.9, .9, .9)))
        entities = {}
        for name, obj in bindings.items():
            source = local_path(obj['model_entrypoint'])
            if source.suffix.lower() not in FORMATS:
                raise ValueError(f'initial scene loader does not support {source.suffix}')
            options = dict(file=str(source), scale=1.0, convexify=False, decimate=False,
                           watertighten=None, collision=False)
            morph = (gs.morphs.MJCF(**options) if source.suffix == '.xml' else
                     gs.morphs.URDF(**options, fixed=True) if source.suffix == '.urdf' else
                     gs.morphs.Mesh(**options, fixed=True))
            entities[name] = scene.add_entity(morph, material=gs.materials.Rigid(),
                                               vis_mode='visual')
        camera = scene.add_camera(res=(1280, 960), pos=(0, -3, 2), lookat=(0, 0, .4),
                                  fov=35, GUI=False)
        scene.build()
        check_inputs()
        parents = spatial.relations(document)
        geometry, base_positions = measure_geometry(entities, expected_bounds, parents)
        clip.write_json(output/'native_geometry.json', geometry)
        check_inputs()
        layout = plan_layout(geometry)
        clip.write_json(output/'scene_layout.json', layout)
        clip.write_json(output/'support_surfaces.json', layout['support_surfaces'])
        for obj in layout['objects']:
            name = obj['object_id']
            entities[name].set_pos(base_positions[name]+np.asarray(obj['translation_m']))
        for obj in layout['objects']:
            bounds = np.asarray(official.bounds(entity_mesh(entities[obj['object_id']])[0]))
            error = float(np.max(np.abs(bounds-np.asarray(obj['world_visual_bounds_m']))))
            if error > 1e-5:
                raise ValueError('placed asset geometry does not match the planned translation')
            report['geometry'].append(dict(object_id=obj['object_id'], max_bounds_error_m=error))
        boxes = np.array([o['world_visual_bounds_m'] for o in layout['objects']])
        low, high = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)
        center = (low+high)/2
        distance = float(np.linalg.norm(high-low)/2/np.sin(np.deg2rad(35/2))*1.15)
        for name, direction, up in [('overview', [0, -1, .7], [0, 0, 1]),
                                     ('top', [0, 0, 1], [0, 1, 0]),
                                     ('side', [1, -1, .7], [0, 0, 1])]:
            vector = np.array(direction, float)
            camera.set_pose(pos=(center+distance*vector/np.linalg.norm(vector)).tolist(),
                            lookat=center.tolist(), up=up)
            rgb, _, seg, _ = camera.render(rgb=True, segmentation=True)
            visible = {}
            for object_id, entity in entities.items():
                ids = [int(i) for i, key in scene.visualizer.segmentation_idx_dict.items()
                       if int(key) == int(entity.idx)]
                visible[object_id] = int(np.isin(np.asarray(seg).squeeze(), ids).sum())
            path = output/f'{name}.png'
            Image.fromarray(np.asarray(rgb, np.uint8)).save(path)
            report['views'].append(dict(name=name, visible_pixels=visible,
                                        image=official.fingerprint(path, output.parent)))
        if any(max(v['visible_pixels'][n] for v in report['views']) < 64 for n in bindings):
            raise ValueError('an asset is not visible enough in any initial view')
        check_inputs()
        report['status'] = 'passed'
    finally:
        clip.write_json(output/'render_report.json', report)
        try:
            if initialized:
                gs.destroy()
        finally:
            gs.Scene.step = original_step
    return report


def run(scene_dir, clip_index, *, planner='llm', llm_config=None, seed=42,
        provider=None, renderer=None, expected_manifest_sha256=None):
    task = TaskOutput(scene_dir)
    if planner not in {'llm', 'rule'}:
        raise ValueError('unknown scene planner')
    clip.separate(task.root, Path(clip_index).resolve().parent,
                  getattr(provider, 'cache_dir', CACHE_DIR))
    with task.lock():
        if (expected_manifest_sha256 is not None
                and library.sha256(task.root/'manifest.json') != expected_manifest_sha256):
            raise ValueError('asset-stage manifest changed before automatic scene construction')
        task.start_scene()
        started = time.perf_counter()
        report = dict(schema_version=SCHEMA, status='error', physics_steps=0, model_calls=0,
                      planner=planner, seed=seed)
        config = None
        if isinstance(provider, ScenePlanner):
            provider.reset_evidence()
        try:
            index, _ = clip.load_index(clip_index)
            index_hash = library.sha256(clip_index)
            document = library.read_json(task.stage('objects')/'asset_request.json')
            spatial.relations(document)
            if planner == 'llm' and provider is None:
                config = load_llm_provider_config(llm_config)
                provider = ScenePlanner(config)
            bindings, expected_bounds = {}, {}
            for obj in document['objects']:
                name = obj['object_id']
                directory = task.stage('objects')/'asset_selection'/name
                bindings[name] = verified_binding(obj, directory, index, index_hash)
                asset = next(a for a in index['assets']
                             if a['asset_id'] == bindings[name]['asset_id'])
                preview_root = local_path(asset.get('preview_root',
                    Path(index['official_index']['path']).parent))
                record = library.read_json(preview_root/asset['record'])
                preview = library.read_json(preview_root/record['preview_result'])
                expected_bounds[name] = (preview['loaded_bounds'] if 'loaded_bounds' in preview
                                        else preview['geometry']['loaded_bounds_m'])
            def check_inputs():
                task.verify_scene_inputs()
                if library.sha256(clip_index) != index_hash:
                    raise ValueError('CLIP index changed during scene construction')
                for obj in document['objects']:
                    verified_binding(obj, task.stage('objects')/'asset_selection'/obj['object_id'],
                                     index, index_hash)
            def plan_layout(geometry):
                check_inputs()
                for target in set(spatial.relations(document).values()):
                    if 'surface' not in geometry[target]:
                        reason = geometry[target].get('support_error', 'no measured surface')
                        raise spatial.UnsupportedScene(f'{target}: {reason}')
                if planner == 'llm':
                    return provider.plan(document, bindings, geometry, seed, check_inputs,
                                         task.stage('scene'))
                proposal = dict(object_ids=[o['object_id'] for o in document['objects']],
                                relations=document['relations'], preferences=[])
                graph = spatial.graph_for(document, bindings, proposal)
                graph['preference_source'] = 'rule'
                clip.write_json(task.stage('scene')/'scene_graph.json', graph)
                layout = layout_objects(document, bindings, geometry)
                layout.update(scene_graph_sha256=clip.digest(graph),
                              native_geometry_sha256=clip.digest(geometry))
                clip.write_json(task.stage('scene')/'planning_evidence.json',
                                dict(status='passed', planner='rule', calls=0))
                return layout
            check_inputs()
            result = (renderer or render_scene)(document, bindings, expected_bounds,
                                                task.stage('scene'), check_inputs,
                                                plan_layout=plan_layout)
            if result['status'] != 'passed':
                raise ValueError('scene rendering did not pass')
            check_inputs()
            report.update(status='scene_built', object_count=len(bindings), render=result)
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            secret = config.api_key if config else getattr(
                getattr(provider, 'config', None), 'api_key', None)
            report['error'] = message.replace(secret, '[REDACTED]') if secret else message
        finally:
            if planner == 'llm' and provider is not None:
                report['model_calls'] = provider.evidence['calls']
                report['planning_cache'] = provider.evidence['cache']
                clip.write_json(task.stage('scene')/'planning_evidence.json', provider.evidence)
            report['total_s'] = time.perf_counter()-started
            clip.write_json(task.stage('scene')/'build_report.json', report)
            task.finish_scene(report)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', required=True, type=Path)
    parser.add_argument('--clip-index', required=True, type=Path)
    parser.add_argument('--planner', choices=['llm', 'rule'], default='llm')
    parser.add_argument('--llm-config', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    report = run(**vars(parser.parse_args(argv)))
    print(dict(status=report['status'], error=report.get('error'), total_s=report['total_s']))
    return 0 if report['status'] == 'scene_built' else 1


if __name__ == '__main__':
    raise SystemExit(main())
