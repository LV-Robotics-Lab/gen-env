"""Small self-contained finite box assets for real graph-workflow regression."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.physics_math import rotation


def package(root, document):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    objects, geometry = [], {}
    for source in document['objects']:
        n = source['object_id']
        folder = root/'assets'/n
        folder.mkdir(parents=True)
        v = np.asarray(source['geometry_vertices_m'])
        size = np.ptp(v, axis=0)
        assert np.allclose(v.min(0), -v.max(0))
        mass = float(np.prod(size)*500)
        inertia = mass/12*np.array([size[1]**2+size[2]**2, size[0]**2+size[2]**2,
                                   size[0]**2+size[1]**2])
        robot = ET.Element('robot', name=n)
        link = ET.SubElement(robot, 'link', name='body')
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', ixx=str(inertia[0]), iyy=str(inertia[1]),
                      izz=str(inertia[2]), ixy='0', ixz='0', iyz='0')
        for tag in ('visual', 'collision'):
            part = ET.SubElement(link, tag)
            shape = ET.SubElement(part, 'geometry')
            ET.SubElement(shape, 'box', size=' '.join(map(str, size)))
        ET.ElementTree(robot).write(folder/'body.urdf')
        official.write_json(folder/'physics.json', dict(friction=.8,
                            source='explicit regression fixture, uniform density 500 kg/m3'))
        files = [official.fingerprint(p, folder) for p in sorted(folder.iterdir())]
        asset = dict(schema_version='genenv.standard_urdf_asset.v1', asset_id='fixture_'+n,
                     entrypoint='body.urdf', physics_file='physics.json', files=files,
                     source=dict(object_id=n, kind='committed box fixture'), category='box')
        official.write_json(folder/'asset.json', asset)
        world = official.bounds(v @ rotation(source['orientation_wxyz']).T
                                + source['translation_m']).tolist()
        local = official.bounds(v).tolist()
        obj = {k: source[k] for k in ('object_id', 'fixed', 'translation_m', 'orientation_wxyz')}
        obj.update(category='box', asset_id=asset['asset_id'], scale=1,
                   standard_package=f'assets/{n}/asset.json',
                   standard_package_sha256=lib.sha256(folder/'asset.json'),
                   model_entrypoint=f'assets/{n}/body.urdf', source_files=files,
                   friction=.8, mass_kg=mass, intended_dynamic=not source['fixed'],
                   local_visual_bounds_m=local, world_visual_bounds_m=world,
                   source_velocity_mps=[0, 0, 0], source_angular_velocity_radps=[0, 0, 0])
        objects.append(obj)
        geometry[n] = dict(bounds=local, world_bounds=world,
                           visual_vertex_count=8, collision_vertex_count=8)
    env = dict(ground=None, position_m=[0, 0, 0], orientation_wxyz=[1, 0, 0, 0],
               visible=False, z_m=0)
    graph = dict(schema_version=spatial.SCHEMA, nodes=[{k: o[k] for k in (
        'object_id', 'asset_id', 'category')} for o in objects], edges=document['relations'],
        preferences=[], environment=env)
    layout = dict(schema_version=spatial.SCHEMA, objects=objects, relations=document['relations'],
                  support_surfaces={s['surface_id']: s for s in document['surfaces']},
                  environment=env, scene_graph_sha256=clip.digest(graph),
                  native_geometry_sha256=clip.digest(geometry), physics_status='not_run')
    for name, value in [('scene_graph', graph), ('scene_layout', layout),
                        ('native_geometry', geometry)]:
        official.write_json(root/(name+'.json'), value)
    official.write_json(root/'manifest.json', dict(schema_version=imported.VERSION,
                        files=[official.fingerprint(p, root) for p in sorted(root.rglob('*'))
                               if p.is_file()]))
    imported.verify(root)
    return root
