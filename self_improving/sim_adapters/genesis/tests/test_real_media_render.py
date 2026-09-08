"""Real zero-step rendering of an asymmetric fixture at a validated-style final pose."""
import os

import pytest

from self_improving.sim_adapters.genesis import import_simfoundry_scene as scenes
from self_improving.sim_adapters.genesis.tests.test_simfoundry_scene import case, run


@pytest.mark.skipif(os.environ.get('GENESIS_MEDIA_RENDER_REAL') != '1',
                    reason='explicit real renderer opt-in required')
def test_render_uses_final_wxyz_pose_and_produces_real_orbit(tmp_path):
    fixture = case.__wrapped__(tmp_path)
    run(fixture)
    original = scenes.verify(fixture[2])
    final = {'objects': {'iter_0': {'position': [1, 2, 3.5],
                                  'orientation_wxyz': [1, 0, 0, 0]}}}
    report = scenes.preview(fixture[2], tmp_path / 'render', final_state=final, orbit=True,
        reference_camera={'cam2world': [[1, 0, 0, 1], [0, -1, 0, 2],
                                        [0, 0, -1, 5], [0, 0, 0, 1]],
                          'resolution': [320, 240],
                          'intrinsics': [[300, 0, 160], [0, 300, 120], [0, 0, 1]]})
    assert report['status'] == 'passed'
    assert report['physics_steps'] == 0
    assert report['geometry'][0]['world_vertex_error_m'] < 1e-5
    assert report['video']['total_frames'] == 120
    assert report['video']['unique_frames'] > 1
    assert (tmp_path / 'render/reference.png').is_file()
    assert scenes.verify(fixture[2]) == original
