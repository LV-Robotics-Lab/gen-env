"""Exercise the actual upstream image adapter with FFmpeg, without loading GPU models."""
import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.parametrize('size', [(17, 19), (18, 20)])
def test_single_image_preserves_frame_and_encodes_even_video(tmp_path, size):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg is required for the real encoding test')
    root = Path(__file__).resolve().parents[4]
    stage = (root / 'external/SimFoundry/scripts/pipeline/A_reconstruction/stages'
             / '1b_process_raw_video.py')
    if not stage.is_file():
        pytest.skip('SimFoundry submodule is not initialized')
    tree = ast.parse(stage.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'process_single_image')
    namespace = dict(Path=Path, shutil=shutil, Image=Image, subprocess=subprocess)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(stage), 'exec'), namespace)
    source = tmp_path / 'source.png'
    Image.new('RGB', size, (100, 150, 200)).save(source)
    video, frames = tmp_path / 'video', tmp_path / 'frames'
    video.mkdir()
    frames.mkdir()
    _, movie = namespace['process_single_image'](source, video, frames)
    with Image.open(source) as original, Image.open(frames / 'frame_0001.png') as frame:
        assert frame.size == size
        assert frame.tobytes() == original.tobytes()
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', movie]))['streams'][0]
    assert (info['width'], info['height']) == tuple(n + n % 2 for n in size)
    assert int(info['nb_frames']) == 1
