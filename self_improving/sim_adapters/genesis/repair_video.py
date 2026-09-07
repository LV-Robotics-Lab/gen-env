"""Hash-bound Genesis replay in 03 and passed-only final camera orbit in 04."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import repair_assets
from self_improving.sim_adapters.genesis import repair_geometry as geo
from self_improving.sim_adapters.genesis import repair_physics as physics
from self_improving.sim_adapters.genesis import validate_asset_scene as native
from self_improving.sim_adapters.genesis.physics_math import rotation
from self_improving.sim_adapters.genesis.storage_paths import local_path

RES = (960, 720)


def encode(path, fps):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{RES[0]}x{RES[1]}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def verify_video(path, count, fps):
    import cv2

    cap = cv2.VideoCapture(str(path))
    decoded, unique = 0, set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape != (RES[1], RES[0], 3):
            raise ValueError("decoded video dimensions mismatch")
        unique.add(hashlib.sha256(frame.tobytes()).hexdigest())
        decoded += 1
    cap.release()
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)], text=True
        )
    )
    duration = float(probe["format"]["duration"])
    if decoded != count or abs(duration - count / fps) > 0.01:
        raise ValueError("decoded video frame count/duration mismatch")
    return dict(
        file=path.name,
        frame_count=decoded,
        unique_decoded_frames=len(unique),
        fps=fps,
        duration_s=duration,
        resolution=list(RES),
        sha256=lib.sha256(path),
    )


def replay_timing(dt, fps=50):
    """Name replay speed from recorded integration time without dropping samples."""
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid replay timestep or frame rate")
    speed = float(dt * fps)
    slowdown = 1 / speed
    if np.isclose(speed, 1.0, rtol=0, atol=1e-12):
        name = "physics_replay_realtime.mp4"
    elif speed < 1:
        name = f"physics_replay_{slowdown:g}x_slow.mp4"
    else:
        name = f"physics_replay_{speed:g}x_fast.mp4"
    return dict(filename=name, playback_speed=speed, slowdown_factor=slowdown)


def save_frame(out, index, frame):
    path = out / "frames" / f"frame_{index:04d}.png"
    Image.fromarray(frame).save(path)
    return dict(png_path=path.relative_to(out).as_posix(), png_sha256=lib.sha256(path))


def render(data, trace, out, check, *, final=False, physics_passed=False):
    if final and not physics_passed:
        raise ValueError("final render requires passed physics")
    rows = [json.loads(line) for line in Path(trace).open()]
    if not rows:
        raise ValueError("no physical samples to render")
    input_path = Path(trace).parent / "physics_input.json"
    if lib.read_json(input_path) != data:
        raise ValueError("video input differs from frozen physical input")
    if final and not physics.evaluate(data, rows)["passed"]:
        raise ValueError("final render requires independently passed trace")
    timing = replay_timing(data["settings"]["dt"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "frames").mkdir()
    gs = repair_assets.init_genesis()
    process = None
    report = dict(
        status="running",
        physics_steps=0,
        source_trace_sha256=lib.sha256(trace),
        source_input_sha256=lib.sha256(input_path),
        source_input_canonical_sha256=clip.digest(data),
        camera_motion_only=final,
        source_sample_count=len(rows),
        physics_passed=physics_passed,
        purpose="final_orbit" if final else "physics_replay",
    )
    old_step = gs.Scene.step

    def forbidden(*args, **kwargs):
        raise RuntimeError("physics steps forbidden in saved trajectory replay")

    gs.Scene.step = forbidden
    try:
        scene = gs.Scene(
            show_viewer=False,
            show_FPS=False,
            renderer=gs.renderers.Rasterizer(),
            vis_options=gs.options.VisOptions(
                background_color=(1, 1, 1),
                ambient_light=(0.4, 0.4, 0.4),
                lights=official.LIGHTS,
                shadow=False,
            ),
        )
        scene.add_entity(
            gs.morphs.Plane(collision=False), surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9))
        )
        entities = {}
        for n, a in data["assets"].items():
            options = dict(
                file=str(local_path(a["model_entrypoint"])),
                scale=a["scale"],
                convexify=False,
                decimate=False,
                watertighten=None,
                collision=False,
            )
            morph = (
                gs.morphs.MJCF(**options)
                if Path(a["model_entrypoint"]).suffix == ".xml"
                else (gs.morphs.Mesh(**options, fixed=True))
            )
            entities[n] = scene.add_entity(morph, material=gs.materials.Rigid(), vis_mode="visual")
        boxes = [
            geo.bounds(geo.transform(a["hull"], rows[-1]["objects"][n]))
            for n, a in data["assets"].items()
        ]
        low, high = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
        center = (low + high) / 2
        distance = np.linalg.norm(high - low) / 2 / np.sin(np.deg2rad(35 / 2)) * 1.1
        offset = np.array([0.0, -1.0, 0.8])
        offset *= distance / np.linalg.norm(offset)
        camera = scene.add_camera(
            res=RES, pos=(center + offset).tolist(), lookat=center.tolist(), fov=35, GUI=False
        )
        scene.build()
        references = {n: native.pose(e) for n, e in entities.items()}
        check()
        selected = [rows[-1]] * 180 if final else rows
        fps = 30 if final else 50
        path = out / ("orbit.mp4" if final else timing["filename"])
        process = encode(path, fps)
        hashes = set()
        max_geometry_error = 0.0
        with (out / "frames.jsonl").open("x") as ledger:
            for i, row in enumerate(selected):
                for n, e in entities.items():
                    a, state = data["assets"][n], row["objects"][n]
                    delta = rotation(state["orientation_wxyz"])
                    reference = references[n]
                    e.set_quat(
                        np.asarray(geo.quat(delta @ rotation(reference["orientation_wxyz"])))
                    )
                    e.set_pos(
                        np.asarray(state["position"])
                        + delta @ (np.asarray(reference["position"]) - a["anchor_m"])
                    )
                    if i in {0, len(selected) - 1}:
                        vertices, _ = repair_assets.visible_geometry(e, a["model_entrypoint"])
                        source = np.load(a["geometry_file"])["vertices"]
                        expected = geo.transform(source, state)
                        if vertices.shape != expected.shape:
                            raise ValueError("visual vertex set changed")
                        error = float(np.max(np.abs(vertices - expected)))
                        max_geometry_error = max(max_geometry_error, error)
                        if error > 1e-5:
                            raise ValueError(f"{n}: replay visual transform mismatch {error}")
                if final:
                    theta = 2 * np.pi * i / len(selected)
                    pos = center + np.array(
                        [np.sin(theta) * -offset[1], np.cos(theta) * offset[1], offset[2]]
                    )
                    camera.set_pose(pos=pos.tolist(), lookat=center.tolist(), up=[0, 0, 1])
                rgb, _, _, _ = camera.render(rgb=True, force_render=True)
                frame = np.ascontiguousarray(rgb, dtype=np.uint8)
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                hashes.add(digest)
                process.stdin.write(frame.tobytes())
                saved = save_frame(out, i, frame)
                ledger.write(
                    json.dumps(
                        dict(
                            frame=i,
                            source_step=row["step"],
                            source_time_s=row["time_s"],
                            raw_rgb_sha256=digest,
                            **saved,
                        )
                    )
                    + "\n"
                )
                if i in {0, len(selected) - 1}:
                    Image.fromarray(frame).save(out / f"frame_{i:04d}.png")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("video encoder failed")
        process = None
        if final:
            final_images = []
            for name, direction, up in (
                ("overview", [0, -1, 0.8], [0, 0, 1]),
                ("top", [0, 0, 1], [0, 1, 0]),
                ("side", [1, -1, 0.35], [0, 0, 1]),
            ):
                direction = np.asarray(direction, dtype=float)
                camera.set_pose(
                    pos=(center + distance * direction / np.linalg.norm(direction)).tolist(),
                    lookat=center.tolist(), up=up,
                )
                rgb, _, _, _ = camera.render(rgb=True, force_render=True)
                picture = out / f"{name}.png"
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(picture)
                final_images.append(official.fingerprint(picture, out))
            report["final_images"] = final_images
        report.update(
            verify_video(path, len(selected), fps),
            unique_raw_frames=len(hashes),
            maximum_visual_error_m=max_geometry_error,
            playback_speed=None if final else timing["playback_speed"],
            slowdown_factor=None if final else timing["slowdown_factor"],
            saved_frame_count=len(selected),
            frames_directory="frames",
            physics_duration_s=rows[-1]["time_s"] - rows[0]["time_s"],
            status="passed",
        )
        check()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        gs.Scene.step = old_step
        gs.destroy()
        clip.write_json(out / "render_report.json", report)
    return report
