"""Finite support observation, retrieval and materialization for media reconstruction."""

from __future__ import annotations

import base64
import io
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

from scene_gen.llm_provider import load_llm_provider_config
from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis.storage_paths import local_path

OBSERVATION_SCHEMA = "genenv.media_support_observation.v1"
SELECTION_SCHEMA = "genenv.media_support_selection.v1"
MATERIALIZATION_SCHEMA = "genenv.media_support_asset.v1"
SUPPORTED_CATEGORIES = {"desk", "table", "counter"}
SUPPORTED_FORMATS = {"glb", "gltf"}
SUPPORT_ID = "support_0"


def _safe_copy(source: Path, destination: Path) -> None:
    source, destination = Path(source).resolve(), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"invalid source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def observe(scene_dir, output_dir):
    """Bind stage-3 finite pixels/points to SimFoundry's stage-4 world frame."""
    scene, out = Path(scene_dir).resolve(), Path(output_dir).resolve()
    stage3, stage4 = scene / "s3_ground", scene / "s4_frame"
    if out.exists():
        raise FileExistsError(out)
    selection_path = stage3 / "frame_selection.json"
    if selection_path.exists():
        selection = lib.read_json(selection_path)
        index = int(selection.get("resolved_img_idx", selection.get("img_idx")))
    else:
        matches = sorted(stage3.glob("image_*_floor_info.json"))
        if len(matches) != 1:
            raise ValueError("cannot resolve SimFoundry support frame")
        index = int(matches[0].stem.split("_")[1])
    floor_path = stage3 / f"image_{index}_floor_info.json"
    floor = lib.read_json(floor_path)
    if floor.get("floor_category") not in SUPPORTED_CATEGORIES:
        raise ValueError("unsupported support category")
    npz_path = stage3 / floor.get("support_observation", "support_observation.npz")
    mask_path = stage3 / floor.get("support_mask", "support_mask.png")
    transform_path = stage4 / f"image_{index}_cam2world.npy"
    for path in (npz_path, mask_path, transform_path):
        if not path.is_file():
            raise ValueError(f"missing finite support evidence: {path.name}")
    arrays = np.load(npz_path, allow_pickle=False)
    required = {
        "rgb", "depth", "intrinsics", "mask", "points_camera", "plane_inlier_indices"
    }
    if set(arrays.files) != required:
        raise ValueError("unexpected support observation arrays")
    rgb, mask = arrays["rgb"], arrays["mask"].astype(bool)
    if rgb.shape[:2] != mask.shape or mask.ndim != 2 or not mask.any():
        raise ValueError("invalid support image or mask")
    points = np.asarray(arrays["points_camera"], float)
    indices = np.asarray(arrays["plane_inlier_indices"], int)
    if points.ndim != 2 or points.shape[1] != 3 or len(indices) < 3:
        raise ValueError("insufficient support plane points")
    if indices.min() < 0 or indices.max() >= len(points):
        raise ValueError("invalid support inlier indices")
    transform = np.load(transform_path, allow_pickle=False)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("invalid camera-to-world transform")
    inliers = points[indices]
    world = inliers @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError("nonfinite support world points")
    low, high = world[:, :2].min(0), world[:, :2].max(0)
    ys, xs = np.nonzero(mask)
    crop_box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    out.mkdir(parents=True)
    _safe_copy(mask_path, out / "support_mask.png")
    np.savez_compressed(
        out / "support_observation.npz",
        rgb=rgb.astype(np.uint8),
        depth=np.asarray(arrays["depth"], np.float32),
        intrinsics=np.asarray(arrays["intrinsics"], np.float64),
        mask=mask.astype(np.uint8),
        points_camera=points.astype(np.float32),
        plane_inlier_indices=indices.astype(np.int64),
        points_world=world.astype(np.float32),
    )
    masked = np.full_like(rgb, 255)
    masked[mask] = rgb[mask]
    Image.fromarray(masked).crop(crop_box).save(out / "support_crop.png")
    censored = floor.get(
        "boundary_censored",
        {
            "top": bool(mask[0].any()),
            "bottom": bool(mask[-1].any()),
            "left": bool(mask[:, 0].any()),
            "right": bool(mask[:, -1].any()),
        },
    )
    report = dict(
        schema_version=OBSERVATION_SCHEMA,
        category=floor["floor_category"],
        frame_index=index,
        plane=dict(origin=floor["origin"], z_dir=floor["z_dir"]),
        image_size_hw=list(mask.shape),
        crop_box_xyxy=crop_box,
        boundary_censored=censored,
        mask_pixels=int(mask.sum()),
        plane_inlier_pixels=len(indices),
        plane_inlier_ratio=float(len(indices) / max(len(points), 1)),
        visible_footprint_world_xy_m=[low.tolist(), high.tolist()],
        visible_extents_m=(high - low).tolist(),
        extent_evidence=[
            "inferred" if censored["left"] or censored["right"] else "measured",
            "inferred" if censored["top"] or censored["bottom"] else "measured",
        ],
        limitations=[
            "single-view hidden support geometry is inferred",
            "monocular depth scale follows SimFoundry reconstruction",
        ],
        inputs={
            "floor_info": official.fingerprint(floor_path, scene),
            "stage3_observation": official.fingerprint(npz_path, scene),
            "cam2world": official.fingerprint(transform_path, scene),
        },
    )
    official.write_json(out / "support_observation.json", report)
    return report


def _mesh(source):
    import trimesh

    loaded = trimesh.load(source, force="scene", process=False)
    geometry = loaded.to_geometry()
    if not len(geometry.vertices) or not len(geometry.faces):
        raise ValueError("empty support mesh")
    return geometry


def geometry_gate(source):
    """Find the dominant finite terminal plane and an axis mapping to Genesis +Z."""
    mesh = _mesh(source)
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    triangles = vertices[faces]
    normals = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)
    bounds = np.asarray(mesh.bounds, float)
    best = None
    for axis in range(3):
        span = bounds[1, axis] - bounds[0, axis]
        for sign in (-1, 1):
            level = bounds[1, axis] if sign > 0 else bounds[0, axis]
            centers = triangles.mean(1)[:, axis]
            selected = (normals[:, axis] * sign > 0.94) & (
                np.abs(centers - level) <= max(0.01, span * 0.03)
            )
            area = float(areas[selected].sum())
            row = (area, axis, sign, selected)
            if best is None or row[:3] > best[:3]:
                best = row
    area, axis, sign, selected = best
    if area <= 0 or not selected.any():
        raise ValueError("no continuous horizontal terminal surface")
    order = [a for a in range(3) if a != axis] + [axis]
    aligned = vertices[:, order].copy()
    aligned[:, 2] *= sign
    top_faces = faces[selected]
    top_points = aligned[np.unique(top_faces)]
    low, high = aligned.min(0), aligned.max(0)
    top_low, top_high = top_points[:, :2].min(0), top_points[:, :2].max(0)
    extents = top_high - top_low
    if np.any(extents <= 0.05) or float(np.prod(extents)) < 0.01:
        raise ValueError("support surface is too small")
    return dict(
        up_axis=axis,
        up_sign=sign,
        axis_order=order,
        source_bounds_m=bounds.tolist(),
        aligned_bounds_m=[low.tolist(), high.tolist()],
        top_bounds_xy_m=[top_low.tolist(), top_high.tolist()],
        top_extents_m=extents.tolist(),
        top_area_m2=area,
        vertex_count=len(vertices),
        face_count=len(faces),
    )


def _source_root(asset):
    root = local_path(asset["source_root"]).resolve()
    source = (root / asset["entrypoint"]).resolve()
    if not source.is_relative_to(root):
        raise ValueError("unsafe support asset entrypoint")
    official.verify_files(root, asset["source_files"])
    return root, source


def rank(clip_index, crop_path, category, *, top_k=3, encoder=None):
    if category not in SUPPORTED_CATEGORIES or not 1 <= top_k <= 3:
        raise ValueError("invalid support retrieval request")
    index, vectors = clip.load_index(clip_index)
    encoder = encoder or clip.get_encoder(str(clip.WEIGHTS_DIR.resolve()))
    query = f"{category}, fixed rigid table or counter with a broad horizontal support top"
    text = clip.normalize(encoder.encode_text(query))[0]
    with Image.open(crop_path) as image:
        image_vector = clip.normalize(encoder.encode_images([image.convert("RGB")]))[0]
    semantic_scores = clip.normalize(vectors) @ text
    visual_scores = clip.normalize(vectors) @ image_vector
    grouped = {}
    for row, semantic, visual in zip(
        index["rows"], semantic_scores, visual_scores, strict=True
    ):
        slot = grouped.setdefault(row["asset_id"], {"semantic": [], "visual": [], "views": []})
        slot["semantic"].append(float(semantic))
        slot["visual"].append(float(visual))
        slot["views"].append(row)
    assets = {a["asset_id"]: a for a in index["assets"]}
    candidates, rejected = [], []
    for asset_id, score in grouped.items():
        asset = assets[asset_id]
        if asset.get("model_format") not in SUPPORTED_FORMATS:
            continue
        try:
            root, source = _source_root(asset)
            geometry = geometry_gate(source)
        except Exception as exc:
            rejected.append({"asset_id": asset_id, "reason": str(exc)})
            continue
        semantic, visual = max(score["semantic"]), max(score["visual"])
        combined = 0.4 * semantic + 0.6 * visual
        views = sorted(
            zip(score["visual"], score["views"], strict=True),
            key=lambda row: (-row[0], row[1]["view"]),
        )
        candidates.append(
            dict(
                asset_id=asset_id,
                score=float(combined),
                semantic_score=semantic,
                visual_score=visual,
                asset=asset,
                source_root=str(root),
                model_entrypoint=str(source),
                geometry=geometry,
                selected_views=[row for _, row in views[:2]],
            )
        )
    candidates.sort(key=lambda row: (-row["score"], row["asset_id"]))
    candidates = candidates[:top_k]
    if not candidates:
        raise ValueError("no physically qualified Genesis support candidate")
    numbers = {
        asset_id: i + 1 for i, asset_id in enumerate(sorted(c["asset_id"] for c in candidates))
    }
    for candidate in candidates:
        candidate["candidate_id"] = numbers[candidate["asset_id"]]
    return query, candidates, rejected


def _data_url(path):
    with Image.open(path) as image:
        stream = io.BytesIO()
        image.convert("RGB").save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def choose(candidates, crop_path, preview_root, vlm_config, *, vlm=None):
    config = load_llm_provider_config(vlm_config)
    content = [
        {
            "type": "text",
            "text": (
                "Select the one candidate that best matches the finite support object in the "
                "first image. Reject if none is a desk/table/counter. Return strict JSON: "
                "{status:selected,candidate_id:int,reason:str,visible_differences:[str]} or "
                "{status:rejected,reason:str,visible_differences:[str]}."
            ),
        },
        {"type": "text", "text": "Observed support"},
        {"type": "image_url", "image_url": {"url": _data_url(crop_path), "detail": "high"}},
    ]
    root = Path(preview_root)
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        content.append({"type": "text", "text": f"Candidate {candidate['candidate_id']}"})
        for view in candidate["selected_views"]:
            path = official.safe_file(root, view["image"]["path"])
            official.verify_files(root, [view["image"]])
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(path), "detail": "high"}}
            )
    client = vlm or clip.ChatVisionClient(config)
    raw = client(
        [
            {"role": "system", "content": "You are a conservative 3D support asset matcher."},
            {"role": "user", "content": content},
        ]
    )
    selection = clip.validate_selection(raw, {c["candidate_id"] for c in candidates})
    if selection["status"] != "selected":
        raise ValueError("VLM rejected all support candidates")
    selected = next(c for c in candidates if c["candidate_id"] == selection["candidate_id"])
    return selected, dict(raw_response=raw, selection=selection, config=config.safe_dict())


def target_footprint(observation, foreground_layout, margin=0.05):
    boxes = np.asarray(
        [obj["world_visual_bounds_m"] for obj in foreground_layout["objects"] if not obj["fixed"]],
        float,
    )
    if not len(boxes):
        raise ValueError("no dynamic foreground footprint")
    object_low, object_high = boxes[:, 0, :2].min(0), boxes[:, 1, :2].max(0)
    visible = np.asarray(observation["visible_footprint_world_xy_m"], float)
    low = np.minimum(visible[0], object_low - margin)
    high = np.maximum(visible[1], object_high + margin)
    return dict(
        center_xy_m=((low + high) / 2).tolist(),
        extents_xy_m=(high - low).tolist(),
        bounds_xy_m=[low.tolist(), high.tolist()],
        margin_m=margin,
        axes_evidence=observation["extent_evidence"],
    )


def _rgb_to_lab(rgb):
    value = np.asarray(rgb, float) / 255.0
    value = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    xyz = value @ np.array(
        [[0.4124564, 0.2126729, 0.0193339],
         [0.3575761, 0.7151522, 0.1191920],
         [0.1804375, 0.0721750, 0.9503041]]
    )
    xyz /= [0.95047, 1.0, 1.08883]
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e_ciede2000(rgb1, rgb2):
    """CIEDE2000 for two sRGB colors, kL=kC=kH=1."""
    l1, a1, b1 = _rgb_to_lab(rgb1)
    l2, a2, b2 = _rgb_to_lab(rgb2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar**7 / (cbar**7 + 25**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)
    def hue(a, b):
        return math.degrees(math.atan2(b, a)) % 360 if a or b else 0

    h1, h2 = hue(ap1, b1), hue(ap2, b2)
    dl, dc = l2 - l1, cp2 - cp1
    dh = h2 - h1
    if cp1 * cp2 == 0:
        dh = 0
    elif dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    d_h = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh / 2))
    lbar, cpbar = (l1 + l2) / 2, (cp1 + cp2) / 2
    if cp1 * cp2 == 0:
        hbar = h1 + h2
    elif abs(h1 - h2) <= 180:
        hbar = (h1 + h2) / 2
    elif h1 + h2 < 360:
        hbar = (h1 + h2 + 360) / 2
    else:
        hbar = (h1 + h2 - 360) / 2
    t = (
        1 - 0.17 * math.cos(math.radians(hbar - 30))
        + 0.24 * math.cos(math.radians(2 * hbar))
        + 0.32 * math.cos(math.radians(3 * hbar + 6))
        - 0.20 * math.cos(math.radians(4 * hbar - 63))
    )
    sl = 1 + 0.015 * (lbar - 50) ** 2 / math.sqrt(20 + (lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * cpbar, 1 + 0.015 * cpbar * t
    rt = -2 * math.sqrt(cpbar**7 / (cpbar**7 + 25**7)) * math.sin(
        math.radians(60 * math.exp(-((hbar - 275) / 25) ** 2))
    )
    return math.sqrt(
        (dl / sl) ** 2 + (dc / sc) ** 2 + (d_h / sh) ** 2
        + rt * (dc / sc) * (d_h / sh)
    )


def _mean_color(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((128, 128))).reshape(-1, 3)
    pixels = pixels[np.max(pixels, axis=1) < 248]
    return np.median(pixels, axis=0) if len(pixels) else np.array([180, 180, 180])


def materialize(candidate, footprint, crop_path, preview_root, output_dir):
    """Create one self-contained fixed, collidable standard URDF package."""
    import trimesh

    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(candidate["model_entrypoint"]).resolve()
    mesh = _mesh(source)
    gate = candidate["geometry"]
    vertices = np.asarray(mesh.vertices, float)[:, gate["axis_order"]]
    vertices[:, 2] *= gate["up_sign"]
    faces = np.asarray(mesh.faces, int)
    current = np.asarray(gate["top_extents_m"], float)
    requested = np.asarray(footprint["extents_xy_m"], float)
    xy_scale = np.maximum(requested / current, 1e-6)
    z_scale = math.sqrt(float(xy_scale[0] * xy_scale[1]))
    vertices *= [xy_scale[0], xy_scale[1], z_scale]
    vertices[:, :2] -= (vertices[:, :2].min(0) + vertices[:, :2].max(0)) / 2
    vertices[:, 2] -= vertices[:, 2].max()
    source_color = _mean_color(
        official.safe_file(Path(preview_root), candidate["selected_views"][0]["image"]["path"])
    )
    observed_color = _mean_color(crop_path)
    delta = delta_e_ciede2000(source_color, observed_color)
    visual = mesh.visual.copy()
    transformed = trimesh.Trimesh(
        vertices=vertices, faces=faces, visual=visual, process=False
    )
    if delta > 12:
        transformed.visual.vertex_colors = np.tile(
            np.append(np.asarray(observed_color, np.uint8), 255), (len(vertices), 1)
        )
    out.mkdir(parents=True)
    (out / "urdf").mkdir()
    transformed.export(out / "urdf/support_visual.glb")
    transformed.export(out / "urdf/support_collision.obj")
    (out / "source").mkdir()
    _safe_copy(source, out / "source" / source.name)

    appearance = dict(
        method="preserve_source_material" if delta <= 12 else "deterministic_lab_target",
        ciede2000=float(delta),
        threshold=12.0,
        source_rgb=np.asarray(source_color, int).tolist(),
        observed_rgb=np.asarray(observed_color, int).tolist(),
    )
    official.write_json(
        out / "physics.json", dict(friction=0.8, fixed=True, collision="authored_full_mesh")
    )
    bounds = np.asarray(transformed.bounds, float)
    size = bounds[1] - bounds[0]
    mass = 100.0
    inertia = mass / 12 * np.array(
        [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2,
         size[0] ** 2 + size[1] ** 2]
    )
    robot = ET.Element("robot", name=SUPPORT_ID)
    link = ET.SubElement(robot, "link", name="support")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(
        inertial, "inertia", ixx=str(inertia[0]), ixy="0", ixz="0",
        iyy=str(inertia[1]), iyz="0", izz=str(inertia[2])
    )
    for kind, filename in (
        ("visual", "support_visual.glb"),
        ("collision", "support_collision.obj"),
    ):
        node = ET.SubElement(link, kind)
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "mesh", filename=filename, scale="1 1 1")
    ET.ElementTree(robot).write(out / "urdf/support.urdf", encoding="unicode")
    provenance = dict(
        schema_version=MATERIALIZATION_SCHEMA,
        selected_asset_id=candidate["asset_id"],
        source_sha256=lib.sha256(source),
        geometry_gate=gate,
        footprint=footprint,
        applied_scale_xyz=[float(xy_scale[0]), float(xy_scale[1]), z_scale],
        top_alignment_z_m=0.0,
        appearance=appearance,
        limitations=["hidden dimensions completed from selected Genesis asset"],
    )
    official.write_json(out / "provenance.json", provenance)
    files = [
        official.fingerprint(path, out)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "asset.json"
    ]
    package = dict(
        schema_version="genenv.standard_urdf_asset.v1",
        asset_id=f"media_support_{candidate['asset_id']}",
        category="support",
        entrypoint="urdf/support.urdf",
        physics_file="physics.json",
        files=files,
        source=dict(asset_id=candidate["asset_id"], root=str(candidate["source_root"])),
        path_mapping=[],
    )
    official.write_json(out / "asset.json", package)
    standard.verify_package(out / "asset.json")
    return package, provenance


def augment_scene(base_scene, support_package, observation, output_dir):
    """Replace the infinite plane with support_0 and explicit finite on relations."""
    from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported

    source, package_root, out = map(
        lambda p: Path(p).resolve(), (base_scene, support_package, output_dir)
    )
    if out.exists():
        raise FileExistsError(out)
    foreground = imported.verify(source)
    footprint = target_footprint(observation, foreground)
    shutil.copytree(source, out)
    (out / "manifest.json").unlink()
    target = out / "assets" / SUPPORT_ID
    shutil.copytree(package_root, target)
    package, entry, physics = standard.verify_package(target / "asset.json")
    measured = standard.inspect(entry)
    bounds = official.bounds(measured["visual"])
    center = np.asarray(footprint["center_xy_m"], float)
    translation = [float(center[0]), float(center[1]), 0.0]
    world_bounds = bounds + np.asarray(translation)
    graph = lib.read_json(out / "scene_graph.json")
    layout = lib.read_json(out / "scene_layout.json")
    geometry = lib.read_json(out / "native_geometry.json")
    relations = [
        dict(
            relation="on", source=obj["object_id"], target=SUPPORT_ID,
            evidence="observed finite support contact candidate"
        )
        for obj in layout["objects"] if not obj["fixed"]
    ]
    support = dict(
        object_id=SUPPORT_ID,
        category=observation["category"],
        translation_m=translation,
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        scale=1.0,
        orientation_policy="baked_selected_asset_up_to_world_z",
        intended_dynamic=False,
        fixed=True,
        support=None,
        source_velocity_mps=[0.0, 0.0, 0.0],
        source_angular_velocity_radps=[0.0, 0.0, 0.0],
        asset_id=package["asset_id"],
        model_format="urdf",
        source_root=f"assets/{SUPPORT_ID}",
        source_files=package["files"],
        model_entrypoint=f"assets/{SUPPORT_ID}/{package['entrypoint']}",
        standard_package=f"assets/{SUPPORT_ID}/asset.json",
        standard_package_sha256=lib.sha256(target / "asset.json"),
        local_visual_bounds_m=bounds.tolist(),
        world_visual_bounds_m=world_bounds.tolist(),
        mass_kg=measured["mass"],
        friction=physics["friction"],
    )
    for obj in layout["objects"]:
        if not obj["fixed"]:
            obj["support"] = SUPPORT_ID
    layout["objects"].append(support)
    layout["relations"] = relations
    layout["environment"] = dict(
        ground=None, position_m=[0.0, 0.0, 0.0], z_m=0.0, visible=False,
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0]
    )
    local_polygon = [
        [float(bounds[0, 0]), float(bounds[0, 1])],
        [float(bounds[1, 0]), float(bounds[0, 1])],
        [float(bounds[1, 0]), float(bounds[1, 1])],
        [float(bounds[0, 0]), float(bounds[1, 1])],
    ]
    layout["support_surfaces"] = {
        SUPPORT_ID: {
            "z_m": 0.0,
            "polygon_xy_m": local_polygon,
            "world_bounds_xy_m": footprint["bounds_xy_m"],
            "source": "measured_visible_plus_inferred_completion",
        }
    }
    graph["nodes"].append(
        {
            "object_id": SUPPORT_ID,
            "category": observation["category"],
            "asset_id": package["asset_id"],
        }
    )
    graph["edges"] = relations
    graph["relations_status"] = "observed_primary_support_declared"
    graph["environment"] = layout["environment"]
    geometry[SUPPORT_ID] = dict(
        bounds=bounds.tolist(), world_bounds=world_bounds.tolist(),
        visual_vertex_count=len(measured["visual"]),
        collision_vertex_count=len(measured["collision"])
    )
    layout["scene_graph_sha256"] = clip.digest(graph)
    layout["native_geometry_sha256"] = clip.digest(geometry)
    layout["meaning"] = "authored foreground poses with one finite fixed collidable support"
    for filename, document in (
        ("scene_graph.json", graph),
        ("scene_layout.json", layout),
        ("native_geometry.json", geometry),
        ("support_surfaces.json", layout["support_surfaces"]),
    ):
        official.write_json(out / filename, document)
    official.write_json(
        out / "manifest.json",
        dict(
            schema_version=imported.VERSION,
            files=[
                official.fingerprint(path, out)
                for path in sorted(out.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            ],
        ),
    )
    imported.verify(out)
    return layout, footprint
