"""Fully automated local material & wallpaper painter for 3D meshes.

Uses local Ollama (qwen2.5vl:3b) + facebook/sam-vit-base (SAM on MPS) to:
1. Discover distinct surface parts/materials from a reference photo.
2. Segment each part and extract a clean, seamless 2x2 tileable swatch.
3. Render shaded clay views of the 3D mesh, detect the same parts on each
   render with Qwen-VL + SAM, vote labels onto 3D triangles, and smooth.
4. Assign clean box-projected UVs per part and export multi-material GLB + FBX.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab"))
from photo_finish import camera, raster

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
VLM_MODEL = "qwen2.5vl:3b"
SAM_MODEL_ID = "facebook/sam-vit-base"


def _model_image(source: Image.Image) -> tuple[str, tuple[int, int, int, int]]:
    """Resize/pad image onto a 28px-aligned canvas for qwen2.5vl:3b."""
    if source.mode == "RGBA":
        bg = Image.new("RGB", source.size, "#e8eaed")
        bg.paste(source, mask=source.getchannel("A"))
        rgb = bg
    else:
        rgb = source.convert("RGB")
    ratio = max(1 / 4, min(4, rgb.width / rgb.height))
    grid_w = round(math.sqrt(1050 * ratio))
    grid_h = math.ceil(1050 / grid_w)
    width, height = grid_w * 28, grid_h * 28
    scale = min(width / rgb.width, height / rgb.height)
    cw, ch = max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))
    left, top = (width - cw) // 2, (height - ch) // 2
    canvas = Image.new("RGB", (width, height), "#e8eaed")
    canvas.paste(rgb.resize((cw, ch), Image.Resampling.LANCZOS), (left, top))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii"), (left, top, left + cw, top + ch)


def _ask_vlm(image: Image.Image, prompt: str, num_predict: int = 400) -> list[dict]:
    """Query local Ollama qwen2.5vl:3b and parse JSON list response."""
    b64, content = _model_image(image)
    payload = json.dumps({
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "keep_alive": "60s",
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.loads(resp.read())["message"]["content"].strip()

    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    parsed = json.loads(match[1] if match else text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    dx, dy = content[2] - content[0], content[3] - content[1]
    out = []
    for row in parsed:
        if not isinstance(row, dict):
            continue
        coords = row.get("bbox_2d")
        if not isinstance(coords, list):
            continue
        if coords and isinstance(coords[0], list):
            coords = coords[0]
        if len(coords) != 4:
            continue
        x0 = max(0, math.floor((coords[0] - content[0]) * image.width / dx))
        y0 = max(0, math.floor((coords[1] - content[1]) * image.height / dy))
        x1 = min(image.width, math.ceil((coords[2] - content[0]) * image.width / dx))
        y1 = min(image.height, math.ceil((coords[3] - content[1]) * image.height / dy))
        if x1 - x0 >= 12 and y1 - y0 >= 12:
            row = dict(row)
            row["pixel_box"] = (x0, y0, x1, y1)
            out.append(row)
    return out


def _sam_mask(
    image: Image.Image,
    box: tuple[int, int, int, int],
    model: SamModel,
    processor: SamProcessor,
    device: str,
) -> np.ndarray:
    """Segment a single bounding box with SAM and return a boolean mask."""
    rgb = image.convert("RGB")
    raw_inputs = processor(rgb, input_boxes=[[[list(box)]]], return_tensors="pt")
    inputs = {
        k: (v.to(torch.float32) if torch.is_tensor(v) and v.dtype == torch.float64 else v).to(device)
        if torch.is_tensor(v) else v
        for k, v in raw_inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]
    best_idx = int(np.argmax(outputs.iou_scores[0, 0].cpu().numpy()))
    return masks[best_idx].numpy().astype(bool)


def _extract_seamless_swatch(image: Image.Image, mask: np.ndarray, size: int = 512) -> Image.Image:
    """Find the deepest interior patch inside mask and 2x2 mirror-tile it seamlessly."""
    rgb = np.asarray(image.convert("RGB"))
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(int(np.argmax(dist)), dist.shape)
    r = max(16, min(160, int(dist[py, px] * 0.65)))
    y0, y1 = max(0, py - r), min(rgb.shape[0], py + r)
    x0, x1 = max(0, px - r), min(rgb.shape[1], px + r)
    patch = Image.fromarray(rgb[y0:y1, x0:x1])
    half = size // 2
    p00 = patch.resize((half, half), Image.Resampling.LANCZOS)
    p01 = ImageOps.mirror(p00)
    p10 = ImageOps.flip(p00)
    p11 = ImageOps.flip(p01)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(p00, (0, 0))
    canvas.paste(p01, (half, 0))
    canvas.paste(p10, (0, half))
    canvas.paste(p11, (half, half))
    return canvas


def _render_clay_view(
    verts: np.ndarray,
    faces: np.ndarray,
    f_norms: np.ndarray,
    yaw: float,
    elev: float,
    size: int = 512,
) -> tuple[Image.Image, np.ndarray]:
    """Render a shaded clay view and return (PIL RGB, triangle_id_buffer)."""
    proj, direction, aspect = camera(verts, yaw, elev)
    pad = 36
    if aspect >= 1:
        rw = size - 2 * pad
        rh = max(1, int(rw / aspect))
    else:
        rh = size - 2 * pad
        rw = max(1, int(rh * aspect))
    ox, oy = (size - rw) // 2, (size - rh) // 2
    scr = proj.copy()
    scr[:, 0] = scr[:, 0] * (rw - 1) + ox
    scr[:, 1] = scr[:, 1] * (rh - 1) + oy
    fi, _, _ = raster(scr, faces, size, size)
    hit = fi >= 0
    img = np.full((size, size, 3), (225, 228, 232), np.uint8)
    if hit.any():
        shade = np.clip((f_norms[fi[hit]] @ direction) * 0.65 + 0.35, 0.18, 1.0)
        img[hit] = (shade[:, None] * 215).astype(np.uint8)
    return Image.fromarray(img), fi


def _smooth_face_labels(mesh: trimesh.Trimesh, labels: np.ndarray, iters: int = 4) -> np.ndarray:
    """Majority-vote smoothing across mesh face adjacency to remove speckles."""
    adj = mesh.face_adjacency
    n_faces = len(labels)
    n_classes = int(labels.max()) + 1
    cur = labels.copy()
    for _ in range(iters):
        counts = np.zeros((n_faces, n_classes), dtype=np.int32)
        np.add.at(counts, (adj[:, 0], cur[adj[:, 1]]), 1)
        np.add.at(counts, (adj[:, 1], cur[adj[:, 0]]), 1)
        counts[np.arange(n_faces), cur] += 1
        cur = np.argmax(counts, axis=1).astype(np.int32)
    return cur


def _box_uv(sub: trimesh.Trimesh, scale: float = 1.6) -> np.ndarray:
    """Dominant-normal triplanar box UVs."""
    verts = np.asarray(sub.vertices, dtype=np.float32)
    vn = np.asarray(sub.vertex_normals, dtype=np.float32)
    an = np.abs(vn)
    uv = np.zeros((len(verts), 2), dtype=np.float32)
    top = (an[:, 1] >= an[:, 0]) & (an[:, 1] >= an[:, 2])
    side = ~top & (an[:, 0] >= an[:, 2])
    front = ~top & ~side
    uv[top] = verts[top][:, [0, 2]] * scale
    uv[side] = verts[side][:, [2, 1]] * scale
    uv[front] = verts[front][:, [0, 1]] * scale
    return uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=Path, required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    t0 = time.time()
    jid = args.job_id or uuid.uuid4().hex
    out_dir = ROOT / "outputs" / "studio" / jid
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {SAM_MODEL_ID} on {device}...", flush=True)
    processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    sam.eval()

    with Image.open(args.photo) as raw:
        photo = ImageOps.exif_transpose(raw)
        if photo.mode == "RGBA":
            bg = Image.new("RGB", photo.size, "#e8eaed")
            bg.paste(photo, mask=photo.getchannel("A"))
            photo_rgb = bg
        else:
            photo_rgb = photo.convert("RGB")

    # 1. Discover parts & materials from photo
    discover_prompt = (
        "Identify the 3 to 5 distinct surface parts/materials of the main object in this image "
        "(for example: wood_frame, tufted_headboard, bedspread, pillows). "
        "Return a JSON list of objects, each with keys: "
        "name (short snake_case), roughness (float 0.2 to 0.9), and bbox_2d ([x1, y1, x2, y2] tightly enclosing that part)."
    )
    parts_raw = _ask_vlm(photo, discover_prompt, num_predict=500)
    parts = []
    seen_names = set()
    for row in parts_raw:
        name = re.sub(r"[^a-z0-9_]+", "_", str(row.get("name", "part")).lower()).strip("_") or "part"
        if name in seen_names:
            continue
        seen_names.add(name)
        rough = float(np.clip(float(row.get("roughness", 0.7)), 0.15, 0.95))
        mask = _sam_mask(photo_rgb, row["pixel_box"], sam, processor, device)
        if int(mask.sum()) < 256:
            continue
        swatch = _extract_seamless_swatch(photo_rgb, mask, size=512)
        swatch.save(out_dir / f"swatch_{name}.png")
        parts.append({"name": name, "roughness": rough, "box": row["pixel_box"], "swatch": swatch})

    if len(parts) < 2:
        raise RuntimeError(f"Expected >= 2 valid parts from VLM, got {len(parts)}: {parts_raw}")
    part_names = [p["name"] for p in parts]
    print(f"Discovered {len(parts)} parts: {part_names}", flush=True)

    # 2. Load 3D mesh and render 5 clay views
    loaded = trimesh.load(args.shape, process=False)
    mesh = list(loaded.geometry.values())[0] if isinstance(loaded, trimesh.Scene) else loaded
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    f_norms = np.asarray(mesh.face_normals, dtype=np.float32)

    votes = np.zeros((len(faces), len(parts)), dtype=np.float32)
    render_angles = [(0, 12), (180, 12), (90, 12), (-90, 12), (35, 25)]
    locate_prompt = (
        f"This is a 3D clay render of the object. Locate any of these parts visible here: {', '.join(part_names)}. "
        f"Return a JSON list of objects with keys: name (one of {', '.join(part_names)}) and bbox_2d ([x1, y1, x2, y2])."
    )

    for yaw, elev in render_angles:
        clay_img, fi_buf = _render_clay_view(verts, faces, f_norms, yaw, elev, size=512)
        preds = _ask_vlm(clay_img, locate_prompt, num_predict=400)
        print(f"  Render ({yaw}, {elev}) -> {[(p.get('name'), p['pixel_box']) for p in preds]}", flush=True)
        for pred in preds:
            pname = re.sub(r"[^a-z0-9_]+", "_", str(pred.get("name", "")).lower()).strip("_")
            if pname not in part_names:
                continue
            pid = part_names.index(pname)
            rmask = _sam_mask(clay_img, pred["pixel_box"], sam, processor, device)
            hit_tris = fi_buf[rmask & (fi_buf >= 0)]
            if hit_tris.size:
                np.add.at(votes[:, pid], hit_tris, 1.0)

    voted_mask = votes.sum(axis=1) > 0
    if not voted_mask.any():
        raise RuntimeError("No 3D triangles received any VLM+SAM votes")

    face_labels = np.zeros(len(faces), dtype=np.int32)
    face_labels[voted_mask] = np.argmax(votes[voted_mask], axis=1)
    if not voted_mask.all():
        centers = np.asarray(mesh.triangles_center, dtype=np.float32)
        tree = cKDTree(centers[voted_mask])
        _, nn_idx = tree.query(centers[~voted_mask])
        face_labels[~voted_mask] = face_labels[voted_mask][nn_idx]

    face_labels = _smooth_face_labels(mesh, face_labels, iters=4)

    # 3. Export multi-material GLB & FBX
    scene = trimesh.Scene()
    part_counts = {}
    for pid, part in enumerate(parts):
        fidx = np.flatnonzero(face_labels == pid)
        if not fidx.size:
            continue
        sub = mesh.submesh([fidx], append=True, repair=False)
        sub.unmerge_vertices()
        uv = _box_uv(sub, scale=1.6)
        mat = trimesh.visual.material.PBRMaterial(
            name=part["name"],
            baseColorTexture=part["swatch"],
            metallicFactor=0.0,
            roughnessFactor=part["roughness"],
        )
        sub.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        scene.add_geometry(sub, node_name=part["name"], geom_name=part["name"])
        part_counts[part["name"]] = int(len(fidx))
        print(f"  part {part['name']}: {len(fidx)} faces", flush=True)

    out_glb = out_dir / "model.glb"
    scene.export(str(out_glb))
    subprocess.run([str(ROOT / ".fbx-venv/bin/python"), str(ROOT / "app/convert_fbx.py"), str(out_dir)], check=True)

    elapsed = round(time.time() - t0, 1)
    meta = {"parts": [{"name": p["name"], "roughness": p["roughness"], "box": p["box"]} for p in parts], "face_counts": part_counts, "seconds": elapsed}
    (out_dir / "auto_paint_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    job_data = {
        "id": jid,
        "status": "complete",
        "stage": "Ready to download",
        "glb": f"/api/files/{jid}/model.glb",
        "fbx": f"/api/files/{jid}/model.fbx",
        "bundle": f"/api/files/{jid}/model_fbx_with_textures.zip",
        "details": {
            "model": "Auto-Material Paint (Qwen2.5-VL + SAM)",
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "texture_size": 512,
            "watertight": bool(mesh.is_watertight),
            "seconds": elapsed,
        },
    }
    (out_dir / "job.json").write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    print(f"AUTO_PAINT_DONE job={jid} time={elapsed}s", flush=True)


if __name__ == "__main__":
    main()
