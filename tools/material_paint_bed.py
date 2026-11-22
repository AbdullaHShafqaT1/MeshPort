"""Segment a clean 3D bed mesh into physical regions and apply seamless photo-derived wallpaper materials."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageOps


DEFAULT_SHAPE = Path("outputs/studio/7a748b3626b6437f80d681e19d24592c/shape.glb")
DEFAULT_CLOSEUP = Path(
    "/Users/legacy-as/.gemini/antigravity/brain/bc7e52b9-b49e-4179-a4cd-3b06af782e73"
    "/.user_uploaded/media_1795348137391_ecebfc18.jpg"
)
DEFAULT_FRONT = Path("outputs/bed_cleaned/clean_1_media_1795348137396_7c5332ce.png")


def _seamless_mirror_tile(patch: Image.Image, out_size: int) -> Image.Image:
    """Build a 2x2 mirrored tile so UV wrapping has zero hard seams."""
    half = out_size // 2
    p00 = patch.resize((half, half), Image.Resampling.LANCZOS)
    p01 = ImageOps.mirror(p00)
    p10 = ImageOps.flip(p00)
    p11 = ImageOps.flip(p01)
    canvas = Image.new("RGB", (out_size, out_size))
    canvas.paste(p00, (0, 0))
    canvas.paste(p01, (half, 0))
    canvas.paste(p10, (0, half))
    canvas.paste(p11, (half, half))
    return canvas


def build_swatches(closeup_path: Path, front_path: Path, out_dir: Path) -> dict[str, Image.Image]:
    """Extract clean, tileable fabric & wood swatches directly from the reference photos."""
    with Image.open(closeup_path) as raw:
        cu = ImageOps.exif_transpose(raw).convert("RGB")
    cw, ch = cu.size

    with Image.open(front_path) as raw:
        fr = ImageOps.exif_transpose(raw).convert("RGB")
    fw, fh = fr.size

    # 1. Tufted cream upholstery (central diamond-tufted region of headboard)
    tuft_crop = cu.crop((int(0.35 * cw), int(0.565 * ch), int(0.68 * cw), int(0.72 * ch)))
    tufted = _seamless_mirror_tile(tuft_crop, 1024)

    # 2. Navy floral bedspread wallpaper (clean flat patch from front view center)
    spread_crop = fr.crop((int(0.35 * fw), int(0.41 * fh), int(0.65 * fw), int(0.55 * fh)))
    bedspread = _seamless_mirror_tile(spread_crop, 1024)

    # 3. Floral pillow wallpaper
    pillow_crop = cu.crop((int(0.38 * cw), int(0.78 * ch), int(0.62 * cw), int(0.87 * ch)))
    pillows = _seamless_mirror_tile(pillow_crop, 512)

    # 4. Dark espresso walnut wood grain
    rng = np.random.default_rng(42)
    y_idx, x_idx = np.mgrid[0:512, 0:512]
    grain = (
        np.sin(x_idx * 0.18 + np.sin(y_idx * 0.03) * 3.0) * 4.5
        + np.sin(x_idx * 0.55 + y_idx * 0.02) * 2.0
        + rng.normal(0, 1.5, (512, 512))
    )
    wood_arr = np.zeros((512, 512, 3), dtype=np.float32)
    wood_arr[:, :, 0] = 38.0 + grain
    wood_arr[:, :, 1] = 22.0 + grain * 0.65
    wood_arr[:, :, 2] = 16.0 + grain * 0.45
    wood = Image.fromarray(np.clip(wood_arr, 0, 255).astype(np.uint8), mode="RGB")

    swatches = {
        "tufted_upholstery": tufted,
        "bedspread": bedspread,
        "pillows": pillows,
        "wood_frame": wood,
    }
    for name, img in swatches.items():
        img.save(out_dir / f"swatch_{name}.png")
    return swatches


def _profile_lookup(xs: np.ndarray, ys: np.ndarray, qx: np.ndarray, bins: int = 40) -> np.ndarray:
    """Compute smooth upper silhouette Y(X) for a structural slice (headboard or footboard)."""
    x_min, x_max = float(xs.min()), float(xs.max())
    edges = np.linspace(x_min - 1e-4, x_max + 1e-4, bins + 1)
    prof = np.full(bins, float(np.median(ys)), dtype=np.float32)
    for i in range(bins):
        sel = (xs >= edges[i]) & (xs <= edges[i + 1])
        if sel.any():
            prof[i] = float(ys[sel].max())
    idx = np.clip(np.searchsorted(edges[1:-1], qx), 0, bins - 1)
    return prof[idx]


def segment_bed_faces(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    """Assign each triangle index to wood_frame, tufted_upholstery, pillows, or bedspread."""
    c = np.asarray(mesh.triangles_center, dtype=np.float32)
    n = np.asarray(mesh.face_normals, dtype=np.float32)
    cx, cy, cz = c[:, 0], c[:, 1], c[:, 2]
    nz = n[:, 2]

    hb_mask = cz < -0.77
    fb_mask = cz > 0.78
    hb_top = _profile_lookup(cx[hb_mask], cy[hb_mask], cx) if hb_mask.any() else np.zeros_like(cy)
    fb_top = _profile_lookup(cx[fb_mask], cy[fb_mask], cx) if fb_mask.any() else np.zeros_like(cy)

    labels = np.full(len(c), -1, dtype=np.int32)
    # 0 = wood_frame, 1 = tufted_upholstery, 2 = pillows, 3 = bedspread

    # Legs / under-frame
    labels[cy < -0.46] = 0

    # Headboard: wood frame around back, sides, top crown, and lower rail; tufted inside front
    hb_wood = hb_mask & (
        (nz < 0.25)
        | (np.abs(cx) > 0.78)
        | (cy > hb_top - 0.085)
        | (cy < -0.15)
    )
    hb_tuft = hb_mask & ~hb_wood & (labels < 0)
    labels[hb_wood] = 0
    labels[hb_tuft] = 1

    # Footboard: wood frame around top crown, sides, bottom rail, and inner back; tufted outer front
    fb_wood = fb_mask & (
        (nz < -0.35)
        | (np.abs(cx) > 0.79)
        | (cy > fb_top - 0.065)
        | (cy < -0.44)
    )
    fb_tuft = fb_mask & (nz >= 0.25) & ~fb_wood & (labels < 0)
    labels[fb_wood] = 0
    labels[fb_tuft] = 1

    # Pillows: raised mound right in front of the headboard
    pillow_mask = (labels < 0) & (cz >= -0.77) & (cz <= -0.38) & (cy > -0.095)
    labels[pillow_mask] = 2

    # Everything else on the mattress / side drapes -> bedspread
    labels[labels < 0] = 3

    names = ("wood_frame", "tufted_upholstery", "pillows", "bedspread")
    return {name: np.flatnonzero(labels == i) for i, name in enumerate(names) if (labels == i).any()}


def _compute_part_uv(sub: trimesh.Trimesh, part_name: str) -> np.ndarray:
    """Compute clean planar or box-projected UVs ('wallpaper' wrapping) for a sub-mesh."""
    verts = np.asarray(sub.vertices, dtype=np.float32)
    if part_name == "tufted_upholstery":
        # Map headboard & footboard cushions independently so each fits the tufted grid cleanly
        uv = np.zeros((len(verts), 2), dtype=np.float32)
        for mask in (verts[:, 2] < 0, verts[:, 2] >= 0):
            if not mask.any():
                continue
            vx, vy = verts[mask, 0], verts[mask, 1]
            u = (vx - vx.min()) / max(float(vx.max() - vx.min()), 1e-4)
            v = (vy - vy.min()) / max(float(vy.max() - vy.min()), 1e-4)
            uv[mask, 0] = u * 1.0
            uv[mask, 1] = v * (0.65 if verts[mask, 2].mean() > 0 else 1.0)
        return uv

    if part_name == "pillows":
        vx, vz = verts[:, 0], verts[:, 2]
        u = (vx - vx.min()) / max(float(vx.max() - vx.min()), 1e-4) * 2.0
        v = (vz - vz.min()) / max(float(vz.max() - vz.min()), 1e-4) * 1.0
        return np.stack([u, v], axis=1)

    if part_name == "bedspread":
        # Smooth top-down + side-drape box projection scaled in world units
        # Split vertices per-face isn't needed if we use triplanar dominant axis on vertex normals
        vn = np.asarray(sub.vertex_normals, dtype=np.float32)
        an = np.abs(vn)
        uv = np.zeros((len(verts), 2), dtype=np.float32)
        top = an[:, 1] >= 0.45
        side = ~top & (an[:, 0] >= an[:, 2])
        front = ~top & ~side
        scale = 1.6
        uv[top] = verts[top][:, [0, 2]] * scale
        uv[side] = verts[side][:, [2, 1]] * scale
        uv[front] = verts[front][:, [0, 1]] * scale
        return uv

    # wood_frame: box projection
    vn = np.asarray(sub.vertex_normals, dtype=np.float32)
    an = np.abs(vn)
    uv = np.zeros((len(verts), 2), dtype=np.float32)
    top = (an[:, 1] >= an[:, 0]) & (an[:, 1] >= an[:, 2])
    side = ~top & (an[:, 0] >= an[:, 2])
    front = ~top & ~side
    uv[top] = verts[top][:, [0, 2]] * 2.5
    uv[side] = verts[side][:, [2, 1]] * 2.5
    uv[front] = verts[front][:, [0, 1]] * 2.5
    return uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=Path, default=DEFAULT_SHAPE)
    parser.add_argument("--closeup", type=Path, default=DEFAULT_CLOSEUP)
    parser.add_argument("--front", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    jid = args.job_id or uuid.uuid4().hex
    out_dir = Path("outputs/studio") / jid
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(args.shape, process=False)
    mesh = list(loaded.geometry.values())[0] if isinstance(loaded, trimesh.Scene) else loaded

    swatches = build_swatches(args.closeup, args.front, out_dir)
    regions = segment_bed_faces(mesh)

    mat_props = {
        "wood_frame": dict(metallicFactor=0.05, roughnessFactor=0.38),
        "tufted_upholstery": dict(metallicFactor=0.0, roughnessFactor=0.78),
        "pillows": dict(metallicFactor=0.0, roughnessFactor=0.85),
        "bedspread": dict(metallicFactor=0.0, roughnessFactor=0.85),
    }

    scene = trimesh.Scene()
    for part_name, face_idx in regions.items():
        sub = mesh.submesh([face_idx], append=True, repair=False)
        # Unweld vertices for clean planar/box UV seams
        sub.unmerge_vertices()
        uv = _compute_part_uv(sub, part_name)
        mat = trimesh.visual.material.PBRMaterial(
            name=part_name,
            baseColorTexture=swatches[part_name],
            **mat_props[part_name],
        )
        sub.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        scene.add_geometry(sub, node_name=part_name, geom_name=part_name)
        print(f"  part {part_name}: {len(sub.faces)} faces", flush=True)

    out_glb = out_dir / "model.glb"
    scene.export(str(out_glb))

    subprocess.run([".fbx-venv/bin/python", "app/convert_fbx.py", str(out_dir)], check=True)

    job_data = {
        "id": jid,
        "status": "complete",
        "stage": "Ready to download",
        "glb": f"/api/files/{jid}/model.glb",
        "fbx": f"/api/files/{jid}/model.fbx",
        "bundle": f"/api/files/{jid}/model_fbx_with_textures.zip",
        "details": {
            "model": "Hunyuan3D-2mv + Region Wallpaper Paint",
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "texture_size": 1024,
            "watertight": bool(mesh.is_watertight),
        },
    }
    (out_dir / "job.json").write_text(json.dumps(job_data, indent=2), encoding="utf-8")
    print(f"JOB_ID={jid}", flush=True)


if __name__ == "__main__":
    main()
