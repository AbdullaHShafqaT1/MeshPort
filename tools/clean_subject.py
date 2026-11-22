"""Isolate a single foreground subject (e.g. a bed) from cluttered room photos.

Uses local Ollama (`qwen2.5vl:3b`) to locate the subject bounding box and
`facebook/sam-vit-base` (SAM) on Apple Silicon MPS to segment only that
subject, stripping walls, floors, wardrobes, nightstands, and reflections.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label as cc_label
from transformers import SamModel, SamProcessor

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
VLM_MODEL = "qwen2.5vl:3b"
SAM_MODEL_ID = "facebook/sam-vit-base"


def _model_image(source: Image.Image) -> tuple[str, tuple[int, int, int, int]]:
    """Resize/pad image onto a 28px-aligned canvas for qwen2.5vl:3b."""
    ratio = max(1 / 4, min(4, source.width / source.height))
    grid_w = round(math.sqrt(1050 * ratio))
    grid_h = math.ceil(1050 / grid_w)
    width, height = grid_w * 28, grid_h * 28
    scale = min(width / source.width, height / source.height)
    cw = max(1, round(source.width * scale))
    ch = max(1, round(source.height * scale))
    left, top = (width - cw) // 2, (height - ch) // 2
    canvas = Image.new("RGB", (width, height), "#f0f0f0")
    canvas.paste(source.convert("RGB").resize((cw, ch), Image.Resampling.LANCZOS), (left, top))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii"), (left, top, left + cw, top + ch)


def _locate_subject_box(image: Image.Image, subject: str) -> tuple[int, int, int, int]:
    """Ask local Ollama qwen2.5vl:3b for a tight bbox_2d around the foreground subject."""
    import urllib.request

    b64, content = _model_image(image)
    prompt = (
        f"Locate ONLY the main foreground {subject} itself in this room photo. "
        f"Return one JSON object with key bbox_2d: [x1, y1, x2, y2] tightly enclosing the {subject} "
        f"(headboard, mattress, bedding, and footboard). "
        f"Strictly EXCLUDE nightstands, side tables, wardrobes, mirrors/reflections, walls, ceiling, and floor."
    )
    payload = json.dumps({
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "keep_alive": "60s",
        "options": {"temperature": 0, "num_ctx": 3072, "num_predict": 150},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.loads(resp.read())["message"]["content"].strip()

    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    parsed = json.loads(match[1] if match else text)
    if isinstance(parsed, list):
        parsed = parsed[0]
    coords = parsed["bbox_2d"]
    if coords and isinstance(coords[0], list):
        coords = coords[0]
    dx, dy = content[2] - content[0], content[3] - content[1]
    x0 = max(0, math.floor((coords[0] - content[0]) * image.width / dx))
    y0 = max(0, math.floor((coords[1] - content[1]) * image.height / dy))
    x1 = min(image.width, math.ceil((coords[2] - content[0]) * image.width / dx))
    y1 = min(image.height, math.ceil((coords[3] - content[1]) * image.height / dy))
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError(f"Degenerate subject box {coords} -> {(x0, y0, x1, y1)}")
    return (x0, y0, x1, y1)


def _segment_with_sam(
    image: Image.Image,
    box: tuple[int, int, int, int],
    model: SamModel,
    processor: SamProcessor,
    device: str,
) -> np.ndarray:
    """Run SAM with a bounding-box prompt and return the best cleaned binary mask."""
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
    scores = outputs.iou_scores[0, 0].cpu().numpy()
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].numpy().astype(bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    mask = binary_fill_holes(mask)
    comp, n_comp = cc_label(mask)
    if n_comp > 1:
        areas = np.bincount(comp.ravel())
        areas[0] = 0
        largest = int(areas.max())
        mask = areas[comp] >= 0.05 * largest
    return mask


def _extract_rgba(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Apply mask, decontaminate border colors, and tight-crop RGBA cutout."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    ys, xs = np.where(mask)
    if not ys.size or not xs.size:
        raise ValueError("Empty segmentation mask")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgba[y0:y1, x0:x1].copy()
    m = crop[:, :, 3] > 0
    if m.any() and not m.all():
        near = distance_transform_edt(~m, return_distances=False, return_indices=True)
        crop[~m, :3] = crop[near[0][~m], near[1][~m], :3]
    return Image.fromarray(crop, mode="RGBA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="bed")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {SAM_MODEL_ID} on {device}...", flush=True)
    processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    model.eval()

    tiles: list[tuple[str, Image.Image]] = []
    for idx, src_path in enumerate(args.inputs, 1):
        with Image.open(src_path) as raw:
            im = ImageOps.exif_transpose(raw).convert("RGB")
        box = _locate_subject_box(im, args.subject)
        mask = _segment_with_sam(im, box, model, processor, device)
        cutout = _extract_rgba(im, mask)
        out_name = f"clean_{idx}_{src_path.stem}.png"
        out_path = args.out_dir / out_name
        cutout.save(out_path, format="PNG")
        print(f"[{idx}/{len(args.inputs)}] {src_path.name} -> {out_name} (box={box}, size={cutout.size})", flush=True)
        tiles.append((out_name, cutout))

    cols = min(3, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    W, H = 480, 360
    sheet = Image.new("RGB", (cols * W, rows * H), (40, 44, 52))
    draw = ImageDraw.Draw(sheet)
    for i, (name, cutout) in enumerate(tiles):
        thumb = cutout.copy()
        thumb.thumbnail((W - 24, H - 48), Image.Resampling.LANCZOS)
        tx, ty = (i % cols) * W, (i // cols) * H
        ox = tx + (W - thumb.width) // 2
        oy = ty + 32 + (H - 36 - thumb.height) // 2
        sheet.paste(thumb, (ox, oy), thumb)
        draw.text((tx + 12, ty + 10), f"{name} ({cutout.width}x{cutout.height})", fill=(255, 220, 120))
    preview_path = args.out_dir / "preview_cutouts.png"
    sheet.save(preview_path)
    print(f"Preview saved: {preview_path}", flush=True)


if __name__ == "__main__":
    main()
