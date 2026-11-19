"""Locate views with the locally installed Qwen vision model."""

import base64
import io
import json
import math
import re

import httpx
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .sheet_panels import Box, split_gutters


MODEL = "qwen2.5vl:3b"
OLLAMA = "http://127.0.0.1:11434"
ANGLES = ("front", "rear", "left", "right", "top", "bottom")
EXTRA_ANGLES = ("front_three_quarter", "rear_three_quarter", "oblique", "unknown")
ORIENTED_ANGLES = tuple(f'{end}_three_quarter_nose_{side}' for end in ('front','rear') for side in ('left','right'))
VIEW_LABELS = {
    "front": "front", "rear": "rear", "back": "rear",
    "left": "left", "left side": "left", "right": "right", "right side": "right",
    "top": "top", "overhead": "top", "bottom": "bottom", "underside": "bottom",
    "underbody": "bottom", "front three quarter": "front_three_quarter",
    "rear three quarter": "rear_three_quarter", "oblique": "oblique", "perspective": "oblique",
    "three quarter": "oblique", "uncaptioned perspective": "oblique",
    "nose left": "left", "nose right": "right", "side": "unknown", "unknown": "unknown",
}
View = tuple[str, Box]

TILE_PROMPT = (
    "Identify the single object viewpoint in this tile. Read the angle heading when visible. "
    "If it shows the front and a side together, label front three quarter. If it shows the rear and a side "
    "together, label rear three quarter. For a car in true side profile, report nose left or nose right according "
    "to where its nose points in this image. Other labels are front, rear, top, bottom, unknown. "
    "Return one JSON object with the tight complete bbox_2d around the object and the label. "
    "If there is no object, return []. Exclude the caption from the bounding box."
)
SHEET_PROMPT = (
    "Locate each distinct depiction of the main subject. Output each tight bbox_2d and its view label "
    "in JSON format. Include the uncaptioned perspective view and captioned views. Label via the caption "
    "if present and the actual direction shown otherwise. Exclude text and graphics from each bounding box."
)
HEADING_PROMPT = "Locate each object, report bbox_2d in JSON format, and label each with its visible heading or perspective."
PERSPECTIVE_PROMPT = (
    "Locate the largest uncaptioned depiction of the main subject. "
    "Return its tight bounding box as bbox_2d and angle as label in JSON format."
)


class DetectionError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def angle_of(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip().replace("3 4", "three quarter")
    return VIEW_LABELS.get(text.removesuffix(" view"), "")


def rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        result = Image.new("RGB", image.size, "#f0f0f0")
        result.paste(image, mask=image.getchannel("A"))
        return result
    return image


def encode(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getbuffer()).decode("ascii")


def model_image(source: Image.Image) -> tuple[str, Box]:
    # Ollama uses a 28 px patch stride with a 1024-token image floor for this model.
    # Keep the canvas above that floor so predicted pixel coordinates map to it.
    ratio = max(1 / 4, min(4, source.width / source.height))
    grid_width = round(math.sqrt(1050 * ratio))
    grid_height = math.ceil(1050 / grid_width)
    width, height = grid_width * 28, grid_height * 28
    if ratio == source.width / source.height:
        canvas = rgb(source.resize((width, height), Image.Resampling.LANCZOS))
        return encode(canvas), (0, 0, width, height)
    scale = min(width / source.width, height / source.height)
    content_width = max(1, round(source.width * scale))
    content_height = max(1, round(source.height * scale))
    left, top = (width - content_width) // 2, (height - content_height) // 2
    canvas = Image.new("RGB", (width, height), "#f0f0f0")
    canvas.paste(rgb(source.resize((content_width, content_height), Image.Resampling.LANCZOS)), (left, top))
    return encode(canvas), (left, top, left + content_width, top + content_height)


def model_list(content: str) -> list[dict]:
    # Some otherwise accurate replies wrap their JSON in a Markdown fence.
    content = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    result = json.loads(match[1] if match else content)
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list) or any(not isinstance(row, dict) for row in result):
        raise ValueError("Expected a JSON list of objects")
    return result


async def vision(
    client: httpx.AsyncClient, image: str, prompt: str, keep_alive: str | int = 0, num_predict: int = 512
) -> list[dict]:
    response = await client.post(
        "/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [image]}],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_ctx": 3072, "num_predict": num_predict},
        },
    )
    if response.status_code != 200:
        raise DetectionError(502, "Local Ollama could not run the vision model. Check Ollama and try again.")
    try:
        output = response.json()
        if output.get("done_reason") == "length":
            raise ValueError("Response reached the token limit")
        return model_list(output["message"]["content"])
    except (AttributeError, KeyError, TypeError, ValueError):
        raise DetectionError(502, "The local model returned an unreadable view list. Try a clearer reference sheet.")


def intersection(a: Box, b: Box) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


def source_box(coords: list, content: Box, size: tuple[int, int]) -> Box | None:
    if len(coords) != 4 or any(type(value) is not int or abs(value) > 2 * max(content[2:]) for value in coords):
        return None
    left, top, right, bottom = coords
    if right - left < 20 or bottom - top < 20 or intersection(tuple(coords), content) < 0.85 * area(tuple(coords)):
        return None
    width, height = size
    dx, dy = content[2] - content[0], content[3] - content[1]
    x_pad, y_pad = max(3, round(0.045 * (right - left))), max(3, round(0.065 * (bottom - top)))
    return (
        max(0, math.floor((left - x_pad - content[0]) * width / dx)),
        max(0, math.floor((top - y_pad - content[1]) * height / dy)),
        min(width, math.ceil((right + x_pad - content[0]) * width / dx)),
        min(height, math.ceil((bottom + y_pad - content[1]) * height / dy)),
    )


def grounded_views(rows: list[dict], content: Box, size: tuple[int, int]) -> list[View]:
    candidates = []
    for row in rows:
        label, coords = row.get("label", row.get("view")), row.get("bbox_2d")
        if not isinstance(label, str) or not isinstance(coords, list):
            continue
        angle, box = angle_of(label), source_box(coords, content, size)
        if angle and box:
            candidates.append((angle, box))
    # Prefer specific, non-overlapping detections over a large box spanning several views.
    views = []
    for angle, box in sorted(candidates, key=lambda view: area(view[1])):
        if angle in ANGLES and any(angle == previous for previous, _ in views):
            continue
        if any(intersection(box, prior) > 0.35 * min(area(box), area(prior)) for _, prior in views):
            continue
        views.append((angle, box))
    return views[:12]


def add_perspective(views: list[View], rows: list[dict], content: Box, size: tuple[int, int]) -> None:
    # The second pass can recover an uncropped hero image from a captioned sheet.
    # Require a single independent region, clearly larger than the captioned crops.
    unassigned = []
    for row in rows:
        coords = row.get("bbox_2d")
        box = source_box(coords, content, size) if isinstance(coords, list) else None
        if box and all(intersection(box, seen) < 0.1 * min(area(box), area(seen)) for _, seen in views):
            unassigned.append(box)
    if len(unassigned) == 1 and area(unassigned[0]) > 1.5 * max(area(box) for _, box in views):
        views.append(("oblique", unassigned[0]))


async def detect_panels(client: httpx.AsyncClient, source: Image.Image, panels: list[Box]) -> list[View]:
    views = []
    for index, panel in enumerate(panels):
        with source.crop(panel) as tile:
            image, content = await run_in_threadpool(model_image, tile)
            rows = await vision(
                client, image, TILE_PROMPT, keep_alive="30s" if index < len(panels) - 1 else 0, num_predict=150,
            )
            localized = grounded_views(rows, content, tile.size)
            if localized:
                angle, (left, top, right, bottom) = max(localized, key=lambda view: area(view[1]))
                x, y = panel[:2]
                views.append((angle, (left + x, top + y, right + x, bottom + y)))
    return views


async def detect_sheet(client: httpx.AsyncClient, source: Image.Image) -> list[View]:
    image, content = await run_in_threadpool(model_image, source)
    rows = await vision(client, image, SHEET_PROMPT, keep_alive="30s")
    views = grounded_views(rows, content, source.size)
    canonical = sum(angle in ANGLES for angle, _ in views)
    if 2 <= canonical < 5:
        try:
            rows = await vision(client, image, HEADING_PROMPT, keep_alive="30s")
        except (DetectionError, httpx.RequestError):
            pass
        else:
            alternate = grounded_views(rows, content, source.size)
            if sum(angle in ANGLES for angle, _ in alternate) > canonical:
                # Keep a tighter primary crop only when both passes agree on its location and extent.
                primary = dict(views)
                refined = []
                for angle, box in alternate:
                    existing = primary.get(angle)
                    if existing and area(existing) <= area(box) and intersection(existing, box) >= 0.65 * area(box):
                        box = existing
                    refined.append((angle, box))
                views = refined
    if len(views) >= 3 and all(angle in ANGLES for angle, _ in views):
        try:
            rows = await vision(client, image, PERSPECTIVE_PROMPT)
        except (DetectionError, httpx.RequestError):
            pass
        else:
            add_perspective(views, rows, content, source.size)
    return views


async def identify_angle(source: Image.Image, client: httpx.AsyncClient) -> str:
    image, _ = await run_in_threadpool(model_image, source)
    rows = await vision(client,image,
        'Classify the camera view of the single main object. Use a readable angle caption when present. '
        'If the front bumper AND the side doors are clearly visible, output front three quarter. '
        'If the rear bumper AND side doors are clearly visible, output rear three quarter. '
        'Use front or rear only for a head-on or straight rear view. For a profile use nose left or nose right. '
        'Other valid labels: top, bottom, unknown. Return one JSON object with keys label and nose. '
        'For a vehicle, nose is the horizontal side of the image containing its front bumper: left, right, or center. '
        'Use unknown for nonvehicles.',
        keep_alive='30s',num_predict=50)
    label = rows[0].get('label') if len(rows)==1 else None
    angle = (angle_of(label) if isinstance(label,str) else '') or 'unknown'
    nose = rows[0].get('nose') if len(rows)==1 else None
    if angle.endswith('three_quarter') and nose in ('left','right'):
        return f'{angle}_nose_{nose}'
    return angle


async def detect_views(source: Image.Image) -> list[View]:
    blank, panels = await run_in_threadpool(split_gutters, source)
    if blank:
        return []
    async with httpx.AsyncClient(base_url=OLLAMA, timeout=httpx.Timeout(180, connect=4), trust_env=False) as client:
        tags = await client.get("/api/tags", timeout=4)
        if tags.status_code != 200:
            raise DetectionError(503, "Local Ollama did not list its installed models.")
        try:
            installed = any(
                model.get("name") == MODEL and not model.get("remote_host") and not model.get("remote_model")
                for model in tags.json()["models"]
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            raise DetectionError(503, "Local Ollama returned an unreadable model list.")
        if not installed:
            raise DetectionError(503, f"Local Ollama needs the installed {MODEL} model for this preview.")
        if panels:
            views = await detect_panels(client, source, panels)
        else:
            views = await detect_sheet(client, source)
    return sorted(views, key=lambda view: (ANGLES + EXTRA_ANGLES).index(view[0]))
