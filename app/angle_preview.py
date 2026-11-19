"""Loopback-only preview of the source crops found in one reference sheet."""

import asyncio
import base64
import io
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from .view_detection import ANGLES, MODEL, DetectionError, detect_views


router = APIRouter()
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 16 * 1024 * 1024
MAX_PIXELS = 16_000_000
processing = asyncio.Lock()


@router.get("/angle-preview")
def preview_page():
    return FileResponse(STATIC / "angle-preview.html")


def open_source(raw: bytearray) -> Image.Image:
    with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG", "WEBP"]) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a single image of at most 16 megapixels")
        oriented = ImageOps.exif_transpose(image)
        mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
        return oriented.convert(mode)


def render_crops(source: Image.Image, views: list[tuple[str, tuple[int, int, int, int]]]):
    crops = []
    for angle, box in views:
        with io.BytesIO() as buffer:
            with source.crop(box) as crop:
                crop.info.clear()
                crop.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getbuffer()).decode("ascii")
        crops.append({"angle": angle, "box": box, "image": "data:image/png;base64," + encoded})
    return crops


@router.post("/api/angle-preview")
async def separate_sheet(request: Request):
    if processing.locked():
        raise HTTPException(409, "One reference sheet is already being analyzed. Try again when it finishes.")
    async with processing:
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > MAX_UPLOAD:
                raise HTTPException(413, "Choose an image of 16 MiB or less.")
            raw.extend(chunk)
        try:
            source = await run_in_threadpool(open_source, raw)
        except (ValueError, OSError, Image.DecompressionBombError):
            raise HTTPException(400, "Choose a valid single-frame JPG, PNG, or WebP image up to 16 megapixels.")
        del raw
        with source:
            try:
                views = await asyncio.wait_for(detect_views(source), timeout=180)
            except (asyncio.TimeoutError, httpx.TimeoutException):
                raise HTTPException(504, "Local Ollama took too long to analyze this sheet.")
            except httpx.RequestError:
                raise HTTPException(503, "Could not reach local Ollama. Open Ollama and try again.")
            except DetectionError as error:
                raise HTTPException(error.status, str(error))
            crops = await run_in_threadpool(render_crops, source, views)
            detected = {angle for angle, _ in views}
            return {
                "model": MODEL,
                "width": source.width,
                "height": source.height,
                "views": crops,
                "missing": [angle for angle in ANGLES if angle not in detected],
            }
