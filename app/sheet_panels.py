"""Find panels separated by clear, solid gutters in a contact sheet."""

import re
from collections import deque

from PIL import Image


Box = tuple[int, int, int, int]
MAX_PANELS = 8


def blank_runs(image: Image.Image) -> list[tuple[int, int]]:
    width, height = image.size
    data = image.tobytes()
    blank = bytearray()
    for y in range(height):
        row = data[y * width:(y + 1) * width]
        blank.append(max(row) - min(row) <= 8)
    return [match.span() for match in re.finditer(rb"\x01+", blank)]


def split_gutters(source: Image.Image) -> tuple[bool, list[Box]]:
    """Return whether the image is blank and any separate, unambiguous panels."""
    scale = min(1.0, 500 / max(source.size))
    size = tuple(max(1, round(side * scale)) for side in source.size)
    preview = source.resize(size, Image.Resampling.BILINEAR)
    if preview.mode == "RGBA":
        background = Image.new("RGB", size, "#f0f0f0")
        background.paste(preview, mask=preview.getchannel("A"))
        preview = background
    preview = preview.convert("L")
    darkest, lightest = preview.getextrema()
    if lightest - darkest <= 8:
        return True, []

    pending = deque([(0, 0, *size)])
    panels = []
    while pending and len(pending) + len(panels) < MAX_PANELS:
        left, top, right, bottom = pending.popleft()
        tile = preview.crop((left, top, right, bottom))
        width, height = tile.size
        splits = []
        for axis, view in enumerate((tile.transpose(Image.Transpose.TRANSPOSE), tile)):
            length = (width, height)[axis]
            for start, end in blank_runs(view):
                if start >= 0.22 * length and end <= 0.78 * length and end - start >= max(3, 0.009 * length):
                    splits.append(((end - start) / length, axis, start, end))
        if not splits:
            panels.append((left, top, right, bottom))
            continue
        _, axis, start, end = max(splits)
        if axis == 0:
            children = [(left, top, left + start, bottom), (left + end, top, right, bottom)]
        else:
            children = [(left, top, right, top + start), (left, top + end, right, bottom)]
        # Process children breadth-first so no region can consume the full panel budget.
        pending.extend(children)

    panels.extend(pending)
    if len(panels) <= 1:
        return False, []
    sx, sy = source.width / size[0], source.height / size[1]
    return False, [
        (round(left * sx), round(top * sy), round(right * sx), round(bottom * sy))
        for left, top, right, bottom in panels
    ]
