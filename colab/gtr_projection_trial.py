"""GT-R projection trial for the saved Hunyuan mesh and canonical crops.

Positions and normals use Blender coordinates: Z up, front facing -Y.
Registration boxes are measured on the original, unscaled GT-R crops.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter


VIEWS = (
    # image, screen horizontal, screen down, direction toward camera, crop bounds
    (1, (0, -1), (2, -1), (1, -1), (23, 20, 413, 264)),
    (2, (0, 1), (2, -1), (1, 1), (21, 18, 415, 270)),
    (3, (1, 1), (2, -1), (0, -1), (22, 13, 655, 198)),
    (4, (1, -1), (2, -1), (0, 1), (33, 13, 685, 201)),
    (5, (1, 1), (0, 1), (2, 1), (24, 12, 628, 276)),
)


def fit_proportions(vertices):
    """Fit one shared width and height to all five registered silhouettes."""
    ratios = []
    system = []
    for _, horizontal, vertical, _, (left, top, right, bottom) in VIEWS:
        row = np.zeros(3)
        row[horizontal[0]] = 1
        row[vertical[0]] = -1
        system.append(row[[0, 2]])  # Keep the vehicle length as the scale anchor.
        ratios.append(np.log((right - left) / (bottom - top)))
    width, height = np.exp(np.linalg.lstsq(system, ratios, rcond=None)[0])
    extents = np.ptp(vertices, axis=0)
    scales = np.array([width * extents[1] / extents[0], 1, height * extents[1] / extents[2]])
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    return (vertices - center) * scales + center, scales


def project_colors(vertices, normals, directory):
    """Blend near-facing, visible reference colors into a vertex-color layer."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    normalized = (vertices - lo) / (hi - lo)
    colors = np.zeros_like(vertices, dtype=np.float32)
    coverage = np.zeros(len(vertices), dtype=np.float32)
    for index, horizontal, vertical, facing, box in VIEWS:
        with Image.open(directory / f"input-{index}.png") as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        x_axis, x_sign = horizontal
        y_axis, y_sign = vertical
        depth_axis, depth_sign = facing
        left, top, right, bottom = box
        u = normalized[:, x_axis] if x_sign > 0 else 1 - normalized[:, x_axis]
        v = normalized[:, y_axis] if y_sign > 0 else 1 - normalized[:, y_axis]
        x = np.rint(left + u * (right - left)).astype(int)
        y = np.rint(top + v * (bottom - top)).astype(int)
        depth = normalized[:, depth_axis] * depth_sign
        depth_map = np.full(pixels.shape[:2], -np.inf, dtype=np.float32)
        np.maximum.at(depth_map, (y, x), depth)
        visible = depth >= maximum_filter(depth_map, size=5)[y, x] - 0.065
        weight = np.maximum(normals[:, depth_axis] * depth_sign, 0) ** 5
        sampled = pixels[y, x]
        # Reject strongly red background, while retaining saturated red lamps.
        background = (
            (sampled[:, 0] < 0.24)
            & (sampled[:, 0] > 1.3 * sampled[:, 1] + 0.015)
        )
        weight *= visible & ~background
        colors += weight[:, None] * sampled
        coverage += weight
    mapped = coverage > 0.001
    colors[mapped] /= coverage[mapped, None]
    colors[~mapped] = (0.17, 0.17, 0.17)
    return colors, mapped
