"""Lift the finish of a single-atlas MeshPort GLB, preserving its mesh buffers."""

import argparse
from pathlib import Path
import io, struct, json, uuid, sys
import numpy as np
from PIL import Image, ImageDraw
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
from photo_finish import camera, raster, interpolate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "model", type=Path, help="MeshPort GLB with one embedded RGB base-color PNG"
)
args = parser.parse_args()
studio = Path(__file__).resolve().parents[1] / "outputs/studio"
d = studio / uuid.uuid4().hex
d.mkdir(parents=True)


def parse(b):
    n = struct.unpack_from("<I", b, 12)[0]
    return json.loads(b[20 : 20 + n]), b[28 + n :]


source = args.model.read_bytes()
j, data = parse(source)
if (
    len(j.get("images", [])) != 1
    or j["images"][0].get("mimeType") != "image/png"
    or len(j["buffers"]) != 1
):
    raise ValueError("Expected one embedded PNG atlas in a single-buffer GLB")
img_id = j["images"][0]["bufferView"]
view = j["bufferViews"][img_id]
begin = view["byteOffset"]
end = begin + view["byteLength"]
atlas = Image.open(io.BytesIO(data[begin:end]))
if atlas.mode != "RGB":
    raise ValueError("Expected an RGB atlas")
old = np.asarray(atlas)
# Smooth tonal lift. Keep black and white fixed and retain the ordering of shadows.
x = np.arange(256) / 255
curve = np.rint(255 * x + 12 * np.sin(np.pi * x) ** 1.35).clip(0, 255).astype("uint8")
assert np.all(np.diff(curve.astype(int)) >= 0) and np.all(curve[:16] >= np.arange(16))
new = curve[old]
ioimg = io.BytesIO()
Image.fromarray(new).save(ioimg, format="PNG")
png = ioimg.getvalue()
pad = (view["byteLength"] - len(png)) % 4
new_bin = data[:begin] + png + b"\0" * pad + data[end:]
delta = len(new_bin) - len(data)
view["byteLength"] = len(png)
for v in j["bufferViews"]:
    if v["byteOffset"] >= end:
        v["byteOffset"] += delta
j["buffers"][0]["byteLength"] = len(new_bin)
header = json.dumps(j, separators=(",", ":")).encode()
header += b" " * (-len(header) % 4)
dst = d / "model.glb"
dst.write_bytes(
    struct.pack(
        "<4sIIII", b"glTF", 2, 28 + len(header) + len(new_bin), len(header), 0x4E4F534A
    )
    + header
    + struct.pack("<II", len(new_bin), 0x004E4942)
    + new_bin
)
orig, ob = parse(source)
out, nb = parse(dst.read_bytes())
for i, (v, z) in enumerate(zip(orig["bufferViews"], out["bufferViews"])):
    if i != img_id:
        assert (
            ob[v["byteOffset"] : v["byteOffset"] + v["byteLength"]]
            == nb[z["byteOffset"] : z["byteOffset"] + z["byteLength"]]
        )
assert (
    orig["meshes"] == out["meshes"]
    and orig["materials"] == out["materials"]
    and orig["accessors"] == out["accessors"]
)
m = trimesh.load(dst, force="mesh", process=False)
tex = np.array(m.visual.material.baseColorTexture.convert("RGB"))
assert np.array_equal(tex, new)
uv = m.visual.uv.copy()
uv[:, 1] = 1 - uv[:, 1]
uv *= np.array(tex.shape[:2][::-1]) - 1
sheet = Image.new("RGB", (1080, 480), "white")
for i, (yaw, el) in enumerate(
    ((0, 0), (180, 0), (48, 12), (90, 0), (270, 0), (90, 90))
):
    q, _, asp = camera(m.vertices, yaw, el)
    sz = np.array([min(350, 109 * asp), min(109, 350 / asp)])
    q[:, :2] = q[:, :2] * sz + ([360, 120] - sz) / 2
    face, bc, _ = raster(q, m.faces, 120, 360)
    mask = face >= 0
    x, y = interpolate(uv, m.faces[face[mask]], bc[mask]).round().astype(int).T
    for k, a in enumerate((old, tex)):
        frame = np.full((120, 360, 3), 225, np.uint8)
        frame[mask] = a[y, x]
        tile = Image.fromarray(frame)
        ImageDraw.Draw(tile).text(
            (4, 3), "Original" if k == 0 else "Lifted", fill=(60, 60, 60)
        )
        sheet.paste(tile, (i % 3 * 360, (i // 3 * 2 + k) * 120))
sheet.save(d / "contrast-comparison.jpg", quality=95)
job = dict(
    status="complete", stage="Ready to download", details=dict(faces=len(m.faces))
)
job.update(id=d.name, glb=f"/api/files/{d.name}/model.glb")
job["details"].update(variant="Subtle contrast lift", texture_source=args.model.name)
(d / "job.json").write_text(json.dumps(job, indent=2))
print(
    d.name,
    dst.stat().st_size,
    "PNG",
    len(png),
    "curve",
    curve[[0, 8, 16, 32, 64, 96, 128, 180, 220, 255]],
)
