"""Rebuild the calibrated drift-car texture from the clean mesh and six reference photos."""

import argparse, json, struct, sys, uuid
from pathlib import Path
import cv2, numpy as np, trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from rembg import new_session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
from photo_finish import camera, raster, interpolate, foreground, unlit

studio = Path(__file__).resolve().parents[1] / "outputs/studio"
parser = argparse.ArgumentParser(
    description="Retexture the saved drift car from a clean GLB and its six original PNGs."
)
parser.add_argument("--clean-glb", type=Path, required=True)
parser.add_argument("--photos", type=Path, required=True)
args = parser.parse_args()
original = args.photos
# Calibrations of these six photos; side wheel hubs and oblique landmarks are below.
poses = [
    {"yaw": 0.0, "elevation": 0.0, "visible_window": [0.0, 1.0, 0.0, 1.0]},
    {"yaw": 180.0, "elevation": -8.0, "visible_window": [0.0, 1.0, 0.0, 1.0]},
    {
        "yaw": 90.0,
        "elevation": 8.0,
        "visible_window": [0.07236350058704119, 0.7105459976518352, 0.0, 1.0],
    },
    {
        "yaw": 282.0,
        "elevation": 0.0,
        "visible_window": [0.0, 0.6963541720958801, 0.0, 1.0],
    },
    {
        "yaw": 90.0,
        "elevation": 90.0,
        "visible_window": [0.035411898047346346, 0.9645881019526537, 0.0, 1.0],
    },
    {"yaw": 48.0, "elevation": 12.0, "visible_window": [0.0, 1.0, 0.0, 1.0]},
]


def folder():
    d = studio / uuid.uuid4().hex
    d.mkdir(parents=True)
    return d


def load(path):
    return trimesh.load(path, force="mesh", process=False)


def doc(path):
    b = path.read_bytes()
    return json.loads(b[20 : 20 + struct.unpack_from("<I", b, 12)[0]])


def record(d, name, mesh):
    (d / "job.json").write_text(
        json.dumps(
            dict(
                id=d.name,
                status="complete",
                stage="Ready to download",
                glb=f"/api/files/{d.name}/model.glb",
                details=dict(
                    faces=len(mesh.faces), variant=name, geometry_source=original.name
                ),
            )
        )
    )


def compare(mesh, base):
    for a, b in [
        (mesh.vertices, base.vertices),
        (mesh.faces, base.faces),
        (mesh.visual.uv, base.visual.uv),
    ]:
        assert np.array_equal(a, b)


clean_file = args.clean_glb
assert not doc(clean_file).get("images") and not doc(clean_file).get("textures")
base = load(clean_file)
images = [
    Image.open(p).convert("RGBA")
    for p in [original / f"input-{i}.png" for i in range(1, 7)]
]
session = new_session("u2net", providers=["CPUExecutionProvider"])
pics = [foreground(im, session)[0] for im in images]
del session
# Rectify the actual pixels of the small photographed arch decal. Remove the
# same ink from the general projection so overlapping cameras cannot double it.
decal_quad = np.float32([[627, 84], [653, 81], [662, 99], [634, 104]]) - [30, 23]
decal_quad = decal_quad.astype("f")
decal_map = cv2.getPerspectiveTransform(
    decal_quad, np.float32([[0, 0], [127, 0], [127, 39], [0, 39]])
)
decal = cv2.warpPerspective(pics[5][:, :, :3], decal_map, (128, 40))
ink = np.clip((decal.min(2).astype("f") - 90) / 85, 0, 1)
mask = np.zeros(pics[5].shape[:2], np.uint8)
cv2.fillConvexPoly(mask, np.rint(decal_quad).astype("int32"), 255)
pics[5] = pics[5].copy()
pics[5][:, :, :3] = cv2.inpaint(pics[5][:, :, :3].copy(), mask, 4, cv2.INPAINT_TELEA)
size = 2048


def paint():
    m = load(clean_file)
    v = m.vertices.astype("f")
    f = m.faces.astype("i")
    n = m.vertex_normals.astype("f")
    uv = m.visual.uv.astype("f").copy()
    uv[:, 1] = 1 - uv[:, 1]
    uv *= size - 1
    face, bc, _ = raster(np.column_stack((uv, np.zeros(len(v), "f"))), f, size, size)
    ys, xs = np.where(face >= 0)
    tri = f[face[ys, xs]]
    pos = interpolate(v, tri, bc[ys, xs])
    norm = interpolate(n, tri, bc[ys, xs])
    accum = np.zeros_like(pos)
    ws = np.zeros(len(pos), "f")
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        v = m.vertices.astype("f").copy()
        src_i = {6: 2, 7: 3, 8: 5}.get(i, i)
        r, pic = poses[src_i], pics[src_i]
        if i in (6, 7):
            # Mirror the observed front wheel onto its unobserved opposite face only.
            v = v.copy()
            v[:, 0] = v[:, 0].min() + v[:, 0].max() - v[:, 0]

        side = i in (2, 3, 6, 7)
        yaw, elev = r["yaw"], r["elevation"]
        if i == 1:
            yaw, elev = 180, 0
        if side:
            yaw, elev = (90 if i in (2, 6) else 270), 0
        q, eye, aspect = camera(v, yaw, elev)
        ph, pw = pic.shape[:2]
        if side:
            # The near front hub in the nose-left crop and near rear hub in
            # the nose-right crop anchor the partial photos (original pixels).
            scale = (ph - 1) * aspect
            low = v[:, 1] < v[:, 1].min() + 0.06 * np.ptp(v[:, 1])
            wheel = (v[:, 2] > 0) if i in (2, 6) else (v[:, 2] < 0)
            hub = np.median(q[low & wheel, 0])
            crop_x = 27 if i in (2, 6) else 74 - 12
            q[:, 0] = (q[:, 0] - hub) * scale + crop_x
            q[:, 1] *= ph - 1
        else:
            x0, sw, y0, sh = r["visible_window"]
            q[:, :2] = (q[:, :2] - [x0, y0]) / [sw, sh] * [pw - 1, ph - 1]
        if i == 8:
            # The oblique image is the only direct observation of the near rear
            # quarter decal. Rectify that small side panel using the ends of two
            # vertical tire radii; the body-wide camera is less accurate here.
            q, eye, _ = camera(v, 90, 0)
            foot = v[:, 1] < v[:, 1].min() + 0.06 * np.ptp(v[:, 1])
            zs = [np.median(v[foot & cond, 2]) for cond in (v[:, 2] > 0, v[:, 2] < 0)]
            hub_y = [
                np.quantile(v[foot & cond, 1], 0.1) + 0.071
                for cond in (v[:, 2] > 0, v[:, 2] < 0)
            ]
            src = np.float32(
                [(z, y + dy) for z, y in zip(zs, hub_y) for dy in (0, 0.075)]
            )
            dst = np.float32([[448, 250], [448, 175], [652, 196], [652, 141]]) - [
                30,
                23,
            ]
            H = cv2.getPerspectiveTransform(src, dst.astype("f"))
            q[:, :2] = cv2.perspectiveTransform(
                v[:, [2, 1]].reshape(-1, 1, 2), H
            ).reshape(-1, 2)
        if i == 5:
            from scipy.optimize import least_squares

            # Solve the perspective camera on the original photo coordinates.
            # Use visible headlight centers, plate center, and the two near wheel hubs.
            fp, _, _ = camera(v, 0, 0)
            fp[:, :2] = fp[:, :2] * [188, 116] + [11, 7]
            owner, bweights, _ = raster(fp, f, 132, 211)
            pts = np.array(
                [[43, 69], [58, 69], [153, 69], [169, 69], [104, 93], [105, 26]]
            )
            yy, xx = pts[:, 1], pts[:, 0]
            xyz = list(interpolate(v, f[owner[yy, xx]], bweights[yy, xx]))
            pts2 = [
                [80, 183],
                [104, 184],
                [271, 191],
                [300, 193],
                [151, 252],
                [232, 78],
            ]
            for front, uv2 in [(True, [448, 250]), (False, [652, 196])]:
                bottom = (
                    (v[:, 1] < v[:, 1].min() + 0.06 * np.ptp(v[:, 1]))
                    & ((v[:, 2] > 0) if front else (v[:, 2] < 0))
                    & (v[:, 0] > 0)
                )
                wz = np.median(v[bottom, 2])
                slab = (np.abs(v[:, 2] - wz) < 0.025) & (v[:, 1] < -0.045)
                xyz.append(
                    [
                        np.quantile(v[slab, 0], 0.99),
                        np.quantile(v[bottom, 1], 0.1) + 0.071,
                        wz,
                    ]
                )
                pts2.append(uv2)
            xyz = np.asarray(xyz, "d")
            pts2 = np.array(pts2, "d")

            def project(par, xyz):
                rr, _ = cv2.Rodrigues(par[:3])
                cp = xyz @ rr.T + par[3:6]
                return cp[:, :2] * np.exp(par[6]) / cp[:, 2:] + [353, 180], cp[:, 2]

            K = np.array([[750.0, 0, 353], [0, 750, 180], [0, 0, 1.0]])
            _, rot, tr = cv2.solvePnP(xyz, pts2, K, None, flags=cv2.SOLVEPNP_EPNP)
            par = least_squares(
                lambda a: (project(a, xyz)[0] - pts2).ravel(),
                np.r_[rot.ravel(), tr.ravel(), np.log(750)],
                loss="soft_l1",
                f_scale=3,
                bounds=([-np.inf] * 6 + [np.log(450)], [np.inf] * 6 + [np.log(2000)]),
            ).x
            uvphoto, depth = project(par, v)
            q = np.column_stack([uvphoto - [30, 23], -depth]).astype("f")
            rr, _ = cv2.Rodrigues(par[:3])
            cam = -rr.T @ par[3:6]
            direction = cam - pos
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True), 1e-7
            )
            print(
                "Perspective errors",
                np.linalg.norm(project(par, xyz)[0] - pts2, axis=1).round(1),
                flush=True,
            )
        _, _, z = raster(q, f, ph, pw)
        qp = interpolate(q, tri, bc[ys, xs])
        u, b = qp[:, :2].T
        ix = np.clip(np.rint(u).astype(int), 0, pw - 1)
        iy = np.clip(np.rint(b).astype(int), 0, ph - 1)
        inside = (u >= 0) & (u <= pw - 1) & (b >= 0) & (b <= ph - 1)
        if i == 5:
            angle = np.clip((norm * direction).sum(1), 0, 1)
        elif i in (6, 7):
            angle = np.clip(-norm @ eye, 0, 1)
        else:
            angle = np.clip(norm @ eye, 0, 1)
        # Accept the visible shell only. Prefer the side profile on door panels,
        # including the curved sheet metal over the rear wheel.
        w = angle**8
        if i in (0, 1):
            # Deep bumper recesses can face away from the averaged vertex normal
            # while still being the first surface seen by the straight-on camera.
            end = (pos[:, 2] > 0.31) if i == 0 else (pos[:, 2] < -0.30)
            center = np.clip((0.18 - np.abs(pos[:, 0])) / 0.035, 0, 1)
            bumper = np.clip((-0.010 - pos[:, 1]) / 0.035, 0, 1) * np.clip(
                (angle - 0.02) / 0.14, 0, 1
            )
            w = np.maximum(w, end * center * bumper * 0.65)
        w *= inside & (pic[iy, ix, 3] > 205) & (np.abs(z[iy, ix] - qp[:, 2]) < 0.012)
        if i in (6, 7):
            wz = np.median(v[low & wheel, 2])
            wheel_area = (
                (pos[:, 2] - wz) ** 2 + (pos[:, 1] + 0.067) ** 2 < 0.083**2
            ) & ((pos[:, 0] < -0.11) if i == 6 else (pos[:, 0] > 0.11))
            w *= wheel_area * 2
        elif i == 8:
            # Limit to the photographed near-side panel and blend its edge.
            region = (
                np.clip((-pos[:, 2] - 0.16) / 0.055, 0, 1)
                * np.clip((pos[:, 1] + 0.018) / 0.022, 0, 1)
                * np.clip((pos[:, 0] - 0.12) / 0.035, 0, 1)
            )
            w *= region * 25
        elif side:
            w *= 6
        elif i == 5:
            w *= 1.2
        ix0 = np.clip(np.floor(u).astype(int), 0, pw - 1)
        iy0 = np.clip(np.floor(b).astype(int), 0, ph - 1)
        ix1 = np.minimum(ix0 + 1, pw - 1)
        iy1 = np.minimum(iy0 + 1, ph - 1)
        du = np.clip(u - ix0, 0, 1)[:, None]
        dv = np.clip(b - iy0, 0, 1)[:, None]
        sample = (pic[iy0, ix0, :3] * (1 - du) + pic[iy0, ix1, :3] * du) * (1 - dv) + (
            pic[iy1, ix0, :3] * (1 - du) + pic[iy1, ix1, :3] * du
        ) * dv
        accum += sample * w[:, None]
        ws += w
    good = ws > 0.0005
    color = accum / np.maximum(ws[:, None], 1e-10)
    # Extend average local finish briefly around occlusion edges, fade to charcoal
    # on distant hidden surfaces. Do not repeat lettering into unseen regions.
    support = np.flatnonzero(good)[::8]
    tree = cKDTree(pos[support])
    bad = ~good
    for inds in np.array_split(np.flatnonzero(bad), 16):
        ds, js = tree.query(pos[inds], k=12)
        fade = np.exp(-((ds[:, 0] / 0.035) ** 2))[:, None]
        color[inds] = np.median(color[support[js]], axis=1) * fade + (1 - fade) * [
            40,
            37,
            42,
        ]
    # Apply the measured lettering once to the visible outer panel over the rear hub.
    side_q, _, _ = camera(m.vertices, 90, 0)
    side_q[:, :2] *= [999, 319]
    _, _, outer = raster(side_q, f, 320, 1000)
    q = interpolate(side_q, tri, bc[ys, xs])
    ix, iy = np.rint(q[:, :2]).astype(int).T
    exposed = np.abs(q[:, 2] - outer[iy, ix]) < 0.012
    zcenter = np.median(
        m.vertices[
            (
                m.vertices[:, 1]
                < m.vertices[:, 1].min() + 0.06 * np.ptp(m.vertices[:, 1])
            )
            & (m.vertices[:, 2] < 0),
            2,
        ]
    )
    u = (zcenter + 0.058 - pos[:, 2]) / 0.116
    vv = (0.050 - pos[:, 1]) / 0.030
    eligible = (
        exposed & (u >= 0) & (u <= 1) & (vv >= 0) & (vv <= 1) & (norm[:, 0] > 0.12)
    )
    dx = np.clip(np.rint(u * 127).astype(int), 0, 127)
    dy = np.clip(np.rint(vv * 39).astype(int), 0, 39)
    opacity = (eligible * ink[dy, dx])[:, None]
    color = color * (1 - opacity) + decal[dy, dx] * opacity
    atlas = np.zeros((size, size, 3), np.uint8)
    atlas[ys, xs] = np.uint8(color.clip(0, 255))
    near = distance_transform_edt(face < 0, return_distances=False, return_indices=True)
    atlas = atlas[near[0], near[1]]
    m.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(atlas),
        baseColorFactor=[255] * 4,
        metallicFactor=0,
        roughnessFactor=1,
    )
    d = folder()
    m.export(d / "model.glb", tree_postprocessor=unlit)
    record(d, "Calibrated drift car", m)
    compare(load(d / "model.glb"), base)
    print(d, "covered UV fraction", float(good.mean()), flush=True)
    return d


def render(d, yaw, elev):
    m = load(d / "model.glb")
    v = m.vertices.astype("f")
    f = m.faces.astype("i")
    atlas = np.asarray(m.visual.material.baseColorTexture.convert("RGB"))
    uv = m.visual.uv.copy()
    uv[:, 1] = 1 - uv[:, 1]
    uv *= size - 1
    q, _, ar = camera(v, yaw, elev)
    wh = np.array([min(480, 170 * ar), min(170, 480 / ar)])
    q[:, :2] = q[:, :2] * wh + ([500, 180] - wh) / 2
    face, bc, _ = raster(q, f, 180, 500)
    hit = face >= 0
    x, y = interpolate(uv, f[face[hit]], bc[hit]).round().astype(int).T
    img = np.full((180, 500, 3), 226, np.uint8)
    img[hit] = atlas[y, x]
    return Image.fromarray(img)


p2 = paint()
sheet = Image.new("RGB", (1500, 820), "white")
pen = ImageDraw.Draw(sheet)
for i, (name, yaw, el) in enumerate(
    [
        ("Front", 0, 0),
        ("Rear", 180, 0),
        ("Left profile", 90, 0),
        ("Right profile", 270, 0),
        ("Roof", 90, 90),
        ("Three-quarter", 48, 12),
    ]
):
    x = i % 3 * 500
    y = i // 3 * 410
    img = images[i].copy()
    img.thumbnail((490, 180))
    sheet.paste(img, (x + (500 - img.width) // 2, y))
    pen.text((x + 10, y + 182), name + " reference", fill="black")
    sheet.paste(render(p2, yaw, el), (x, y + 205))
    pen.text((x + 10, y + 387), "Clean mesh + six photos", fill="black")
sheet.save(p2 / "comparison.jpg", quality=96)
