"""Register foreground photos to a generated textured mesh and refine its UV atlas."""
import math

import cv2
import numpy as np
from numba import njit
from PIL import Image
from scipy.ndimage import distance_transform_edt, label as cc_label
from scipy.spatial import cKDTree

CARDINAL_VIEWS = {'front', 'rear', 'left', 'right', 'top', 'bottom'}


@njit
def raster(screen, faces, height, width):
    """Nearest triangle and barycentric coordinates at each covered pixel."""
    owner = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 2), np.float32)
    depth = np.full((height, width), -np.inf, np.float32)
    for fi, (i, j, k) in enumerate(faces):
        ax, ay, az = screen[i]
        bx, by, bz = screen[j] - screen[i]
        cx, cy, cz = screen[k] - screen[i]
        det = bx*cy - by*cx
        if abs(det) < 1e-9:
            continue
        for y in range(max(0, math.ceil(ay+min(0., by, cy))), min(height, math.floor(ay+max(0., by, cy))+1)):
            for x in range(max(0, math.ceil(ax+min(0., bx, cx))), min(width, math.floor(ax+max(0., bx, cx))+1)):
                u = ((x-ax)*cy-(y-ay)*cx)/det
                v = (bx*(y-ay)-by*(x-ax))/det
                z = az+u*bz+v*cz
                if u >= 0 and v >= 0 and u+v <= 1 and z > depth[y,x]:
                    owner[y,x] = fi
                    bary[y,x,0], bary[y,x,1] = u,v
                    depth[y,x] = z
    return owner, bary, depth


@njit
def raster_uv(uv, faces, height, width):
    """Conservative UV triangle rasterizer with a 1-texel border skirt for thin islands."""
    owner = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 2), np.float32)
    exact = np.zeros((height, width), np.uint8)
    best_dist = np.full((height, width), np.inf, np.float32)
    for fi, (i, j, k) in enumerate(faces):
        ax, ay = uv[i, 0], uv[i, 1]
        bx, by = uv[j, 0] - ax, uv[j, 1] - ay
        cx, cy = uv[k, 0] - ax, uv[k, 1] - ay
        det = bx * cy - by * cx
        if abs(det) < 1e-9:
            continue
        y0 = max(0, math.floor(ay + min(0.0, by, cy)) - 1)
        y1 = min(height, math.ceil(ay + max(0.0, by, cy)) + 2)
        x0 = max(0, math.floor(ax + min(0.0, bx, cx)) - 1)
        x1 = min(width, math.ceil(ax + max(0.0, bx, cx)) + 2)
        for y in range(y0, y1):
            for x in range(x0, x1):
                u = ((x - ax) * cy - (y - ay) * cx) / det
                v = (bx * (y - ay) - by * (x - ax)) / det
                if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                    owner[y, x] = fi
                    bary[y, x, 0], bary[y, x, 1] = u, v
                    exact[y, x] = 1
                    best_dist[y, x] = 0.0
                elif exact[y, x] == 0 and u >= -0.35 and v >= -0.35 and u + v <= 1.35:
                    uc = max(0.0, u)
                    vc = max(0.0, v)
                    uv_sum = uc + vc
                    if uv_sum > 1.0:
                        uc /= uv_sum
                        vc /= uv_sum
                    dist = (u - uc) ** 2 + (v - vc) ** 2
                    if dist < best_dist[y, x]:
                        owner[y, x] = fi
                        bary[y, x, 0], bary[y, x, 1] = uc, vc
                        best_dist[y, x] = dist
    return owner, bary


def camera(vertices, yaw, elevation):
    a,b = np.deg2rad([yaw,elevation])
    right = np.array([np.cos(a),0,-np.sin(a)])
    direction = np.array([np.sin(a)*np.cos(b),np.sin(b),np.cos(a)*np.cos(b)])
    basis = np.stack([right,np.cross(right,direction),direction]).astype('f')
    projected = vertices @ basis.T
    extent = np.maximum(np.ptp(projected[:,:2],axis=0),1e-6)
    projected[:,:2] = (projected[:,:2]-projected[:,:2].min(0))/extent
    return projected, direction, extent[0]/extent[1]


def interpolate(values, triangles, weights):
    a,b,c = values[triangles].transpose(1,0,2)
    return a+(b-a)*weights[:,0:1]+(c-a)*weights[:,1:2]


def foreground(photo, session):
    """Extract cleaned RGBA object crop with border decontamination and edge-clip flags."""
    photo = photo.copy()
    photo.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    if photo.mode != 'RGBA' or photo.getchannel('A').getextrema()[0] == 255:
        from rembg import remove
        photo = remove(photo, session=session)
    rgba = np.asarray(photo.convert('RGBA')).copy()
    mask = rgba[:, :, 3] > 128
    comp, n_comp = cc_label(mask)
    if n_comp > 1:
        areas = np.bincount(comp.ravel())
        areas[0] = 0
        largest = int(areas.max())
        keep = areas[comp] >= 0.02 * largest
        rgba[~keep, 3] = 0
        mask = rgba[:, :, 3] > 128
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None, (False,)*4
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if min(x1 - x0 + 1, y1 - y0 + 1) < 8:
        return None, (False,)*4
    crop = rgba[y0:y1 + 1, x0:x1 + 1].copy()
    opaque = crop[:, :, 3] >= 235
    if opaque.any() and not opaque.all():
        near = distance_transform_edt(~opaque, return_distances=False, return_indices=True)
        crop[~opaque, :3] = crop[near[0][~opaque], near[1][~opaque], :3]
    m = crop[:, :, 3] > 128
    sides = (m[:, 0], m[:, -1], m[0], m[-1])
    at_edge = (x0 == 0, x1 + 1 == rgba.shape[1], y0 == 0, y1 + 1 == rgba.shape[0])
    return crop, tuple(bool(edge and side.mean() > .08) for edge, side in zip(at_edge, sides))


def visible_windows(edges, photo_aspect, model_aspect):
    """Candidate regions of the projected mesh visible inside a clipped photo."""
    left,right,top,bottom = edges
    horizontal,vertical = left or right,top or bottom
    if horizontal == vertical:
        yield (0.,1.,0.,1.)
        return
    span = float(min(1., photo_aspect/model_aspect if horizontal else model_aspect/photo_aspect))
    low,high = (left,right) if horizontal else (top,bottom)
    starts = np.linspace(0,1-span,13) if low and high else [1-span if low else 0.]
    for start in starts:
        yield (float(start),span,0.,1.) if horizontal else (0.,1.,float(start),span)


def joint_cameras(scores, labels):
    """Choose the globally consistent camera pose assignment across labeled reference views."""
    offsets = {'front': 0, 'right': -90, 'rear': 180, 'left': 90}

    def allowed(label, front, candidate):
        yaw,elev = candidate[1:3]
        base,_,nose = label.partition('_nose_')
        signed = (yaw-front-offsets.get(base,0)+180)%360-180
        delta = abs(signed)
        if nose=='left' and signed<=0 or nose=='right' and signed>=0:
            return False
        if base in offsets:
            return delta == 0 and elev == 0
        if base in ('top','bottom'):
            expected_elev = 90 if base == 'top' else -90
            return elev == expected_elev and ((yaw - front) % 90 == 0)
        if base.endswith('three_quarter'):
            return (20 <= delta <= 70 if base.startswith('front') else 110 <= delta <= 160) and (-10 <= elev <= 50)
        return True

    # The generated object's forward axis is unknown. Score complete assignments
    # around one shared heading, including the side profiles that reveal its nose.
    best = None
    for front in range(0,360,30):
        picks = [max((c for c in row if allowed(label,front,c)), default=None)
                 for row,label in zip(scores,labels)]
        quality = sum(p[0] if p else -1 for p in picks)
        if best is None or quality > best[0]:
            best = quality,picks
    return best[1]


def _wheel_centroids(mask):
    """Locate left and right ground-contact wheel horizontal centroids from a side silhouette mask."""
    ph, pw = mask.shape
    bot = np.array(
        [float(np.flatnonzero(mask[:, x])[-1]) if mask[:, x].any() else 0.0 for x in range(pw)],
        dtype=np.float32,
    )
    centers = []
    for a, b in ((int(0.05 * pw), int(0.45 * pw)), (int(0.55 * pw), int(0.95 * pw))):
        if b <= a:
            return None
        sub = bot[a:b]
        w = np.maximum(0.0, sub - np.quantile(sub, 0.78)) ** 2
        w_sum = float(w.sum())
        if w_sum <= 0.0:
            return None
        xs = np.arange(a, b, dtype=np.float32)
        centers.append(float((xs * w).sum() / w_sum))
    return centers[0], centers[1]


def _symmetry_axis(m_mask, pic, axis):
    """Return (m_sym, p_sym, body_scale) when mesh silhouette and photo RGB are bilaterally symmetric."""
    m_s = m_mask if axis == 1 else m_mask.T
    gray = cv2.cvtColor(pic[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    p_hi = pic[:, :, 3] > 180
    p_al = pic[:, :, 3] > 128
    g_s = gray if axis == 1 else gray.T
    pm_s = p_hi if axis == 1 else p_hi.T
    pa_s = p_al if axis == 1 else p_al.T

    orth_len, lat_len = m_s.shape
    m_centers, m_in, p_in = [], [], []
    o0, o1 = int(0.10 * orth_len), int(0.90 * orth_len)
    i0, i1 = int(0.15 * orth_len), int(0.85 * orth_len)
    for o in range(o0, o1):
        idx = np.flatnonzero(m_s[o])
        if idx.size > 0:
            m_centers.append(0.5 * float(idx[0] + idx[-1]))
            if i0 <= o < i1:
                m_in.append(float(idx[-1] - idx[0]))
        if i0 <= o < i1:
            pidx = np.flatnonzero(pa_s[o])
            if pidx.size > 0:
                p_in.append(float(pidx[-1] - pidx[0]))
    if len(m_centers) < 10 or not m_in or not p_in:
        return None

    m_sym = float(np.median(m_centers))
    ks = np.arange(lat_len)
    ms_mesh = np.rint(2.0 * m_sym - ks).astype(int)
    ok_m = (ms_mesh >= 0) & (ms_mesh < lat_len)
    mirror_iou = float((m_s[:, ks[ok_m]] & m_s[:, ms_mesh[ok_m]]).sum()) / max(
        int((m_s[:, ks[ok_m]] | m_s[:, ms_mesh[ok_m]]).sum()), 1
    )
    if mirror_iou < 0.86:
        return None

    min_ov = 0.35 * float(pm_s.sum())
    best_sym = None
    for c2 in range(int(round(0.70 * lat_len)), int(round(1.30 * lat_len)) + 1):
        c = 0.5 * c2
        ms = np.rint(2.0 * c - ks).astype(int)
        ok = (ms >= 0) & (ms < lat_len) & (ks < ms)
        k1, k2 = ks[ok], ms[ok]
        ov = pm_s[:, k1] & pm_s[:, k2]
        if float(ov.sum()) < min_ov:
            continue
        v1, v2 = g_s[:, k1][ov], g_s[:, k2][ov]
        err = float(np.abs(v1 - v2).mean())
        if best_sym is None or err < best_sym[0]:
            best_sym = (err, c, v1, v2)
    if best_sym is None:
        return None
    _, p_sym, v1, v2 = best_sym
    if v1.size < 16 or float(v1.std()) < 1.0 or float(v2.std()) < 1.0:
        return None
    corr = float(np.corrcoef(v1, v2)[0, 1])
    if not np.isfinite(corr) or corr < 0.72:
        return None

    m_arr = np.asarray(m_in, dtype=np.float32)
    p_arr = np.asarray(p_in, dtype=np.float32)
    m_keep = m_arr[m_arr <= np.quantile(m_arr, 0.85)]
    p_keep = p_arr[p_arr <= np.quantile(p_arr, 0.85)]
    m_span = float(np.median(m_keep)) if m_keep.size else float(np.median(m_arr))
    p_span = float(np.median(p_keep)) if p_keep.size else float(np.median(p_arr))
    body_scale = float(np.clip(p_span / max(m_span, 1.0), 0.92, 1.18))
    return m_sym, p_sym, body_scale


def _align_symmetric(projected, m_small, p_small, raw_iou, shape_hw, sym, lat_axis, min_gain):
    """Lock the lateral RGB symmetry centerline while optimizing orthogonal alignment."""
    ph, pw = shape_hw
    sh_grid, sw_grid = p_small.shape
    cx_s, cy_s = 0.5 * (sw_grid - 1), 0.5 * (sh_grid - 1)
    m_sym, p_sym, body_scale = sym

    if lat_axis == 1:
        m_g = m_sym * (sw_grid - 1) / max(pw - 1, 1)
        p_g = p_sym * (sw_grid - 1) / max(pw - 1, 1)
        sx_vals = np.linspace(0.95, 1.17, 12)
        sy_lo, sy_hi = 0.94, 1.18
        sy_vals = np.linspace(0.95, 1.17, 12)
        t_vals = np.linspace(-0.08, 0.08, 9)
        min_center_shift = 1.0
    else:
        m_g = m_sym * (sh_grid - 1) / max(ph - 1, 1)
        p_g = p_sym * (sh_grid - 1) / max(ph - 1, 1)
        sx_vals = np.linspace(0.95, 1.15, 11)
        sy_lo = max(0.94, body_scale - 0.03)
        sy_hi = min(1.16, body_scale + 0.03)
        sy_vals = np.linspace(sy_lo, sy_hi, 5)
        t_vals = np.linspace(-0.06, 0.06, 9)
        min_center_shift = 1.5

    def score_pose(sx, sy, t_orth):
        if lat_axis == 1:
            mat = np.array(
                [[sx, 0.0, p_g - sx * m_g], [0.0, sy, cy_s * (1.0 - sy) + t_orth * sh_grid]],
                dtype=np.float32,
            )
        else:
            mat = np.array(
                [[sx, 0.0, cx_s * (1.0 - sx) + t_orth * sw_grid], [0.0, sy, p_g - sy * m_g]],
                dtype=np.float32,
            )
        warped = cv2.warpAffine(m_small, mat, (sw_grid, sh_grid), flags=cv2.INTER_NEAREST) > 0
        return float((warped & p_small).sum()) / max(int((warped | p_small).sum()), 1)

    best = (-1.0, 1.0, 1.0, 0.0)
    for sx in sx_vals:
        for sy in sy_vals:
            for t_orth in t_vals:
                iou = score_pose(float(sx), float(sy), float(t_orth))
                if iou > best[0]:
                    best = (iou, float(sx), float(sy), float(t_orth))

    for step in (0.01, 0.005):
        _, bsx, bsy, bt = best
        for dsx in (-step, 0.0, step):
            sx = float(np.clip(bsx + dsx, 0.94, 1.18))
            for dsy in (-step, 0.0, step):
                sy = float(np.clip(bsy + dsy, sy_lo, sy_hi))
                for dt in (-step, 0.0, step):
                    t_orth = float(np.clip(bt + dt, -0.09, 0.09))
                    iou = score_pose(sx, sy, t_orth)
                    if iou > best[0]:
                        best = (iou, sx, sy, t_orth)

    best_iou, bsx, bsy, bt = best
    if best_iou < raw_iou + min_gain and abs(p_sym - m_sym) < min_center_shift:
        return projected, False

    aligned = projected.copy()
    cx, cy = 0.5 * (pw - 1), 0.5 * (ph - 1)
    if lat_axis == 1:
        aligned[:, 0] = (aligned[:, 0] - m_sym) * bsx + p_sym
        aligned[:, 1] = (aligned[:, 1] - cy) * bsy + cy + bt * ph
    else:
        aligned[:, 0] = (aligned[:, 0] - cx) * bsx + cx + bt * pw
        aligned[:, 1] = (aligned[:, 1] - m_sym) * bsy + p_sym
    return aligned, True


def _align_silhouette(projected, m_mask, pic, base_label, min_gain):
    """Align unclipped orthographic projections with symmetry locking or 4-DOF silhouette IoU."""
    ph, pw = m_mask.shape
    p_mask = pic[:, :, 3] > 128
    sw_grid = min(pw, 128)
    sh_grid = max(16, int(round(ph * sw_grid / max(pw, 1))))
    m_small = (cv2.resize(m_mask.astype('uint8'), (sw_grid, sh_grid), interpolation=cv2.INTER_NEAREST) > 0).astype('uint8')
    p_small = cv2.resize(p_mask.astype('uint8'), (sw_grid, sh_grid), interpolation=cv2.INTER_NEAREST) > 0
    raw_iou = float(((m_small > 0) & p_small).sum()) / max(int(((m_small > 0) | p_small).sum()), 1)

    lat_axis = 1 if base_label in ('front', 'rear') else (0 if pw >= ph else 1)
    sym = _symmetry_axis(m_mask, pic, lat_axis)
    if sym is not None:
        return _align_symmetric(projected, m_small, p_small, raw_iou, (ph, pw), sym, lat_axis, min_gain)

    cx_s, cy_s = 0.5 * (sw_grid - 1), 0.5 * (sh_grid - 1)

    def score_pose(sx, sy, tx_frac, ty_frac):
        mat = np.array(
            [
                [sx, 0.0, cx_s * (1.0 - sx) + tx_frac * sw_grid],
                [0.0, sy, cy_s * (1.0 - sy) + ty_frac * sh_grid],
            ],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(m_small, mat, (sw_grid, sh_grid), flags=cv2.INTER_NEAREST) > 0
        return float((warped & p_small).sum()) / max(int((warped | p_small).sum()), 1)

    best = (-1.0, 1.0, 1.0, 0.0, 0.0)
    for sx in np.linspace(0.95, 1.17, 12):
        for sy in np.linspace(0.95, 1.17, 12):
            for ty_frac in np.linspace(-0.08, 0.08, 9):
                iou = score_pose(float(sx), float(sy), 0.0, float(ty_frac))
                if iou > best[0]:
                    best = (iou, float(sx), float(sy), 0.0, float(ty_frac))

    prev_tx = best[3]
    _, bsx, bsy, _, bty = best
    for tx_frac in np.linspace(-0.08, 0.08, 9):
        iou = score_pose(bsx, bsy, float(tx_frac), bty)
        if iou > best[0]:
            best = (iou, bsx, bsy, float(tx_frac), bty)
    if best[3] != prev_tx:
        _, bsx, bsy, btx, _ = best
        for ty_frac in np.linspace(-0.08, 0.08, 9):
            iou = score_pose(bsx, bsy, btx, float(ty_frac))
            if iou > best[0]:
                best = (iou, bsx, bsy, btx, float(ty_frac))

    for step in (0.01, 0.005):
        _, bsx, bsy, btx, bty = best
        for dsx in (-step, 0.0, step):
            sx = float(np.clip(bsx + dsx, 0.94, 1.18))
            for dsy in (-step, 0.0, step):
                sy = float(np.clip(bsy + dsy, 0.94, 1.18))
                for dtx in (-step, 0.0, step):
                    tx = float(np.clip(btx + dtx, -0.09, 0.09))
                    for dty in (-step, 0.0, step):
                        ty = float(np.clip(bty + dty, -0.09, 0.09))
                        iou = score_pose(sx, sy, tx, ty)
                        if iou > best[0]:
                            best = (iou, sx, sy, tx, ty)

    best_iou, bsx, bsy, btx, bty = best
    if best_iou < raw_iou + min_gain:
        return projected, False
    aligned = projected.copy()
    cx, cy = 0.5 * (pw - 1), 0.5 * (ph - 1)
    aligned[:, 0] = (aligned[:, 0] - cx) * bsx + cx + btx * pw
    aligned[:, 1] = (aligned[:, 1] - cy) * bsy + cy + bty * ph
    return aligned, True


def _align_side_wheelbase(projected, m_mask, p_mask):
    """Align unclipped side projections via rigid horizontal wheelbase affine fit."""
    ph, pw = m_mask.shape
    mc = _wheel_centroids(m_mask)
    pc = _wheel_centroids(p_mask)
    if mc is None or pc is None or (mc[1] - mc[0]) <= 1.0:
        return projected, False
    sx = (pc[1] - pc[0]) / (mc[1] - mc[0])
    tx = pc[0] - sx * mc[0]
    if not (0.92 <= sx <= 1.12 and abs(tx) <= 0.08 * pw and (abs(sx - 1.0) > 0.02 or abs(tx) > 2.5)):
        return projected, False
    raw_iou = float((m_mask & p_mask).sum()) / max(int((m_mask | p_mask).sum()), 1)
    mat = np.array([[sx, 0.0, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
    warped = cv2.warpAffine(m_mask.astype('uint8'), mat, (pw, ph), flags=cv2.INTER_NEAREST) > 0
    new_iou = float((warped & p_mask).sum()) / max(int((warped | p_mask).sum()), 1)
    if new_iou < raw_iou - 0.015:
        return projected, False
    aligned = projected.copy()
    aligned[:, 0] = aligned[:, 0] * sx + tx
    return aligned, True


def _align_projection(projected, depth, pic, base_label, window):
    """Align orthographic projections to compensate protruding appendages or cropped mirrors."""
    if window[1] < 0.90 or window[3] < 0.90 or base_label not in CARDINAL_VIEWS:
        return projected, False
    m_mask = np.isfinite(depth)
    if base_label in ('front', 'rear'):
        return _align_silhouette(projected, m_mask, pic, base_label, 0.020)
    if base_label in ('top', 'bottom'):
        return _align_silhouette(projected, m_mask, pic, base_label, 0.015)
    return _align_side_wheelbase(projected, m_mask, pic[:, :, 3] > 128)


def _depth_reach(base_label, q_z, z_min, z_span, u_img, v_img, pw, ph):
    """Down-weight far-side grazing regions where orthogonal views own the object's flanks or end fascias."""
    z_norm = np.clip((q_z - z_min) / z_span, 0.0, 1.0).astype(np.float32)
    u_norm = np.abs(2.0 * (u_img / max(float(pw - 1), 1.0)) - 1.0).astype(np.float32)
    if base_label in ('front', 'rear'):
        flank = np.clip((u_norm - 0.52) / 0.28, 0.0, 1.0)
        z_cut = 0.46 + 0.26 * flank
        return np.clip((z_norm - (z_cut - 0.14)) / 0.14, 0.005, 1.0).astype(np.float32)
    if base_label in ('left', 'right'):
        return np.clip((z_norm - 0.34) / 0.16, 0.005, 1.0).astype(np.float32)
    if base_label in ('top', 'bottom'):
        base_r = np.clip((z_norm - 0.28) / 0.18, 0.01, 1.0).astype(np.float32)
        long_norm = (
            u_norm
            if pw >= ph
            else np.abs(2.0 * (v_img / max(float(ph - 1), 1.0)) - 1.0).astype(np.float32)
        )
        end_cap = np.clip((long_norm - 0.56) / 0.20, 0.0, 1.0)
        low_fascia = np.clip((0.76 - z_norm) / 0.22, 0.0, 1.0)
        taper = 1.0 - 0.96 * end_cap * low_fascia
        return (base_r * taper).astype(np.float32)
    return np.ones_like(z_norm, dtype=np.float32)


def _sample_rgba_bilinear(pic_f, u_img, v_img):
    """Sample RGBA float32 image bilinearly at subpixel image coordinates."""
    ph, pw = pic_f.shape[:2]
    uc = np.clip(u_img, 0.0, float(pw - 1))
    vc = np.clip(v_img, 0.0, float(ph - 1))
    ix0 = np.floor(uc).astype(int)
    iy0 = np.floor(vc).astype(int)
    ix1 = np.minimum(ix0 + 1, pw - 1)
    iy1 = np.minimum(iy0 + 1, ph - 1)
    du = (uc - ix0)[:, None].astype(np.float32)
    dv = (vc - iy0)[:, None].astype(np.float32)
    w00 = (1.0 - du) * (1.0 - dv)
    w01 = du * (1.0 - dv)
    w10 = (1.0 - du) * dv
    w11 = du * dv
    return (
        pic_f[iy0, ix0] * w00
        + pic_f[iy0, ix1] * w01
        + pic_f[iy1, ix0] * w10
        + pic_f[iy1, ix1] * w11
    )


def _inpaint_unobserved(fused, mix, vertices, faces, owner, bary, valid, model_diag):
    """Fill shadowed or grazing texels from nearby observed 3D surface colors and dilate UV seams."""
    ys, xs = np.where(valid)
    good_idx = np.flatnonzero(mix[ys, xs] >= 0.65)
    bad_idx = np.flatnonzero(mix[ys, xs] < 0.65)
    if good_idx.size >= 16 and bad_idx.size > 0:
        pos = interpolate(vertices, faces[owner[ys, xs]], bary[ys, xs])
        stride = max(1, good_idx.size // 60000)
        sup = good_idx[::stride]
        tree = cKDTree(pos[sup])
        sup_rgb = fused[ys[sup], xs[sup]]
        radius = max(model_diag * 0.16, 1e-4)
        k = min(12, sup.size)
        chunks = max(1, math.ceil(bad_idx.size / 65536))
        for chunk in np.array_split(bad_idx, chunks):
            ds, js = tree.query(pos[chunk], k=k)
            if k == 1:
                ds = ds[:, None]
                js = js[:, None]
            fill_rgb = np.median(sup_rgb[js], axis=1).astype(np.float32)
            fill_blend = np.exp(-((ds[:, 0] / radius) ** 2)).astype(np.float32)
            existing = mix[ys[chunk], xs[chunk]]
            improve = fill_blend > existing
            if not improve.any():
                continue
            sel = chunk[improve]
            cy, cx = ys[sel], xs[sel]
            bw = (fill_blend[improve] * (1.0 - existing[improve]))[:, None]
            fused[cy, cx] = fused[cy, cx] * (1.0 - bw) + fill_rgb[improve] * bw
            mix[cy, cx] = fill_blend[improve]
    if valid.any() and not valid.all():
        near = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        fused[~valid] = fused[near[0][~valid], near[1][~valid]]


def refine(mesh, images, labels=None):
    """Register reference photos to mesh cameras and project high-resolution photo detail into UV atlas."""
    mat = mesh.visual.material
    atlas = np.asarray(mat.baseColorTexture.convert('RGB'))
    h,w = atlas.shape[:2]
    uv = np.asarray(mesh.visual.uv,dtype='f').copy()
    uv[:,1] = 1-uv[:,1]
    uv *= [w-1,h-1]
    vertices = np.asarray(mesh.vertices,dtype='f')
    faces = np.asarray(mesh.faces,dtype=np.int32)
    session = None
    if any(im.mode != 'RGBA' or im.getchannel('A').getextrema()[0] == 255 for im in images):
        from rembg import new_session
        session = new_session('u2net', providers=['CPUExecutionProvider'])
    photos,edges = zip(*(foreground(im, session) for im in images))
    del session
    refs = [None if p is None else cv2.resize(p,(64,64),interpolation=cv2.INTER_AREA) for p in photos]
    scores = [[] for _ in photos]
    grid_y,grid_x = np.indices((64,64),dtype='f')

    def evaluate(yaw,elev,indices):
        screen,_,aspect = camera(vertices,yaw,elev)
        screen[:,:2] *= 63
        face,bary,_ = raster(screen,faces,64,64)
        hit = face>=0
        tex = interpolate(uv,faces[face[hit]],bary[hit]).round().astype(int)
        render = np.zeros((64,64,3),np.float32)
        render[hit] = atlas[tex[:,1].clip(0,h-1),tex[:,0].clip(0,w-1)]
        for i in indices:
            ref = refs[i]
            if ref is None:
                continue
            mask = ref[:,:,3]>128
            ratio = photos[i].shape[1]/photos[i].shape[0]
            best = None
            for window in visible_windows(edges[i],ratio,aspect):
                sx,sw,sy,sh = window
                if sw==sh==1:
                    visible,colors = hit,render
                else:
                    mx,my = grid_x*sw+63*sx,grid_y*sh+63*sy
                    visible = cv2.remap(hit.astype('uint8'),mx,my,cv2.INTER_NEAREST)>0
                    colors = cv2.remap(render,mx,my,cv2.INTER_LINEAR)
                overlap = mask & visible
                if not overlap.any():
                    continue
                iou = overlap.sum()/(mask|visible).sum()
                target = ref[:,:,:3][overlap].astype('f')
                projected = colors[overlap]
                gain = np.clip((target.mean()+1)/(projected.mean()+1),.6,1.7)
                error = np.abs(target-projected*gain).mean()/255
                aspect_error = abs(np.log(aspect*sw/(ratio*sh)))
                score = iou-.72*error-.18*min(aspect_error,2)
                candidate = (score,yaw,elev,iou,error,window)
                if best is None or score>best[0]:
                    best = candidate
            if best is not None:
                scores[i].append(best)

    for elev in (-90,-45,-20,0,20,45,90):
        for yaw in range(0,360,30):
            evaluate(yaw,elev,range(len(refs)))
    supplied_labels = labels or ['unknown']*len(images)
    labels = list(supplied_labels)
    for i,(ref,label) in enumerate(zip(refs,labels)):
        if ref is None or 'three_quarter' not in label:
            continue
        # Cross-check oblique predictions on bilaterally symmetric end views.
        gray = cv2.cvtColor(ref[:,:,:3],cv2.COLOR_RGB2GRAY).astype('f')
        mask = ref[:,:,3]>128
        overlap = mask & mask[:,::-1]
        values, mirror = gray[overlap],gray[:,::-1][overlap]
        if (overlap.sum() > .87*mask.sum() and values.size and values.std()>5
                and np.corrcoef(values,mirror)[0,1]>.72):
            labels[i] = label.split('_')[0]
    for i,match in enumerate(joint_cameras(scores,labels)):
        if match and labels[i].partition('_nose_')[0] not in CARDINAL_VIEWS:
            a,b = match[1:3]
            for da in (-12,0,12):
                for db in (-8,0,8):
                    evaluate(a+da,np.clip(b+db,-90,90),[i])

    matches = joint_cameras(scores, labels)
    report = []
    selected = []
    for pic, match, label, candidates in zip(photos, matches, labels, scores):
        if not match:
            report.append(dict(applied=False, label=label, reason='no eligible camera'))
            selected.append(None)
            continue
        score, yaw, elev, iou, error, window = match
        if label in ('unknown', 'oblique') and abs(elev) < 65 and any(
            abs((other[1] - yaw + 180) % 360 - 180) > 140 and abs(other[2] - elev) < 25
            and other[0] >= score - .025 for other in candidates
        ):
            report.append(dict(applied=False, label=label, reason='ambiguous opposite directions'))
            selected.append(None)
            continue
        applied = bool(iou >= .70 and score >= .52 and error < .36)
        item = dict(
            applied=applied,
            label=label,
            yaw=float(yaw),
            elevation=float(elev),
            silhouette_iou=round(float(iou), 3),
            score=round(float(score), 3),
        )
        if applied:
            item['visible_window'] = [float(v) for v in window]
            selected.append((pic, match, label))
        else:
            selected.append(None)
        report.append(item)

    cardinal_applied = sum(
        1 for item in report
        if item.get('applied') and item['label'].partition('_nose_')[0] in CARDINAL_VIEWS
    )

    owner, bary = raster_uv(uv, faces, h, w)
    v_normals = np.asarray(mesh.vertex_normals, dtype='f')
    f_normals = np.asarray(mesh.face_normals, dtype='f')
    model_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tolerance = model_diag * 0.012
    weighted = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    max_conf = np.zeros((h, w), np.float32)

    def project_entry(entry):
        pic, match, label = entry
        base_label = label.partition('_nose_')[0]
        is_cardinal = base_label in CARDINAL_VIEWS
        view_priority = np.float32(0.002 if (not is_cardinal and cardinal_applied >= 4) else 1.0)
        conf_scale = np.float32(1.0 if (is_cardinal or cardinal_applied < 4) else 0.75)

        score, yaw, elev, _, _, window = match
        ph, pw = pic.shape[:2]
        pic_f = pic.astype(np.float32)
        projected, direction, _ = camera(vertices, yaw, elev)
        direction = np.asarray(direction, dtype='f')
        sx, sw, sy, sh = window
        if base_label not in CARDINAL_VIEWS or sw < 0.90 or sh < 0.90:
            projected[:, :2] = (projected[:, :2] - [sx, sy]) / [sw, sh]
        projected[:, :2] *= [pw - 1, ph - 1]
        _, _, depth = raster(projected, faces, ph, pw)
        projected, changed = _align_projection(projected, depth, pic, base_label, window)
        if changed:
            _, _, depth = raster(projected, faces, ph, pw)

        hit = np.isfinite(depth)
        if hit.any():
            pad = float(np.ptp(depth[hit])) + 10.0
            hi_sent = float(depth[hit].max()) + pad
            lo_sent = float(depth[hit].min()) - pad
            d_min = cv2.erode(np.where(hit, depth, hi_sent).astype('f'), np.ones((3, 3), np.uint8))
            d_max = cv2.dilate(np.where(hit, depth, lo_sent).astype('f'), np.ones((3, 3), np.uint8))
            d_min_ok = d_min < hi_sent - 0.5 * pad
            d_max_ok = d_max > lo_sent + 0.5 * pad
            jump = np.where(hit & d_min_ok & d_max_ok, np.maximum(d_max - d_min, 0.0), 0.0)
        else:
            d_min = np.full_like(depth, 1e6)
            d_min_ok = np.zeros_like(hit)
            jump = np.zeros_like(depth)
        jump_map = np.clip(
            1.0 - np.maximum(jump - tolerance * 2.0, 0.0) / max(tolerance * 8.0, 1e-6),
            0.15,
            1.0,
        ).astype('f')

        dist_in = distance_transform_edt(pic[:, :, 3] > 140).astype('f')
        border_pad = max(3.0, min(ph, pw) * 0.018)
        border_map = (np.clip(dist_in / border_pad, 0.04, 1.0).astype('f')) ** 2

        z_min = float(projected[:, 2].min())
        z_span = max(float(np.ptp(projected[:, 2])), 1e-6)
        score_pow = np.float32(max(float(score), 0.0) ** 4)

        for row in range(0, h, 64):
            block = owner[row:row + 64]
            yy, xx = np.where(block >= 0)
            if yy.size == 0:
                continue
            yy += row
            fi = owner[yy, xx]
            tri = faces[fi]
            frac = bary[yy, xx]
            q = interpolate(projected, tri, frac)
            u_img, v_img = q[:, 0], q[:, 1]
            inside = (u_img >= 0.0) & (u_img <= pw - 1.0) & (v_img >= 0.0) & (v_img <= ph - 1.0)
            px = np.clip(np.rint(u_img).astype(int), 0, pw - 1)
            py = np.clip(np.rint(v_img).astype(int), 0, ph - 1)

            d_lo = d_min[py, px]
            d_hi = depth[py, px]
            d_lo_fin = d_min_ok[py, px]
            d_hi_fin = hit[py, px]
            dz_front = np.where(d_lo_fin, d_lo - q[:, 2], np.inf)
            dz_exact = np.where(d_hi_fin, np.where(d_hi_fin, d_hi, 0.0) - q[:, 2], np.inf)
            front_shell = inside & (dz_front <= tolerance) & (dz_exact <= tolerance * 2.5)
            cavity_mask = inside & d_hi_fin & (dz_exact > 0.0) & (dz_exact < tolerance * 14.0)
            dz_cavity = np.clip(np.where(cavity_mask, dz_exact, 0.0), 0.0, tolerance * 14.0)
            cavity_vis = np.where(cavity_mask, 0.015 * np.exp(-dz_cavity / max(tolerance * 4.0, 1e-6)), 0.0)
            vis_factor = np.where(front_shell, 1.0, cavity_vis).astype('f')

            facing_v = interpolate(v_normals, tri, frac) @ direction
            facing_f = f_normals[fi] @ direction
            facing_pos = np.maximum(np.maximum(facing_v, 0.75 * facing_f), 0.0)
            facing_eff = np.maximum(np.where(front_shell, np.maximum(facing_pos, 0.06), facing_pos), 0.0).astype('f')
            if base_label in ('front', 'rear'):
                z_norm_pt = np.clip((q[:, 2] - z_min) / z_span, 0.0, 1.0)
                u_norm_pt = np.abs(2.0 * (u_img / max(float(pw - 1), 1.0)) - 1.0)
                v_norm_pt = v_img / max(float(ph - 1), 1.0)
                end_fascia = (
                    front_shell
                    & (z_norm_pt > 0.70)
                    & (v_norm_pt > 0.20)
                    & (u_norm_pt < 0.92)
                    & (facing_pos > 0.10)
                )
                facing_eff = np.where(end_fascia, np.maximum(facing_eff, 0.92), facing_eff).astype('f')
                fascia_boost = np.where(end_fascia, 3.5, 1.0).astype('f')
            else:
                fascia_boost = np.float32(1.0)

            reach_w = _depth_reach(base_label, q[:, 2], z_min, z_span, u_img, v_img, pw, ph)
            rgba = _sample_rgba_bilinear(pic_f, u_img, v_img)
            alpha_ok = np.clip((rgba[:, 3] - 128.0) / 100.0, 0.0, 1.0).astype('f')

            w_cam = (
                (facing_eff ** 6)
                * fascia_boost
                * vis_factor
                * alpha_ok
                * border_map[py, px]
                * jump_map[py, px]
                * reach_w
                * score_pow
                * view_priority
            )
            conf_cam = np.where(
                front_shell & (alpha_ok > 0.25),
                np.clip(facing_eff / 0.08, 0.35, 1.0) * alpha_ok * conf_scale,
                0.0,
            ).astype('f')
            weighted[yy, xx] += w_cam[:, None] * rgba[:, :3]
            weights[yy, xx] += w_cam
            max_conf[yy, xx] = np.maximum(max_conf[yy, xx], conf_cam)

    horizontal_entries = []
    vertical_entries = []
    for idx, entry in enumerate(selected):
        if entry is None:
            continue
        if entry[2].partition('_nose_')[0] in ('top', 'bottom'):
            vertical_entries.append((idx, entry))
        else:
            horizontal_entries.append(entry)

    for entry in horizontal_entries:
        project_entry(entry)

    if vertical_entries and (weights > 1e-8).any():
        pass1_mix = np.where(weights > 1e-8, np.clip(max_conf / 0.65, 0.0, 1.0), 0.0).astype('f')
        pass1_rgb = weighted / np.maximum(weights[:, :, None], 1e-10)
        pass1_atlas = np.clip(
            atlas.astype(np.float32) * (1.0 - pass1_mix[:, :, None]) + pass1_rgb * pass1_mix[:, :, None],
            0,
            255,
        )
        for idx, (pic, match, label) in vertical_entries:
            ref = refs[idx]
            if ref is None:
                continue
            mask_ref = ref[:, :, 3] > 128
            ratio = pic.shape[1] / pic.shape[0]
            base_yaw, elev = match[1], match[2]
            best_v = None
            for dy in (0, 90, 180, 270):
                cand_yaw = (base_yaw + dy) % 360
                scr, _, aspect = camera(vertices, cand_yaw, elev)
                scr[:, :2] *= 63
                fi_r, bary_r, _ = raster(scr, faces, 64, 64)
                hit_r = fi_r >= 0
                tex_r = interpolate(uv, faces[fi_r[hit_r]], bary_r[hit_r]).round().astype(int)
                ren = np.zeros((64, 64, 3), np.float32)
                ren[hit_r] = pass1_atlas[tex_r[:, 1].clip(0, h - 1), tex_r[:, 0].clip(0, w - 1)]
                ov = mask_ref & hit_r
                if not ov.any():
                    continue
                iou_v = ov.sum() / (mask_ref | hit_r).sum()
                target_v = ref[:, :, :3][ov].astype('f')
                proj_v = ren[ov]
                gain_v = np.clip((target_v.mean() + 1) / (proj_v.mean() + 1), 0.6, 1.7)
                err_v = np.abs(target_v - proj_v * gain_v).mean() / 255.0
                aerr_v = abs(np.log(aspect / ratio))
                sc_v = iou_v - 0.72 * err_v - 0.18 * min(aerr_v, 2)
                cand_v = (sc_v, cand_yaw, elev, iou_v, err_v, match[5])
                if best_v is None or sc_v > best_v[0]:
                    best_v = cand_v
            if best_v is not None:
                match = best_v
                report[idx]['yaw'] = float(best_v[1])
                report[idx]['silhouette_iou'] = round(float(best_v[3]), 3)
                report[idx]['score'] = round(float(best_v[0]), 3)
            project_entry((pic, match, label))
    else:
        for _, entry in vertical_entries:
            project_entry(entry)

    valid = owner >= 0
    mix = np.where(weights > 1e-8, np.clip(max_conf / 0.65, 0.0, 1.0), 0.0).astype('f')
    photo_rgb = weighted / np.maximum(weights[:, :, None], 1e-10)
    fused = atlas.astype(np.float32) * (1.0 - mix[:, :, None]) + photo_rgb * mix[:, :, None]
    _inpaint_unobserved(fused, mix, vertices, faces, owner, bary, valid, model_diag)

    mat.baseColorTexture = Image.fromarray(np.uint8(np.clip(fused, 0, 255)))
    for item,label in zip(report,supplied_labels):
        item['supplied_label'] = label
    return dict(
        reference_registration=report,
        photo_texture_fraction=round(float(np.count_nonzero(mix[valid] > .5) / max(1, int(valid.sum()))), 4),
    )


def unlit(document):
    document.setdefault('extensionsUsed',[]).append('KHR_materials_unlit')
    for material in document['materials']:
        material.setdefault('extensions',{})['KHR_materials_unlit'] = {}
