"""Local Apple Silicon MLX 3D generation engine for MeshPort.

Runs Hunyuan3D-2.1 MLX in isolated per-stage subprocesses so each stage
releases 100% of its Metal unified memory on exit (critical for 16 GB Macs),
then refines the 4096x4096 PBR UV atlas with MeshPort's multi-view photo
projection (`colab/photo_finish.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / '.engines' / 'Hunyuan3D-2.1-mlx'
SHAPE_DIR = ENGINE_ROOT / 'hy3dshape'
PAINT_DIR = ENGINE_ROOT / 'hy3dpaint'
MV_ENGINE_DIR = ROOT / '.engines' / 'Hunyuan3D-2'
MV_PYTHON = ROOT / '.mv-venv' / 'bin' / 'python'
COLAB_DIR = ROOT / 'colab'

TARGET_FACES = int(os.environ.get('MESHPORT_TARGET_FACES', '40000'))
TEXTURE_SIZE = int(os.environ.get('MESHPORT_TEXTURE_SIZE', '4096'))
OCTREE_RESOLUTION = int(os.environ.get('MESHPORT_OCTREE_RESOLUTION', '256'))
SHAPE_STEPS = int(os.environ.get('MESHPORT_SHAPE_STEPS', '50'))
SEED = int(os.environ.get('MESHPORT_SEED', '42'))

_GPU_LOCK = asyncio.Lock()


def is_local_available() -> bool:
    """Return True when Apple Silicon MLX and the local Hunyuan3D-2.1-mlx engine are ready."""
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return False
    if not (SHAPE_DIR / 'hy3dshape' / 'pipeline_mlx.py').is_file():
        return False
    if not (PAINT_DIR / 'textureGenPipeline_mlx.py').is_file():
        return False
    try:
        import mlx.core as mx
        return bool(mx.metal.is_available())
    except Exception:
        return False


def is_multiview_shape_available() -> bool:
    """Return True when the isolated PyTorch MPS multi-view shape engine is installed."""
    return MV_PYTHON.is_file() and (MV_ENGINE_DIR / 'hy3dgen' / 'shapegen' / 'pipelines.py').is_file()


def select_hero_index(labels: list[str]) -> int:
    """Pick the best reference view index for single-image shape + base PBR synthesis."""
    priorities = (
        'front_three_quarter',
        'front_three_quarter_nose_left',
        'front_three_quarter_nose_right',
        'front',
        'oblique',
        'rear_three_quarter',
        'rear_three_quarter_nose_left',
        'rear_three_quarter_nose_right',
    )
    for target in priorities:
        for index, label in enumerate(labels):
            if label == target:
                return index
    return 0


def _ensure_foreground_rgba(image_path: Path, out_rgba_path: Path) -> Path:
    """Ensure the image is a clean, floater-free, square-padded RGBA cutout."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from scipy.ndimage import distance_transform_edt, label as cc_label

    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode == 'RGBA' and im.getchannel('A').getextrema()[0] < 250
        if has_alpha:
            cutout = im.copy()
        else:
            from rembg import new_session, remove
            session = new_session('u2net', providers=['CPUExecutionProvider'])
            cutout = remove(im.convert('RGB'), session=session).convert('RGBA')
        rgba = np.asarray(cutout.convert('RGBA')).copy()
        mask = rgba[:, :, 3] > 128
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
        if opened.any():
            mask = opened
        comp, n_comp = cc_label(mask)
        if n_comp > 1:
            areas = np.bincount(comp.ravel())
            areas[0] = 0
            largest = int(areas.max())
            mask = areas[comp] >= 0.02 * largest
        rgba[~mask, 3] = 0
        ys, xs = np.where(mask)
        if ys.size and xs.size:
            rgba = rgba[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
            opaque = rgba[:, :, 3] >= 235
            if opaque.any() and not opaque.all():
                near = distance_transform_edt(~opaque, return_distances=False, return_indices=True)
                rgba[~opaque, :3] = rgba[near[0][~opaque], near[1][~opaque], :3]
        cutout = Image.fromarray(rgba, mode='RGBA')
        side = max(1, int(round(max(cutout.width, cutout.height) * 1.15)))
        canvas = Image.new('RGBA', (side, side), (0, 0, 0, 0))
        canvas.paste(cutout, ((side - cutout.width) // 2, (side - cutout.height) // 2), cutout)
        canvas.save(out_rgba_path, format='PNG')
        return out_rgba_path


def _decimate_mesh(in_glb: Path, out_glb: Path, target_faces: int) -> tuple[int, int]:
    """Remove small floaters and run quadric edge-collapse decimation to target_faces."""
    import pymeshlab
    import trimesh

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(in_glb))
    source_faces = int(ms.current_mesh().face_number())
    try:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(15.0)
        )
    except Exception:
        pass
    if ms.current_mesh().face_number() > target_faces > 0:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
        )
    cur = ms.current_mesh()
    mesh = trimesh.Trimesh(vertices=cur.vertex_matrix(), faces=cur.face_matrix(), process=False)
    mesh.export(str(out_glb))
    return source_faces, int(len(mesh.faces))


def _register_mlx_shape_packages() -> None:
    """Register hy3dshape package paths in sys.modules to bypass legacy PyTorch __init__.py imports."""
    import types
    base = SHAPE_DIR / 'hy3dshape'
    for name, rel in (
        ('hy3dshape', base),
        ('hy3dshape.models', base / 'models'),
        ('hy3dshape.models.autoencoders', base / 'models' / 'autoencoders'),
        ('hy3dshape.models.denoisers', base / 'models' / 'denoisers'),
    ):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(rel)]
            sys.modules[name] = pkg


def stage_shape(
    image_path: Path,
    out_glb: Path,
    meta_path: Path,
    target_faces: int = TARGET_FACES,
    octree_resolution: int = OCTREE_RESOLUTION,
    steps: int = SHAPE_STEPS,
    seed: int = SEED,
) -> None:
    """Stage 1: Generate 3D mesh from hero image via MLX and decimate to target_faces."""
    _register_mlx_shape_packages()
    sys.path.insert(0, str(SHAPE_DIR))
    from hy3dshape.pipeline_mlx import ShapePipeline

    t0 = time.monotonic()
    hero_rgba = out_glb.parent / 'hero_rgba.png'
    _ensure_foreground_rgba(image_path, hero_rgba)

    pipe = ShapePipeline.from_pretrained('dgrauet/hunyuan3d-2.1-mlx')
    import mlx.core as mx
    orig_set_timesteps = pipe.scheduler.set_timesteps

    def _fixed_set_timesteps(num_inference_steps: int, sigmas=None):
        orig_set_timesteps(num_inference_steps, sigmas=sigmas)
        if sigmas is not None and len(sigmas) > 1 and sigmas[-1] > sigmas[0]:
            pipe.scheduler.sigmas = mx.concatenate(
                [pipe.scheduler.sigmas[:-1], mx.ones((1,), dtype=pipe.scheduler.sigmas.dtype)]
            )

    pipe.scheduler.set_timesteps = _fixed_set_timesteps
    mesh = pipe(
        str(hero_rgba),
        num_inference_steps=steps,
        guidance_scale=7.5,
        octree_resolution=octree_resolution,
        seed=seed,
    )
    raw_glb = out_glb.parent / 'shape_raw.glb'
    mesh.export(str(raw_glb))
    source_faces, final_faces = _decimate_mesh(raw_glb, out_glb, target_faces)
    try:
        raw_glb.unlink()
    except OSError:
        pass

    meta = {
        'shape_model': 'dgrauet/hunyuan3d-2.1-mlx',
        'source_faces': source_faces,
        'target_faces': target_faces,
        'decimated_faces': final_faces,
        'octree_resolution': octree_resolution,
        'shape_steps': steps,
        'seed': seed,
        'shape_seconds': round(time.monotonic() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')


def stage_shape_mv(
    image_paths: list[Path],
    labels: list[str],
    hero_image: Path,
    out_glb: Path,
    meta_path: Path,
    target_faces: int = TARGET_FACES,
    octree_resolution: int = 380,
    steps: int = 35,
    seed: int = SEED,
) -> None:
    """Stage 1 (Multi-View): Generate 3D mesh from cardinal views via Hunyuan3D-2mv on MPS."""
    import torch
    from PIL import Image

    sys.path.insert(0, str(MV_ENGINE_DIR))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.monotonic()
    hero_rgba = out_glb.parent / 'hero_rgba.png'
    _ensure_foreground_rgba(hero_image, hero_rgba)

    label_map = {'front': 'front', 'rear': 'back', 'left': 'left', 'right': 'right'}
    views = {}
    for path, label in zip(image_paths, labels):
        mv_key = label_map.get(label)
        if mv_key and mv_key not in views:
            clean_path = out_glb.parent / f'mv_{mv_key}.png'
            _ensure_foreground_rgba(path, clean_path)
            with Image.open(clean_path) as im:
                views[mv_key] = im.convert('RGBA')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv',
        device=device,
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    mesh = pipe(
        image=views,
        num_inference_steps=steps,
        octree_resolution=octree_resolution,
        num_chunks=20_000,
        generator=generator,
        output_type='trimesh',
    )[0]
    raw_glb = out_glb.parent / 'shape_raw.glb'
    mesh.export(str(raw_glb))
    source_faces, final_faces = _decimate_mesh(raw_glb, out_glb, target_faces)
    try:
        raw_glb.unlink()
    except OSError:
        pass

    meta = {
        'shape_model': 'tencent/Hunyuan3D-2mv',
        'mv_views': sorted(views.keys()),
        'source_faces': source_faces,
        'target_faces': target_faces,
        'decimated_faces': final_faces,
        'octree_resolution': octree_resolution,
        'shape_steps': steps,
        'seed': seed,
        'shape_seconds': round(time.monotonic() - t0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')


def stage_paint(
    shape_glb: Path,
    hero_image: Path,
    out_obj: Path,
    texture_size: int = TEXTURE_SIZE,
    max_num_view: int = 6,
    resolution: int = 512,
) -> None:
    """Stage 2: Synthesize 4K PBR textures (albedo + metallic/roughness) via MLX."""
    sys.path.insert(0, str(PAINT_DIR))
    sys.path.insert(0, str(SHAPE_DIR))
    from textureGenPipeline_mlx import Hunyuan3DPaintConfigMLX, Hunyuan3DPaintPipelineMLX

    hero_rgba = shape_glb.parent / 'hero_rgba.png'
    ref_image = hero_rgba if hero_rgba.is_file() else hero_image

    cfg = Hunyuan3DPaintConfigMLX(max_num_view=max_num_view, resolution=resolution)
    cfg.texture_size = texture_size
    cfg.mlx_num_inference_steps = int(os.environ.get('MESHPORT_PAINT_STEPS', '25'))
    pipe = Hunyuan3DPaintPipelineMLX(cfg)
    pipe(
        mesh_path=str(shape_glb),
        image_path=str(ref_image),
        output_mesh_path=str(out_obj),
        use_remesh=False,
        save_glb=True,
    )


def stage_refine(
    textured_glb: Path,
    image_paths: list[Path],
    labels: list[str],
    final_glb: Path,
    manifest_path: Path,
    meta_path: Path,
    started_at: float,
) -> None:
    """Stage 3: Project multi-view reference photos onto the 4K UV atlas and export final GLB."""
    import trimesh
    from PIL import Image, ImageOps

    sys.path.insert(0, str(COLAB_DIR))
    from photo_finish import refine, unlit

    loaded = trimesh.load(str(textured_glb), process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) == 1:
            mesh = geometries[0]
        else:
            mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    pil_images = []
    digests = []
    for path in image_paths:
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as im:
            pil_images.append(ImageOps.exif_transpose(im).convert('RGBA'))

    orig_tex = getattr(getattr(mesh.visual, 'material', None), 'baseColorTexture', None)
    if orig_tex is not None:
        orig_tex = orig_tex.copy()

    finish = refine(mesh, pil_images, labels)
    high_conf_cardinal = sum(
        1 for item in finish.get('reference_registration', [])
        if item.get('applied')
        and item.get('label', '').partition('_nose_')[0] in ('front', 'rear', 'left', 'right', 'top', 'bottom')
        and float(item.get('silhouette_iou', 0.0)) >= 0.80
    )
    mat = getattr(mesh.visual, 'material', None)
    if high_conf_cardinal < 2 and orig_tex is not None and mat is not None:
        mat.baseColorTexture = orig_tex
        finish['photo_finish_applied'] = False
    else:
        finish['photo_finish_applied'] = True

    if mat is not None:
        if hasattr(mat, 'metallicRoughnessTexture'):
            mat.metallicRoughnessTexture = None
        if hasattr(mat, 'metallicFactor'):
            mat.metallicFactor = 0.0
        if hasattr(mat, 'roughnessFactor'):
            mat.roughnessFactor = 1.0
    mesh.export(str(final_glb), tree_postprocessor=unlit)

    stage1_meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.is_file() else {}
    tex = getattr(getattr(mesh.visual, 'material', None), 'baseColorTexture', None)
    tex_size = tex.size[0] if tex is not None else TEXTURE_SIZE

    details = {
        'model': stage1_meta.get('shape_model', 'Hunyuan3D-2.1-mlx'),
        'backend': 'local-mlx',
        'seed': stage1_meta.get('seed', SEED),
        'steps': stage1_meta.get('shape_steps', SHAPE_STEPS),
        'texture_size': tex_size,
        'input_count': len(image_paths),
        'conditioning': 'local-mlx-pbr-photofinish',
        'input_sha256': digests,
        'faces': int(len(mesh.faces)),
        'vertices': int(len(mesh.vertices)),
        'watertight': bool(mesh.is_watertight),
        'source_faces': stage1_meta.get('source_faces'),
        'target_faces': stage1_meta.get('target_faces', TARGET_FACES),
        'seconds': round(time.time() - started_at, 1),
    }
    details.update(finish)
    manifest_path.write_text(json.dumps(details, indent=2), encoding='utf-8')


async def _run_stage_subprocess(
    args: list[str],
    log_path: Path,
    stage_name: str,
    python_exe: str | None = None,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        python_exe or sys.executable,
        '-m',
        'app.local_engine',
        *args,
        cwd=str(ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await proc.communicate()
    with log_path.open('ab') as log:
        log.write(f'\n=== {stage_name} (exit {proc.returncode}) ===\n'.encode('utf-8'))
        log.write(stdout or b'')
    if proc.returncode != 0:
        tail = (stdout or b'').decode('utf-8', errors='replace').strip().splitlines()[-12:]
        detail = ' | '.join(tail) if tail else f'{stage_name} exited with code {proc.returncode}'
        raise RuntimeError(f'Local {stage_name} failed: {detail}')


async def run_local_generation(
    folder: Path,
    image_paths: list[Path],
    labels: list[str],
    update_stage: Callable[[str], None],
) -> dict:
    """Run all local MLX stages sequentially in isolated subprocesses under the GPU lock."""
    async with _GPU_LOCK:
        started_at = time.time()
        log_path = folder / 'local_generation.log'
        shape_glb = folder / 'shape.glb'
        meta_path = folder / 'shape_meta.json'
        textured_obj = folder / 'textured.obj'
        textured_glb = folder / 'textured.glb'
        final_glb = folder / 'model.glb'
        manifest_path = folder / 'manifest.json'

        hero_idx = select_hero_index(labels)
        hero_path = image_paths[hero_idx]
        cardinal_mv = {label for label in labels if label in ('front', 'rear', 'left', 'right')}

        if is_multiview_shape_available() and len(cardinal_mv) >= 2:
            update_stage(f'Generating {TARGET_FACES:,}-face multi-view 3D geometry on Apple Silicon (MPS)')
            await _run_stage_subprocess(
                [
                    'shape_mv',
                    '--images', *[str(p) for p in image_paths],
                    '--labels', *labels,
                    '--hero', str(hero_path),
                    '--out', str(shape_glb),
                    '--meta', str(meta_path),
                    '--target-faces', str(TARGET_FACES),
                ],
                log_path,
                'shape_mv',
                python_exe=str(MV_PYTHON),
            )
        else:
            update_stage(f'Generating {TARGET_FACES:,}-face 3D geometry on Apple Silicon (MLX)')
            await _run_stage_subprocess(
                [
                    'shape',
                    '--image', str(hero_path),
                    '--out', str(shape_glb),
                    '--meta', str(meta_path),
                    '--target-faces', str(TARGET_FACES),
                    '--octree', str(OCTREE_RESOLUTION),
                    '--steps', str(SHAPE_STEPS),
                    '--seed', str(SEED),
                ],
                log_path,
                'shape',
            )

        update_stage(f'Synthesizing {TEXTURE_SIZE}px PBR textures on Apple Silicon (MLX)')
        await _run_stage_subprocess(
            [
                'paint',
                '--shape', str(shape_glb),
                '--image', str(hero_path),
                '--out', str(textured_obj),
                '--texture-size', str(TEXTURE_SIZE),
            ],
            log_path,
            'paint',
        )
        if not textured_glb.is_file():
            raise RuntimeError('Local texture synthesis did not produce textured.glb')

        update_stage('Projecting reference photos onto 4K UV atlas')
        await _run_stage_subprocess(
            [
                'refine',
                '--textured', str(textured_glb),
                '--out', str(final_glb),
                '--manifest', str(manifest_path),
                '--meta', str(meta_path),
                '--started-at', str(started_at),
                '--images', *[str(p) for p in image_paths],
                '--labels', *labels,
            ],
            log_path,
            'refine',
        )

        if not manifest_path.is_file():
            raise RuntimeError('Local refinement did not write manifest.json')
        return json.loads(manifest_path.read_text(encoding='utf-8'))


def _cli() -> None:
    parser = argparse.ArgumentParser(description='MeshPort local MLX engine stages')
    sub = parser.add_subparsers(dest='stage', required=True)

    p_shape = sub.add_parser('shape')
    p_shape.add_argument('--image', type=Path, required=True)
    p_shape.add_argument('--out', type=Path, required=True)
    p_shape.add_argument('--meta', type=Path, required=True)
    p_shape.add_argument('--target-faces', type=int, default=TARGET_FACES)
    p_shape.add_argument('--octree', type=int, default=OCTREE_RESOLUTION)
    p_shape.add_argument('--steps', type=int, default=SHAPE_STEPS)
    p_shape.add_argument('--seed', type=int, default=SEED)

    p_shape_mv = sub.add_parser('shape_mv')
    p_shape_mv.add_argument('--images', type=Path, nargs='+', required=True)
    p_shape_mv.add_argument('--labels', nargs='+', required=True)
    p_shape_mv.add_argument('--hero', type=Path, required=True)
    p_shape_mv.add_argument('--out', type=Path, required=True)
    p_shape_mv.add_argument('--meta', type=Path, required=True)
    p_shape_mv.add_argument('--target-faces', type=int, default=TARGET_FACES)
    p_shape_mv.add_argument('--octree', type=int, default=380)
    p_shape_mv.add_argument('--steps', type=int, default=35)
    p_shape_mv.add_argument('--seed', type=int, default=SEED)

    p_paint = sub.add_parser('paint')
    p_paint.add_argument('--shape', type=Path, required=True)
    p_paint.add_argument('--image', type=Path, required=True)
    p_paint.add_argument('--out', type=Path, required=True)
    p_paint.add_argument('--texture-size', type=int, default=TEXTURE_SIZE)

    p_refine = sub.add_parser('refine')
    p_refine.add_argument('--textured', type=Path, required=True)
    p_refine.add_argument('--out', type=Path, required=True)
    p_refine.add_argument('--manifest', type=Path, required=True)
    p_refine.add_argument('--meta', type=Path, required=True)
    p_refine.add_argument('--started-at', type=float, required=True)
    p_refine.add_argument('--images', type=Path, nargs='+', required=True)
    p_refine.add_argument('--labels', nargs='+', required=True)

    args = parser.parse_args()
    if args.stage == 'shape':
        stage_shape(
            args.image,
            args.out,
            args.meta,
            target_faces=args.target_faces,
            octree_resolution=args.octree,
            steps=args.steps,
            seed=args.seed,
        )
    elif args.stage == 'shape_mv':
        stage_shape_mv(
            args.images,
            args.labels,
            args.hero,
            args.out,
            args.meta,
            target_faces=args.target_faces,
            octree_resolution=args.octree,
            steps=args.steps,
            seed=args.seed,
        )
    elif args.stage == 'paint':
        stage_paint(
            args.shape,
            args.image,
            args.out,
            texture_size=args.texture_size,
        )
    elif args.stage == 'refine':
        stage_refine(
            args.textured,
            args.images,
            args.labels,
            args.out,
            args.manifest,
            args.meta,
            args.started_at,
        )


if __name__ == '__main__':
    _cli()
