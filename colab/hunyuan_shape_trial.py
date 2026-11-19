"""Isolated shape trial. Run with Hunyuan3D-2 installed in a CUDA environment."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for angle in ("front", "back", "left", "right"):
        parser.add_argument(f"--{angle}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".glb" or args.output.exists():
        parser.error("--output must be a new .glb file")

    import torch
    from hy3dgen.rembg import BackgroundRemover
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from PIL import Image

    remove_background = BackgroundRemover()
    views = {}
    for angle in ("front", "back", "left", "right"):
        with Image.open(getattr(args, angle)) as image:
            views[angle] = remove_background(image.convert("RGB"))

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mv",
        subfolder="hunyuan3d-dit-v2-mv",
        use_safetensors=True,
        device="cuda",
    )
    mesh = pipeline(
        image=views,
        num_inference_steps=30,
        octree_resolution=380,
        num_chunks=20_000,
        generator=torch.Generator(device="cuda").manual_seed(42),
        output_type="trimesh",
    )[0]
    if not len(mesh.faces) or not len(mesh.vertices):
        raise RuntimeError("Hunyuan returned an empty mesh")
    with args.output.open("xb") as output:
        output.write(mesh.export(file_type="glb"))


if __name__ == "__main__":
    main()
