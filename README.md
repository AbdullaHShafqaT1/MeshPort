# MeshPort

Turn reference images into downloadable 3D models through a local browser interface and a cloud GPU service.

MeshPort runs the interface, preview, and FBX conversion on your computer. The included Google Colab notebook runs Microsoft TRELLIS on an A100 GPU. Paste the notebook connection URL into MeshPort, upload one image or up to six views of the same object, and download GLB, textured FBX, or an FBX texture bundle.

## Start on Windows

Requirements: Windows, Git, Python 3.10 (64-bit) with the Python launcher, an internet connection, and access to a Google Colab A100 GPU. The local computer does not need an NVIDIA GPU. Colab GPU availability and charges depend on your Google account.

Clone the repository and run:

```powershell
git clone https://github.com/AbdullaHShafqaT1/MeshPort.git
cd MeshPort
.\setup.ps1
.\run.ps1
```

Open http://127.0.0.1:8000. Use `.\run.ps1 -Port 8001` to choose another port. Setup creates a local `.venv` and installs the Blender Python converter into `.fbx-tools`; it does not install Python, CUDA, local inference engines, or model weights. The converter requires CPython 3.10. Initial dependency downloads include the large Blender module.

1. Download the notebook from the interface, or open `colab/meshport_colab.ipynb` in Google Colab.
2. Select an A100 GPU and run all cells. Initial cloud setup compiles GPU dependencies and downloads model weights; it can take several minutes and requires sufficient Colab GPU access.
3. Paste the complete final URL, including `#key=...`, into MeshPort. The studio connects automatically.
4. Upload a reference sheet or up to six photos of one object. Automatic generation is enabled by default. The studio checks a single upload for multiple angles, labels individual views using local Ollama when available, and sends the images and camera labels together. Disable automatic generation to review the queue before starting. Each input can be up to 16 megapixels with a combined limit of 24 MiB.
5. Preview the result and download GLB, FBX, or FBX with textures.

The included notebook provides an authenticated `trycloudflare.com` service URL. Generation uses TRELLIS image large, seed 42, 16 sampling steps, and 2048 textures. With two or more images, the updated notebook blends every view at each geometry and appearance sampling step. This takes more GPU time as you add views. Earlier compatible notebooks can still rotate through the views using stochastic sampling. The result reports the mode and number of views when the connected service provides them.

Use consistent views of the same subject, with the object fully visible. For cars, front, rear, side and elevated views can guide detailing across the body. TRELLIS uses the images as visual guidance and infers geometry and surfaces outside their coverage.

## Preview views from a reference sheet

Run MeshPort and click **Separate the views locally** on the studio page. The preview uses the already-installed `qwen2.5vl:3b` model in Ollama on port 11434. You can use it without a Colab connection or the FBX converter.

Upload one JPG, PNG, or WebP image of at most 16 megapixels and 16 MiB. The preview shows detected crop boundaries on the original sheet, labels the views it recognizes, and offers a PNG download for each crop. Views the model cannot identify appear in the coverage list. Clear gaps and printed angle headings help the model distinguish views. Crops retain the original pixels inside the displayed rectangles. Select the crops to use and open them in the studio. The studio uses the same automatic generation setting for imported crops.

## Your files and session

Images are uploaded to the connected cloud runtime. Inputs, models, conversion reports, and job metadata are saved locally under `outputs/studio/<job-id>/`. Keep any files you need before deleting that directory. Job listings and the cloud connection live in server memory and reset when the local server restarts; files remain on disk.

Keep Colab running while generating. Restarting the cloud runtime creates a new URL and key. Treat the complete connection URL as a credential and clear notebook outputs before sharing. MeshPort holds the key in memory and redacts it from persisted job errors. The local server binds to loopback and rejects foreign hosts and browser origins.

Generation is limited to one job at a time, a one-hour polling window, and a 256 MB model download. FBX conversion has a five-minute timeout. If conversion fails, the downloaded GLB remains available. FBX conversion checks a reimport for geometry, UVs, texture sizes, and bounds.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest test_angle_preview test_studio -v
.\.venv\Scripts\python.exe colab/build_notebook.py
node test_uploads.cjs  # optional browser-handler checks, requires Node.js
python -m unittest test_view_handoff -v  # optional check, requires Playwright and Chromium
```

Tests exercise the local HTTP boundaries and a synthetic textured model through a simulated cloud transport and the real Blender converter. They do not verify live Colab inference. The notebook is generated from the Python sources in `colab/`; edit those sources and regenerate it. Do not commit notebook outputs, session keys, generated assets, or virtual environments.

## Project layout

- `app/studio.py`: local API and cloud bridge.
- `app/static/`: browser interface and bundled Three.js preview.
- `app/convert_fbx.py`: Blender conversion and verification.
- `colab/`: cloud service, environment setup, and generated notebook.

MeshPort is a standalone application. This repository contains the local interface, cloud connection, Colab notebook, and export workflow. Model weights and generated assets are downloaded or created at runtime.

## Optional local texture tools

Install the CPU texture dependencies in a separate Python 3.10 environment with `python -m pip install -r requirements-texture.txt`. Run these commands from the repository root:

```powershell
python tools/lift_finish.py path/to/textured-model.glb
python tools/retexture_drift_car.py --clean-glb path/to/clean.glb --photos path/to/six-views
```

Each tool writes a new job under `outputs/studio/` and prints its directory. The contrast tool expects a MeshPort GLB with a single embedded RGB PNG atlas. The drift-car recipe is calibrated to the original clean mesh and its original six PNG crops, named `input-1.png` through `input-6.png` in front, rear, left, right, roof, and three-quarter order. Transfer those private inputs separately to reproduce that car on another device. For other subjects use the studio and Colab notebook. The local recipe downloads the background-removal weights on first use.

## Troubleshooting

- If setup cannot find Python, install Python 3.10 (64-bit) with the Python launcher and rerun setup.
- If the default port is busy, run `.\run.ps1 -Port 8001`.
- If the cloud connection stops working, check that Colab is still running and paste its current complete connection URL again.
- If cloud setup fails, inspect the notebook output and `/content/trellis-setup.log`. Some upstream dependencies are unpinned and may need maintenance.
- If FBX conversion fails, download the GLB result. Local conversion logs are saved with the job under `outputs/studio/`.

Local bridge tests do not establish that a fresh Colab runtime will complete inference. Live Colab generation requires separate verification.

## License and dependencies

Project code retains the MIT license in `LICENSE`. See `THIRD_PARTY_NOTICES.md` for bundled Three.js attribution. Blender/bpy is installed separately under GPL-3.0; TRELLIS and its cloud dependencies and model weights retain their respective upstream licenses. Review those terms for your intended use and distribution.
