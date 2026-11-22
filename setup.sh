#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Ensuring Python 3.11 is available..."
uv python install 3.11

echo "Setting up .venv..."
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-mac.txt

echo "Setting up isolated .fbx-venv (Blender bpy 4.3.0)..."
uv venv --python 3.11 .fbx-venv
uv pip install --python .fbx-venv/bin/python "bpy==4.3.0" "numpy==1.26.4"

echo "Setting up isolated .mv-venv (PyTorch MPS multi-view shape)..."
uv venv --python 3.11 .mv-venv
uv pip install --python .mv-venv/bin/python torch torchvision "diffusers==0.31.0" "transformers==4.46.3" einops omegaconf safetensors huggingface-hub trimesh scikit-image PyMCubes scipy numpy pymeshlab opencv-python-headless rembg onnxruntime Pillow tqdm pyyaml

mkdir -p .engines
if [ ! -d .engines/Hunyuan3D-2.1-mlx/.git ]; then
  echo "Cloning Hunyuan3D-2.1-mlx engine..."
  git clone --depth 1 https://github.com/dgrauet/Hunyuan3D-2.1-mlx.git .engines/Hunyuan3D-2.1-mlx
fi
if [ ! -d .engines/Hunyuan3D-2/.git ]; then
  echo "Cloning Hunyuan3D-2 multi-view shape engine..."
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git .engines/Hunyuan3D-2
fi

if command -v ollama >/dev/null 2>&1; then
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    open -a Ollama >/dev/null 2>&1 || true
    sleep 3
  fi
  echo "Ensuring Ollama model qwen2.5vl:3b is pulled..."
  ollama pull qwen2.5vl:3b || echo "Warning: could not pull qwen2.5vl:3b; ensure Ollama is running."
fi

echo "Pre-downloading Hunyuan3D-2.1-mlx weights..."
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('dgrauet/hunyuan3d-2.1-mlx')"

echo "MeshPort macOS setup complete. Run ./run.sh and open http://127.0.0.1:8000"
