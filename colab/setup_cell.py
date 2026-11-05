# @title 1. Install an isolated TRELLIS environment (rerunnable)
import os, pathlib, subprocess, textwrap
setup_script = r'''
set -euo pipefail
cd /content
export MAMBA_ROOT_PREFIX=/content/mamba
if [ ! -x /content/bin/micromamba ]; then
  curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest -o /content/micromamba.tar.bz2
  tar -xjf /content/micromamba.tar.bz2 -C /content bin/micromamba
fi
if [ ! -x /content/trellis-env/bin/python ]; then
  /content/bin/micromamba create -y -p /content/trellis-env -c conda-forge -c nvidia python=3.10 pip cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-cccl=12.1 gcc_linux-64=12 gxx_linux-64=12
fi
export PATH=/content/trellis-env/bin:$PATH
export CUDA_HOME=/content/trellis-env
export CC=/content/trellis-env/bin/x86_64-conda-linux-gnu-cc
export CXX=/content/trellis-env/bin/x86_64-conda-linux-gnu-c++
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
export ATTN_BACKEND=xformers SPCONV_ALGO=native
if [ ! -d /content/TRELLIS/.git ]; then
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /content/TRELLIS
fi
cd /content/TRELLIS
git rev-parse HEAD
printf 'numpy==1.26.4
scipy==1.14.1
torch==2.4.0
torchvision==0.19.0
xformers==0.0.27.post2
pymeshfix==0.17.0
pyvista==0.44.2
vtk==9.3.1
transformers==4.46.3
huggingface-hub==0.25.2
fastapi==0.115.6
pydantic==2.10.6
starlette==0.41.3
' > /content/trellis-constraints.txt
export PIP_CONSTRAINT=/content/trellis-constraints.txt
python -m pip install --disable-pip-version-check 'setuptools<70' wheel ninja packaging
python -m pip install torch==2.4.0 torchvision==0.19.0 xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 scipy==1.14.1 pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless rembg==2.0.61 onnxruntime trimesh open3d xatlas pyvista==0.44.2 pymeshfix==0.17.0 igraph transformers==4.46.3 huggingface-hub==0.25.2 spconv-cu120 gradio==4.44.1 fastapi==0.115.6 pydantic==2.10.6 starlette==0.41.3
python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
python -m pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
export CPATH=/content/trellis-env/targets/x86_64-linux/include
for cuda_include in /content/trellis-env/lib/python3.10/site-packages/nvidia/*/include; do
  export CPATH="$CPATH:$cuda_include"
done
export LIBRARY_PATH=/content/trellis-env/targets/x86_64-linux/lib
mkdir -p /content/trellis-extensions
if [ ! -d /content/trellis-extensions/nvdiffrast/.git ]; then
  git clone --branch v0.3.3 https://github.com/NVlabs/nvdiffrast.git /content/trellis-extensions/nvdiffrast
fi
if [ ! -d /content/trellis-extensions/mip-splatting/.git ]; then
  git clone --recursive https://github.com/autonomousvision/mip-splatting.git /content/trellis-extensions/mip-splatting
fi
if ! python -c 'import nvdiffrast' >/dev/null 2>&1; then
  python -m pip install --no-build-isolation /content/trellis-extensions/nvdiffrast
fi
if ! python -c 'import diff_gaussian_rasterization' >/dev/null 2>&1; then
  python -m pip install --no-build-isolation /content/trellis-extensions/mip-splatting/submodules/diff-gaussian-rasterization
fi
python -m pip check
python -u -c "import torch,numpy,scipy,kaolin,nvdiffrast.torch,diff_gaussian_rasterization; from trellis.pipelines import TrellisImageTo3DPipeline; from trellis.utils import postprocessing_utils; assert torch.cuda.is_available(); nvdiffrast.torch.RasterizeCudaContext(); print('VERIFIED',torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name(0))"
printf '
TRELLIS dependency imports passed. Run Cell 2.
'
'''
pathlib.Path('/content/trellis-setup.sh').write_text(textwrap.dedent(setup_script))
with open('/content/trellis-setup.log', 'a') as log:
    proc = subprocess.Popen(['bash', '/content/trellis-setup.sh'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log.write(line); log.flush()
        print(line, end='')
    rc = proc.wait()
if rc:
    raise RuntimeError(f'Setup stopped with exit code {rc}; see /content/trellis-setup.log. Completed packages are reusable.')
