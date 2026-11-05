# @title 0. Verify runtime before setup
import sys, subprocess, pathlib
print("Colab kernel Python:", sys.version)
for cmd in [["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"], ["nvcc", "--version"]]:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, result.stderr)
print("Existing TRELLIS:", pathlib.Path("/content/TRELLIS").exists())
print("Input images:", [str(p) for p in pathlib.Path("/content").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
