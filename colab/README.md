# MeshPort cloud service

Open `meshport_colab.ipynb` in Google Colab, select an A100 GPU, and run all cells. Paste the complete final URL into the local MeshPort interface. Keep the runtime connected.

`verify_cell.py` and `setup_cell.py` own the environment cells; `service.py` owns generation; `build_notebook.py` embeds those sources and owns the launch and tunnel cells. Regenerate from the project root with `python colab/build_notebook.py`. The resulting notebook is self-contained and contains no saved cell outputs.

The service supports one GPU job at a time, one to six views, authenticated polling, and GLB download. MeshPort converts GLB to textured FBX locally. The existing `form-studio-v1` wire protocol and `STUDIO_KEY` environment name are retained for compatibility with already running notebooks.

Cloud setup downloads upstream tools, compiles extensions, and installs the TRELLIS environment. Some upstream sources are unpinned, so a fresh runtime may require dependency maintenance. Live Colab execution must be verified separately from local bridge tests. Restart the runtime to apply cloud source changes. Clear notebook outputs before sharing: the printed URL includes the authentication key.
