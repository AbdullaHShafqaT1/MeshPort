# MeshPort cloud service

Open `meshport_colab.ipynb` in Google Colab, select an A100 GPU, and run all cells. Paste the complete final URL into the local MeshPort interface. Keep the runtime connected.

`verify_cell.py` and `setup_cell.py` own the environment cells; `service.py` owns generation; `build_notebook.py` embeds those sources and owns the launch and tunnel cells. Regenerate from the project root with `python colab/build_notebook.py`. The resulting notebook is self-contained and contains no saved cell outputs.

The notebook embeds the camera matching and texture refinement code from `photo_finish.py`. After baking a 2K generated atlas, the worker estimates camera angles from silhouette and color agreement, projects visible foreground pixels, and retains generated appearance outside the matched regions. The GLB uses an unlit material. Job details record per-view matches and the fraction of occupied texels receiving more than half-strength photo color.

The service supports one GPU job at a time, one to six views, authenticated polling, and GLB download. It blends all submitted views at each geometry and appearance sampling step using TRELLIS MultiDiffusion. The one-image path uses the standard TRELLIS sampler. MeshPort converts GLB to textured FBX locally. The existing `form-studio-v1` wire protocol and `STUDIO_KEY` environment name preserve compatibility with older notebooks.

Cloud setup downloads upstream tools, compiles extensions, and installs the TRELLIS environment. Some upstream sources are unpinned, so a fresh runtime may require dependency maintenance. Live Colab execution must be verified separately from local bridge tests. Restart the runtime to apply cloud source changes. Clear notebook outputs before sharing: the printed URL includes the authentication key.
