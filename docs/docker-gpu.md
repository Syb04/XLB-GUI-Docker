# Docker / GPU setup

## Windows + WSL2

Windows NVIDIA driver and WSL2 must already work (`wsl nvidia-smi`). The Linux CUDA driver must **not** be installed inside WSL. This project uses the Windows driver through WSL's GPU bridge.

For a fresh Ubuntu WSL distribution, the included script installs Docker Engine, the Compose plugin and NVIDIA Container Toolkit from their official signed apt repositories. It starts Docker through systemd. Run it as root:

```powershell
wsl -d Ubuntu-24.04 -u root -- bash /mnt/h/20260902_XLB/xlb_workbench/scripts/setup-docker-ubuntu.sh
```

The path above is specific to this checkout. On a different machine, substitute its actual WSL path. On an existing Docker host use the vendor's installation and upgrade instructions to avoid conflicting engines.

Build and launch from PowerShell:

```powershell
cd H:\20260902_XLB\xlb_workbench
.\start-docker.ps1 -Gpu -LocalData
```

`-Gpu` includes CUDA 12 libraries and enables NVIDIA device access. `-LocalData` mounts `data/`, including existing projects. Omit it to use the default named volume. The wrapper uses a native Docker CLI when present, otherwise the selected WSL distribution. This PC's WSL Docker commands run as root; the application inside the image runs as UID 10001.

## Linux / Docker Desktop

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d --wait
docker compose -f compose.yaml -f compose.gpu.yaml exec workbench python -c "import jax; print(jax.devices('gpu'))"
```

Open http://127.0.0.1:8766 and select `cuda:0` under study settings. Merely starting the GPU image does not change saved CPU studies. CPU-only operation uses `docker compose up --build -d --wait`.

The default CUDA path runs LBM, thermal finite-volume calculations, Smagorinsky LES and temperature-dependent material evaluation on the GPU. Voxel meshing and file output remain on CPU. During stepping, only scalar reductions are downloaded; final fields are downloaded in one batched call for artifact output. A missing GPU is an error for a `cuda:0` study, never a silent CPU fallback. GPU memory preallocation is disabled in Compose so the solver does not reserve most device memory at startup.

`XLB_COMPUTE_BACKEND=auto` selects the device pipeline for CUDA and the NumPy reference thermal pipeline for CPU. Set `reference` in `.env` to explicitly benchmark the old host thermal path, or `device` to force the JAX pipeline on either selected device. Recreate the container after changing it. LBM storage remains FP32; thermal/material/LES arithmetic uses FP64. See [GPU pipeline measurements](gpu-pipeline.md).

The solver explicitly uses `jax_default_matmul_precision=highest`. XLB's JAX moments and equilibrium use tensor contractions; reduced-precision GPU defaults caused measurable CPU/GPU differences in the verification case. The setting requests full float32 arithmetic for those contractions, while distribution storage remains float32. See [JAX's precision documentation](https://docs.jax.dev/en/latest/201/precision.html).

Ports are bound to loopback. Set `XLB_PORT=8767` to run on a different host port. Persistent data survives `docker compose down`; removing its volume would remove the saved projects.

## Verification

With the server running, `python scripts/verify_gpu.py` submits the same LES/solid heat-transfer model to CPU and CUDA, checks the saved fields and compares results. It writes `docs/gpu-verification.json` and a portable project archive under `examples/`. This is an execution/agreement check, not a validated turbulent-flow benchmark.

The recorded test environment and numerical checks are in [verification.md](verification.md).

Official installation sources: [Docker Ubuntu installation](https://docs.docker.com/engine/install/ubuntu/), [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
