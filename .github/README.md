# genefer22 — CUDA / HIP port

A GPU-backend port of [genefer22](https://github.com/galloty/genefer22) (Yves Gallot's
Generalized Fermat Number primality tester), adding **CUDA** and **HIP/ROCm** backends
alongside the original OpenCL — bit-exact with upstream — plus V100-tuned optimizations
and a CUDA-MPS multi-task throughput toolkit.

> Fork of **galloty/genefer22**. Upstream's README: [`README.md`](../README.md).
> Port design + results: **[CUDA_PORT.md](../CUDA_PORT.md)**.

## What's different in this fork
- **CUDA backend** (`cuda/kernel.cu`, `src/cu.h`): Driver API + ahead-of-time multi-arch
  fatbins (`sm_50…sm_120f`, no NVRTC in the shipped binary), CUDA Graphs, and a Blackwell
  `sm_120` codegen workaround (verified on an RTX 5070).
- **HIP/ROCm backend** sharing the CUDA kernel source.
- **Optimizations** (bit-exact, V100): per-size radix-split decompositions — `OPT-15`
  (n=22/23, ~6–8%) plus n=18/n=21 tuning. ~3–14% faster than the OpenCL backend at 1×.
- **MPS concurrency tooling** (`dev/genefer-mps-sweep.sh`): run several tasks per GPU —
  up to ~2.8× aggregate throughput at small GFN sizes on a V100.

## Build
    bash dev/build.sh --cuda     # AOT CUDA fatbin binary (build_dev/genefercu), BOINC on
    bash dev/build.sh --opencl   # original OpenCL
    bash dev/build.sh --hip      # HIP/ROCm

See [CUDA_PORT.md](../CUDA_PORT.md) for the design, fatbin/Blackwell scheme, and MPS results.
