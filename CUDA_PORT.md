# genefer — CUDA backend

This fork adds a **native CUDA backend** (`genefercu`) alongside the existing OpenCL
backend (`geneferg`). Both build from the same tree; the OpenCL path is byte-for-byte the
upstream behaviour. The CPU backends (`genefer`) are untouched. A **HIP / AMD** backend
also exists — see `HIP_PORT.md`.

## Why

Eliminate OpenCL host-side launch/queue overhead and open up CUDA-specific optimisation
headroom (CUDA Graphs, occupancy control, native vectorisation). The number-theoretic
transform itself is identical maths — the wins are in the host layer and in tuning.

## Results (Tesla V100, full clock, quick-test b=1000000)

| n | OpenCL `geneferg` | CUDA `genefercu` | speedup |
|---|---|---|---|
| 14 | 6s | 5s | 1.20× |
| 15 | 14s | 12s | 1.17× |
| 16 | 31s | 27s | 1.15× |
| 17 | 73s | 64s | 1.14× |

(min of 2 interleaved runs per backend; V100 measured at 1530 MHz with no throttle flags.
Use this controlled methodology — an earlier uncontrolled run produced ~2× higher absolute
times for both backends; cause not established, so disregard it. These are *kernel*
throughput; the move from runtime compile to AOT below changes only startup, not
steady-state speed.)

All residues (`res64`) are **bit-exact identical** to the OpenCL/CPU backends across every
transform-size branch (`<2,false> <2,true> <3,false> <3,true>`). The integer NTT is
deterministic, so res64 is the exact correctness oracle. The full proof path (`-p`
Pietrzak-Li proof → `-s` certificate) also matches: both backends produce the identical
server key (e.g. b=1000000 n=13 → `57A0281D3EB48FBE`).

Two optimisations got CUDA ahead of OpenCL (a naïve port was ~16–33% *slower* at large n):
1. **CUDA Graphs** — the per-squaring kernel sequence is identical every iteration (only
   `normalize1`'s `dup` arg changes), so it is captured once per `dup` value and replayed,
   removing per-launch CPU/driver overhead (~175K launches/s).
2. **Vector struct alignment** — `__align__(8/16)` on the `uint2_32`/`uint4_32` structs so
   the compiler emits coalesced 64/128-bit loads instead of scalar ones (matching OpenCL's
   native `uint2`/`uint4` vectorisation). ~34% kernel speedup. (The HIP backend gets the
   same effect with native `ext_vector_type` — see `HIP_PORT.md`.)

## Build — ahead-of-time, one script

The build is driven by **`dev/build.sh`** (not the makefile). For CUDA it produces a
**self-contained binary with every GPU architecture's machine code (SASS) baked in** — no
runtime compilation, and the CUDA toolkit is **not** needed to run it (only the driver /
`libcuda`).

```sh
bash dev/build.sh --cuda                 # -> build_dev/genefercu   (BOINC on by default)
bash dev/build.sh --cuda --no-boinc      # standalone binary, no BOINC libs
```

Toolchain locations are overridable via env vars (defaults shown). `BOINC_DIR` in
particular defaults to a developer path, so set it or use `--no-boinc`:

```sh
CUDA=/usr/local/cuda-12.9 \
BOINC_DIR=/path/to/boinc \
bash dev/build.sh --cuda
```

**Prerequisites:** the CUDA toolkit (12.9 recommended — provides `nvcc`/`ptxas`/`fatbinary`/
`cuobjdump`), `libgmp-dev`, a C++17 `g++`, an NVIDIA GPU + driver (the build runs a one-shot
device query while specialising the kernels), and BOINC static libs (or `--no-boinc`). The
same script builds the other backends: `--opencl` (→ `geneferg`; needs an OpenCL ICD, and
the Khronos headers are bundled in `Khronos/`) and `--hip` (→ `genefer_hip`; needs ROCm /
`hipcc`).

`genefercu -q -b 1000000 -n 12 -d 0` runs a quick PRP test on CUDA device 0; `genefercu -h`
runs the built-in validation sweep. The CUDA app selects its GPU via `aid.gpu_device_num`
(not `boinc_get_opencl_ids`), so `libboinc_opencl.a` is *not* linked (verified against
BOINC 8.3.0; runs bit-exact standalone).

## Design — build-time specialisation + embedded fatbins

genefer specialises the kernel per FFT size by injecting ~50 `#define`s (primes, Montgomery
constants, block sizes). **Originally this was done at runtime with NVRTC** — a source
string compiled on the fly, mirroring `clCreateProgramWithSource`. The current scheme moves
all of that to **build time**:

1. **Specialise** — for each of the 32 shipped configs (n ∈ 16..23 × the four RNS branches
   `<2,false> <2,true> <3,false> <3,true>`), `dev/dump_src.cpp` emits the fully specialised
   kernel source with the `#define`s baked in.
2. **Compile offline** — `nvcc` compiles each source to a **multi-arch fatbin** holding SASS
   for `sm_50 … sm_90`, `sm_100f`, `sm_120f`.
3. **Embed** — all 32 fatbins are linked into the binary via `.incbin` (`build_dev/fatbins.s`)
   plus a generated `(n, rns, is32) → blob` lookup table (`src/cuda/fatbins.h`).
4. **Load** — at runtime `transformGPU.h` looks up the fatbin for the active config and loads
   it with `cuModuleLoadData`; the **driver auto-selects** the SASS for the present GPU. No
   compilation, no PTX-JIT, no toolkit at runtime.

**Why AOT instead of NVRTC:** runtime NVRTC on Blackwell hit (a) a codegen bug (below) and
(b) PTX-version/driver coupling (newer-toolkit PTX rejected by older drivers). Shipping
finished SASS for every arch sidesteps both, and BOINC volunteers then need only the driver.

The Driver API still mirrors OpenCL object-for-object (`CUcontext`/`CUstream`/`CUmodule`/
`CUfunction`/`CUdeviceptr`), and `cuda/kernel.cu` is a hand translation of `ocl/kernel.cl`
(no macro shim). The NVRTC runtime-compile path still exists in `src/cu.h` (under `#else`,
i.e. when *not* built with `GENEFER_EMBED_FATBINS`) as a non-default fallback; the legacy
makefile `cuda:` target builds that variant.

### Blackwell (sm_120) codegen workaround

Compiling the hot `mulmod` kernel for `compute_120` makes nvcc/NVRTC's `cicc` (NVVM front
end) rewrite its 32×32→64 multiply from a single `mul.wide.u32` into `cvt.u64.u32 +
mul.lo.s64 + {lo,hi}-split` — a target-dependent consequence of packing the `uint2`/`uint4`
twiddle lanes into 64-bit registers and reading a lane via `shr.u64 …,32`. On `sm_120` this
produces **wrong results** (observed on an RTX 5070: the `-h` self-test passes n=16 and
fails n=17). The *same source* compiled through `compute_89` (Ada) is **correct** on the
same GPU.

So `build.sh` builds the Blackwell SASS through the clean `compute_89` front end and lets
`ptxas` retarget it (note: `nvcc -gencode arch=compute_89,code=sm_120f` is rejected, so it
is done in two steps):

```
nvcc -ptx -arch=compute_89   ->   ptxas -arch=sm_120f   (and -arch=sm_100f)
```

then `fatbinary`-merges those cubins with the directly-compiled non-Blackwell arches. The
divergent codegen persists on CUDA 12.9 **and** 13.2. The trigger is `cuda/kernel.cu:162`
(`mulmod`), reached via the `uint2`/`uint4` scalar-broadcast overloads. Full analysis,
SASS/PTX evidence, a reproducer, and an NVIDIA bug report are under `nvidia_bug/`.

### File layout

| File | Role |
|---|---|
| `dev/build.sh` | unified build (`--cuda`/`--opencl`/`--hip`): dump → fatbin → embed → link |
| `dev/dump_src.cpp` | build-time generator: emits the specialised kernel source per (n, RNS) |
| `cuda/kernel.cu` | CUDA kernels (hand-translated from `ocl/kernel.cl`) |
| `src/cu.h` | CUDA Driver-API wrapper (mirrors `src/ocl.h`); embedded-fatbin loader + CUDA-graph capture/replay; NVRTC fallback |
| `src/cuda/fatbins.h` | generated `(n,rns,is32) → fatbin blob` lookup table |
| `src/transform_cu.cpp` | `transform::create_cuda` factory (mirrors `transform_ocl.cpp`) |
| `src/transformGPU.h` | shared orchestration; `#if defined(CUDA)` aliases + the fatbin lookup/load seam + graph capture |
| `src/cuda/kernel.h` | `cuda/kernel.cu` embedded as a C++ string (used by the NVRTC fallback / source dump) |
| `src/transform.h`, `src/genefer.h`, `src/main.cpp` | shared; small `#if defined(CUDA)` device-selection branches |
| `genefer/makefile_linux_x64` | legacy `cuda:` target (NVRTC variant, `-lnvrtc`) |

## Correctness notes

- No floating point on the correctness path (integer NTT over RNS + Montgomery), so
  fast-math/FMA hazards do not apply.
- `cu.h` keeps all work (kernels + async H2D/D2H copies) on a single non-blocking stream —
  in-order like OpenCL's queue, and stream-capturable for graphs.
- `localWorkSize==0` (OpenCL driver-chosen) launches pick the largest power-of-two ≤256
  dividing the global size (those kernels have no bounds guard).

## Status / next steps

- Done: bit-exact CUDA backend, dual/tri build, CUDA Graphs + alignment (faster than
  OpenCL), per-GPU `TUNE` autotuner (builds + bit-exact), **AOT multi-arch fatbin build**
  (driver-only, no NVRTC in the shipped binary), **Blackwell `compute_89` workaround**,
  BOINC build (verified against BOINC 8.3.0).
- Pending: confirm the Blackwell fatbin on real `sm_120` hardware (the workaround SASS comes
  from the `compute_89` path an RTX 5070 already validated via arch override); run-test under
  a live BOINC client (GPU assignment) + server-side app version; Windows makefile (CUDA has
  no macOS).
- Investigated and declined (not worth the risk on this algorithm): Shoup modmul (~1 SASS
  slot), normalize→backward fusion (blocked — backward writes strided FFT output, the base-b
  carry needs contiguous order), on-the-fly twiddles (complicated by bit-reversed root storage).
- Further optimisation ideas: warp-shuffle butterflies, per-kernel occupancy tuning.

## Platform note

CUDA has no macOS support; the Mac makefiles keep OpenCL. The shipped `genefercu` embeds
finished SASS for `sm_50 … sm_90` + `sm_100f`/`sm_120f` and the driver selects the right one
at load — there is no PTX-JIT. A GPU whose architecture is not in that set (and is not
family-compatible with one of the embedded `*f` targets) would have no code to run, since
the AOT binary cannot JIT from PTX; add its arch to `ARCHES` in `dev/build.sh` and rebuild.
