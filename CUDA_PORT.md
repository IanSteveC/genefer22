# genefer — CUDA backend

This fork adds a **native CUDA backend** (`genefercu`) alongside the existing OpenCL
backend (`geneferg`). Both are buildable from the same tree; the OpenCL path is
byte-for-byte the upstream behaviour. The CPU backends (`genefer`) are untouched.

## Why

Eliminate OpenCL host-side launch/queue overhead and open up CUDA-specific
optimisation headroom (CUDA Graphs, occupancy control, native vectorisation).
The number-theoretic transform itself is identical maths — the wins are in the
host layer and in tuning.

## Results (Tesla V100, full clock, quick-test b=1000000)

| n | OpenCL `geneferg` | CUDA `genefercu` | speedup |
|---|---|---|---|
| 14 | 6s | 5s | 1.20× |
| 15 | 14s | 12s | 1.17× |
| 16 | 31s | 27s | 1.15× |
| 17 | 73s | 64s | 1.14× |

(min of 2 interleaved runs per backend; V100 measured at 1530 MHz with no throttle
flags. Use this controlled methodology — an earlier uncontrolled run produced
~2× higher absolute times for both backends; cause not established, so disregard it.)

All residues (`res64`) are **bit-exact identical** to the OpenCL/CPU backends across
every transform-size branch (`<2,false> <2,true> <3,false> <3,true>`). The integer
NTT is deterministic, so res64 is the exact correctness oracle. The full proof path
(`-p` Pietrzak-Li proof → `-s` certificate) also matches: both backends produce the
identical server key (e.g. b=1000000 n=13 → `57A0281D3EB48FBE`).

Two optimisations got CUDA ahead of OpenCL (a naïve port was ~16–33% *slower* at large n):
1. **CUDA Graphs** — the per-squaring kernel sequence is identical every iteration
   (only `normalize1`'s `dup` arg changes), so it is captured once per `dup` value
   and replayed, removing per-launch CPU/driver overhead (~175K launches/s).
2. **Vector struct alignment** — `__align__(8/16)` on the `uint2_32`/`uint4_32`
   structs so the compiler emits coalesced 64/128-bit loads instead of scalar ones
   (matching OpenCL's native `uint2`/`uint4` vectorisation). ~34% kernel speedup.

## Build

Requires the CUDA toolkit (NVRTC + driver). The CUDA target is opt-in so the default
OpenCL build needs no CUDA installed:

```sh
cd genefer
make -f makefile_linux_x64 cuda            # builds ../bin/genefercu
# (BOINC off for local dev:)  make -f makefile_linux_x64 BOINC= cuda
```

`genefercu -q -b 1000000 -n 12 -d 0` runs a quick PRP test on CUDA device 0.

## Design

- **Runtime compilation via NVRTC + the CUDA Driver API.** genefer specialises the
  kernel per FFT size by injecting ~50 `#define`s (primes, Montgomery constants,
  block sizes) as a text prelude at runtime. NVRTC consumes a source string exactly
  like `clCreateProgramWithSource`, so this model is preserved; the Driver API mirrors
  OpenCL object-for-object (`CUcontext/CUstream/CUmodule/CUfunction/CUdeviceptr`).
- **Separate kernel source, shared orchestration.** `cuda/kernel.cu` is a hand
  translation of `ocl/kernel.cl` (no macro shim). The host orchestration
  (`transformGPU.h`) is shared and selects the backend with `#if defined(CUDA)`.

### File layout

| File | Role |
|---|---|
| `src/cu.h` | CUDA Driver-API + NVRTC runtime wrapper (mirrors `src/ocl.h`); CUDA-graph capture/replay |
| `cuda/kernel.cu` | CUDA kernels (translated from `ocl/kernel.cl`) |
| `src/cuda/kernel.h` | `cuda/kernel.cu` embedded as a C++ string (for BOINC/offline; regenerated at runtime otherwise) |
| `src/transform_cu.cpp` | `transform::create_cuda` factory (mirrors `transform_ocl.cpp`) |
| `src/transformGPU.h` | shared orchestration; `#if defined(CUDA)` aliases (`gpu_mem`/`gpu_kernel`/…) + the source seam + graph capture |
| `src/transform.h`, `src/genefer.h`, `src/main.cpp` | shared; small `#if defined(CUDA)` device-selection branches |
| `genefer/makefile_linux_x64` | `cuda:` target (`-lcuda -lnvrtc`) |

## Correctness notes

- No floating point on the correctness path (integer NTT over RNS + Montgomery), so
  fast-math/FMA hazards do not apply.
- `cu.h` keeps all work (kernels + async H2D/D2H copies) on a single non-blocking
  stream — in-order like OpenCL's queue, and stream-capturable for graphs.
- `localWorkSize==0` (OpenCL driver-chosen) launches pick the largest power-of-two
  ≤256 dividing the global size (those kernels have no bounds guard).

## Status / next steps

- Done: bit-exact CUDA backend, dual build, CUDA Graphs + alignment (faster than OpenCL).
- Not yet ported: the `TUNE` autotuner, full BOINC CUDA device wiring (a minimal
  `aid.gpu_device_num` path exists), Windows/macOS makefiles (CUDA has no macOS).
- Further optimisation ideas: warp-shuffle butterflies, per-kernel occupancy tuning.

## Platform note

CUDA has no macOS support; the Mac makefiles keep OpenCL. The CUDA backend targets
the device's compute capability via NVRTC PTX-JIT (V100 = `compute_70`).
