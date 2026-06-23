# genefer — HIP backend

This fork adds a **HIP backend** (`genefer_hip`) for AMD GPUs, alongside the OpenCL
(`geneferg`) and CUDA (`genefercu`) backends. All three build from the same tree; the
OpenCL and CPU paths are untouched. HIP targets RDNA/CDNA — including the Radeon VII —
and the hipRTC runtime compiles the kernel for whatever GPU is installed.

## Why

Same motivation as the CUDA backend, for AMD hardware: a native runtime (hipRTC + the
HIP runtime API) instead of OpenCL, with the CUDA optimisations carried over (graphs,
native-vector loads, the per-GPU autotuner). HIP is a near-clone of the CUDA API, so the
backend reuses **the same kernel source** (`cuda/kernel.cu`) — HIP is just a second
dialect of that source, not a third translation.

## Results (AMD Radeon RX 6800 XT, gfx1030, b=1000000)

**Correctness — bit-exact vs CUDA/OpenCL** (the integer NTT is deterministic, so `res64`
is an exact oracle). Validated on every code path:

| n | path exercised | check | result |
|---|---|---|---|
| 12 | `square64` | full PRP res64 | `6B30BF7EDE3EC2D5` ✓ |
| 16 | `square1024` | full PRP res64 | `3C0FD2F0B3C3C3EA` ✓ |
| 18 | `forward256`/`backward256` + `square1024` + proof | full PRP res64 | `1E856ADFAE74B7C0` ✓ |
| 20 | `square4096` | 40-squaring residue hash | ✓ |
| 21 | `forward64_9` multi-stage + `square512` | 40-squaring residue hash | ✓ |
| 22 | 3-prime RNS (`<3,false>`) + `square4096` + `reduce96` | 40-squaring residue hash | ✓ |

All identical to `genefercu` on a V100 and to `geneferg` (OpenCL) on the same AMD GPU.

**Performance** (steady-state, n=18): HIP **0.058 ms/bit** vs OpenCL **0.052 ms/bit** —
within ~11%, carrying the same hipGraph + native-vector optimisations as the CUDA port.

## The one non-obvious bug: clang miscompiles the `__align__` struct vectors

The CUDA backend represents `uint2_32`/`uint4_32` as `__align__(8/16)` **structs** (so
NVCC emits coalesced 64/128-bit loads). On AMD, **ROCm clang's loop unroller (-O2+)
miscompiles the square/mul NTT helpers** (`_square4x2v4` etc.) when the vector operands
are those structs: the residue is correct for the first few squarings (sparse data) then
diverges once both halves of a vector hold non-zero data.

Bisected with evidence: `-O0`/`-O1` correct, `-O2`/`-O3` wrong, `-fno-unroll-loops`
correct at full opt; `-fwrapv` and `-fno-strict-aliasing` do **not** fix it (so it is not
signed-overflow or aliasing UB). It is specific to the struct representation under clang —
NVRTC does not have it.

**Fix:** under `__HIP__`, define the vector types as clang `ext_vector_type` natives — the
very same `uint2`/`uint4`/`int4`/`long4` that OpenCL uses (OpenCL runs correctly on this GPU).
Native vectors unroll correctly **and** compile to packed `dwordx2/x4` loads, so they are
both correct at full `-O3` *and* ~2× faster than the `-fno-unroll-loops` workaround (0.058
vs 0.106 ms/bit). CUDA/NVRTC keeps the structs (NVRTC has no `ext_vector_type` and no bug).

```c
#if defined(__HIP__)
typedef uint_32 uint4_32 __attribute__((ext_vector_type(4)));   // native uint4, like OpenCL
#else
struct __align__(16) uint4_32 { uint_32 s0, s1, s2, s3; };      // NVCC coalesced load
#endif
```

## Build

Requires ROCm/HIP (hipcc + hipRTC). Opt-in, so the default build needs no ROCm:

```sh
cd genefer
make -f makefile_linux_x64 BOINC= hip      # local dev, no BOINC -> ../bin/genefer_hip
```

`genefer_hip -q -b 1000000 -n 12 -d 0` runs a quick PRP test on HIP device 0. (For local
validation without the BOINC tree, `dev/build_hip.sh` builds the same binary into `build_dev/`.)

**With BOINC:** like CUDA, the HIP app selects its GPU via `aid.gpu_device_num` (not
`boinc_get_opencl_ids`), so `libboinc_opencl.a` is *not* linked:

```sh
make -f makefile_linux_x64 hip BOINC_DIR=/path/to/boinc
```

## Design

- **Shared kernel source.** `cuda/kernel.cu` is compiled at runtime by both NVRTC (CUDA)
  and hipRTC (HIP); the only HIP-specific divergences are guarded by `#if defined(__HIP__)`:
  the native vector types (above) and turning off the NVIDIA inline-PTX carry chains
  (`PTX_ASM`) in favour of the portable C path.
- **`src/hip.h`** is the hipRTC + HIP runtime-API wrapper (mirrors `src/cu.h`): hipGraph
  capture/replay, async copies on one non-blocking stream, and runtime arch selection
  (`--gpu-architecture=<gcnArchName>` queried from the device, so one binary serves
  gfx900/gfx906 Radeon VII, gfx10xx RDNA, gfx9xx CDNA, …).
- **`src/cuda/kernel.h`** is `cuda/kernel.cu` embedded as a C++ string; `dev/gen_kernel_h.sh`
  regenerates it from the `.cu` and the `cuda`/`hip` make targets run it automatically, so
  the embedded source can never drift from the edited kernel.

### File layout (HIP-specific)

| File | Role |
|---|---|
| `src/hip.h` | hipRTC + HIP runtime-API wrapper (mirrors `src/cu.h`); hipGraph capture/replay |
| `src/transform_hip.cpp` | `transform::create_hip` factory (mirrors `transform_cu.cpp`) |
| `cuda/kernel.cu` | shared kernel source; native ext_vector types under `#if defined(__HIP__)` |
| `dev/gen_kernel_h.sh` | regenerates `src/cuda/kernel.h` from `cuda/kernel.cu` (run by the build) |
| `genefer/makefile_linux_x64` | `hip:` target (hipcc, `-lhiprtc`) |

## Status / next steps

- Done: bit-exact HIP backend across all transform-size branches, native-vector fix for the
  clang unroller miscompile, hipGraph + autotuner carried over, BOINC GPU-selection wired
  (`aid.gpu_device_num`), `make hip` target.
- Untested (no hardware here): **wavefront-64** parts (CDNA, Radeon VII gfx906). The kernels
  use explicit block barriers and explicit block sizes (not warp-implicit sync), and the
  OpenCL build runs on those GPUs, so wave64 is expected to work — but it has only been
  validated on RDNA2 (gfx1030, wave32). Worth a res64 check on a Radeon VII / MI-series card.
- Remaining for deployment: run-test under a live BOINC client; server-side app version /
  plan class; closing the last ~11% vs OpenCL (per-kernel occupancy tuning).
