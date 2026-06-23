# NVIDIA Blackwell (sm_120) mulmod codegen bug — reproducer

genefer22 (PrimeGrid GFN, CUDA backend) computes WRONG residues on NVIDIA
Blackwell (RTX 50xx, `sm_120`) when the GPU kernels are compiled for
`compute_120`. The same source compiled for `compute_89` is correct on the same
`sm_120` hardware. The fault is in the `sm_120` code path for the Montgomery
32×32→64 multiply inside `mulmod()`.

* Toolkit observed: CUDA 12.9 (V12.9.86), `nvcc`/`ptxas`.
* Affected kernel (representative): `square2048` in
  `build_dev/_fatsrc/n17_rns3_is0.cu` (and every other kernel that uses `mulmod`).

---

## The divergence (what differs)

`mulmod()` does a 32×32→64 unsigned multiply and uses **both** halves of the
product:

```c
const uint_64 t  = lhs * (uint_64)(rhs);          // 32x32 -> 64
const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
const uint_32 mp = __umulhi(lo * pq.s1, pq.s0);
return submod(hi, mp, pq.s0);
```

`nvcc -ptx` lowers that one multiply differently per target:

| target        | PTX for the multiply                                   |
|---------------|--------------------------------------------------------|
| `compute_89`  | a single `mul.wide.u32 %rd, lhs, rhs`                  |
| `compute_120` | `cvt.u64.u32` + **`mul.lo.s64`** + extract `{lo,hi}`   |

Root cause visible in the PTX: on `compute_120` NVVM **packs the vectorized
`v2.u32` twiddle load into a 64-bit register** (`mov.b64 {lo,hi}`) and takes the
high word with `shr.u64 …, 32`. That high word is now 64-bit-typed, so the
following `lhs * (uint64)w.s1` can no longer fold to `mul.wide.u32` and becomes
`mul.lo.s64`. `compute_89` never does this packing (`mov.b64` count = **0** in
the whole module) so it keeps two clean 32-bit lanes and emits `mul.wide.u32`.

Whole-module counts (CUDA 12.9, `nvcc -ptx`, `n17_rns3_is0.cu`):

```
compute_89 : mul.lo.s64=0    mul.wide.u32=3982   mov.b64=0
compute_120: mul.lo.s64=56   mul.wide.u32=3977   mov.b64=3620
```

The two forms are arithmetically identical. On real `sm_120` the `compute_120`
build gives wrong results, so the `sm_120` lowering/scheduling of the
`mul.lo.s64` form (under the kernel's register pressure) is the suspect.

See `evidence_square2048.txt` for the exact side-by-side PTX.

---

## Files

| file                     | what it is                                                                 |
|--------------------------|----------------------------------------------------------------------------|
| `repro_minimal.cu`       | minimal probe kernels (`probe_force64`, `probe_narrow`, `probe_vec_pressure`) |
| `verify_ptx.sh`          | **no GPU needed** — compiles probes + full kernel both ways and prints the PTX counts |
| `harness.cu`             | runtime A/B host harness (CPU reference + GPU; PASS/FAIL)                   |
| `check_on_blackwell.sh`  | **run on a 50xx** — builds `harness.cu` two ways, runs both; `--full` adds the real `square2048` |
| `evidence_square2048.txt`| extracted side-by-side PTX from the real kernel                            |
| `_ptx/`                  | generated PTX (incl. `full_120.ptx`, `full_89.ptx`) produced by `verify_ptx.sh` |

---

## 1. Verify the PTX divergence locally (no GPU)

```bash
cd nvidia_bug/repro
./verify_ptx.sh          # uses CUDA=/usr/local/cuda-12.9 by default
```

Expected (the load-bearing line is the full kernel row):

```
[2] FULL real kernel  (n17_rns3_is0.cu)  -- whole module
  n17_rns3_is0 (module)  120[lo.s64=56 wide=3977 cvt64=166 movb64=3620]  89[lo.s64=0 wide=3982 cvt64=186 movb64=0]
```

`compute_120` has `mul.lo.s64 > 0` and a large `mov.b64` count; `compute_89` has
both at 0. That is the bug-relevant codegen divergence, confirmed locally.

### Note on the minimal probes (honest status)

* `probe_force64` reproduces the **exact** suspect `compute_120` idiom
  (`cvt.u64.u32 + mul.lo.s64 + extract{lo,hi}`) byte-for-byte — but it emits it
  on **both** `compute_89` and `compute_120` (16/16), i.e. it does **not** show
  the asymmetric "89=wide vs 120=lo.s64" crossover that the full kernel shows.
* `probe_vec_pressure` reproduces the **mechanism** (the 64-bit packing:
  `mov.b64` appears only on `compute_120`, 0 on `compute_89`) but at small scale
  ptxas still folds the multiply back to `mul.wide.u32`.

In short: **a small standalone kernel did not, by itself, reproduce the
asymmetric PTX divergence** the way the full kernel does. The asymmetry is driven
by NVVM's target-dependent 2×32→64 packing decision, which only fires under the
full kernel's structure/pressure. The reliable reproducer is therefore the full
kernel (below). The minimal probe is still useful as the *exact* PTX form to
hand NVIDIA, and as a runtime A/B candidate on real hardware.

---

## 2. Confirm wrong RESULTS on a Blackwell GPU (5070 / sm_120)

**Runtime wrong-result confirmation requires `sm_120` hardware** (an RTX 50xx).
It could not be done on the development box (Tesla V100, `sm_70`); the suspect
lowering only mis-executes on `sm_120`. On the developer's box `ptxas` recovers
`IMAD.WIDE.U32` for the minimal kernel's `mul.lo.s64`, and the harness PASSes —
which proves nothing about `sm_120`.

On the Blackwell machine:

```bash
cd nvidia_bug/repro
./check_on_blackwell.sh            # minimal probe, both build paths
./check_on_blackwell.sh --full     # ALSO runs the real square2048 both paths
```

It builds the harness two ways and runs both against an identical CPU reference:

* **path A (SUSPECT)** : `nvcc -gencode arch=compute_120,code=sm_120`
  → mulmod multiply = `mul.lo.s64` form
* **path B (REFERENCE)**: `nvcc -gencode arch=compute_89,code=sm_120`
  (equivalently `nvcc -ptx -arch=compute_89` then `ptxas -arch=sm_120`)
  → mulmod multiply = `mul.wide.u32` form

If the bug bites, **path A FAILs and path B PASSes** against the same CPU
reference (minimal probe), and for `--full` the two `square2048` output hashes
**differ**. If the minimal probe does NOT diverge on the 50xx (likely — see the
honesty note above), use the full-kernel fallback.

---

## 3. Full-kernel fallback repro (the reliable one)

The genefer kernels are the dependable reproducer. Two ways, easiest first.

### 3a. Run the genefer app (end-to-end, definitive)

```bash
cd build_dev
./genefercu -h          # runs a built-in validation sweep across FFT sizes
```

`-h` validates `b^{2^16}+1 … b^{2^23}+1`. The `2^17` case is `n17` (the kernel in
`_fatsrc/n17_rns3_is0.cu`). On Blackwell the `compute_120` build mis-validates
(`n17` fails / wrong residue) while a `compute_89`-built binary validates
cleanly on the same GPU. (The CUDA backend JIT-compiles these kernels via NVRTC
for the device arch; build/run a `compute_120` vs `compute_89` variant to
compare. The genefer source/makefiles are under the repo root and `build_dev/`.)

### 3b. Compile the real kernel two ways and diff codegen / output

```bash
cd nvidia_bug/repro
CUDA=/usr/local/cuda-12.9
SRC=/home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu

# PTX divergence (no GPU): see counts differ
$CUDA/bin/nvcc -ptx -arch=compute_120 -diag-suppress 177 $SRC -o full_120.ptx
$CUDA/bin/nvcc -ptx -arch=compute_89  -diag-suppress 177 $SRC -o full_89.ptx
grep -c 'mul.lo.s64'   full_120.ptx   # 56
grep -c 'mul.lo.s64'   full_89.ptx    # 0

# On a Blackwell GPU: build the real square2048 both ways and compare outputs
./check_on_blackwell.sh --full
```

`check_on_blackwell.sh --full` links a tiny driver against the real `square2048`
from `n17_rns3_is0.cu`, runs it under both build paths on identical input, and
compares output hashes. Differing hashes on the 50xx ⇒ the codegen divergence is
observable at runtime. (The driver launches a single block sized to the kernel's
`maxThreadsPerBlock`; the genefer app in 3a exercises the full grid and is the
authoritative check.)

---

## Environment

* Working dir: `/home/ian/builds/primegrid/genefer22`
* CUDA: `/usr/local/cuda-12.9` (override with `CUDA=…`)
* Override the full-kernel path with `FULL=/path/to/n17_rns3_is0.cu`
