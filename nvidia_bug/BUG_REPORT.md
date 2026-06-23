# Wrong results on sm_120 (Blackwell): cicc emits `mul.lo.s64` + `cvt` instead of `mul.wide.u32` for a provably-zero-upper 32x32->64 multiply in an integer-NTT kernel (CUDA 12.9 / 13.2)

## 1. Summary

An ahead-of-time-compiled (fatbin, no JIT) integer-NTT kernel produces a wrong
result on Blackwell (`sm_120`) when its virtual architecture is `compute_120`,
but produces the correct result on the same GPU when compiled through the
`compute_89` (Ada) virtual architecture and then assembled to `sm_120`. The same
problem sizes are correct on `sm_70` (V100). At the codegen level the only
relevant divergence we can demonstrate is in the NVVM/cicc front end: for
`compute_120` it rewrites exactly six 32x32->64 multiplies in the hot kernel from
a single `mul.wide.u32` into `cvt.u64.u32 + mul.lo.s64 + {lo,hi}-split`, whereas
`compute_89` keeps `mul.wide.u32`. All divergent forms are arithmetically
equivalent on paper; we have not been able to point at a single SASS instruction
that is wrong on its face, but the wrong runtime result tracks the `compute_120`
codegen path exactly.

Because the failing binary uses AOT SASS embedded via fatbin
(`cuModuleLoadData`), this is **not** a PTX-version / JIT-version mismatch. The
same wrong result was also originally observed with NVRTC runtime compilation
(CUDA 12.9); nvcc and NVRTC share the cicc/NVVM front end.

## 2. Affected component and kernel

- **Application:** genefer (genefer22), a Generalized-Fermat-Number PRP primality
  tester.
- **Kernel:** `square2048` — an integer NTT (Number-Theoretic Transform) that
  performs a fused mini forward-transform + pointwise square + inverse-transform
  on a 2048-element tile. The arithmetic uses an RNS/Montgomery modular-reduction
  core built from 32x32->64 unsigned multiplies.
- **Translation unit under analysis:** the n=17 problem-size variant,
  `build_dev/_fatsrc/n17_rns3_is0.cu`.

## 3. Environment / versions

| Item | Value |
| --- | --- |
| Failing GPUs (symptom OBSERVED) | NVIDIA GeForce RTX 5070, `sm_120`, 48 SMs, driver 13.0; plus a second, different Blackwell card |
| Dev box (static analysis only) | NVIDIA Tesla V100, `sm_70` — codegen/SASS inspection only; the wrong result was NOT reproduced here |
| OS | Linux (6.17.x kernel on dev box) |
| Toolkit A | CUDA 12.9 — `nvcc` V12.9.86 (`cuda_12.9.r12.9/compiler.36037853_0`), `ptxas` V12.9.86 |
| Toolkit B | CUDA 13.2 — `nvcc` V13.2.78 (`cuda_13.2.r13.2/compiler.37668154_0`), `ptxas` V13.2.78 |

Note on hardware split: every **runtime** ("wrong result") claim below is labeled
"observed on reporter's RTX 5070". Every **codegen** (PTX/SASS) claim is labeled
"statically verified on dev box (sm_70 V100)" and was produced purely by
`nvcc`/`cicc` PTX generation, `ptxas` SASS assembly, and `cuobjdump --dump-sass`
disassembly. No kernel was executed on Blackwell hardware by the analyst; the
runtime symptom comes from the reporter running on the RTX 5070.

## 4. Observed symptom (observed on reporter's RTX 5070)

Running the application's built-in self-test (`genefercu -h`), which is a
validation sweep:

- Test `550000000^{2^16}+1` (n=16) **PASSES**.
- The very next test `400000000^{2^17}+1` (n=17) reports **"test failed!"** — a
  **wrong residue**.

Key qualifications:

- The failing binary uses **AOT-compiled SASS embedded via fatbin**
  (`cuModuleLoadData`); there is **no runtime JIT**. So this is not a
  PTX-version / driver-JIT issue.
- The same wrong result was **also originally seen with NVRTC** runtime
  compilation under CUDA 12.9. nvcc and NVRTC share the cicc/NVVM front end.
- The same problem sizes **pass on `sm_70` (V100)**.
- The reporter confirmed the n=17 case **passes on the same Blackwell card** when
  the kernel is compiled through the **`compute_89` (Ada) virtual arch** instead
  of `compute_120` (arch override). So: `compute_120` codegen path -> WRONG on
  `sm_120`; `compute_89` codegen path -> CORRECT on `sm_120`.

## 5. Root-cause analysis

### 5.1 PTX divergence (statically verified on dev box)

For identical source (`n17_rns3_is0.cu`), `cicc` emits divergent PTX as a pure
function of the virtual architecture. Within the `square2048` entry body
(`compute_120` PTX lines 24987-28291; `compute_89` PTX lines 26439-29925):

| PTX idiom | compute_120 | compute_89 |
| --- | --- | --- |
| `mul.lo.s64` | 6 | 0 |
| `mul.wide.u32` | 203 | 213 |
| `cvt.u64.u32` | 10 | 3 |
| `mul.hi.*` | 192 | 192 |

`cicc` rewrote exactly **six** multiplies: `compute_89` emits a single
`mul.wide.u32`; `compute_120` emits `cvt.u64.u32` (widen the 32-bit operand) +
`mul.lo.s64` + a `{lo,hi}` split (`cvt.u32.u64` for the low word,
`mov.b64 {tmp,hi}` for the high word). Both forms appear in the SAME reduction
idiom on adjacent lines, e.g.:

```
mul.wide.u32 %rd119, %r1285, %r1094;                 // compute_89 form (kept wide)
cvt.u64.u32  %rd120, %r1154;                          // compute_120 form
mul.lo.s64   %rd121, %rd98, %rd120;                   //   where %rd98 = shr.u64 %rd97,32
```

In all six sites the operands are **provably zero in their upper 32 bits** (one is
`shr.u64 ...,32`, the other is `cvt.u64.u32`), so the signed low-64 product equals
the unsigned wide product — i.e. the rewrite is arithmetically equivalent on
paper.

The mechanism (from the full kernel's PTX): on `compute_120`, NVVM packs the
vectorized `v2.u32` twiddle load into a 64-bit register (`mov.b64 {lo,hi}`) and
takes the high lane via `shr.u64 ...,32` (now 64-bit-typed). Because the operand
is now 64-bit-typed, `lhs * (uint64)w.s1` can no longer fold to `mul.wide.u32`
and becomes `mul.lo.s64`. `compute_89` never performs this packing
(`mov.b64` = 0 across the whole module), so it keeps two 32-bit lanes and emits
`mul.wide.u32`. This is an NVVM **target-dependent packing heuristic**.

Whole-module counts (full kernel, statically verified on dev box):
`compute_120` `mul.lo.s64` = 56, `mov.b64` = 3620; `compute_89` `mul.lo.s64` = 0,
`mov.b64` = 0.

Evidence: `nvidia_bug/analysis/ptx_divergence_excerpt_120.txt`,
`nvidia_bug/analysis/ptx_mul_lo_s64_sites_120.txt`,
`nvidia_bug/repro/_ptx/full_120.ptx`, `nvidia_bug/repro/_ptx/full_89.ptx`.

### 5.2 SASS divergence (statically verified on dev box)

Normalized instruction streams (address + encoding bytes stripped): **2888**
instructions for `compute_120 -> sm_120` vs **2872** for `compute_89 -> sm_120`
(+16 in the buggy path). Opcode deltas (120 minus 89):

| Opcode | compute_120 | compute_89 | delta |
| --- | --- | --- | --- |
| IADD3 | 1055 | 1041 | +14 |
| HFMA2 | 7 | 2 | +5 |
| MOV | 6 | 2 | +4 |
| IADD.64 | 6 | 3 | +3 |
| MOV.64 | 1 | 0 | +1 |
| IMAD | 194 | 193 | +1 |
| IMAD.WIDE.U32 | 208 | 212 | -4 |
| IMAD.SHL.U32 | 5 | 16 | -11 |
| SHF.R.U32.HI | 7 | 10 | -3 |

**Critical qualitative result:** the modular-reduction (RNS/Montgomery) core is
**instruction-for-instruction identical in selection** between the two builds.
Both use only **unsigned** multiplies (`IMAD.WIDE.U32` present in both,
`IMAD.HI.U32` = 192 in both, `SHF.R.S32.HI` = 356 in both). There are **zero**
signed `IMAD.WIDE` / `IMAD.HI` in either build, and no narrow/signed multiply
appears. So the reduction core is **NOT** a candidate, and **no SASS instruction
was found that is wrong on its face**.

The real difference is confined to **address/index strength-reduction**. The
`compute_89` path forms addresses as `base + index*<immediate>` (`IMAD.WIDE.U32`
with an immediate power-of-two multiplier: 19 sites; plus 16 `IMAD.SHL.U32`),
while the `compute_120` path materializes the scale into a register
(`MOV Rk,0x4`) and/or pre-adds the index (`IADD3`) and uses a register-multiplier
`IMAD.WIDE.U32` (201 vs 191 register-multiplier sites; only 7 immediate-multiplier
sites). The remainder of the ~5254-line diff is register-allocation / scheduling
cascade noise, not selection differences.

**Strongest static candidate** (the region where the two builds actually diverge):
the `compute_120` register-multiplier `IMAD.WIDE.U32` index path that only the
`mul.lo.s64`-derived PTX produces, with its feeding `IADD3`/`MOV` scaffolding.
First concrete instances in `nvidia_bug/analysis/sass_square2048_120.txt`:

```
IADD3 R26, PT, PT, R2, R29, RZ;
IMAD.WIDE.U32 R26, R26, R27, UR10;
MOV R33, 0x4;
IMAD.WIDE.U32 R32, R2, R33, UR10;
```

The arithmetically-equivalent `compute_89 -> sm_120` forms are:

```
IMAD.SHL.U32 R9, R29, 0x20000, RZ;
IMAD.WIDE.U32 R34, R9, 0x4, R34;
IMAD.WIDE.U32 R24, R27, 0x4, R34;
```

Evidence: `nvidia_bug/analysis/sass_square2048.diff`,
`nvidia_bug/analysis/sass_reduction_core_windows.txt`,
`nvidia_bug/analysis/sass_square2048_120.txt`,
`nvidia_bug/analysis/sass_square2048_89.txt`.

### 5.3 cicc-vs-ptxas assessment (qualified)

The divergence **originates in cicc (PTX generation)**, not ptxas: the two PTX
files already differ for identical source (`mul.wide.u32` vs
`cvt.u64.u32 + mul.lo.s64 + {lo,hi}-split`), and the choice is a pure function of
the virtual arch (Blackwell-class: `sm_100`/`sm_120`/`sm_120a` all = 6 rewritten
multiplies; `sm_89`/`sm_90` = 0).

As a **static** matter, ptxas appears to lower **both** PTX forms correctly and
equivalently: the resulting SASS uses only unsigned `IMAD.WIDE.U32` /
`IMAD.HI.U32`, the modular-reduction core is byte-identical in selection, and no
signed/narrow multiply appears. We could not confirm by static inspection that
any instruction miscomputes, because both PTX forms and both SASS forms are
arithmetically equivalent on paper and no runtime oracle was available on the dev
box.

If a real miscompile exists (and the reporter's RTX 5070 wrong result indicates
one does), the qualified static candidates, in order, are:

1. **ptxas** lowering/scheduling of the register-multiplier `IMAD.WIDE.U32` index
   path that only the `compute_120` PTX triggers — i.e. a latent ptxas bug
   exposed only by the `mul.lo.s64`-derived index expressions; or
2. a **cicc** bug — but only if the `mul.lo.s64` rewrite were ever applied to
   operands whose upper 32 bits are **not** provably zero. That is **not** the
   case in these six sites (one operand is `shr.u64` by 32, the other is
   `cvt.u64.u32`; both are provably zero-upper, so signed low-64 == unsigned wide
   product). So on the evidence here, candidate (1) is favored.

In short: cicc's target-dependent decision to emit `mul.lo.s64` for Blackwell
virtual archs is the trigger; ptxas's lowering of that index path is the most
likely place the wrong result is actually produced. This cannot be pinned more
tightly without execution on Blackwell hardware.

## 6. Version matrix (statically verified on dev box)

| Toolkit | nvcc | ptxas | Reproduces codegen divergence | `mul.lo.s64` count (square2048 body) |
| --- | --- | --- | --- | --- |
| CUDA 12.9 | V12.9.86 (`compiler.36037853_0`) | V12.9.86 | yes | 6 |
| CUDA 13.2 | V13.2.78 (`compiler.37668154_0`) | V13.2.78 | yes | 6 |

The divergent cicc codegen persists through the latest toolkit installed on the
dev box (CUDA 13.2).

## 7. Reproduction

### 7.1 PTX codegen divergence — NO GPU needed (reliable; statically verified on dev box)

This is the dependable reproducer of the codegen divergence:

```
cd /home/ian/builds/primegrid/genefer22/nvidia_bug/repro
./verify_ptx.sh
# or manually:
SRC=/home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu
/usr/local/cuda-12.9/bin/nvcc -ptx -arch=compute_120 -diag-suppress 177 $SRC -o full_120.ptx
/usr/local/cuda-12.9/bin/nvcc -ptx -arch=compute_89  -diag-suppress 177 $SRC -o full_89.ptx
grep -c mul.lo.s64 full_120.ptx   # -> 56
grep -c mul.lo.s64 full_89.ptx    # -> 0
grep -c mov.b64    full_120.ptx   # -> 3620 (64-bit packing)
grep -c mov.b64    full_89.ptx    # -> 0
```

Files: `nvidia_bug/repro/verify_ptx.sh`,
`nvidia_bug/repro/_ptx/full_120.ptx`, `nvidia_bug/repro/_ptx/full_89.ptx`.

### 7.2 Minimal standalone kernel — partial (did NOT reproduce the asymmetry)

A small standalone kernel (`nvidia_bug/repro/repro_minimal.cu`) did **not**, on
its own, reproduce the **asymmetric** PTX divergence the full kernel shows. Probe
results (CUDA 12.9, `nvcc -ptx`):

- `probe_force64`: reproduces the **exact** suspect `compute_120` idiom
  byte-for-byte (`cvt.u64.u32` + `mul.lo.s64` + `cvt.u32.u64`/`mov.b64{tmp,hi}`;
  16 `mul.lo.s64` + 8 `cvt.u64.u32`) — but emits it **symmetrically** on both
  `compute_89` AND `compute_120` (16/16). Correct problematic form, but not a
  *divergence*.
- `probe_narrow` (control): both arches emit a single `mul.wide.u32` (the good
  form).
- `probe_vec_pressure`: reproduces the **mechanism** (the target-dependent
  2x32->64 packing — `mov.b64` 16x on `compute_120`, 0x on `compute_89`), but at
  this scale ptxas still folds the multiply to `mul.wide.u32` (17 vs 18), so no
  `mul.lo.s64` surfaces.

Conclusion: the asymmetry is an NVVM target-dependent packing heuristic that only
fires under the full kernel's register pressure/structure; isolating it in a tiny
kernel forces an all-or-nothing outcome. **The full kernel is the reliable
reproducer.** File: `nvidia_bug/repro/repro_minimal.cu`.

### 7.3 On-device A/B check harness (PENDING Blackwell hardware)

A driver harness is provided for someone with a Blackwell GPU:

```
cd /home/ian/builds/primegrid/genefer22/nvidia_bug/repro
./check_on_blackwell.sh --full
```

It links a driver against the real `square2048` from `n17_rns3_is0.cu`, builds it
path A (`compute_120,sm_120`) vs path B (`compute_89,sm_120`), runs identical
input on the GPU, and compares output hashes. Differing hashes on `sm_120` =>
the codegen divergence is observable at runtime. Caveat: the harness launches a
single block clamped to the kernel's `maxThreadsPerBlock` (observed 128 for the
standalone build on the dev box), which may not exercise the exact failing
schedule. Files: `nvidia_bug/repro/check_on_blackwell.sh`,
`nvidia_bug/repro/harness.cu`.

### 7.4 Authoritative full-app reproducer (observed on reporter's RTX 5070)

The definitive, end-to-end wrong-result reproducer is the application's own
validation sweep on a Blackwell GPU:

```
cd /home/ian/builds/primegrid/genefer22/build_dev
./genefercu -h
```

On Blackwell, the `compute_120` build mis-validates the n=17 case
(`400000000^{2^17}+1`, source `n17_rns3_is0.cu`) — "test failed!" — while a
`compute_89`-built binary validates cleanly on the **same** GPU. The full-grid
launch here is authoritative; the single-block harness in 7.3 is only a
supplementary check.

Supporting files: `nvidia_bug/repro/README.md`,
`nvidia_bug/repro/evidence_square2048.txt`,
`nvidia_bug/analysis/findings.md`.

## 8. Workaround

Compile the affected kernel(s) through the **`compute_89` (Ada) virtual
architecture** and let ptxas assemble to the Blackwell target:

```
nvcc -gencode arch=compute_89,code=sm_120 ...
# equivalent to: nvcc -ptx -arch=compute_89   then   ptxas -arch=sm_120
```

This routes the `compute_89` NVVM front end (which emits `mul.wide.u32`, no
`mul.lo.s64` packing) through ptxas for the `sm_120` target. The reporter
confirmed on the RTX 5070 that this path produces the **correct** residue for the
n=17 case, while the native `compute_120` path produces the wrong one. (`sm_120f`
may be substituted for `sm_120` as appropriate for the target.)

## 9. Verification status — what is established at each level

| Claim | Status |
| --- | --- |
| n=16 passes, n=17 reports "test failed!" (wrong residue) | **Observed on reporter's RTX 5070** (driver 13.0); failing binary is AOT fatbin SASS, no JIT |
| Same wrong result also seen via NVRTC (CUDA 12.9) | **Observed on reporter's RTX 5070** |
| Same problem sizes pass on V100 (`sm_70`) | **Observed** (reporter) |
| `compute_89`-built kernel is CORRECT on the same Blackwell GPU; `compute_120`-built is WRONG | **Confirmed on reporter's RTX 5070** — the shipped AOT fatbin whose `sm_120f` SASS is built via `compute_89` passes the self-test on the 5070, on the same GPU where the native `compute_120` fatbin fails (both AOT, identical packaging; only the virtual arch differs) |
| cicc emits `mul.lo.s64`+`cvt` (6 in square2048 body, 56 module-wide) on `compute_120` vs `mul.wide.u32` (0 `mul.lo.s64`) on `compute_89`, identical source | **Statically verified on dev box** (sm_70 V100), CUDA 12.9 and 13.2 |
| The six rewritten operands are provably zero-upper-32 (so the rewrite is arithmetically equivalent on paper) | **Statically verified on dev box** |
| SASS modular-reduction core is selection-identical between builds; only the index/address multiply path diverges; no SASS instruction wrong on its face | **Statically verified on dev box** |
| ptxas lowering/scheduling of the `compute_120` register-multiplier index path is the leading candidate for the actual miscompile | **Hypothesis** — consistent with static evidence + reporter's runtime symptom; NOT proven |
| Minimal standalone kernel reproduces the wrong RESULT | **NOT reproduced** — minimal kernel does not even reproduce the codegen asymmetry; ptxas recovers `IMAD.WIDE.U32` for it on the dev box |
| On-device A/B: `compute_89` fatbin correct vs `compute_120` fatbin wrong on the same `sm_120` | **Confirmed on reporter's RTX 5070** via the full genefer self-test (authoritative). The single-block `check_on_blackwell.sh --full` harness was not separately run (analyst dev box is a V100) |

### Honesty caveat

This analysis is **static only** with respect to codegen: no kernel was executed
on Blackwell by the analyst, and no SASS instruction was found that is wrong on
its face. All divergent PTX and SASS forms are arithmetically equivalent on
paper. The wrong **result** is real and reproducible **on the reporter's RTX 5070
via the full genefer self-test**, and it tracks the `compute_120` codegen path
exactly: the shipped AOT fatbin built via `compute_89` **passes** on the same 5070
where the native `compute_120` fatbin **fails** — an A/B on identical hardware and
packaging in which only the virtual architecture differs. What we ask NVIDIA to
confirm is which stage actually miscomputes — our leading hypothesis is ptxas's
lowering/scheduling of the `mul.lo.s64`-derived index path that only the
`compute_120` cicc output produces.
