# Static analysis: square2048 compute_120 codegen divergence (CUDA 12.9 / 13.2)

STATIC ANALYSIS ONLY. No Blackwell (sm_120) GPU was available; nothing here was
executed on hardware. All conclusions are from nvcc/cicc PTX, ptxas SASS, and
cuobjdump disassembly only. No runtime behavior is claimed or verified.

Source: /home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu
Kernel: square2048
Toolkits: CUDA 12.9 (V12.9.86) and CUDA 13.2 (V13.2.78), both under /usr/local.

---

## 1. Builds (exact commands)

Buggy path (compute_120 -> sm_120 directly):

    /usr/local/cuda-12.9/bin/nvcc -gencode arch=compute_120,code=sm_120 --cubin \
      /home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu -o /tmp/sass120.cubin

Clean reference path (compute_89 PTX, then ptxas to sm_120):

    /usr/local/cuda-12.9/bin/nvcc -ptx -arch=compute_89 \
      /home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu -o /tmp/p89.ptx
    /usr/local/cuda-12.9/bin/ptxas -arch=sm_120 /tmp/p89.ptx -o /tmp/sass89.cubin

PTX for divergence comparison:

    /usr/local/cuda-12.9/bin/nvcc -ptx -arch=compute_120 \
      /home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu -o /tmp/p120.ptx

Disassembly + extraction of the single square2048 body:

    /usr/local/cuda-12.9/bin/cuobjdump --dump-sass /tmp/sass120.cubin > /tmp/sass120_full.txt
    /usr/local/cuda-12.9/bin/cuobjdump --dump-sass /tmp/sass89.cubin  > /tmp/sass89_full.txt
    # square2048 is the function from its "Function : square2048" header to the next "Function :"
    sed -n '61592,67372p' /tmp/sass120_full.txt > analysis/sass_square2048_120.txt
    sed -n '61912,67660p' /tmp/sass89_full.txt  > analysis/sass_square2048_89.txt

All builds returned exit 0 (warnings only; "declared but never referenced").

---

## 2. SASS diff and the candidate miscompiled instruction(s)

Normalization for diffing (strip address column + encoding bytes, keep mnemonics):

    grep -oP '/\*[0-9a-f]{4}\*/\s+\K.*?;' <file> | sed -E 's/\s+/ /g; s/\s+;/;/'

Produces analysis/sass_square2048_{120,89}.instr.txt, then:

    diff -u <89 instr> <120 instr> > analysis/sass_square2048.diff

Instruction-stream sizes: compute_120 = 2888, compute_89->sm_120 = 2872 (+16 in 120).

Opcode-count deltas (120 minus 89) of interest:

    IMAD.WIDE.U32    208  vs 212   (-4)
    IMAD.SHL.U32       5  vs  16   (-11)
    IMAD             194  vs 193   (+1)
    IMAD.HI.U32      192  vs 192   ( 0)
    IADD3           1055  vs 1041  (+14)
    IADD.64            6  vs   3   (+3)
    MOV                6  vs   2   (+4)
    MOV.64             1  vs   0   (+1)
    HFMA2              7  vs   2   (+5)
    SHF.R.U32.HI       7  vs  10   (-3)
    SHF.R.S32.HI     356  vs 356   ( 0)

Key qualitative finding: BOTH builds use only UNSIGNED 64-bit multiply lowering
(IMAD.WIDE.U32, IMAD.HI.U32); there are ZERO signed IMAD.WIDE/IMAD.HI in either.
So ptxas recognized that the mul.lo.s64 operands have zero upper halves and
lowered them to the same unsigned form. The core RNS/Montgomery reduction idiom
is instruction-for-instruction identical in selection between the two builds
(see analysis/sass_reduction_core_windows.txt):

    IMAD.WIDE.U32 ...      ; the 32x32->64 product
    IMAD          ...      ; low-part Montgomery multiply  (mul.lo.s32 in PTX)
    IMAD.HI.U32   ...      ; high-part                     (mul.hi.u32 in PTX)
    IADD3 ... -Rx ...      ; subtract
    SHF.R.S32.HI R, RZ, 0x1f, ... ; sign correction (shr 31)

The counts that DO differ are addressing/index strength-reduction:
IMAD.WIDE.U32 with an immediate power-of-two multiplier:

    immediate-multiplier IMAD.WIDE.U32:  120 = 7    89 = 19
    register-multiplier  IMAD.WIDE.U32:  120 = 201  89 = 191

i.e. the compute_89 path forms array addresses as base + index*<imm 4/8/...>
(single wide IMAD with an immediate), whereas the compute_120 path more often
materializes the scale into a register (MOV Rk, 0x4) and/or pre-adds the index
(IADD3) before a register-multiplier IMAD.WIDE.U32. That accounts for the
+14 IADD3 / +4 MOV / -11 IMAD.SHL.U32 / -4 IMAD.WIDE.U32 deltas.

Candidate miscompiled SASS instruction(s): the IMAD.WIDE.U32 (register-multiplier
form) plus its feeding IADD3 .., R2, R29, RZ / MOV R33, 0x4 that implement the
index expressions originating from the 6 PTX mul.lo.s64 sites. Concrete first
instance in analysis/sass_square2048_120.txt:

    IADD3 R26, PT, PT, R2, R29, RZ;
    IMAD.WIDE.U32 R26, R26, R27, UR10;     <-- candidate
    ...
    MOV R33, 0x4;
    IMAD.WIDE.U32 R32, R2, R33, UR10;      <-- candidate

versus the compute_89->sm_120 equivalents:

    IMAD.SHL.U32 R9, R29, 0x20000, RZ;
    IMAD.WIDE.U32 R34, R9, 0x4, R34;
    IMAD.WIDE.U32 R24, R27, 0x4, R34;

These remain arithmetically equivalent in the disassembly; static inspection
found no signedness or width error in the SASS itself. The divergence is real but
its observable effect (if any) cannot be confirmed without Blackwell execution.

---

## 3. PTX divergence (confirmed, with counts)

square2048 entry ranges: compute_120 PTX lines 24987-28291; compute_89 PTX lines 26439-29925.

    grep -c within square2048 body:
                     compute_120   compute_89
    mul.lo.s64            6            0
    mul.wide.u32        203          213
    cvt.u64.u32          10            3
    mul.hi.             192          192

So cicc rewrote exactly 6 multiplies: compute_89 emits mul.wide.u32; compute_120
emits cvt.u64.u32 (widen the 32-bit operand) + mul.lo.s64 + a {lo,hi} split
(cvt.u32.u64 for low, mov.b64 {tmp,hi} for high). Adjacent lines show the two
forms side-by-side in the SAME reduction idiom (analysis/ptx_divergence_excerpt_120.txt):

    mul.wide.u32 %rd119, %r1285, %r1094;   <- kept wide
    ...
    cvt.u64.u32  %rd120, %r1154;
    mul.lo.s64   %rd121, %rd98, %rd120;     <- rewritten

where %rd98 = shr.u64 %rd97, 32 has a provably-zero upper 32 bits, and
%rd120 = cvt.u64.u32 %r1154 likewise. With both upper halves zero, the low 64
bits of the signed product equal the unsigned 32x32->64 wide product:
arithmetically equivalent. The 6 products feed array-index/address math (e.g.
ki/ko), not the Montgomery multiply itself.

---

## 4. cicc vs ptxas assessment (static only, qualified)

The DIVERGENCE ORIGINATES IN cicc (PTX generation), not ptxas:

- The two PTX inputs already differ (mul.wide.u32 vs cvt+mul.lo.s64+split) for
  the same source, chosen by cicc purely as a function of the virtual arch.
- ptxas appears to lower BOTH PTX forms correctly and equivalently: the resulting
  SASS uses only unsigned IMAD.WIDE.U32/IMAD.HI.U32, the modular-reduction core is
  identical, and no signed/narrow multiply appears. Static inspection of the SASS
  found no instruction that is, on its face, wrong.

Therefore, as a static matter, the suspicious actor is the cicc CHOICE to emit
mul.lo.s64 for Blackwell virtual archs. Whether that choice (or ptxas's lowering
of it) actually miscomputes cannot be determined here -- both PTX forms and both
SASS forms are arithmetically equivalent on paper. IF a real miscompile exists,
the most likely static candidates, in order, are:
  (a) ptxas's lowering/scheduling of the register-multiplier IMAD.WIDE.U32 index
      path that only the compute_120 PTX triggers (a latent ptxas bug exposed only
      by the mul.lo.s64-derived index expressions), or
  (b) a cicc bug if the mul.lo.s64 rewrite is ever applied to operands whose upper
      halves are NOT provably zero (not observed in these 6 sites, which all have
      zero-upper operands).
This is deliberately qualified: no runtime oracle is available.

---

## 5. Version matrix (does the divergent codegen persist on newer toolkits?)

Discovery: ls -d /usr/local/cuda* /opt/cuda* 2>/dev/null
  -> /usr/local/cuda (symlink -> cuda-13.2), /usr/local/cuda-12.9, /usr/local/cuda-13.2

For each toolkit:
  nvcc -ptx -arch=compute_120 <src>; grep -c 'mul.lo.s64' within square2048

    toolkit   nvcc        ptxas       compute_120 mul.lo.s64   reproduces
    12.9      V12.9.86    V12.9.86     6                        YES
    13.2      V13.2.78    V13.2.78     6                        YES   (latest)

compute_89 baseline on both toolkits: mul.lo.s64 = 0 (clean).

Extra arch probes (12.9): compute_90 -> 0; compute_100 -> 6; compute_120a -> 6.
=> the rewrite is gated on the Blackwell-class virtual SM family (sm_100/sm_120),
   not present for sm_89/sm_90.

CRITICAL: the divergent cicc codegen STILL REPRODUCES on the latest installed
toolkit (CUDA 13.2). NVIDIA fixes the latest; the latest is affected.

---

## 6. Artifacts

    analysis/sass_square2048_120.txt          full raw SASS body (compute_120 -> sm_120)
    analysis/sass_square2048_89.txt           full raw SASS body (compute_89 -> sm_120)
    analysis/sass_square2048_120.instr.txt    normalized instruction stream (120)
    analysis/sass_square2048_89.instr.txt     normalized instruction stream (89)
    analysis/sass_square2048.diff             unified diff of the normalized streams
    analysis/sass_reduction_core_windows.txt  side-by-side reduction-core SASS windows
    analysis/ptx_divergence_excerpt_120.txt   PTX: wide.u32 vs lo.s64 in same idiom
    analysis/ptx_mul_lo_s64_sites_120.txt     all 6 mul.lo.s64 PTX sites with context
    analysis/findings.md                      this file

CAVEAT (repeated): static analysis only. No Blackwell GPU; no kernel was run.
No claim is made that square2048 produces wrong results at runtime -- only that
cicc emits divergent (but arithmetically equivalent) PTX for compute_120, that
this persists through CUDA 13.2, and that the resulting SASS differs in the
index/address multiply path while the modular-reduction core is unchanged.
