#!/usr/bin/env bash
# ===========================================================================
# verify_ptx.sh -- VERIFY (no GPU needed) the compute_120 vs compute_89 PTX
#                  codegen divergence for the genefer22 mulmod 32x32->64 multiply.
#
# Requires only a CUDA toolkit (tested with 12.9).  Prints grep counts of
#   mul.lo.s64   (the suspect compute_120 form)
#   mul.wide.u32 (the correct compute_89 form)
#   cvt.u64.u32 / mov.b64  (the 64-bit packing that enables mul.lo.s64)
# for (1) each minimal probe kernel and (2) the FULL real kernel.
# ===========================================================================
set -u
CUDA=${CUDA:-/usr/local/cuda-12.9}
NVCC=$CUDA/bin/nvcc
HERE="$(cd "$(dirname "$0")" && pwd)"
FULL=${FULL:-/home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu}
OUT="$HERE/_ptx"; mkdir -p "$OUT"

count() { local f=$1 pat=$2 n; if [ -f "$f" ]; then n=$(grep -c "$pat" "$f" 2>/dev/null); echo "${n:-0}"; else echo "?"; fi; }

row() { # row <label> <ptx120> <ptx89>
    printf "  %-22s  120[lo.s64=%-3s wide=%-4s cvt64=%-3s movb64=%-4s]  89[lo.s64=%-3s wide=%-4s cvt64=%-3s movb64=%-4s]\n" \
        "$1" \
        "$(count "$2" 'mul.lo.s64')" "$(count "$2" 'mul.wide.u32')" "$(count "$2" 'cvt.u64.u32')" "$(count "$2" 'mov.b64')" \
        "$(count "$3" 'mul.lo.s64')" "$(count "$3" 'mul.wide.u32')" "$(count "$3" 'cvt.u64.u32')" "$(count "$3" 'mov.b64')"
}

echo "============================================================================"
echo " PTX divergence check  (CUDA: $($NVCC --version | sed -n 's/.*release //p'))"
echo "============================================================================"

echo
echo "[1] minimal probes  (repro_minimal.cu)"
$NVCC -ptx -arch=compute_120 "$HERE/repro_minimal.cu" -o "$OUT/min_120.ptx" 2>/dev/null
$NVCC -ptx -arch=compute_89  "$HERE/repro_minimal.cu" -o "$OUT/min_89.ptx"  2>/dev/null
# split per-kernel so the histogram is attributable
for k in probe_force64 probe_narrow probe_vec_pressure; do
    awk -v K="$k" '
        $0 ~ "\\.entry "K"\\(" || $0 ~ "\\.entry "K"$" {on=1}
        on {print}
        on && /^\}/ {on=0}
    ' "$OUT/min_120.ptx" > "$OUT/${k}_120.ptx"
    awk -v K="$k" '
        $0 ~ "\\.entry "K"\\(" || $0 ~ "\\.entry "K"$" {on=1}
        on {print}
        on && /^\}/ {on=0}
    ' "$OUT/min_89.ptx" > "$OUT/${k}_89.ptx"
    row "$k" "$OUT/${k}_120.ptx" "$OUT/${k}_89.ptx"
done

echo
echo "[2] FULL real kernel  (n17_rns3_is0.cu)  -- whole module"
if [ -f "$FULL" ]; then
    $NVCC -ptx -arch=compute_120 -diag-suppress 177 "$FULL" -o "$OUT/full_120.ptx" 2>/dev/null
    $NVCC -ptx -arch=compute_89  -diag-suppress 177 "$FULL" -o "$OUT/full_89.ptx"  2>/dev/null
    row "n17_rns3_is0 (module)" "$OUT/full_120.ptx" "$OUT/full_89.ptx"
    echo
    echo "  --> EXPECT: compute_120 has mul.lo.s64 > 0 and mov.b64 >> 0, while"
    echo "             compute_89 has mul.lo.s64 = 0 and mov.b64 = 0."
    echo "             That is the bug-relevant NVVM codegen divergence."
else
    echo "  (full kernel not found at $FULL -- set FULL=...)"
fi
echo
echo "PTX written under: $OUT"
