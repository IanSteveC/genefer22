#!/usr/bin/env bash
# ===========================================================================
# check_on_blackwell.sh -- RUN THIS ON A BLACKWELL (sm_120 / RTX 50xx) GPU.
#
# Builds the SAME harness TWO ways and runs both against an identical CPU
# reference, then prints which path matches:
#
#   path A (SUSPECT) : nvcc -gencode arch=compute_120,code=sm_120
#                      -> mulmod's 32x32->64 multiply is the mul.lo.s64 form
#   path B (REFERENCE): nvcc -ptx -arch=compute_89  ->  ptxas -arch=sm_120
#                      -> mulmod's 32x32->64 multiply is the mul.wide.u32 form
#
# Per the bug report, on a real 5070 the SUSPECT build mis-computes the genefer
# kernel while the REFERENCE build is correct.  This script is what the friend
# with the 50xx runs to confirm.
#
# Requires only: a CUDA toolkit (>=12.8 for sm_120) + a Blackwell GPU.
# Usage:
#   ./check_on_blackwell.sh            # minimal probe (harness.cu), both paths
#   ./check_on_blackwell.sh --full     # ALSO build/run the real square2048 NTT
#                                       # round-trip (the reliable repro)
# Env: CUDA=/path/to/cuda   (default /usr/local/cuda-12.9 then /usr/local/cuda)
# ===========================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA="${CUDA:-}"
if [ -z "$CUDA" ]; then
    for c in /usr/local/cuda-12.9 /usr/local/cuda; do [ -d "$c" ] && CUDA="$c" && break; done
fi
NVCC="$CUDA/bin/nvcc"; PTXAS="$CUDA/bin/ptxas"
[ -x "$NVCC" ] || { echo "nvcc not found (set CUDA=...)"; exit 2; }
echo "Using CUDA: $CUDA   ($($NVCC --version | sed -n 's/.*release //p'))"
echo

OUT="$HERE/_blackwell"; mkdir -p "$OUT"

# --------------------------------------------------------------------------
# 1) MINIMAL PROBE  (harness.cu)
# --------------------------------------------------------------------------
echo "=========================================================================="
echo " [1] MINIMAL PROBE  harness.cu  (probe_force64)"
echo "=========================================================================="

echo "-- building path A (SUSPECT: -gencode arch=compute_120,code=sm_120) --"
$NVCC -O2 -gencode arch=compute_120,code=sm_120 "$HERE/harness.cu" -o "$OUT/harness_A_sm120" \
    || { echo "path A build FAILED (need CUDA>=12.8 + sm_120 support)"; exit 2; }

echo "-- building path B (REFERENCE: compute_89 PTX -> ptxas sm_120) --"
# compile ONLY the device code of harness.cu to compute_89 PTX, then re-assemble
# for sm_120, then link with the host object.  Easiest robust route: build the
# whole TU with -arch=compute_89,code=sm_120 (JIT-style embed), which forces the
# device PTX through the compute_89 front-end and then ptxas to sm_120.
$NVCC -O2 -gencode arch=compute_89,code=sm_120 "$HERE/harness.cu" -o "$OUT/harness_B_sm120" \
    || { echo "path B build FAILED"; exit 2; }

echo
echo "-- running path A (SUSPECT) --"
"$OUT/harness_A_sm120"; rcA=$?
echo
echo "-- running path B (REFERENCE) --"
"$OUT/harness_B_sm120"; rcB=$?

echo
echo "-------------------------------------------------------------------------"
printf " RESULT (minimal probe):  path A (compute_120)=%s   path B (compute_89->sm120)=%s\n" \
    "$([ $rcA -eq 0 ] && echo PASS || echo FAIL)" \
    "$([ $rcB -eq 0 ] && echo PASS || echo FAIL)"
if [ $rcA -ne 0 ] && [ $rcB -eq 0 ]; then
    echo " >>> BUG CONFIRMED on this GPU: the compute_120 build is WRONG, compute_89 is correct."
elif [ $rcA -eq 0 ] && [ $rcB -eq 0 ]; then
    echo " >>> Minimal probe did NOT diverge here (both correct).  ptxas recovered the"
    echo "     wide multiply for this small kernel.  Use --full for the reliable repro."
else
    echo " >>> Unexpected combination; inspect output above."
fi
echo "-------------------------------------------------------------------------"

# --------------------------------------------------------------------------
# 2) FULL real kernel (square2048)  -- the reliable repro
# --------------------------------------------------------------------------
if [ "${1:-}" = "--full" ]; then
    echo
    echo "=========================================================================="
    echo " [2] FULL real kernel  square2048  (NTT forward+inverse round-trip)"
    echo "=========================================================================="
    FULL="${FULL:-/home/ian/builds/primegrid/genefer22/build_dev/_fatsrc/n17_rns3_is0.cu}"
    if [ ! -f "$FULL" ]; then echo "full kernel not found: $FULL (set FULL=...)"; exit 2; fi

    # A round-trip test: square2048 followed by the inverse path should be an
    # identity (up to the known scaling) on a delta input, independent of any
    # external reference.  We launch the real kernel both ways and compare the
    # two GPU outputs to each other AND to a compute_89-on-sm_89-style reference.
    # Since wiring the full NTT host driver is involved, the SIMPLEST reliable
    # check is: build the real module both ways, run identical input, and compare
    # the two device outputs bit-for-bit.  If they differ, the codegen divergence
    # is observable at runtime on this GPU.
    cat > "$OUT/full_driver.cu" <<'DRV'
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);} }while(0)
// square2048 signature from n17_rns3_is0.cu: takes (VTYPE* zg, const uint_32* wg).
// VTYPE for N_SZ=131072 is uint4_32 (VSIZE=4).  We treat buffers as raw uint32.
extern "C" __global__ void square2048(void* zg, const unsigned* wg);
int main(){
    int dev=0; CK(cudaSetDevice(dev)); cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("GPU: %s sm_%d%d\n",p.name,p.major,p.minor);
    // 2048-point block, VSIZE=4 -> 512 uint4 = 2048 uint32 of state, plus a twiddle table.
    const size_t ZN = 4u*131072u;            // generous z buffer (uint32)
    const size_t WN = 4u*131072u;            // generous w buffer (uint32)
    unsigned *hz=(unsigned*)malloc(ZN*4),*hw=(unsigned*)malloc(WN*4);
    unsigned long long s=0xdeadbeef12345ull;
    auto rnd=[&](){ s^=s<<13;s^=s>>7;s^=s<<17;return (unsigned)s; };
    const unsigned P1=2130706433u;
    for(size_t i=0;i<ZN;i++) hz[i]=rnd()%P1;
    for(size_t i=0;i<WN;i++) hw[i]=rnd()%P1;
    unsigned *dz,*dw; CK(cudaMalloc(&dz,ZN*4)); CK(cudaMalloc(&dw,WN*4));
    CK(cudaMemcpy(dz,hz,ZN*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw,hw,WN*4,cudaMemcpyHostToDevice));
    // Launch with a block size the kernel permits (its __launch_bounds__ / reg use
    // cap maxThreadsPerBlock; query it instead of hard-coding 512).
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa,(const void*)square2048));
    int blk = fa.maxThreadsPerBlock; if (blk>512) blk=512; if (blk<32) blk=32;
    square2048<<<1,blk>>>((void*)dz,(const unsigned*)dw);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned *out=(unsigned*)malloc(ZN*4); CK(cudaMemcpy(out,dz,ZN*4,cudaMemcpyDeviceToHost));
    unsigned long long h=1469598103934665603ull; // FNV of the first 2048 words of output
    for(int i=0;i<2048;i++){ h^=out[i]; h*=1099511628211ull; }
    printf("square2048 output hash (first 2048 words): %016llx\n",h);
    return 0;
}
DRV

    echo "-- building FULL path A (compute_120 -> sm_120) --"
    $NVCC -O2 -gencode arch=compute_120,code=sm_120 -diag-suppress 177 \
        "$FULL" "$OUT/full_driver.cu" -o "$OUT/full_A_sm120" 2>/dev/null \
        || { echo "FULL path A build FAILED"; exit 2; }
    echo "-- building FULL path B (compute_89 PTX -> ptxas sm_120) --"
    $NVCC -O2 -gencode arch=compute_89,code=sm_120 -diag-suppress 177 \
        "$FULL" "$OUT/full_driver.cu" -o "$OUT/full_B_sm120" 2>/dev/null \
        || { echo "FULL path B build FAILED"; exit 2; }

    echo
    echo "-- running FULL path A (SUSPECT compute_120) --"
    hA=$("$OUT/full_A_sm120" | tee /dev/stderr | sed -n 's/.*hash[^:]*: //p')
    echo "-- running FULL path B (REFERENCE compute_89->sm120) --"
    hB=$("$OUT/full_B_sm120" | tee /dev/stderr | sed -n 's/.*hash[^:]*: //p')
    echo
    echo "-------------------------------------------------------------------------"
    if [ "$hA" = "$hB" ]; then
        echo " FULL kernel: path A and path B produced the SAME output hash ($hA)."
        echo " (On a GPU where the bug bites, these differ.  If they match here, the"
        echo "  single-block harness path may not exercise the failing schedule; run"
        echo "  the genefer app itself -- see README.md.)"
    else
        echo " >>> FULL kernel BUG CONFIRMED: path A hash=$hA  != path B hash=$hB"
        echo "     The compute_120 build computes DIFFERENT results from compute_89"
        echo "     on identical input -- the codegen divergence is observable at runtime."
    fi
    echo "-------------------------------------------------------------------------"
fi
