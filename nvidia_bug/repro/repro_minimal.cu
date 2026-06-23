// ===========================================================================
// repro_minimal.cu  --  minimal probes for the genefer22 sm_120 mulmod codegen
//                       divergence (NVIDIA Blackwell / compute_120 vs compute_89)
// ===========================================================================
//
// BACKGROUND
// ----------
// In kernel square2048 (build_dev/_fatsrc/n17_rns3_is0.cu) the Montgomery-style
// 32x32->64 multiply inside mulmod():
//
//     const uint_64 t  = lhs * (uint_64)(rhs);          // 32x32 -> 64
//     const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);  // BOTH halves used
//     const uint_32 mp = __umulhi(lo * pq.s1, pq.s0);
//     return submod(hi, mp, pq.s0);
//
// is emitted by NVVM as ONE  `mul.wide.u32`  for -arch=compute_89, but as
//     cvt.u64.u32 + mul.lo.s64 + (cvt.u32.u64 lo / mov.b64 {tmp,hi})
// for -arch=compute_120.  On real sm_120 (RTX 50xx) the compute_120 build gives
// WRONG results; the compute_89 build is correct.  The two forms are
// arithmetically identical, so the fault is in the sm_120 path (NVVM choosing the
// 64-bit form on compute_120, and/or ptxas's sm_120 lowering of it under the real
// kernel's register pressure).
//
// WHAT WE FOUND LOCALLY (no Blackwell GPU; PTX/SASS inspection only)
// -----------------------------------------------------------------
//  * Full kernel n17_rns3_is0.cu:
//        compute_89 :  mul.lo.s64=0    mul.wide.u32=3982
//        compute_120:  mul.lo.s64=56   mul.wide.u32=3977
//    -> CONFIRMED divergence at the NVVM (PTX) level.
//    Root cause visible in PTX: on compute_120 NVVM packs the vectorized v2.u32
//    twiddle load into a 64-bit register (`mov.b64 {lo,hi}`; compute_89 emits
//    ZERO mov.b64 in the whole module, compute_120 emits ~3620) and then takes
//    the high word with `shr.u64 ...,32`.  That high word is now 64-bit-typed, so
//    the following `lhs * (uint64)hiword` can no longer fold to mul.wide.u32 and
//    becomes cvt.u64.u32 + mul.lo.s64.
//
//  * The TWO behaviours each reproduce in isolation, but the *crossover* (89=wide
//    while 120=lo.s64 at the SAME site) needs the real kernel's structure:
//      - PROBE A (force64): if one factor genuinely comes from a 64-bit value
//        (e.g. (uint64)packed >> 32), BOTH arches emit mul.lo.s64.  This is the
//        exact problematic PTX idiom, but it is symmetric, so it does not by
//        itself prove a *divergence*.   -> kernel `probe_force64`
//      - PROBE B (narrowable): if the factor is a plain 32-bit value, BOTH arches
//        emit mul.wide.u32.                                  -> kernel `probe_narrow`
//      - PROBE C (vec+pressure): vectorized v2 twiddle + register pressure makes
//        compute_120 start packing to 64-bit (mov.b64 appears only on 120) while
//        compute_89 keeps lanes separate; this is the same mechanism as the full
//        kernel, though at small scale ptxas still recovers mul.wide in places.
//                                                            -> kernel `probe_vec_pressure`
//
// HONEST STATUS: a *small standalone* kernel did NOT, on its own, reproduce the
// asymmetric  "89=mul.wide.u32 vs 120=mul.lo.s64 at the same multiply"  PTX
// divergence the way the full kernel does; the asymmetry is driven by NVVM's
// target-dependent 2x32->64 packing decision which only fires under the full
// kernel's pressure.  probe_force64 DOES reproduce the exact problematic
// compute_120 PTX form (cvt.u64.u32 + mul.lo.s64 + extract {lo,hi}) byte-for-byte
// -- but it produces it on BOTH arches, and at SASS level ptxas lowers it back to
// IMAD.WIDE.U32, so this minimal kernel is NOT expected to give a wrong runtime
// result.  Wrong-RESULT confirmation requires Blackwell hardware (see
// check_on_blackwell.sh and the full-kernel fallback in README.md).
//
// VERIFY THE PTX with verify_ptx.sh in this directory.
// ===========================================================================

typedef unsigned int       uint_32;
typedef unsigned long long uint_64;

struct __align__(8)  uint2_32 { uint_32 s0, s1; };
struct __align__(16) uint4_32 { uint_32 s0, s1, s2, s3; };

__device__ __forceinline__ static uint_32 addmod(uint_32 a, uint_32 b, uint_32 p)
{ const uint_32 t = a + b; return t - ((t >= p) ? p : 0); }
__device__ __forceinline__ static uint_32 submod(uint_32 a, uint_32 b, uint_32 p)
{ const uint_32 t = a - b; return t + (((int)(t) < 0) ? p : 0); }

// The exact mulmod from n17_rns3_is0.cu (the 32x32->64 multiply lives here).
__device__ __forceinline__ static uint_32 mulmod(uint_32 lhs, uint_32 rhs, uint2_32 pq)
{
    const uint_64 t  = lhs * (uint_64)(rhs);                  // <-- the multiply
    const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32); // BOTH halves used
    const uint_32 mp = __umulhi(lo * pq.s1, pq.s0);
    return submod(hi, mp, pq.s0);
}
__device__ __forceinline__ static uint_32 sqrmod(uint_32 a, uint2_32 pq) { return mulmod(a, a, pq); }

// ---------------------------------------------------------------------------
// PROBE A -- "force64": one factor comes from a genuine 64-bit value, exactly as
// in the real kernel where the twiddle high word is (uint64)packed >> 32.
// RESULT (compute 12.9): BOTH compute_89 and compute_120 emit the problematic
// idiom  cvt.u64.u32 + mul.lo.s64 + cvt.u32.u64 / mov.b64{tmp,hi}.  This is the
// byte-for-byte match to square2048's compute_120 PTX -- but it is symmetric.
// ---------------------------------------------------------------------------
extern "C" __global__
void probe_force64(uint_32 *zg, const uint_64 *wg, uint2_32 pq, unsigned base)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint_32 wlo[4], whi[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint_64 p = wg[base + t];
        wlo[t] = (uint_32)(p);
        whi[t] = (uint_32)(p >> 32);          // 64-bit-typed high word (mov.b64+shr.u64)
    }
    uint_32 z[16];
    #pragma unroll
    for (int l = 0; l < 16; ++l) z[l] = zg[i * 16 + l];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint_32 *q = &z[g * 4];
        { const uint_32 t = mulmod(q[2], whi[g], pq); q[2] = submod(q[0], t, pq.s0); q[0] = addmod(q[0], t, pq.s0); }
        { const uint_32 t = mulmod(q[3], wlo[g], pq); q[3] = submod(q[1], t, pq.s0); q[1] = addmod(q[1], t, pq.s0); }
        { const uint_32 t = mulmod(q[1], whi[g], pq); q[1] = submod(q[0], t, pq.s0); q[0] = addmod(q[0], t, pq.s0); }
        { const uint_32 t = mulmod(q[3], wlo[g], pq); q[3] = submod(q[2], t, pq.s0); q[2] = addmod(q[2], t, pq.s0); }
    }
    #pragma unroll
    for (int l = 0; l < 16; ++l) zg[i * 16 + l] = z[l];
}

// ---------------------------------------------------------------------------
// PROBE B -- "narrow": plain 32-bit factor.  BOTH arches emit mul.wide.u32.
// (Baseline / control: shows the "good" form.)
// ---------------------------------------------------------------------------
extern "C" __global__
void probe_narrow(uint_32 *out, const uint_32 *a, const uint_32 *b, uint2_32 pq)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = mulmod(a[i], b[i], pq);
}

// ---------------------------------------------------------------------------
// PROBE C -- "vec_pressure": vectorized v2 twiddle read with the real kernel's
// reinterpret-cast idiom ((const uint2_32*)w)[sj], reused across a forward-4
// butterfly chain under register pressure.  On compute_120 NVVM begins packing
// the v2 load into a 64-bit register (mov.b64 appears ONLY on compute_120),
// which is the same mechanism that makes the full kernel diverge.
// ---------------------------------------------------------------------------
__device__ __forceinline__ static void fwd4(uint2_32 pq, uint_32 z[4], uint_32 w1, const uint_32 w2[2])
{
    { const uint_32 t = mulmod(z[2], w1,    pq); z[2] = submod(z[0], t, pq.s0); z[0] = addmod(z[0], t, pq.s0); }
    { const uint_32 t = mulmod(z[3], w1,    pq); z[3] = submod(z[1], t, pq.s0); z[1] = addmod(z[1], t, pq.s0); }
    { const uint_32 t = mulmod(z[1], w2[0], pq); z[1] = submod(z[0], t, pq.s0); z[0] = addmod(z[0], t, pq.s0); }
    { const uint_32 t = mulmod(z[3], w2[1], pq); z[3] = submod(z[2], t, pq.s0); z[2] = addmod(z[2], t, pq.s0); }
}
extern "C" __global__
__launch_bounds__(64)
void probe_vec_pressure(uint_32 *zg, const uint_32 *wg, uint2_32 pq, unsigned sj)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // DECLARE_W1 + DECLARE_W2 exactly as in the kernel:
    const uint_32 w1 = wg[sj];
    uint_32 w2[2]; { const uint2_32 t = ((const uint2_32 *)wg)[sj]; w2[0] = t.s0; w2[1] = t.s1; }
    uint_32 z[16];
    #pragma unroll
    for (int l = 0; l < 16; ++l) z[l] = zg[i * 16 + l];
    #pragma unroll
    for (int g = 0; g < 4; ++g) fwd4(pq, &z[g * 4], w1, w2);
    #pragma unroll
    for (int l = 0; l < 16; ++l) zg[i * 16 + l] = z[l];
}
