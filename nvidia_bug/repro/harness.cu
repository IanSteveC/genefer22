// ===========================================================================
// harness.cu -- runtime A/B test of the mulmod 32x32->64 codegen on a Blackwell GPU.
//
// This file is compiled TWICE by check_on_blackwell.sh:
//   path A:  nvcc -gencode arch=compute_120,code=sm_120   (the suspect build)
//   path B:  nvcc -ptx -arch=compute_89 ; ptxas -arch=sm_120  (the reference build)
// Both binaries run the SAME kernel on the SAME inputs on the Blackwell GPU; the
// host computes a CPU reference (plain 64-bit math) and prints PASS/FAIL.
//
// The kernel `probe_force64` emits, on compute_120, the exact suspect PTX idiom
//   cvt.u64.u32 + mul.lo.s64 + extract{lo,hi}
// for the 32x32->64 multiply inside mulmod(); on compute_89 it emits mul.wide.u32.
// If sm_120's lowering of the compute_120 form is wrong, path A FAILs and path B
// PASSes against the identical CPU reference.
//
// NOTE (honesty): on the developer's non-Blackwell box this kernel's mul.lo.s64
// is recovered to IMAD.WIDE.U32 by ptxas and is EXPECTED to pass on older GPUs.
// The point of this harness is to be RUN ON A 50xx (sm_120) where, per the bug
// report, the full genefer kernel mis-computes.  If this minimal probe does NOT
// FAIL on the 50xx, fall back to the full-kernel test (see README.md / --full).
// ===========================================================================
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

typedef unsigned int       uint_32;
typedef unsigned long long uint_64;
struct __align__(8) uint2_32 { uint_32 s0, s1; };

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(2);} } while(0)

// ---- device arithmetic: byte-for-byte the genefer22 mulmod -------------------
__device__ __forceinline__ static uint_32 d_addmod(uint_32 a, uint_32 b, uint_32 p)
{ const uint_32 t = a + b; return t - ((t >= p) ? p : 0); }
__device__ __forceinline__ static uint_32 d_submod(uint_32 a, uint_32 b, uint_32 p)
{ const uint_32 t = a - b; return t + (((int)(t) < 0) ? p : 0); }
__device__ __forceinline__ static uint_32 d_mulmod(uint_32 lhs, uint_32 rhs, uint2_32 pq)
{
    const uint_64 t  = lhs * (uint_64)(rhs);
    const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
    const uint_32 mp = __umulhi(lo * pq.s1, pq.s0);
    return d_submod(hi, mp, pq.s0);
}

// Same kernel as repro_minimal.cu::probe_force64 (the 64-bit-high-word form).
extern "C" __global__
void probe_force64(uint_32 *zg, const uint_64 *wg, uint2_32 pq, unsigned base, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    uint_32 wlo[4], whi[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint_64 p = wg[(base + t) & 1023];
        wlo[t] = (uint_32)(p);
        whi[t] = (uint_32)(p >> 32);
    }
    uint_32 z[16];
    #pragma unroll
    for (int l = 0; l < 16; ++l) z[l] = zg[i * 16 + l];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint_32 *q = &z[g * 4];
        { const uint_32 t = d_mulmod(q[2], whi[g], pq); q[2] = d_submod(q[0], t, pq.s0); q[0] = d_addmod(q[0], t, pq.s0); }
        { const uint_32 t = d_mulmod(q[3], wlo[g], pq); q[3] = d_submod(q[1], t, pq.s0); q[1] = d_addmod(q[1], t, pq.s0); }
        { const uint_32 t = d_mulmod(q[1], whi[g], pq); q[1] = d_submod(q[0], t, pq.s0); q[0] = d_addmod(q[0], t, pq.s0); }
        { const uint_32 t = d_mulmod(q[3], wlo[g], pq); q[3] = d_submod(q[2], t, pq.s0); q[2] = d_addmod(q[2], t, pq.s0); }
    }
    #pragma unroll
    for (int l = 0; l < 16; ++l) zg[i * 16 + l] = z[l];
}

// ---- host reference (plain 64-bit; matches the *intended* arithmetic) --------
static inline uint_32 h_addmod(uint_32 a, uint_32 b, uint_32 p) { uint_32 t=a+b; return t-((t>=p)?p:0); }
static inline uint_32 h_submod(uint_32 a, uint_32 b, uint_32 p) { uint_32 t=a-b; return t+(((int)(t)<0)?p:0); }
static inline uint_32 h_umulhi(uint_32 a, uint_32 b) { return (uint_32)(((uint_64)a*(uint_64)b) >> 32); }
static inline uint_32 h_mulmod(uint_32 lhs, uint_32 rhs, uint_32 p, uint_32 q)
{
    const uint_64 t  = (uint_64)lhs * (uint_64)rhs;
    const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
    const uint_32 mp = h_umulhi(lo * q, p);
    return h_submod(hi, mp, p);
}

int main(int argc, char** argv)
{
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    const bool isBlackwell = (prop.major >= 12);
    if (!isBlackwell)
        printf("WARNING: this is NOT an sm_120 (Blackwell) GPU; the suspect ptxas "
               "lowering only mis-executes on sm_120, so a PASS here proves nothing.\n");

    // P1/Q1 from genefer22 (Montgomery prime/inverse).
    const uint_32 P = 2130706433u, Q = 2164260865u;
    uint2_32 pq = { P, Q };

    const int N = 1 << 16;                  // 65536 threads
    const int Z = N * 16;                   // z lanes
    const int W = 1024;                     // packed twiddles
    const unsigned base = 7;

    uint_32 *hz = (uint_32*)malloc(sizeof(uint_32)*Z);
    uint_64 *hw = (uint_64*)malloc(sizeof(uint_64)*W);
    // deterministic pseudo-random inputs, reduced mod P
    uint_64 s = 0x12345678abcdef01ull;
    auto rnd = [&](){ s ^= s<<13; s ^= s>>7; s ^= s<<17; return s; };
    for (int i=0;i<Z;i++) hz[i] = (uint_32)(rnd() % P);
    for (int i=0;i<W;i++) { uint_32 a=(uint_32)(rnd()%P), b=(uint_32)(rnd()%P); hw[i]=((uint_64)b<<32)|a; }

    // CPU reference
    uint_32 *ref = (uint_32*)malloc(sizeof(uint_32)*Z);
    for (int i=0;i<N;i++) {
        uint_32 wlo[4], whi[4];
        for (int t=0;t<4;t++){ uint_64 p=hw[(base+t)&1023]; wlo[t]=(uint_32)p; whi[t]=(uint_32)(p>>32); }
        uint_32 z[16]; for(int l=0;l<16;l++) z[l]=hz[i*16+l];
        for (int g=0; g<4; g++){ uint_32 *q=&z[g*4];
            { uint_32 t=h_mulmod(q[2],whi[g],P,Q); q[2]=h_submod(q[0],t,P); q[0]=h_addmod(q[0],t,P); }
            { uint_32 t=h_mulmod(q[3],wlo[g],P,Q); q[3]=h_submod(q[1],t,P); q[1]=h_addmod(q[1],t,P); }
            { uint_32 t=h_mulmod(q[1],whi[g],P,Q); q[1]=h_submod(q[0],t,P); q[0]=h_addmod(q[0],t,P); }
            { uint_32 t=h_mulmod(q[3],wlo[g],P,Q); q[3]=h_submod(q[2],t,P); q[2]=h_addmod(q[2],t,P); }
        }
        for(int l=0;l<16;l++) ref[i*16+l]=z[l];
    }

    uint_32 *dz; uint_64 *dw;
    CK(cudaMalloc(&dz, sizeof(uint_32)*Z));
    CK(cudaMalloc(&dw, sizeof(uint_64)*W));
    CK(cudaMemcpy(dz, hz, sizeof(uint_32)*Z, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw, sizeof(uint_64)*W, cudaMemcpyHostToDevice));

    probe_force64<<<(N+127)/128, 128>>>(dz, dw, pq, base, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    uint_32 *out = (uint_32*)malloc(sizeof(uint_32)*Z);
    CK(cudaMemcpy(out, dz, sizeof(uint_32)*Z, cudaMemcpyDeviceToHost));

    long mism = 0; int first = -1;
    for (int i=0;i<Z;i++) if (out[i]!=ref[i]) { if(first<0) first=i; mism++; }

    printf("probe_force64: %s  (%ld / %d lanes mismatch)\n",
           mism==0 ? "PASS (matches CPU reference)" : "FAIL (DIVERGES from CPU reference)",
           mism, Z);
    if (mism) printf("  first mismatch lane %d: gpu=%u ref=%u\n", first, out[first], ref[first]);

    free(hz); free(hw); free(ref); free(out);
    cudaFree(dz); cudaFree(dw);
    return mism==0 ? 0 : 1;
}
