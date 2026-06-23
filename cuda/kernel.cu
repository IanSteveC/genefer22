/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

----
CUDA kernel source (genefer CUDA backend). Translated from ocl/kernel.cl.
Compiled at runtime by NVRTC. The per-FFT-size #define block (N_SZ, primes,
Montgomery constants, BLK/CHUNK sizes) is prepended by transformGPU.h, exactly
as in the OpenCL path; the `#if !defined(N_SZ)` block below provides defaults so
this file also compiles standalone for testing.

This is a hand translation, NOT a macro shim: OpenCL builtins/types are expressed
in native CUDA. Mapping summary:
  __kernel              -> extern "C" __global__
  __global  (pointers)  -> (dropped; CUDA pointers are global by default)
  __local   (arrays)    -> __shared__   |  (pointers) -> (dropped)
  get_global_id(0)      -> (blockIdx.x*blockDim.x + threadIdx.x)
  barrier(CLK_LOCAL...)  -> __syncthreads()
  uint2/uint4/long4     -> structs uint2_32/uint4_32/int4_64 (fields s0..s3)
  (uintN)(a,b,..)        -> make_uintN_32(a,b,..)
  mul_hi (32-bit)       -> __umulhi
  restrict              -> __restrict__
The integer NTT is bit-exact; results must match the OpenCL/CPU residues.
*/

// ---------------------------------------------------------------------------
// CUDA prelude (replaces OpenCL builtins/types)
// ---------------------------------------------------------------------------

#define INLINE		__device__ __forceinline__ static
#if !defined(__HIP__)
#define PTX_ASM		1	// NVIDIA inline-PTX carry chains (CUDA); HIP uses the portable C fallback
#endif

typedef unsigned int		sz_t;
typedef unsigned int		uint_32;
typedef int					int_32;
typedef unsigned long long	uint_64;
typedef long long			int_64;

#define mul_hi(x, y)	__umulhi((x), (y))

#if defined(__HIP__)
// On HIP use clang's native ext_vector types (the same uint2/uint4/int4/long4 that OpenCL uses):
// they expose .s0..s3 / .s01 swizzles, compile to packed dwordx2/x4/long4 loads, AND unroll
// correctly. ROCm clang's loop unroller MISCOMPILES the __align__ struct form below (the square
// NTT diverges after a few squarings); native vectors avoid the bug and are ~2x faster.
typedef uint_32 uint2_32 __attribute__((ext_vector_type(2)));
typedef uint_32 uint4_32 __attribute__((ext_vector_type(4)));
typedef int_32  int4_32  __attribute__((ext_vector_type(4)));
typedef int_64  int4_64  __attribute__((ext_vector_type(4)));
#else
// CUDA/NVRTC: __align__ structs so NVCC emits coalesced 64/128-bit loads/stores. (NVRTC does not
// support ext_vector_type and does not have the unroller bug, so the struct form is correct there.)
struct __align__(8)  uint2_32 { uint_32 s0, s1; };
struct __align__(16) uint4_32 { uint_32 s0, s1, s2, s3; };
struct __align__(16) int4_32  { int_32  s0, s1, s2, s3; };
struct __align__(16) int4_64  { int_64  s0, s1, s2, s3; };
#endif

// Brace-init is valid for both the aggregate structs (CUDA) and the ext_vector types (HIP).
INLINE uint2_32 make_uint2_32(const uint_32 s0, const uint_32 s1) { uint2_32 r = { s0, s1 }; return r; }
INLINE uint4_32 make_uint4_32(const uint_32 s0, const uint_32 s1, const uint_32 s2, const uint_32 s3) { uint4_32 r = { s0, s1, s2, s3 }; return r; }
INLINE int4_32 make_int4_32(const int_32 s0, const int_32 s1, const int_32 s2, const int_32 s3) { int4_32 r = { s0, s1, s2, s3 }; return r; }
INLINE int4_64 make_int4_64(const int_64 s0, const int_64 s1, const int_64 s2, const int_64 s3) { int4_64 r = { s0, s1, s2, s3 }; return r; }

// ---------------------------------------------------------------------------
// Default config (overridden by the #define block injected at runtime)
// ---------------------------------------------------------------------------

#if !defined(N_SZ)
#define N_SZ		65536u
#define LN_SZ		16
#define RNS_SZ		3
#define VSIZE		4
#define LVSIZE		2
// #define IS32		1
#define P1			2130706433u
#define Q1			2164260865u
#define RSQ1		402124772u
#define IM1			2063729671u
#define MFIM1		1930170389u
#define SQRTI1		1626730317u
#define ISQRTI1		856006302u
#define P2			2113929217u
#define Q2			2181038081u
#define RSQ2		2111798781u
#define IM2			530075385u
#define MFIM2		1036950657u
#define SQRTI2		338852760u
#define ISQRTI2		1090446030u
#define P3			2013265921u
#define Q3			2281701377u
#define RSQ3		1172168163u
#define IM3			473486609u
#define MFIM3		734725699u
#define SQRTI3		1032137103u
#define ISQRTI3		1964242958u
#define INVP2_P1	2130706177u
#define INVP3_P1	608773230u
#define INVP3_P2	1409286102u
#define P1P2P3L		1962934273u
#define P1P2P3H		2111326211158966273ul
#define P1P2P3_2L	3128950784u
#define P1P2P3_2H	1055663105579483136ul
#define NORM1		2130641409u
#define NORM2		2113864705u
#define NORM3		2013204481u
#define W_SHFT		65536u
#define WI_SHFT		32768u
// #define USE_WI		1
#define BLK32		32
#define BLK64		16
#define BLK128		8
#define BLK256		4
#define BLK512		2
#define BLK1024		1
#define CHUNK64		4
#define CHUNK256	4
#define CHUNK1024	1
// #define SHORT_FUNC	1
#define ALL_FUNC	1
#define NORM_WG_SZ	32
#define MAX_WG_SZ	256
#endif

// --- modular arithmetic

#define	PQ1		make_uint2_32(P1, Q1)
#define	PQ2		make_uint2_32(P2, Q2)
#define	PQ3		make_uint2_32(P3, Q3)

__constant__ uint2_32 g_pq[3] = { { P1, Q1 }, { P2, Q2 }, { P3, Q3 } };
__constant__ uint4_32 g_f0[3] = { { RSQ1, MFIM1, SQRTI1, ISQRTI1 }, { RSQ2, MFIM2, SQRTI2, ISQRTI2 }, { RSQ3, MFIM3, SQRTI3, ISQRTI3 } };
__constant__ uint4_32 g_b0[3] = { { ISQRTI1, SQRTI1, IM1, 0 }, { ISQRTI2, SQRTI2, IM2, 0 }, { ISQRTI3, SQRTI3, IM3, 0 } };

INLINE uint_32 addmod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
#if defined(IS32)
	return lhs + rhs - ((lhs >= p - rhs) ? p : 0);
#else
	const uint_32 t = lhs + rhs;
	return t - ((t >= p) ? p : 0);
#endif
}

INLINE uint_32 submod(const uint_32 lhs, const uint_32 rhs, const uint_32 p)
{
#if defined(IS32)
	return lhs - rhs + ((lhs < rhs) ? p : 0);
#else
	const uint_32 t = lhs - rhs;
	return t + (((int_32)(t) < 0) ? p : 0);
#endif
}

// 2 mul + 2 mul_hi
INLINE uint_32 mulmod(const uint_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	const uint_64 t = lhs * (uint_64)(rhs);
	const uint_32 lo = (uint_32)(t), hi = (uint_32)(t >> 32);
	const uint_32 mp = mul_hi(lo * pq.s1, pq.s0);
	return submod(hi, mp, pq.s0);
}

INLINE uint_32 sqrmod(const uint_32 lhs, const uint2_32 pq) { return mulmod(lhs, lhs, pq); }

INLINE int_32 get_int(const uint_32 n, const uint_32 p) { return (int_32)(n - ((n >= p / 2) ? p : 0)); }
INLINE uint_32 set_int(const int_32 i, const uint_32 p) { return (uint_32)(i + ((i < 0) ? p : 0)); }

// --- v2

INLINE uint2_32 addmod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p)
{
	return make_uint2_32(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 submod2(const uint2_32 lhs, const uint2_32 rhs, const uint_32 p)
{
	return make_uint2_32(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p));
}

INLINE uint2_32 mulmod2(const uint2_32 lhs, const uint2_32 rhs, const uint2_32 pq)
{
	return make_uint2_32(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq));
}

// scalar-rhs overload: reproduces OpenCL's implicit scalar->vector broadcast of the twiddle
INLINE uint2_32 mulmod2(const uint2_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	return make_uint2_32(mulmod(lhs.s0, rhs, pq), mulmod(lhs.s1, rhs, pq));
}

// --- v4

INLINE uint4_32 addmod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p)
{
	return make_uint4_32(addmod(lhs.s0, rhs.s0, p), addmod(lhs.s1, rhs.s1, p), addmod(lhs.s2, rhs.s2, p), addmod(lhs.s3, rhs.s3, p));
}

INLINE uint4_32 submod4(const uint4_32 lhs, const uint4_32 rhs, const uint_32 p)
{
	return make_uint4_32(submod(lhs.s0, rhs.s0, p), submod(lhs.s1, rhs.s1, p), submod(lhs.s2, rhs.s2, p), submod(lhs.s3, rhs.s3, p));
}

INLINE uint4_32 mulmod4(const uint4_32 lhs, const uint4_32 rhs, const uint2_32 pq)
{
	return make_uint4_32(mulmod(lhs.s0, rhs.s0, pq), mulmod(lhs.s1, rhs.s1, pq), mulmod(lhs.s2, rhs.s2, pq), mulmod(lhs.s3, rhs.s3, pq));
}

// scalar-rhs overload: reproduces OpenCL's implicit scalar->vector broadcast of the twiddle
INLINE uint4_32 mulmod4(const uint4_32 lhs, const uint_32 rhs, const uint2_32 pq)
{
	return make_uint4_32(mulmod(lhs.s0, rhs, pq), mulmod(lhs.s1, rhs, pq), mulmod(lhs.s2, rhs, pq), mulmod(lhs.s3, rhs, pq));
}

// --- uint96/int96 ---

typedef struct { uint_32 s0; uint_64 s1; } uint96;
typedef struct { uint_32 s0; int_64 s1; } int96;

INLINE int96 uint96_i(const uint96 x) { int96 r; r.s0 = x.s0; r.s1 = (int_64)(x.s1); return r; }
INLINE uint96 int96_u(const int96 x) { uint96 r; r.s0 = x.s0; r.s1 = (uint_64)(x.s1); return r; }

INLINE uint96 uint96_set(const uint_32 s0, const uint_64 s1) { uint96 r; r.s0 = s0; r.s1 = s1; return r; }

INLINE int96 int96_set_si(const int_64 n) { int96 r; r.s0 = (uint_32)(n); r.s1 = n >> 32; return r; }
INLINE int_64 int96_get_si(const int96 x) { return (int_64)(x.s0 | (x.s1 << 32)); }

INLINE bool int96_is_neg(const int96 x) { return (x.s1 < 0); }

INLINE bool uint96_is_greater(const uint96 x, const uint96 y) { return (x.s1 > y.s1) || ((x.s1 == y.s1) && (x.s0 > y.s0)); }

INLINE uint96 uint96_add_64(const uint96 x, const uint_64 y)
{
	const uint_32 yl = (uint_32)(y); const uint_64 yh = y >> 32;
	uint96 r;
#if defined(PTX_ASM)
	// Single asm block so the 32->64 carry flag persists between the two ops.
	asm volatile ("add.cc.u32 %0, %2, %3;\n\t addc.u64 %1, %4, %5;" : "=r" (r.s0), "=l" (r.s1) : "r" (x.s0), "r" (yl), "l" (x.s1), "l" (yh));
#else
	const uint_32 s0 = x.s0 + yl;
	r.s0 = s0; r.s1 = x.s1 + yh + ((s0 < x.s0) ? 1 : 0);
#endif
	return r;
}

INLINE int96 int96_add(const int96 x, const int96 y)
{
	int96 r;
#if defined(PTX_ASM)
	asm volatile ("add.cc.u32 %0, %2, %3;\n\t addc.s64 %1, %4, %5;" : "=r" (r.s0), "=l" (r.s1) : "r" (x.s0), "r" (y.s0), "l" (x.s1), "l" (y.s1));
#else
	const uint_32 s0 = x.s0 + y.s0;
	r.s0 = s0; r.s1 = x.s1 + y.s1 + ((s0 < x.s0) ? 1 : 0);
#endif
	return r;
}

INLINE uint96 uint96_sub(const uint96 x, const uint96 y)
{
	uint96 r;
#if defined(PTX_ASM)
	asm volatile ("sub.cc.u32 %0, %2, %3;\n\t subc.u64 %1, %4, %5;" : "=r" (r.s0), "=l" (r.s1) : "r" (x.s0), "r" (y.s0), "l" (x.s1), "l" (y.s1));
#else
	r.s0 = x.s0 - y.s0; r.s1 = (int_64)(x.s1 - y.s1 - ((x.s0 < y.s0) ? 1 : 0));
#endif
	return r;
}

INLINE uint96 int96_abs(const int96 x)
{
	const bool is_neg = int96_is_neg(x);
	const uint96 mask = uint96_set(is_neg ? ~0u : 0u, is_neg ? ~0ul : 0ul);
	const uint96 t = uint96_set(x.s0 ^ mask.s0, (uint_64)(x.s1) ^ mask.s1);
	return uint96_sub(t, mask);
}

INLINE uint96 uint96_mul_64_32(const uint_64 x, const uint_32 y)
{
	const uint_64 l = (uint_32)(x) * (uint_64)(y);
	uint96 r; r.s0 = (uint_32)(l); r.s1 = (x >> 32) * y + (l >> 32);
	return r;
}

// --- transform/macro ---

#define FWD2(z0, z1, w) \
{ \
	const uint_32 t = mulmod(z1, w, pq); \
	z1 = submod(z0, t, pq.s0); z0 = addmod(z0, t, pq.s0); \
}

#define BCK2(z0, z1, win) \
{ \
	const uint_32 t = submod(z1, z0, pq.s0); z0 = addmod(z0, z1, pq.s0); \
	z1 = mulmod(t, win, pq); \
}

#define SQR2(z0, z1, w) \
{ \
	const uint_32 t = mulmod(sqrmod(z1, pq), w, pq); \
	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \
	z0 = addmod(sqrmod(z0, pq), t, pq.s0); \
}

#define SQR2N(z0, z1, w) \
{ \
	const uint_32 t = mulmod(sqrmod(z1, pq), w, pq); \
	z1 = mulmod(addmod(z0, z0, pq.s0), z1, pq); \
	z0 = submod(sqrmod(z0, pq), t, pq.s0); \
}

#define MUL2(z0, z1, zp0, zp1, w) \
{ \
	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = addmod(mulmod(z0, zp0, pq), t, pq.s0); \
}

#define MUL2N(z0, z1, zp0, zp1, w) \
{ \
	const uint_32 t = mulmod(mulmod(z1, zp1, pq), w, pq); \
	z1 = addmod(mulmod(z0, zp1, pq), mulmod(zp0, z1, pq), pq.s0); \
	z0 = submod(mulmod(z0, zp0, pq), t, pq.s0); \
}

#define FWD2v2(z0, z1, w) \
{ \
	const uint2_32 t = mulmod2(z1, w, pq); \
	z1 = submod2(z0, t, pq.s0); z0 = addmod2(z0, t, pq.s0); \
}

#define BCK2v2(z0, z1, win) \
{ \
	const uint2_32 t = submod2(z1, z0, pq.s0); z0 = addmod2(z0, z1, pq.s0); \
	z1 = mulmod2(t, win, pq); \
}

#define FWD2v4(z0, z1, w) \
{ \
	const uint4_32 t = mulmod4(z1, w, pq); \
	z1 = submod4(z0, t, pq.s0); z0 = addmod4(z0, t, pq.s0); \
}

#define BCK2v4(z0, z1, win) \
{ \
	const uint4_32 t = submod4(z1, z0, pq.s0); z0 = addmod4(z0, z1, pq.s0); \
	z1 = mulmod4(t, win, pq); \
}

INLINE void _loadg1(const sz_t n, uint_32 * const zl, const uint_32 * __restrict__ const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _loadl1(const sz_t n, uint_32 * const zl, const uint_32 * __restrict__ const Z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
INLINE void _storeg1(const sz_t n, uint_32 * __restrict__ const z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
INLINE void _storel1(const sz_t n, uint_32 * __restrict__ const Z, const sz_t s, const uint_32 * const zl) { for (sz_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

INLINE void _loadg2(const sz_t n, uint2_32 * const zl, const uint2_32 * __restrict__ const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _loadl2(const sz_t n, uint2_32 * const zl, const uint2_32 * __restrict__ const Z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
INLINE void _storeg2(const sz_t n, uint2_32 * __restrict__ const z, const sz_t s, const uint2_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
INLINE void _storel2(const sz_t n, uint2_32 * __restrict__ const Z, const sz_t s, const uint2_32 * const zl) { for (sz_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

INLINE void _loadg4(const sz_t n, uint4_32 * const zl, const uint4_32 * __restrict__ const z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = z[l * s]; }
INLINE void _loadl4(const sz_t n, uint4_32 * const zl, const uint4_32 * __restrict__ const Z, const sz_t s) { for (sz_t l = 0; l < n; ++l) zl[l] = Z[l * s]; }
INLINE void _storeg4(const sz_t n, uint4_32 * __restrict__ const z, const sz_t s, const uint4_32 * const zl) { for (sz_t l = 0; l < n; ++l) z[l * s] = zl[l]; }
INLINE void _storel4(const sz_t n, uint4_32 * __restrict__ const Z, const sz_t s, const uint4_32 * const zl) { for (sz_t l = 0; l < n; ++l) Z[l * s] = zl[l]; }

// ---

INLINE void _forward4x1(const uint2_32 pq, uint_32 z[4], const uint_32 w1, const uint_32 w2[2])
{
	FWD2(z[0], z[2], w1); FWD2(z[1], z[3], w1);
	FWD2(z[0], z[1], w2[0]); FWD2(z[2], z[3], w2[1]);
}

INLINE void _backward4x1(const uint2_32 pq, uint_32 z[4], const uint_32 win1, const uint_32 win2[2])
{
	BCK2(z[0], z[1], win2[0]); BCK2(z[2], z[3], win2[1]);
	BCK2(z[0], z[2], win1); BCK2(z[1], z[3], win1);
}

INLINE void _forward4x1_0(const uint2_32 pq, const uint4_32 f0, uint_32 z[4])
{
	const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod(z[0], rsq, pq); z[1] = mulmod(z[1], rsq, pq);
	FWD2(z[0], z[2], mfim); FWD2(z[1], z[3], mfim);
	FWD2(z[0], z[1], sqrti); FWD2(z[2], z[3], isqrti);
}

INLINE void _backward4x1_0(const uint2_32 pq, const uint4_32 b0, uint_32 z[4])
{
	const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
	BCK2(z[0], z[1], isqrti); BCK2(z[2], z[3], sqrti);
	BCK2(z[0], z[2], im); BCK2(z[1], z[3], im);
}

INLINE void _square2x2(const uint2_32 pq, uint_32 z[4], const uint_32 w)
{
	SQR2(z[0], z[1], w); SQR2N(z[2], z[3], w);
}

INLINE void _square4(const uint2_32 pq, uint_32 z[4], const uint_32 w, const uint_32 win)
{
	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);
	_square2x2(pq, z, w);
	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);
}

INLINE void _fwd4(const uint2_32 pq, uint_32 z[4], const uint_32 w)
{
	FWD2(z[0], z[2], w); FWD2(z[1], z[3], w);
}

INLINE void _mul2x2(const uint2_32 pq, uint_32 z[4], const uint_32 zp[4], const uint_32 w)
{
	MUL2(z[0], z[1], zp[0], zp[1], w); MUL2N(z[2], z[3], zp[2], zp[3], w);
}

INLINE void _mul4(const uint2_32 pq, uint_32 z[4], const uint_32 zp[4], const uint_32 w, const uint_32 win)
{
	_fwd4(pq, z, w);
	_mul2x2(pq, z, zp, w);
	BCK2(z[0], z[2], win); BCK2(z[1], z[3], win);
}

// --- v2

INLINE void _forward4x2(const uint2_32 pq, uint2_32 z[4], const uint_32 w1, const uint_32 w2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _backward4x2(const uint2_32 pq, uint2_32 z[4], const uint_32 win1, const uint_32 win2[2])
{
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

INLINE void _forward4x2_0(const uint2_32 pq, const uint4_32 f0, uint2_32 z[4])
{
	const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod2(z[0], rsq, pq); z[1] = mulmod2(z[1], rsq, pq);
	FWD2v2(z[0], z[2], mfim); FWD2v2(z[1], z[3], mfim);
	FWD2v2(z[0], z[1], sqrti); FWD2v2(z[2], z[3], isqrti);
}

INLINE void _backward4x2_0(const uint2_32 pq, const uint4_32 b0, uint2_32 z[4])
{
	const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
	BCK2v2(z[0], z[1], isqrti); BCK2v2(z[2], z[3], sqrti);
	BCK2v2(z[0], z[2], im); BCK2v2(z[1], z[3], im);
}

INLINE void _square4x2(const uint2_32 pq, uint2_32 z[4], const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
	SQR2(z[0].s0, z[0].s1, w2[0]); SQR2N(z[1].s0, z[1].s1, w2[0]);
	SQR2(z[2].s0, z[2].s1, w2[1]); SQR2N(z[3].s0, z[3].s1, w2[1]);
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _square8(const uint2_32 pq, uint2_32 z[4], const uint_32 w1, const uint_32 win1, const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_square4x2(pq, z, w2, win2);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

INLINE void _fwd4x2(const uint2_32 pq, uint2_32 z[4], const uint_32 w2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
}

INLINE void _fwd8(const uint2_32 pq, uint2_32 z[4], const uint_32 w1, const uint_32 w2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_fwd4x2(pq, z, w2);
}

INLINE void _mul4x2(const uint2_32 pq, uint2_32 z[4], const uint2_32 zp[4], const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v2(z[0], z[1], w2[0]); FWD2v2(z[2], z[3], w2[1]);
	MUL2(z[0].s0, z[0].s1, zp[0].s0, zp[0].s1, w2[0]); MUL2N(z[1].s0, z[1].s1, zp[1].s0, zp[1].s1, w2[0]);
	MUL2(z[2].s0, z[2].s1, zp[2].s0, zp[2].s1, w2[1]); MUL2N(z[3].s0, z[3].s1, zp[3].s0, zp[3].s1, w2[1]);
	BCK2v2(z[0], z[1], win2[0]); BCK2v2(z[2], z[3], win2[1]);
}

INLINE void _mul8(const uint2_32 pq, uint2_32 z[4], const uint2_32 zp[4], const uint_32 w1, const uint_32 win1, const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v2(z[0], z[2], w1); FWD2v2(z[1], z[3], w1);
	_mul4x2(pq, z, zp, w2, win2);
	BCK2v2(z[0], z[2], win1); BCK2v2(z[1], z[3], win1);
}

// --- v4

INLINE void _forward4x4(const uint2_32 pq, uint4_32 z[4], const uint_32 w1, const uint_32 w2[2])
{
	FWD2v4(z[0], z[2], w1); FWD2v4(z[1], z[3], w1);
	FWD2v4(z[0], z[1], w2[0]); FWD2v4(z[2], z[3], w2[1]);
}

INLINE void _backward4x4(const uint2_32 pq, uint4_32 z[4], const uint_32 win1, const uint_32 win2[2])
{
	BCK2v4(z[0], z[1], win2[0]); BCK2v4(z[2], z[3], win2[1]);
	BCK2v4(z[0], z[2], win1); BCK2v4(z[1], z[3], win1);
}

INLINE void _forward4x4_0(const uint2_32 pq, const uint4_32 f0, uint4_32 z[4])
{
	const uint_32 rsq = f0.s0, mfim = f0.s1, sqrti = f0.s2, isqrti = f0.s3;
	z[0] = mulmod4(z[0], rsq, pq); z[1] = mulmod4(z[1], rsq, pq);
	FWD2v4(z[0], z[2], mfim); FWD2v4(z[1], z[3], mfim);
	FWD2v4(z[0], z[1], sqrti); FWD2v4(z[2], z[3], isqrti);
}

INLINE void _backward4x4_0(const uint2_32 pq, const uint4_32 b0, uint4_32 z[4])
{
	const uint_32 isqrti = b0.s0, sqrti = b0.s1, im = b0.s2;
	BCK2v4(z[0], z[1], isqrti); BCK2v4(z[2], z[3], sqrti);
	BCK2v4(z[0], z[2], im); BCK2v4(z[1], z[3], im);
}

// _square4x2v4: original passes z[i].s01/.s23 (uint2_32) as lvalues to FWD2v2/BCK2v2.
// CUDA structs have no .s01/.s23 lvalue, so use uint2_32 temporaries and write back.
INLINE void _square4x2v4(const uint2_32 pq, uint4_32 z[2], const uint_32 w2[2], const uint_32 win2[2])
{
	for (sz_t i = 0; i < 2; ++i)
	{
		uint2_32 zs01 = make_uint2_32(z[i].s0, z[i].s1), zs23 = make_uint2_32(z[i].s2, z[i].s3);
		FWD2v2(zs01, zs23, w2[i]);
		SQR2(zs01.s0, zs01.s1, w2[i]); SQR2N(zs23.s0, zs23.s1, w2[i]);
		BCK2v2(zs01, zs23, win2[i]);
		z[i].s0 = zs01.s0; z[i].s1 = zs01.s1; z[i].s2 = zs23.s0; z[i].s3 = zs23.s1;
	}
}

INLINE void _square8v4(const uint2_32 pq, uint4_32 z[2], const uint_32 w1, const uint_32 win1, const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v4(z[0], z[1], w1);
	_square4x2v4(pq, z, w2, win2);
	BCK2v4(z[0], z[1], win1);
}

// _fwd4x2v4: same .s01/.s23 lvalue rewrite as _square4x2v4.
INLINE void _fwd4x2v4(const uint2_32 pq, uint4_32 z[2], const uint_32 w2[2])
{
	for (sz_t i = 0; i < 2; ++i)
	{
		uint2_32 zs01 = make_uint2_32(z[i].s0, z[i].s1), zs23 = make_uint2_32(z[i].s2, z[i].s3);
		FWD2v2(zs01, zs23, w2[i]);
		z[i].s0 = zs01.s0; z[i].s1 = zs01.s1; z[i].s2 = zs23.s0; z[i].s3 = zs23.s1;
	}
}

INLINE void _fwd8v4(const uint2_32 pq, uint4_32 z[2], const uint_32 w1, const uint_32 w2[2])
{
	FWD2v4(z[0], z[1], w1);
	_fwd4x2v4(pq, z, w2);
}

// _mul4x2v4: same .s01/.s23 lvalue rewrite as _square4x2v4.
INLINE void _mul4x2v4(const uint2_32 pq, uint4_32 z[2], const uint4_32 zp[2], const uint_32 w2[2], const uint_32 win2[2])
{
	for (sz_t i = 0; i < 2; ++i)
	{
		uint2_32 zs01 = make_uint2_32(z[i].s0, z[i].s1), zs23 = make_uint2_32(z[i].s2, z[i].s3);
		FWD2v2(zs01, zs23, w2[i]);
		MUL2(zs01.s0, zs01.s1, zp[i].s0, zp[i].s1, w2[i]); MUL2N(zs23.s0, zs23.s1, zp[i].s2, zp[i].s3, w2[i]);
		BCK2v2(zs01, zs23, win2[i]);
		z[i].s0 = zs01.s0; z[i].s1 = zs01.s1; z[i].s2 = zs23.s0; z[i].s3 = zs23.s1;
	}
}

INLINE void _mul8v4(const uint2_32 pq, uint4_32 z[2], const uint4_32 zp[2], const uint_32 w1, const uint_32 win1, const uint_32 w2[2], const uint_32 win2[2])
{
	FWD2v4(z[0], z[1], w1);
	_mul4x2v4(pq, z, zp, w2, win2);
	BCK2v4(z[0], z[1], win1);
}

// --- inverse of roots is wi[s + j] or w[s + s - j - 1] ---

#define DECLARE_W1(sj)			const uint_32 w1 = w[sj];
#define DECLARE_W2(sj)			uint_32 w2[2]; { const uint2_32 t = ((const uint2_32 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; }
#define DECLARE_W12(sj)			DECLARE_W1(sj); DECLARE_W2(sj);
#define DECLARE_W1_2(sj)		uint_32 w1[2]; { const uint2_32 t = ((const uint2_32 *)w)[sj]; w1[0] = t.s0; w1[1] = t.s1; }
#define DECLARE_W2_4(sj)		uint_32 w2[4]; { const uint4_32 t = ((const uint4_32 *)w)[sj]; w2[0] = t.s0; w2[1] = t.s1; w2[2] = t.s2; w2[3] = t.s3; }
#define DECLARE_W12_24(sj)		DECLARE_W1_2(sj); DECLARE_W2_4(sj);

#define DECLARE_WIN1(sji)		const uint_32 win1 = wi[sji];
#if defined(USE_WI)
#define DECLARE_IVAR(s, j)		const sz_t sji = s + j; const uint_32 * __restrict__ const wi = &w[WI_SHFT];
#define DECLARE_WIN2(sji)		uint_32 win2[2]; { const uint2_32 t = ((const uint2_32 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; }
#define DECLARE_WIN1_2(sji)		uint_32 win1[2]; { const uint2_32 t = ((const uint2_32 *)wi)[sji]; win1[0] = t.s0; win1[1] = t.s1; }
#define DECLARE_WIN2_4(sji)		uint_32 win2[4]; { const uint4_32 t = ((const uint4_32 *)wi)[sji]; win2[0] = t.s0; win2[1] = t.s1; win2[2] = t.s2; win2[3] = t.s3; }
#else
#define DECLARE_IVAR(s, j)		const sz_t sji = s + s - j - 1; const uint_32 * __restrict__ const wi = w;
#define DECLARE_WIN2(sji)		uint_32 win2[2]; { const uint2_32 t = ((const uint2_32 *)wi)[sji]; win2[0] = t.s1; win2[1] = t.s0; }
#define DECLARE_WIN1_2(sji)		uint_32 win1[2]; { const uint2_32 t = ((const uint2_32 *)wi)[sji]; win1[0] = t.s1; win1[1] = t.s0; }
#define DECLARE_WIN2_4(sji)		uint_32 win2[4]; { const uint4_32 t = ((const uint4_32 *)wi)[sji]; win2[0] = t.s3; win2[1] = t.s2; win2[2] = t.s1; win2[3] = t.s0; }
#endif
#define DECLARE_WIN12(sj)		DECLARE_WIN1(sj); DECLARE_WIN2(sj);
#define DECLARE_WIN12_24(sj)	DECLARE_WIN1_2(sj); DECLARE_WIN2_4(sj);

// --- vector size (1, 2 or 4) ---

#if VSIZE == 4
#define VTYPE				uint4_32
#define _loadg				_loadg4
#define _loadl				_loadl4
#define _storeg				_storeg4
#define _storel				_storel4
#define _forward4			_forward4x4
#define _backward4			_backward4x4
#define _forward4_0			_forward4x4_0
#define _backward4_0		_backward4x4_0
#elif VSIZE == 2
#define VTYPE				uint2_32
#define _loadg				_loadg2
#define _loadl				_loadl2
#define _storeg				_storeg2
#define _storel				_storel2
#define _forward4			_forward4x2
#define _backward4			_backward4x2
#define _forward4_0			_forward4x2_0
#define _backward4_0		_backward4x2_0
#else
#define VTYPE				uint_32
#define _loadg				_loadg1
#define _loadl				_loadl1
#define _storeg				_storeg1
#define _storel				_storel1
#define _forward4			_forward4x1
#define _backward4			_backward4x1
#define _forward4_0			_forward4x1_0
#define _backward4_0		_backward4x1_0
#endif

// --- transform/inline global mem ---

#if defined(ALL_FUNC)

INLINE void forward4io(const uint2_32 pq, const sz_t m, VTYPE * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_forward4(pq, zl, w1, w2);
	_storeg(4, z, m, zl);
}

INLINE void backward4io(const uint2_32 pq, const sz_t m, VTYPE * __restrict__ const z, const uint_32 * __restrict__ const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_backward4(pq, zl, win1, win2);
	_storeg(4, z, m, zl);
}

INLINE void forward4io_0(const uint2_32 pq, const uint4_32 f0, VTYPE * __restrict__ const z)
{
	const sz_t m = N_SZ / 4 / VSIZE;
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_forward4_0(pq, f0, zl);
	_storeg(4, z, m, zl);
}

INLINE void backwardio_0(const uint2_32 pq, const uint4_32 b0, VTYPE * __restrict__ const z)
{
	const sz_t m = N_SZ / 4 / VSIZE;
	VTYPE zl[4]; _loadg(4, zl, z, m);
	_backward4_0(pq, b0, zl);
	_storeg(4, z, m, zl);
}

// --- v1

INLINE void square2x2io(const uint2_32 pq, uint_32 * __restrict__ const z, const uint_32 w)
{
	uint_32 zl[4]; _loadg1(4, zl, z, 1);
	_square2x2(pq, zl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void square4x1io(const uint2_32 pq, uint_32 * __restrict__ const z, const uint_32 w, const uint_32 win)
{
	uint_32 zl[4]; _loadg1(4, zl, z, 1);
	_square4(pq, zl, w, win);
	_storeg1(4, z, 1, zl);
}

INLINE void fwd4x1io(const uint2_32 pq, uint_32 * __restrict__ const z, const uint_32 w)
{
	uint_32 zl[4]; _loadg1(4, zl, z, 1);
	_fwd4(pq, zl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void mul2x2io(const uint2_32 pq, uint_32 * __restrict__ const z, const uint_32 * __restrict__ const zp, const uint_32 w)
{
	uint_32 zpl[4]; _loadg1(4, zpl, zp, 1);
	uint_32 zl[4]; _loadg1(4, zl, z, 1);
	_mul2x2(pq, zl, zpl, w);
	_storeg1(4, z, 1, zl);
}

INLINE void mul4x1io(const uint2_32 pq, uint_32 * __restrict__ const z, const uint_32 * __restrict__ const zp, const uint_32 w, const uint_32 win)
{
	uint_32 zpl[4]; _loadg1(4, zpl, zp, 1);
	uint_32 zl[4]; _loadg1(4, zl, z, 1);
	_mul4(pq, zl, zpl, w, win);
	_storeg1(4, z, 1, zl);
}

// --- v2

INLINE void square4x2io(const uint2_32 pq, uint2_32 * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_square4x2(pq, zl, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void square8x1io(const uint2_32 pq, uint2_32 * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_square8(pq, zl, w1, win1, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void fwd4x2io(const uint2_32 pq, uint2_32 * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W2(sj);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_fwd4x2(pq, zl, w2);
	_storeg2(4, z, 1, zl);
}

INLINE void fwd8x1io(const uint2_32 pq, uint2_32 * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_fwd8(pq, zl, w1, w2);
	_storeg2(4, z, 1, zl);
}

INLINE void mul4x2io(const uint2_32 pq, uint2_32 * __restrict__ const z, const uint2_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint2_32 zpl[4]; _loadg2(4, zpl, zp, 1);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_mul4x2(pq, zl, zpl, w2, win2);
	_storeg2(4, z, 1, zl);
}

INLINE void mul8x1io(const uint2_32 pq, uint2_32 * __restrict__ const z, const uint2_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint2_32 zpl[4]; _loadg2(4, zpl, zp, 1);
	uint2_32 zl[4]; _loadg2(4, zl, z, 1);
	_mul8(pq, zl, zpl, w1, win1, w2, win2);
	_storeg2(4, z, 1, zl);
}

// --- v4

INLINE void square4x4io(const uint2_32 pq, uint4_32 * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);
	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void square8x2io(const uint2_32 pq, uint4_32 * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void fwd4x4io(const uint2_32 pq, uint4_32 * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W2_4(sj);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_fwd4x2v4(pq, &zl[0], &w2[0]);
	_fwd4x2v4(pq, &zl[2], &w2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void fwd8x2io(const uint2_32 pq, uint4_32 * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12_24(sj);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);
	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void mul4x4io(const uint2_32 pq, uint4_32 * __restrict__ const z, const uint4_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint4_32 zpl[4]; _loadg4(4, zpl, zp, 1);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

INLINE void mul8x2io(const uint2_32 pq, uint4_32 * __restrict__ const z, const uint4_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint4_32 zpl[4]; _loadg4(4, zpl, zp, 1);
	uint4_32 zl[4]; _loadg4(4, zl, z, 1);
	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storeg4(4, z, 1, zl);
}

// --- v1, v2, v4

INLINE void square4io(const uint2_32 pq, VTYPE * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square4x4io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
	square4x2io(pq, z, w, wi, sj, sji);
#else
	square4x1io(pq, z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4io(const uint2_32 pq, VTYPE * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
#if VSIZE == 4
	fwd4x4io(pq, z, w, sj);
#elif VSIZE == 2
	fwd4x2io(pq, z, w, sj);
#else
	fwd4x1io(pq, z, w[sj]);
#endif
}

INLINE void mul4io(const uint2_32 pq, VTYPE * __restrict__ const z, const VTYPE * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul4x4io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul4x2io(pq, z, zp, w, wi, sj, sji);
#else
	mul4x1io(pq, z, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4

INLINE void square8io(const uint2_32 pq, VTYPE * __restrict__ const z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square8x2io(pq, z, w, wi, sj, sji);
#elif VSIZE == 2
	square8x1io(pq, z, w, wi, sj, sji);
#endif
}

INLINE void fwd8io(const uint2_32 pq, VTYPE * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
#if VSIZE == 4
	fwd8x2io(pq, z, w, sj);
#elif VSIZE == 2
	fwd8x1io(pq, z, w, sj);
#endif
}

INLINE void mul8io(const uint2_32 pq, VTYPE * __restrict__ const z, const VTYPE * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul8x2io(pq, z, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul8x1io(pq, z, zp, w, wi, sj, sji);
#endif
}

#endif // ALL_FUNC
// --- transform/inline local & global mem ---

INLINE void forward_4(const uint2_32 pq, const sz_t m, VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, m);
	_forward4(pq, zl, w1, w2);
	_storel(4, Z, m, zl);
}

INLINE void forward_4i(const uint2_32 pq, const sz_t ml, VTYPE * __restrict__ const Z, const sz_t mg,
	const VTYPE * __restrict__ const z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_forward4(pq, zl, w1, w2);
	_storel(4, Z, ml, zl);
}

INLINE void forward_4i_0(const uint2_32 pq, const uint4_32 f0, const sz_t ml, VTYPE * __restrict__ const Z,
	const sz_t mg, const VTYPE * __restrict__ const z)
{
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_forward4_0(pq, f0, zl);
	_storel(4, Z, ml, zl);
}

INLINE void forward_4o(const uint2_32 pq, const sz_t mg, VTYPE * __restrict__ const z, const sz_t ml,
	const VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, ml);
	_forward4(pq, zl, w1, w2);
	_storeg(4, z, mg, zl);
}

INLINE void backward_4(const uint2_32 pq, const sz_t m, VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, m);
	_backward4(pq, zl, win1, win2);
	_storel(4, Z, m, zl);
}

INLINE void backward_4i(const uint2_32 pq, const sz_t ml, VTYPE * __restrict__ const Z, const sz_t mg,
	const VTYPE * __restrict__ const z, const uint_32 * __restrict__ const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	VTYPE zl[4]; _loadg(4, zl, z, mg);
	_backward4(pq, zl, win1, win2);
	_storel(4, Z, ml, zl);
}

INLINE void backward_4o(const uint2_32 pq, const sz_t mg, VTYPE * __restrict__ const z, const sz_t ml,
	const VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const wi, const sz_t sji)
{
	DECLARE_WIN12(sji);
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, ml);
	_backward4(pq, zl, win1, win2);
	_storeg(4, z, mg, zl);
}

INLINE void backward_4o_0(const uint2_32 pq, const uint4_32 b0, const sz_t mg, VTYPE * __restrict__ const z,
	const sz_t ml, const VTYPE * __restrict__ const Z)
{
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, ml);
	_backward4_0(pq, b0, zl);
	_storeg(4, z, mg, zl);
}

// --- v1

INLINE void square_2x2(const uint2_32 pq, uint_32 * __restrict__ const Z, const uint_32 w)
{
	__syncthreads();
	uint_32 zl[4]; _loadl1(4, zl, Z, 1);
	_square2x2(pq, zl, w);
	_storel1(4, Z, 1, zl);
}

INLINE void square_4x1(const uint2_32 pq, uint_32 * __restrict__ const Z, const uint_32 w, const uint_32 win)
{
	__syncthreads();
	uint_32 zl[4]; _loadl1(4, zl, Z, 1);
	_square4(pq, zl, w, win);
	_storel1(4, Z, 1, zl);
}

INLINE void write_4(const sz_t mg, VTYPE * __restrict__ const z, const VTYPE * __restrict__ const Z)
{
	__syncthreads();
	VTYPE zl[4]; _loadl(4, zl, Z, 1);
	_storeg(4, z, mg, zl);
}

INLINE void fwd4x1_write(const uint2_32 pq, const sz_t mg, uint_32 * __restrict__ const z,
	const uint_32 * __restrict__ const Z, const uint_32 w)
{
	__syncthreads();
	uint_32 zl[4]; _loadl1(4, zl, Z, 1);
	_fwd4(pq, zl, w);
	_storeg1(4, z, mg, zl);
}

INLINE void mul_2x2(const uint2_32 pq, uint_32 * __restrict__ const Z, const sz_t mg,
	const uint_32 * __restrict__ const zp, const uint_32 w)
{
	uint_32 zpl[4]; _loadg1(4, zpl, zp, mg);
	__syncthreads();
	uint_32 zl[4]; _loadl1(4, zl, Z, 1);
	_mul2x2(pq, zl, zpl, w);
	_storel1(4, Z, 1, zl);
}

INLINE void mul_4x1(const uint2_32 pq, uint_32 * __restrict__ const Z, const sz_t mg,
	const uint_32 * __restrict__ const zp, const uint_32 w, const uint_32 win)
{
	uint_32 zpl[4]; _loadg1(4, zpl, zp, mg);
	__syncthreads();
	uint_32 zl[4]; _loadl1(4, zl, Z, 1);
	_mul4(pq, zl, zpl, w, win);
	_storel1(4, Z, 1, zl);
}

// --- v2

INLINE void square_4x2(const uint2_32 pq, uint2_32 * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_square4x2(pq, zl, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void square_8x1(const uint2_32 pq, uint2_32 * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_square8(pq, zl, w1, win1, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void fwd4x2_write(const uint2_32 pq, const sz_t mg, uint2_32 * __restrict__ const z,
	const uint2_32 * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W2(sj);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_fwd4x2(pq, zl, w2);
	_storeg2(4, z, mg, zl);
}

INLINE void fwd8x1_write(const uint2_32 pq, const sz_t mg, uint2_32 * __restrict__ const z,
	const uint2_32 * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12(sj);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_fwd8(pq, zl, w1, w2);
	_storeg2(4, z, mg, zl);
}

INLINE void mul_4x2(const uint2_32 pq, uint2_32 * __restrict__ const Z, const sz_t mg, const uint2_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2(sj);
	DECLARE_WIN2(sji);
	uint2_32 zpl[4]; _loadg2(4, zpl, zp, mg);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_mul4x2(pq, zl, zpl, w2, win2);
	_storel2(4, Z, 1, zl);
}

INLINE void mul_8x1(const uint2_32 pq, uint2_32 * __restrict__ const Z, const sz_t mg, const uint2_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12(sj);
	DECLARE_WIN12(sji);
	uint2_32 zpl[4]; _loadg2(4, zpl, zp, mg);
	__syncthreads();
	uint2_32 zl[4]; _loadl2(4, zl, Z, 1);
	_mul8(pq, zl, zpl, w1, win1, w2, win2);
	_storel2(4, Z, 1, zl);
}

// --- v4

INLINE void square_4x4(const uint2_32 pq, uint4_32 * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_square4x2v4(pq, &zl[0], &w2[0], &win2[0]);
	_square4x2v4(pq, &zl[2], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void square_8x2(const uint2_32 pq, uint4_32 * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_square8v4(pq, &zl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_square8v4(pq, &zl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void fwd4x4_write(const uint2_32 pq, const sz_t mg, uint4_32 * __restrict__ const z,
	const uint4_32 * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W2_4(sj);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_fwd4x2v4(pq, &zl[0], &w2[0]);
	_fwd4x2v4(pq, &zl[2], &w2[2]);
	_storeg4(4, z, mg, zl);
}

INLINE void fwd8x2_write(const uint2_32 pq, const sz_t mg, uint4_32 * __restrict__ const z,
	const uint4_32 * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
	DECLARE_W12_24(sj);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_fwd8v4(pq, &zl[0], w1[0], &w2[0]);
	_fwd8v4(pq, &zl[2], w1[1], &w2[2]);
	_storeg4(4, z, mg, zl);
}

INLINE void mul_4x4(const uint2_32 pq, uint4_32 * __restrict__ const Z, const sz_t mg, const uint4_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W2_4(sj);
	DECLARE_WIN2_4(sji);
	uint4_32 zpl[4]; _loadg4(4, zpl, zp, mg);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_mul4x2v4(pq, &zl[0], &zpl[0], &w2[0], &win2[0]);
	_mul4x2v4(pq, &zl[2], &zpl[2], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

INLINE void mul_8x2(const uint2_32 pq, uint4_32 * __restrict__ const Z, const sz_t mg, const uint4_32 * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
	DECLARE_W12_24(sj);
	DECLARE_WIN12_24(sji);
	uint4_32 zpl[4]; _loadg4(4, zpl, zp, mg);
	__syncthreads();
	uint4_32 zl[4]; _loadl4(4, zl, Z, 1);
	_mul8v4(pq, &zl[0], &zpl[0], w1[0], win1[0], &w2[0], &win2[0]);
	_mul8v4(pq, &zl[2], &zpl[2], w1[1], win1[1], &w2[2], &win2[2]);
	_storel4(4, Z, 1, zl);
}

// --- v1, v2, v4 -- no barrier

INLINE void square_4(const uint2_32 pq, VTYPE * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square_4x4(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
	square_4x2(pq, Z, w, wi, sj, sji);
#else
	square_4x1(pq, Z, w[sj], wi[sji]);
#endif
}

INLINE void fwd4_write(const uint2_32 pq, const sz_t mg, VTYPE * __restrict__ const z,
	const VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
#if VSIZE == 4
	fwd4x4_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
	fwd4x2_write(pq, mg, z, Z, w, sj);
#else
	fwd4x1_write(pq, mg, z, Z, w[sj]);
#endif
}

INLINE void mul_4(const uint2_32 pq, VTYPE * __restrict__ const Z, const sz_t mg, const VTYPE * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul_4x4(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul_4x2(pq, Z, mg, zp, w, wi, sj, sji);
#else
	mul_4x1(pq, Z, mg, zp, w[sj], wi[sji]);
#endif
}

// --- v2, v4 -- no barrier

INLINE void square_8(const uint2_32 pq, VTYPE * __restrict__ const Z,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	square_8x2(pq, Z, w, wi, sj, sji);
#elif VSIZE == 2
	square_8x1(pq, Z, w, wi, sj, sji);
#endif
}

INLINE void fwd8_write(const uint2_32 pq, const sz_t mg, VTYPE * __restrict__ const z,
	const VTYPE * __restrict__ const Z, const uint_32 * __restrict__ const w, const sz_t sj)
{
#if VSIZE == 4
	fwd8x2_write(pq, mg, z, Z, w, sj);
#elif VSIZE == 2
	fwd8x1_write(pq, mg, z, Z, w, sj);
#endif
}

INLINE void mul_8(const uint2_32 pq, VTYPE * __restrict__ const Z, const sz_t mg, const VTYPE * __restrict__ const zp,
	const uint_32 * __restrict__ const w, const uint_32 * __restrict__ const wi, const sz_t sj, const sz_t sji)
{
#if VSIZE == 4
	mul_8x2(pq, Z, mg, zp, w, wi, sj, sji);
#elif VSIZE == 2
	mul_8x1(pq, Z, mg, zp, w, wi, sj, sji);
#endif
}

// --- transform/macro ---

#define DECLARE_VAR_REGv1() \
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x), lid = gid >> (LN_SZ - 2), mid = gid & ~((N_SZ / 4) - 1), id = gid %  (N_SZ / 4); \
	const uint2_32 pq = g_pq[lid]; \
	uint_32 * __restrict__ const z = &zg[4 * mid]; \
	const uint_32 * __restrict__ const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv1() \
	const uint_32 * __restrict__ const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv2() \
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x), lid = gid >> (LN_SZ - 3), mid = gid & ~((N_SZ / 8) - 1), id = gid %  (N_SZ / 8); \
	const uint2_32 pq = g_pq[lid]; \
	uint2_32 * __restrict__ const z = &zg[4 * mid]; \
	const uint_32 * __restrict__ const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv2() \
	const uint2_32 * __restrict__ const zp = &zpg[4 * mid];

#define DECLARE_VAR_REGv4() \
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x), lid = gid >> (LN_SZ - 4), mid = gid & ~((N_SZ / 16) - 1), id = gid %  (N_SZ / 16); \
	const uint2_32 pq = g_pq[lid]; \
	uint4_32 * __restrict__ const z = &zg[4 * mid]; \
	const uint_32 * __restrict__ const w = &wg[lid * W_SHFT];

#define DECLARE_VARP_REGv4() \
	const uint4_32 * __restrict__ const zp = &zpg[4 * mid];

#if VSIZE == 4
#define DECLARE_VAR_REG		DECLARE_VAR_REGv4
#define DECLARE_VARP_REG	DECLARE_VARP_REGv4
#elif VSIZE == 2
#define DECLARE_VAR_REG		DECLARE_VAR_REGv2
#define DECLARE_VARP_REG	DECLARE_VARP_REGv2
#else
#define DECLARE_VAR_REG		DECLARE_VAR_REGv1
#define DECLARE_VARP_REG	DECLARE_VARP_REGv1
#endif

// --- transform without local mem ---

#if defined(ALL_FUNC)
extern "C" __global__
void forward4(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id;
	forward4io(pq, m, &z[k], w, s + j);
}

extern "C" __global__
void backward4(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	DECLARE_VAR_REG();
	const sz_t m = (sz_t)(1) << lm, j = id >> lm, k = 3 * (id & ~(m - 1)) + id; DECLARE_IVAR(s, j);
	backward4io(pq, m, &z[k], wi, sji);
}

extern "C" __global__
void forward4_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t k = id;
	forward4io_0(pq, g_f0[lid], &z[k]);
}

extern "C" __global__
void backward4_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t k = id;
	backwardio_0(pq, g_b0[lid], &z[k]);
}

extern "C" __global__
void square2x2(uint_32 * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REGv1();
	const sz_t j = id, k = 4 * id;
	square2x2io(pq, &z[k], w[N_SZ / 4 + j]);
}

extern "C" __global__
void square4(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	square4io(pq, &z[k], w, wi, sj, sji);
}

extern "C" __global__
void fwd4p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;
	fwd4io(pq, &z[k], w, sj);
}

extern "C" __global__
void mul4(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	mul4io(pq, &z[k], &zp[k], w, wi, sj, sji);
}

// --- v1

extern "C" __global__
void mul2x2(uint_32 * __restrict__ const zg, const uint_32 * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REGv1();
	DECLARE_VARP_REGv1();
	const sz_t j = id, k = 4 * id;
	mul2x2io(pq, &z[k], &zp[k], w[N_SZ / 4 + j]);
}

// --- v2, v4

extern "C" __global__
void square8(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	square8io(pq, &z[k], w, wi, sj, sji);
}

extern "C" __global__
void fwd8p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j;
	fwd8io(pq, &z[k], w, sj);
}

extern "C" __global__
void mul8(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_REG();
	DECLARE_VARP_REG();
	const sz_t j = id, k = 4 * id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j);
	mul8io(pq, &z[k], &zp[k], w, wi, sj, sji);
}

#endif // ALL_FUNC

// --- transform ---

#if !defined(SHORT_FUNC)

#define DECLARE_VAR(B_N, CHUNK_N) \
	/* threadIdx < B_N */ \
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \
	const sz_t i = local_id, chunk_idx = i % CHUNK_N, threadIdx = i / CHUNK_N, blockIdx = group_id * CHUNK_N + chunk_idx; \
	VTYPE * const Zi = &Z[chunk_idx]; \
	\
	const sz_t blockIdx_m = blockIdx >> lm, idx_m = blockIdx_m * B_N + threadIdx; \
	const sz_t blockIdx_mm = blockIdx_m << lm, idx_mm = idx_m << lm; \
	\
	const sz_t ki = blockIdx + blockIdx_mm * (B_N * 3 - 1) + idx_mm, ko = blockIdx - blockIdx_mm + idx_mm * 4; \
	\
	const sz_t sj = s + idx_m; DECLARE_IVAR(s, idx_m);

#define DECLARE_VAR_FORWARD() \
	VTYPE * __restrict__ const zi = &z[ki]; \
	VTYPE * __restrict__ const zo = &z[ko];

#define DECLARE_VAR_BACKWARD() \
	VTYPE * __restrict__ const zi = &z[ko]; \
	VTYPE * __restrict__ const zo = &z[ki];

#define FORWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i(pq, B_N * CHUNK_N, &Z[i], B_N << lm, zi, w, sj / B_N);

#define FORWARD_I_0(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_FORWARD(); \
	\
	forward_4i_0(pq, g_f0[lid], B_N * CHUNK_N, &Z[i], B_N << lm, zi);

#define BACKWARD_I(B_N, CHUNK_N) \
	DECLARE_VAR(B_N, CHUNK_N); \
	DECLARE_VAR_BACKWARD(); \
	\
	backward_4i(pq, 1 * CHUNK_N, &Zi[CHUNK_N * 4 * threadIdx], (sz_t)1 << lm, zi, wi, sji / 1);

// -----------------

#define B_64	(64 / 4)

#if MAX_WG_SZ >= B_64 * CHUNK64
#define ATTR_64() \
	__launch_bounds__(B_64 * CHUNK64)
#else
#define ATTR_64()
#endif

#define FORWARD_64() \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK64, &Zi[CHUNK64 * 4 * threadIdx], w, sj / 1);

#if defined(ALL_FUNC)
extern "C" __global__
void ATTR_64() forward64(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	__shared__ VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I(B_64, CHUNK64);
	FORWARD_64();
}
#endif

extern "C" __global__
void ATTR_64() forward64_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 6; const unsigned int s = 64 / 4;
	__shared__ VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I_0(B_64, CHUNK64);
	FORWARD_64();
}

extern "C" __global__
void ATTR_64() forward64_9(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = 9 - LVSIZE; const unsigned int s = N_SZ / (4 * VSIZE) >> lm;
	__shared__ VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I(B_64, CHUNK64);
	FORWARD_64();
}
extern "C" __global__
ATTR_64()
void forward64_11(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = 11 - LVSIZE; const unsigned int s = N_SZ / (4 * VSIZE) >> lm;
	__shared__ VTYPE Z[4 * B_64 * CHUNK64];
	FORWARD_I(B_64, CHUNK64);
	FORWARD_64();
}

#define BACKWARD_64() \
	__shared__ VTYPE Z[4 * B_64 * CHUNK64]; \
	BACKWARD_I(B_64, CHUNK64); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	backward_4(pq, 4 * CHUNK64, &Zi[CHUNK64 * k4], wi, sji / 4);

#if defined(ALL_FUNC)
extern "C" __global__
ATTR_64()
void backward64(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	BACKWARD_64();
	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);
}
#endif

extern "C" __global__
ATTR_64()
void backward64_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 6; const unsigned int s = 64 / 4;
	BACKWARD_64();
	backward_4o_0(pq, g_b0[lid], B_64 << lm, zo, B_64 * CHUNK64, &Z[i]);
}

extern "C" __global__
ATTR_64()
void backward64_9(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = 9 - LVSIZE; const unsigned int s = N_SZ / (4 * VSIZE) >> lm;
	BACKWARD_64();
	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);
}

extern "C" __global__
ATTR_64()
void backward64_11(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = 11 - LVSIZE; const unsigned int s = N_SZ / (4 * VSIZE) >> lm;
	BACKWARD_64();
	backward_4o(pq, B_64 << lm, zo, B_64 * CHUNK64, &Z[i], wi, sji / B_64);
}

// -----------------

#define B_256	(256 / 4)

#if MAX_WG_SZ >= B_256 * CHUNK256
#define ATTR_256() \
	__launch_bounds__(B_256 * CHUNK256)
#else
#define ATTR_256()
#endif

#define FORWARD_256() \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	forward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], w, sj / 16); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK256, &Zi[CHUNK256 * 4 * threadIdx], w, sj / 1);

#if defined(ALL_FUNC)
extern "C" __global__
ATTR_256()
void forward256(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	__shared__ VTYPE Z[4 * B_256 * CHUNK256];
	FORWARD_I(B_256, CHUNK256);
	FORWARD_256();
}
#endif

extern "C" __global__
ATTR_256()
void forward256_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 8; const unsigned int s = 256 / 4;
	__shared__ VTYPE Z[4 * B_256 * CHUNK256];
	FORWARD_I_0(B_256, CHUNK256);
	FORWARD_256();
}

#define BACKWARD_256() \
	__shared__ VTYPE Z[4 * B_256 * CHUNK256]; \
	BACKWARD_I(B_256, CHUNK256); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	backward_4(pq, 4 * CHUNK256, &Zi[CHUNK256 * k4], wi, sji / 4); \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	backward_4(pq, 16 * CHUNK256, &Zi[CHUNK256 * k16], wi, sji / 16);

#if defined(ALL_FUNC)
extern "C" __global__
ATTR_256()
void backward256(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	BACKWARD_256();
	backward_4o(pq, B_256 << lm, zo, B_256 * CHUNK256, &Z[i], wi, sji / B_256);
}
#endif

extern "C" __global__
ATTR_256()
void backward256_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 8; const unsigned int s = 256 / 4;
	BACKWARD_256();
	backward_4o_0(pq, g_b0[lid], B_256 << lm, zo, B_256 * CHUNK256, &Z[i]);
}

// -----------------

#define B_1024	(1024 / 4)

#if MAX_WG_SZ >= B_1024 * CHUNK1024
#define ATTR_1024() \
	__launch_bounds__(B_1024 * CHUNK1024)
#else
#define ATTR_1024()
#endif

#define FORWARD_1024() \
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64); \
	forward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], w, sj / 64); \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	forward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], w, sj / 16); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	forward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], w, sj / 4); \
	forward_4o(pq, (sz_t)1 << lm, zo, 1 * CHUNK1024, &Zi[CHUNK1024 * 4 * threadIdx], w, sj / 1);

#if defined(ALL_FUNC)
extern "C" __global__
ATTR_1024()
void forward1024(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	__shared__ VTYPE Z[4 * B_1024 * CHUNK1024];
	FORWARD_I(B_1024, CHUNK1024);
	FORWARD_1024();
}
#endif

extern "C" __global__
ATTR_1024()
void forward1024_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 10; const unsigned int s = 1024 / 4;
	__shared__ VTYPE Z[4 * B_1024 * CHUNK1024];
	FORWARD_I_0(B_1024, CHUNK1024);
	FORWARD_1024();
}

#define BACKWARD_1024() \
	__shared__ VTYPE Z[4 * B_1024 * CHUNK1024]; \
	BACKWARD_I(B_1024, CHUNK1024); \
	const sz_t k4 = ((4 * threadIdx) & ~(4 * 4 - 1)) + (threadIdx % 4); \
	backward_4(pq, 4 * CHUNK1024, &Zi[CHUNK1024 * k4], wi, sji / 4); \
	const sz_t k16 = ((4 * threadIdx) & ~(4 * 16 - 1)) + (threadIdx % 16); \
	backward_4(pq, 16 * CHUNK1024, &Zi[CHUNK1024 * k16], wi, sji / 16); \
	const sz_t k64 = ((4 * threadIdx) & ~(4 * 64 - 1)) + (threadIdx % 64); \
	backward_4(pq, 64 * CHUNK1024, &Zi[CHUNK1024 * k64], wi, sji / 64);

#if defined(ALL_FUNC)
extern "C" __global__
ATTR_1024()
void backward1024(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg, const int lm, const unsigned int s)
{
	BACKWARD_1024();
	backward_4o(pq, B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i], wi, sji / B_1024);
}
#endif

extern "C" __global__
ATTR_1024()
void backward1024_0(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	const int lm = LN_SZ - LVSIZE - 10; const unsigned int s = 1024 / 4;
	BACKWARD_1024();
	backward_4o_0(pq, g_b0[lid], B_1024 << lm, zo, B_1024 * CHUNK1024, &Z[i]);
}

// -----------------

#define L32S	(32 / VSIZE)

#define DECLARE_VAR_32() \
	__shared__ VTYPE Z[L32S * BLK32]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L32S / 4 * BLK32), group_id = id / (L32S / 4 * BLK32); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i32 = (local_id & ~(L32S / 4 - 1)) * 4, i8 = local_id % (L32S / 4); \
	const sz_t k32 = group_id * L32S * BLK32 + i32 + i8; \
	\
	VTYPE * __restrict__ const zk = &z[k32]; \
	VTYPE * const Z32 = &Z[i32]; \
	VTYPE * const Zi8 = &Z32[i8]; \
	const sz_t i2 = ((4 * i8) & ~(4 * 2 - 1)) + (i8 % 2); \
	VTYPE * const Zi2 = &Z32[i2]; \
	VTYPE * const Z4 = &Z32[4 * i8];
extern "C" __global__
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__launch_bounds__(L32S / 4 * BLK32)
#endif
void square32(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));
}

#define L64S	(64 / VSIZE)

#define DECLARE_VAR_64() \
	__shared__ VTYPE Z[L64S * BLK64]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L64S / 4 * BLK64), group_id = id / (L64S / 4 * BLK64); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i64 = (local_id & ~(L64S / 4 - 1)) * 4, i16 = local_id % (L64S / 4); \
	const sz_t k64 = group_id * L64S * BLK64 + i64 + i16; \
	\
	VTYPE * __restrict__ const zk = &z[k64]; \
	VTYPE * const Z64 = &Z[i64]; \
	VTYPE * const Zi16 = &Z64[i16]; \
	const sz_t i4 = ((4 * i16) & ~(4 * (L64S / 16) - 1)) + (i16 % (L64S / 16)); \
	VTYPE * const Zi4 = &Z64[i4]; \
	VTYPE * const Z4 = &Z64[4 * i16];

extern "C" __global__
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__launch_bounds__(L64S / 4 * BLK64)
#endif
void square64(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));
	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));
}

#define L128S	(128 / VSIZE)

#define DECLARE_VAR_128() \
	__shared__ VTYPE Z[L128S * BLK128]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L128S / 4 * BLK128), group_id = id / (L128S / 4 * BLK128); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i128 = (local_id & ~(L128S / 4 - 1)) * 4, i32 = local_id % (L128S / 4); \
	const sz_t k128 = group_id * L128S * BLK128 + i128 + i32; \
	\
	VTYPE * __restrict__ const zk = &z[k128]; \
	VTYPE * const Z128 = &Z[i128]; \
	VTYPE * const Zi32 = &Z128[i32]; \
	const sz_t i8 = ((4 * i32) & ~(4 * (L128S / 16) - 1)) + (i32 % (L128S / 16)); \
	VTYPE * const Zi8 = &Z128[i8]; \
	const sz_t i2 = ((4 * i32) & ~(4 * 2 - 1)) + (i32 % 2); \
	VTYPE * const Zi2 = &Z128[i2]; \
	VTYPE * const Z4 = &Z128[4 * i32];

extern "C" __global__
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__launch_bounds__(L128S / 4 * BLK128)
#endif
void square128(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));
	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));
}

#define L256S	(256 / VSIZE)

#define DECLARE_VAR_256() \
	__shared__ VTYPE Z[L256S * BLK256]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L256S / 4 * BLK256), group_id = id / (L256S / 4 * BLK256); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i256 = (local_id & ~(L256S / 4 - 1)) * 4, i64 = local_id % (L256S / 4); \
	const sz_t k256 = group_id * L256S * BLK256 + i256 + i64; \
	\
	VTYPE * __restrict__ const zk = &z[k256]; \
	VTYPE * const Z256 = &Z[i256]; \
	VTYPE * const Zi64 = &Z256[i64]; \
	const sz_t i16 = ((4 * i64) & ~(4 * (L256S / 16) - 1)) + (i64 % (L256S / 16)); \
	VTYPE * const Zi16 = &Z256[i16]; \
	const sz_t i4 = ((4 * i64) & ~(4 * (L256S / 64) - 1)) + (i64 % (L256S / 64)); \
	VTYPE * const Zi4 = &Z256[i4]; \
	VTYPE * const Z4 = &Z256[4 * i64];

extern "C" __global__
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__launch_bounds__(L256S / 4 * BLK256)
#endif
void square256(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));
	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));
	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));
}

#define L512S	(512 / VSIZE)

#define DECLARE_VAR_512() \
	__shared__ VTYPE Z[L512S * BLK512]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L512S / 4 * BLK512), group_id = id / (L512S / 4 * BLK512); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i512 = (local_id & ~(L512S / 4 - 1)) * 4, i128 = local_id % (L512S / 4); \
	const sz_t k512 = group_id * L512S * BLK512 + i512 + i128; \
	\
	VTYPE * __restrict__ const zk = &z[k512]; \
	VTYPE * const Z512 = &Z[i512]; \
	VTYPE * const Zi128 = &Z512[i128]; \
	const sz_t i32 = ((4 * i128) & ~(4 * (L512S / 16) - 1)) + (i128 % (L512S / 16)); \
	VTYPE * const Zi32 = &Z512[i32]; \
	const sz_t i8 = ((4 * i128) & ~(4 * (L512S / 64) - 1)) + (i128 % (L512S / 64)); \
	VTYPE * const Zi8 = &Z512[i8]; \
	const sz_t i2 = ((4 * i128) & ~(4 * 2 - 1)) + (i128 % 2); \
	VTYPE * const Zi2 = &Z512[i2]; \
	VTYPE * const Z4 = &Z512[4 * i128];

extern "C" __global__
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__launch_bounds__(L512S / 4 * BLK512)
#endif
void square512(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));
	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));
	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));
}

#define L1024S	(1024 / VSIZE)

// if BLK1024 != 1 then const sz_t i1024 = (local_id & ~(L1024S / 4 - 1)) * 4, i256 = local_id % (L1024S / 4);
// if BLK1024 = 1 then const sz_t i1024 = 0, i256 = local_id;
#define DECLARE_VAR_1024() \
	__shared__ VTYPE Z[L1024S * BLK1024]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L1024S / 4 * BLK1024), group_id = id / (L1024S / 4 * BLK1024); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i1024 = 0, i256 = local_id; \
	const sz_t k1024 = group_id * L1024S * BLK1024 + i1024 + i256; \
	\
	VTYPE * __restrict__ const zk = &z[k1024]; \
	VTYPE * const Z1024 = &Z[i1024]; \
	VTYPE * const Zi256 = &Z1024[i256]; \
	const sz_t i64 = ((4 * i256) & ~(4 * (L1024S / 16) - 1)) + (i256 % (L1024S / 16)); \
	VTYPE * const Zi64 = &Z1024[i64]; \
	const sz_t i16 = ((4 * i256) & ~(4 * (L1024S / 64) - 1)) + (i256 % (L1024S / 64)); \
	VTYPE * const Zi16 = &Z1024[i16]; \
	const sz_t i4 = ((4 * i256) & ~(4 * (L1024S / 256) - 1)) + (i256 % (L1024S / 256)); \
	VTYPE * const Zi4 = &Z1024[i4]; \
	VTYPE * const Z4 = &Z1024[4 * i256];

extern "C" __global__
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__launch_bounds__(L1024S / 4 * BLK1024)
#endif
void square1024(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));
	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));
	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));
	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));
}

#define L2048S	(2048 / VSIZE)

#define DECLARE_VAR_2048() \
	__shared__ VTYPE Z[L2048S]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L2048S / 4), group_id = id / (L2048S / 4); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i512 = local_id, k2048 = group_id * L2048S + i512; \
	\
	VTYPE * __restrict__ const zk = &z[k2048]; \
	VTYPE * const Zi512 = &Z[i512]; \
	const sz_t i128 = ((4 * i512) & ~(4 * (L2048S / 16) - 1)) + (i512 % (L2048S / 16)); \
	VTYPE * const Zi128 = &Z[i128]; \
	const sz_t i32 = ((4 * i512) & ~(4 * (L2048S / 64) - 1)) + (i512 % (L2048S / 64)); \
	VTYPE * const Zi32 = &Z[i32]; \
	const sz_t i8 = ((4 * i512) & ~(4 * (L2048S / 256) - 1)) + (i512 % (L2048S / 256)); \
	VTYPE * const Zi8 = &Z[i8]; \
	const sz_t i2 = ((4 * i512) & ~(4 * 2 - 1)) + (i512 % 2); \
	VTYPE * const Zi2 = &Z[i2]; \
	VTYPE * const Z4 = &Z[4 * i512];

extern "C" __global__
#if MAX_WG_SZ >= L2048S / 4
	__launch_bounds__(L2048S / 4)
#endif
void square2048(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	square_2x2(pq, Z4, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	square_8(pq, Z4, w, wi, sj, sji);
#endif
	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));
	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

#define L4096S	(4096 / VSIZE)

#define DECLARE_VAR_4096() \
	__shared__ VTYPE Z[L4096S]; \
	\
	DECLARE_VAR_REG(); \
	const sz_t local_id = id % (L4096S / 4), group_id = id / (L4096S / 4); \
	const sz_t j = id, sj = N_SZ / 4 / VSIZE + j; DECLARE_IVAR(N_SZ / 4 / VSIZE, j); \
	\
	const sz_t i1024 = local_id, k4096 = group_id * L4096S + i1024; \
	\
	VTYPE * __restrict__ const zk = &z[k4096]; \
	VTYPE * const Zi1024 = &Z[i1024]; \
	const sz_t i256 = ((4 * i1024) & ~(4 * (L4096S / 16) - 1)) + (i1024 % (L4096S / 16)); \
	VTYPE * const Zi256 = &Z[i256]; \
	const sz_t i64 = ((4 * i1024) & ~(4 * (L4096S / 64) - 1)) + (i1024 % (L4096S / 64)); \
	VTYPE * const Zi64 = &Z[i64]; \
	const sz_t i16 = ((4 * i1024) & ~(4 * (L4096S / 256) - 1)) + (i1024 % (L4096S / 256)); \
	VTYPE * const Zi16 = &Z[i16]; \
	const sz_t i4 = ((4 * i1024) & ~(4 * (L4096S / 1024) - 1)) + (i1024 % (L4096S / 1024)); \
	VTYPE * const Zi4 = &Z[i4]; \
	VTYPE * const Z4 = &Z[4 * i1024];

extern "C" __global__
#if MAX_WG_SZ >= L4096S / 4
	__launch_bounds__(L4096S / 4)
#endif
void square4096(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_4096();

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	square_4(pq, Z4, w, wi, sj, sji);
	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));
	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));
	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));
	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));
	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));
}

// -----------------

extern "C" __global__
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__launch_bounds__(L32S / 4 * BLK32)
#endif
void fwd32p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_32();

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(8, zk, Z4);
#else
	fwd8_write(pq, L32S / 4, zk, Z4, w, sj);
#endif
}

extern "C" __global__
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__launch_bounds__(L64S / 4 * BLK64)
#endif
void fwd64p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_64();

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	fwd4_write(pq, L64S / 4, zk, Z4, w, sj);
}

extern "C" __global__
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__launch_bounds__(L128S / 4 * BLK128)
#endif
void fwd128p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_128();

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(32, zk, Z4);
#else
	fwd8_write(pq, L128S / 4, zk, Z4, w, sj);
#endif
}

extern "C" __global__
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__launch_bounds__(L256S / 4 * BLK256)
#endif
void fwd256p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_256();

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	fwd4_write(pq, L256S / 4, zk, Z4, w, sj);
}

extern "C" __global__
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__launch_bounds__(L512S / 4 * BLK512)
#endif
void fwd512p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_512();

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(128, zk, Z4);
#else
	fwd8_write(pq, L512S / 4, zk, Z4, w, sj);
#endif
}

extern "C" __global__
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__launch_bounds__(L1024S / 4 * BLK1024)
#endif
void fwd1024p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_1024();

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	fwd4_write(pq, L1024S / 4, zk, Z4, w, sj);
}

extern "C" __global__
#if MAX_WG_SZ >= L2048S / 4
	__launch_bounds__(L2048S / 4)
#endif
void fwd2048p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_2048();

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	write_4(512, zk, Z4);
#else
	fwd8_write(pq, L2048S / 4, zk, Z4, w, sj);
#endif
}

extern "C" __global__
#if MAX_WG_SZ >= L4096S / 4
	__launch_bounds__(L4096S / 4)
#endif
void fwd4096p(VTYPE * __restrict__ const zg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_4096();

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	fwd4_write(pq, L4096S / 4, zk, Z4, w, sj);
}

// -----------------

extern "C" __global__
#if MAX_WG_SZ >= L32S / 4 * BLK32
	__launch_bounds__(L32S / 4 * BLK32)
#endif
void mul32(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_32();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k32];

	forward_4i(pq, L32S / 4, Zi8, L32S / 4, zk, w, sj / (L32S / 4));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 8, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L32S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4o(pq, L32S / 4, zk, L32S / 4, Zi8, wi, sji / (L32S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L64S / 4 * BLK64
	__launch_bounds__(L64S / 4 * BLK64)
#endif
void mul64(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_64();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k64];

	forward_4i(pq, L64S / 4, Zi16, L64S / 4, zk, w, sj / (L64S / 4));
	forward_4(pq, L64S / 16, Zi4, w, sj / (L64S / 16));
	mul_4(pq, Z4, L64S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L64S / 16, Zi4, wi, sji / (L64S / 16));
	backward_4o(pq, L64S / 4, zk, L64S / 4, Zi16, wi, sji / (L64S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L128S / 4 * BLK128
	__launch_bounds__(L128S / 4 * BLK128)
#endif
void mul128(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_128();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k128];

	forward_4i(pq, L128S / 4, Zi32, L128S / 4, zk, w, sj / (L128S / 4));
	forward_4(pq, L128S / 16, Zi8, w, sj / (L128S / 16));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 32, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L128S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L128S / 16, Zi8, wi, sji / (L128S / 16));
	backward_4o(pq, L128S / 4, zk, L128S / 4, Zi32, wi, sji / (L128S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L256S / 4 * BLK256
	__launch_bounds__(L256S / 4 * BLK256)
#endif
void mul256(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_256();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k256];

	forward_4i(pq, L256S / 4, Zi64, L256S / 4, zk, w, sj / (L256S / 4));
	forward_4(pq, L256S / 16, Zi16, w, sj / (L256S / 16));
	forward_4(pq, L256S / 64, Zi4, w, sj / (L256S / 64));
	mul_4(pq, Z4, L256S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L256S / 64, Zi4, wi, sji / (L256S / 64));
	backward_4(pq, L256S / 16, Zi16, wi, sji / (L256S / 16));
	backward_4o(pq, L256S / 4, zk, L256S / 4, Zi64, wi, sji / (L256S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L512S / 4 * BLK512
	__launch_bounds__(L512S / 4 * BLK512)
#endif
void mul512(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_512();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k512];

	forward_4i(pq, L512S / 4, Zi128, L512S / 4, zk, w, sj / (L512S / 4));
	forward_4(pq, L512S / 16, Zi32, w, sj / (L512S / 16));
	forward_4(pq, L512S / 64, Zi8, w, sj / (L512S / 64));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 128, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L512S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L512S / 64, Zi8, wi, sji / (L512S / 64));
	backward_4(pq, L512S / 16, Zi32, wi, sji / (L512S / 16));
	backward_4o(pq, L512S / 4, zk, L512S / 4, Zi128, wi, sji / (L512S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L1024S / 4 * BLK1024
	__launch_bounds__(L1024S / 4 * BLK1024)
#endif
void mul1024(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_1024();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k1024];

	forward_4i(pq, L1024S / 4, Zi256, L1024S / 4, zk, w, sj / (L1024S / 4));
	forward_4(pq, L1024S / 16, Zi64, w, sj / (L1024S / 16));
	forward_4(pq, L1024S / 64, Zi16, w, sj / (L1024S / 64));
	forward_4(pq, L1024S / 256, Zi4, w, sj / (L1024S / 256));
	mul_4(pq, Z4, L1024S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L1024S / 256, Zi4, wi, sji / (L1024S / 256));
	backward_4(pq, L1024S / 64, Zi16, wi, sji / (L1024S / 64));
	backward_4(pq, L1024S / 16, Zi64, wi, sji / (L1024S / 16));
	backward_4o(pq, L1024S / 4, zk, L1024S / 4, Zi256, wi, sji / (L1024S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L2048S / 4
	__launch_bounds__(L2048S / 4)
#endif
void mul2048(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_2048();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k2048];

	forward_4i(pq, L2048S / 4, Zi512, L2048S / 4, zk, w, sj / (L2048S / 4));
	forward_4(pq, L2048S / 16, Zi128, w, sj / (L2048S / 16));
	forward_4(pq, L2048S / 64, Zi32, w, sj / (L2048S / 64));
	forward_4(pq, L2048S / 256, Zi8, w, sj / (L2048S / 256));
#if VSIZE == 1
	forward_4(pq, 2, Zi2, w, sj / 2);
	mul_2x2(pq, Z4, 512, zpk, w[sj]);
	backward_4(pq, 2, Zi2, wi, sji / 2);
#else
	mul_8(pq, Z4, L2048S / 4, zpk, w, wi, sj, sji);
#endif
	backward_4(pq, L2048S / 256, Zi8, wi, sji / (L2048S / 256));
	backward_4(pq, L2048S / 64, Zi32, wi, sji / (L2048S / 64));
	backward_4(pq, L2048S / 16, Zi128, wi, sji / (L2048S / 16));
	backward_4o(pq, L2048S / 4, zk, L2048S / 4, Zi512, wi, sji / (L2048S / 4));
}

extern "C" __global__
#if MAX_WG_SZ >= L4096S / 4
	__launch_bounds__(L4096S / 4)
#endif
void mul4096(VTYPE * __restrict__ const zg, const VTYPE * __restrict__ const zpg, const uint_32 * __restrict__ const wg)
{
	DECLARE_VAR_4096();
	DECLARE_VARP_REG();
	const VTYPE * __restrict__ const zpk = &zp[k4096];

	forward_4i(pq, L4096S / 4, Zi1024, L4096S / 4, zk, w, sj / (L4096S / 4));
	forward_4(pq, L4096S / 16, Zi256, w, sj / (L4096S / 16));
	forward_4(pq, L4096S / 64, Zi64, w, sj / (L4096S / 64));
	forward_4(pq, L4096S / 256, Zi16, w, sj / (L4096S / 256));
	forward_4(pq, L4096S / 1024, Zi4, w, sj / (L4096S / 1024));
	mul_4(pq, Z4, L4096S / 4, zpk, w, wi, sj, sji);
	backward_4(pq, L4096S / 1024, Zi4, wi, sji / (L4096S / 1024));
	backward_4(pq, L4096S / 256, Zi16, wi, sji / (L4096S / 256));
	backward_4(pq, L4096S / 64, Zi64, wi, sji / (L4096S / 64));
	backward_4(pq, L4096S / 16, Zi256, wi, sji / (L4096S / 16));
	backward_4o(pq, L4096S / 4, zk, L4096S / 4, Zi1024, wi, sji / (L4096S / 4));
}

#endif	// SHORT_FUNC

// -----------------

INLINE uint_32 barrett(const uint_64 a, const uint_32 b, const uint_32 b_inv, const int b_s, uint_32 * a_p)
{
	// Using notations of Modular SIMD arithmetic in Mathemagix, Joris van der Hoeven, Grégoire Lecerf, Guillaume Quintin, 2014, HAL.
	// n = 31, alpha = 2^{n-2} = 2^29, s = r - 2, t = n + 1 = 32 => h = 1.
	// b < 2^31, alpha = 2^29 => a < 2^29 b
	// 2^{r-1} < b <= 2^r then a < 2^{r + 29} = 2^{s + 31} and (a >> s) < 2^31
	// b_inv = [2^{s + 32} / b]
	// b_inv < 2^{s + 32} / b < 2^{s + 32} / 2^{r-1} = 2^{s + 32} / 2^{s + 1} < 2^31
	// Let h be the number of iterations in Barrett's reduction, we have h = [a / b] - [[a / 2^s] b_inv / 2^32].
	// h = ([a/b] - a/b) + a/2^{s + 32} (2^{s + 32}/b - b_inv) + b_inv/2^32 (a/2^s - [a/2^s]) + ([a/2^s] b_inv / 2^32 - [[a/2^s] b_inv / 2^32])
	// Then -1 + 0 + 0 + 0 < h < 0 + 1/2 (2^{s + 32}/b - b_inv) + b_inv/2^32 + 1,
	// 0 <= h < 1 + 1/2 + 1/2 => h = 1.

	const uint_32 d = mul_hi((uint_32)(a >> b_s), b_inv), r = (uint_32)(a) - d * b;
	const bool o = (r >= b);
	*a_p = d + (o ? 1 : 0);
	return r - (o ? b : 0);
}

INLINE int_32 reduce64(int_64 * f, const uint_32 b, const uint_32 b_inv, const int b_s)
{
	// 1- t < 2^63 => t_h < 2^34. We must have t_h < 2^29 b => b > 32
	// 2- t < 2^23 b^2 => t_h < b^2 / 2^6. If 2 <= b < 32 then t_h < 32^2 / 2^6 = 16 < 2^29 b
	const uint_64 t = llabs(*f);
	const uint_64 t_h = t >> 29;
	const uint_32 t_l = (uint_32)(t) % (1u << 29);

	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;

	const bool s = (*f < 0);
	*f = s ? -(int_64)(d) : (int_64)(d);
	return s ? -(int_32)(r_l) : (int_32)(r_l);
}

INLINE int_32 reduce96(int96 * f, const uint_32 b, const uint_32 b_inv, const int b_s)
{
	const uint96 t = int96_abs(*f);
	const uint_64 t_h = (t.s1 << (32 - 29)) | (t.s0 >> 29);
	const uint_32 t_l = t.s0 % (1u << 29);

	uint_32 d_h, r_h = barrett(t_h, b, b_inv, b_s, &d_h);
	uint_32 d_l, r_l = barrett(((uint_64)(r_h) << 29) | t_l, b, b_inv, b_s, &d_l);
	const uint_64 d = ((uint_64)(d_h) << 29) | d_l;

	const bool s = int96_is_neg(*f);
	*f = int96_set_si(s ? -(int_64)(d) : (int_64)(d));
	return s ? -(int_32)(r_l) : (int_32)(r_l);
}

INLINE int_64 garner2(const uint_32 r1, const uint_32 r2)
{
	const uint_64 P1P2 = P1 * (uint_64)(P2);
	uint_32 u12 = mulmod(submod(r1, r2, P1), INVP2_P1, PQ1);	// P2 < P1
	const uint_64 n = r2 + u12 * (uint_64)(P2);
	const bool b = (n > P1P2 / 2);
	return (int_64)(n - (b ? P1P2 : 0));
}

INLINE int96 garner3(const uint_32 r1, const uint_32 r2, const uint_32 r3)
{
	const uint_32 u13 = mulmod(submod(r1, r3, P1), INVP3_P1, PQ1);
	const uint_32 u23 = mulmod(submod(r2, r3, P2), INVP3_P2, PQ2);
	const uint_32 u123 = mulmod(submod(u13, u23, P1), INVP2_P1, PQ1);
	const uint96 n = uint96_add_64(uint96_mul_64_32(P2 * (uint_64)(P3), u123), u23 * (uint_64)(P3) + r3);
	const bool b = uint96_is_greater(n, uint96_set(P1P2P3_2L, P1P2P3_2H));
	return uint96_i(b ? uint96_sub(n, uint96_set(P1P2P3L, P1P2P3H)) : n);
}

INLINE void write_rns(uint4_32 * __restrict__ const zi, const int4_32 r)
{
	uint4_32 zo1, zo2;
#if RNS_SZ == 3
	uint4_32 zo3;
#endif
	zo1.s0 = set_int(r.s0, P1); zo2.s0 = set_int(r.s0, P2);
#if RNS_SZ == 3
	zo3.s0 = set_int(r.s0, P3);
#endif
	zo1.s1 = set_int(r.s1, P1); zo2.s1 = set_int(r.s1, P2);
#if RNS_SZ == 3
	zo3.s1 = set_int(r.s1, P3);
#endif
	zo1.s2 = set_int(r.s2, P1); zo2.s2 = set_int(r.s2, P2);
#if RNS_SZ == 3
	zo3.s2 = set_int(r.s2, P3);
#endif
	zo1.s3 = set_int(r.s3, P1); zo2.s3 = set_int(r.s3, P2);
#if RNS_SZ == 3
	zo3.s3 = set_int(r.s3, P3);
#endif

	zi[0 * N_SZ / 4] = zo1; zi[1 * N_SZ / 4] = zo2;
#if RNS_SZ == 3
	zi[2 * N_SZ / 4] = zo3;
#endif
}

INLINE int4_32 normalize_1(uint4_32 * __restrict__ const zi, int_64 * __restrict__ const c,
	int_64 * const cl, const sz_t gid, const sz_t lid,
	const uint_32 b, const uint_32 b_inv, const int b_s, const bool dup)
{
	const uint4_32 u1 = mulmod4(zi[0 * N_SZ / 4], NORM1, PQ1), u2 = mulmod4(zi[1 * N_SZ / 4], NORM2, PQ2);
	int4_32 r;

#if RNS_SZ == 2

	int4_64 l = make_int4_64(garner2(u1.s0, u2.s0), garner2(u1.s1, u2.s1), garner2(u1.s2, u2.s2), garner2(u1.s3, u2.s3));
	if (dup) { l.s0 += l.s0; l.s1 += l.s1; l.s2 += l.s2; l.s3 += l.s3; }

	int_64 f = l.s0; r.s0 = reduce64(&f, b, b_inv, b_s);
	f += l.s1; r.s1 = reduce64(&f, b, b_inv, b_s);
	f += l.s2; r.s2 = reduce64(&f, b, b_inv, b_s);
	f += l.s3; r.s3 = reduce64(&f, b, b_inv, b_s);

#else

	const uint4_32 u3 = mulmod4(zi[2 * N_SZ / 4], NORM3, PQ3);

	int96 l0 = garner3(u1.s0, u2.s0, u3.s0), l1 = garner3(u1.s1, u2.s1, u3.s1);
	int96 l2 = garner3(u1.s2, u2.s2, u3.s2), l3 = garner3(u1.s3, u2.s3, u3.s3);

	if (dup) { l0 = int96_add(l0, l0); l1 = int96_add(l1, l1);  l2 = int96_add(l2, l2); l3 = int96_add(l3, l3); }

	int96 f96 = l0; r.s0 = reduce96(&f96, b, b_inv, b_s);
	f96 = int96_add(f96, l1); r.s1 = reduce96(&f96, b, b_inv, b_s);
	f96 = int96_add(f96, l2); r.s2 = reduce96(&f96, b, b_inv, b_s);
	f96 = int96_add(f96, l3); r.s3 = reduce96(&f96, b, b_inv, b_s);
	int_64 f = int96_get_si(f96);

#endif

	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	return r;
}

INLINE void normalize_2(uint4_32 * __restrict__ const zi, int_64 * const cl, const sz_t lid,
	const int4_32 r, const uint_32 b, const uint_32 b_inv, const int b_s)
{
	int_64 f = (lid == 0) ? 0 : cl[lid - 1];
	int4_32 ro;
	f += r.s0; ro.s0 = reduce64(&f, b, b_inv, b_s);
	f += r.s1; ro.s1 = reduce64(&f, b, b_inv, b_s);
	f += r.s2; ro.s2 = reduce64(&f, b, b_inv, b_s);
	f += r.s3; ro.s3 = (sz_t)(f);

	write_rns(zi, ro);
}

extern "C" __global__ void __launch_bounds__(NORM_WG_SZ) normalize1(uint4_32 * __restrict__ const z, int_64 * __restrict__ const c,
	const uint_32 b, const uint_32 b_inv, const int b_s, const int_32 dup)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x), lid = gid % NORM_WG_SZ;
	uint4_32 * __restrict__ const zi = &z[gid];
	__shared__ int_64 cl[NORM_WG_SZ];

	const int4_32 r = normalize_1(zi, c, cl, gid, lid, b, b_inv, b_s, dup != 0);

	__syncthreads();

	normalize_2(zi, cl, lid, r, b, b_inv, b_s);
}

extern "C" __global__
void normalize2(uint4_32 * __restrict__ const z, const int_64 * __restrict__ const c, 
	const uint_32 b, const uint_32 b_inv, const int b_s)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x);
	uint4_32 * __restrict__ const zi = &z[NORM_WG_SZ * gid];

	const uint4_32 u1 = zi[0 * N_SZ / 4];
	int4_32 r;

	int_64 f = c[gid] + get_int(u1.s0, P1);
	r.s0 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s1, P1);
	r.s1 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s2, P1);
	r.s2 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s3, P1);
	r.s3 = (int_32)(f);

	write_rns(zi, r);
}

extern "C" __global__ void __launch_bounds__(NORM_WG_SZ) mulscalar(uint4_32 * __restrict__ const z, int_64 * __restrict__ const c,
	const uint_32 b, const uint_32 b_inv, const int b_s, const int_32 a)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x), lid = gid % NORM_WG_SZ;
	uint4_32 * __restrict__ const zi = &z[gid];
	__shared__ int_64 cl[NORM_WG_SZ];

	const uint4_32 u1 = zi[0 * N_SZ / 4];
	int4_32 r;

	int_64 f = get_int(u1.s0, P1) * (int_64)(a);
	r.s0 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s1, P1) * (int_64)(a);
	r.s1 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s2, P1) * (int_64)(a);
	r.s2 = reduce64(&f, b, b_inv, b_s);
	f += get_int(u1.s3, P1) * (int_64)(a);
	r.s3 = reduce64(&f, b, b_inv, b_s);

	cl[lid] = f;

	if (lid == NORM_WG_SZ - 1)
	{
		const sz_t i = (gid / NORM_WG_SZ + 1) % (N_SZ / 4 / NORM_WG_SZ);
		c[i] = (i == 0) ? -f : f;
	}

	__syncthreads();

	normalize_2(zi, cl, lid, r, b, b_inv, b_s);
}

extern "C" __global__
void set(uint4_32 * __restrict__ const z, const uint_32 a)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x);
	z[gid] = (gid % (N_SZ / 4) == 0) ? make_uint4_32(a, 0, 0, 0) : make_uint4_32(0, 0, 0, 0);
}

extern "C" __global__
void copy(uint4_32 * __restrict__ const z, const sz_t dst, const sz_t src)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x);
	z[dst + gid] = z[src + gid];
}

extern "C" __global__
void copyp(uint4_32 * __restrict__ const zp, const uint4_32 * __restrict__ const z, const sz_t src)
{
	const sz_t gid = (sz_t)(blockIdx.x * blockDim.x + threadIdx.x);
	zp[gid] = z[src + gid];
}
