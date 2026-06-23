/*
Phase 1 unit test: validate the translated arithmetic core in cuda/kernel.cu
against an INDEPENDENT host reference (not the same algorithm).

Checks, per prime P1/P2/P3:
 - mulmod(a,b) == a*b*R^-1 mod p   (Montgomery product, R=2^32) via modinv reference
 - to/from Montgomery round-trip: mulmod(mulmod(a,RSQ), 1) == a   (validates RSQ)
 - addmod/submod == (a+/-b) mod p

Loads the REAL cuda/kernel.cu (default config block) + appended test kernels,
compiles with NVRTC via cu.h, runs on the GPU.

Build:
  g++ -std=c++17 -I src -I/usr/local/cuda/include dev/cu_core_test.cpp \
      -o build_dev/cu_core_test -lcuda -L/usr/local/cuda/lib64 -lnvrtc \
      -Wl,-rpath,/usr/local/cuda/lib64
*/

#include "hip.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>

static const char * TEST_KERNELS = R"CU(
extern "C" __global__ void k_mont(const uint_32 * a, const uint_32 * b, uint_32 * out, const int k, const uint_32 n)
{
    const uint_32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const uint2_32 pq = g_pq[k]; out[i] = mulmod(a[i], b[i], pq); }
}
extern "C" __global__ void k_roundtrip(const uint_32 * a, uint_32 * out, const int k, const uint_32 rsq, const uint_32 n)
{
    const uint_32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const uint2_32 pq = g_pq[k]; out[i] = mulmod(mulmod(a[i], rsq, pq), 1u, pq); }
}
extern "C" __global__ void k_addsub(const uint_32 * a, const uint_32 * b, uint_32 * oa, uint_32 * os, const uint_32 p, const uint_32 n)
{
    const uint_32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { oa[i] = addmod(a[i], b[i], p); os[i] = submod(a[i], b[i], p); }
}
)CU";

// ---- independent host reference ----
static uint64_t modpow(uint64_t b, uint64_t e, uint64_t m)
{
	uint64_t r = 1; b %= m;
	while (e) { if (e & 1) r = (__uint128_t)r * b % m; b = (__uint128_t)b * b % m; e >>= 1; }
	return r;
}
static uint64_t modinv(uint64_t a, uint64_t p) { return modpow(a % p, p - 2, p); }	// p prime

struct Prime { int k; uint32_t p, rsq; };

class core_test : public hipDevice
{
public:
	core_test(const hipPlatform & pf, size_t d) : hipDevice(pf, d, true) {}

	bool run()
	{
		std::ifstream f("cuda/kernel.cu");
		if (!f.is_open()) { std::cerr << "cannot open cuda/kernel.cu (run from repo root)\n"; return false; }
		std::stringstream ss; ss << f.rdbuf();
		std::string src = ss.str() + "\n" + TEST_KERNELS;
		loadProgram(src);

		// P1/P2/P3 and RSQ from the default config block in kernel.cu
		const std::vector<Prime> primes = {
			{ 0, 2130706433u, 402124772u },
			{ 1, 2113929217u, 2111798781u },
			{ 2, 2013265921u, 1172168163u },
		};

		const uint32_t N = 1u << 16;
		const size_t bytes = N * sizeof(uint32_t);
		std::vector<uint32_t> ha(N), hb(N), hout(N);

		hip_mem da = _createBuffer(HIP_MEM_READ_ONLY, bytes);
		hip_mem db = _createBuffer(HIP_MEM_READ_ONLY, bytes);
		hip_mem dout = _createBuffer(HIP_MEM_READ_WRITE, bytes);
		hip_mem dout2 = _createBuffer(HIP_MEM_READ_WRITE, bytes);

		hip_kernel kMont = _createKernel("k_mont");
		hip_kernel kRound = _createKernel("k_roundtrip");
		hip_kernel kAddsub = _createKernel("k_addsub");

		std::srand(12345);
		bool allOk = true;

		for (const Prime & pr : primes)
		{
			for (uint32_t i = 0; i < N; ++i)
			{
				ha[i] = uint32_t((uint64_t(std::rand()) * 2654435761u) % pr.p);
				hb[i] = uint32_t((uint64_t(std::rand()) * 40503u + std::rand()) % pr.p);
			}
			_writeBuffer(da, ha.data(), bytes);
			_writeBuffer(db, hb.data(), bytes);

			const uint64_t Rinv = modinv((1ull << 32) % pr.p, pr.p);

			// --- mulmod == a*b*R^-1 mod p ---
			uint32_t n = N;
			_setKernelArg(kMont, 0, sizeof(hip_mem), &da);
			_setKernelArg(kMont, 1, sizeof(hip_mem), &db);
			_setKernelArg(kMont, 2, sizeof(hip_mem), &dout);
			_setKernelArg(kMont, 3, sizeof(int), &pr.k);
			_setKernelArg(kMont, 4, sizeof(uint32_t), &n);
			_executeKernel(kMont, N, 0);
			_readBuffer(dout, hout.data(), bytes);

			size_t bad = 0; uint32_t firstBad = 0, gotBad = 0, expBad = 0;
			int shown = 0;
			for (uint32_t i = 0; i < N; ++i)
			{
				const uint32_t ref = uint32_t((__uint128_t)((uint64_t(ha[i]) * hb[i]) % pr.p) * Rinv % pr.p);
				if (hout[i] != ref) {
					if (!bad) { firstBad = i; gotBad = hout[i]; expBad = ref; }
					if (pr.k == 0 && shown < 6) { std::cout << "    dbg i=" << i << " a=" << ha[i] << " b=" << hb[i] << " got=" << hout[i] << " exp=" << ref << "\n"; ++shown; }
					++bad;
				}
			}
			std::cout << (bad ? "[FAIL]" : "[PASS]") << " mulmod P" << (pr.k + 1) << " (p=" << pr.p << ")";
			if (bad) std::cout << " : " << bad << " mismatches, first @" << firstBad << " got " << gotBad << " exp " << expBad;
			std::cout << "\n"; allOk &= (bad == 0);

			// --- Montgomery round-trip mulmod(mulmod(a,RSQ),1) == a ---
			_setKernelArg(kRound, 0, sizeof(hip_mem), &da);
			_setKernelArg(kRound, 1, sizeof(hip_mem), &dout2);
			_setKernelArg(kRound, 2, sizeof(int), &pr.k);
			_setKernelArg(kRound, 3, sizeof(uint32_t), &pr.rsq);
			_setKernelArg(kRound, 4, sizeof(uint32_t), &n);
			_executeKernel(kRound, N, 0);
			_readBuffer(dout2, hout.data(), bytes);
			size_t badR = 0;
			for (uint32_t i = 0; i < N; ++i) if (hout[i] != ha[i]) ++badR;
			std::cout << (badR ? "[FAIL]" : "[PASS]") << " Montgomery round-trip P" << (pr.k + 1) << (badR ? "" : " (RSQ validated)") << "\n";
			allOk &= (badR == 0);

			// --- addmod/submod ---
			_setKernelArg(kAddsub, 0, sizeof(hip_mem), &da);
			_setKernelArg(kAddsub, 1, sizeof(hip_mem), &db);
			_setKernelArg(kAddsub, 2, sizeof(hip_mem), &dout);
			_setKernelArg(kAddsub, 3, sizeof(hip_mem), &dout2);
			_setKernelArg(kAddsub, 4, sizeof(uint32_t), &pr.p);
			_setKernelArg(kAddsub, 5, sizeof(uint32_t), &n);
			_executeKernel(kAddsub, N, 0);
			std::vector<uint32_t> hadd(N), hsub(N);
			_readBuffer(dout, hadd.data(), bytes);
			_readBuffer(dout2, hsub.data(), bytes);
			size_t badA = 0, badS = 0;
			for (uint32_t i = 0; i < N; ++i)
			{
				if (hadd[i] != uint32_t((uint64_t(ha[i]) + hb[i]) % pr.p)) ++badA;
				if (hsub[i] != uint32_t((uint64_t(ha[i]) + pr.p - hb[i]) % pr.p)) ++badS;
			}
			std::cout << (badA ? "[FAIL]" : "[PASS]") << " addmod P" << (pr.k + 1) << "  "
					  << (badS ? "[FAIL]" : "[PASS]") << " submod P" << (pr.k + 1) << "\n";
			allOk &= (badA == 0 && badS == 0);
		}

		_releaseBuffer(da); _releaseBuffer(db); _releaseBuffer(dout); _releaseBuffer(dout2);
		clearProgram();
		return allOk;
	}
};

int main()
{
	try
	{
		hipPlatform pf;
		if (pf.getDeviceCount() == 0) { std::cerr << "no CUDA device\n"; return 1; }
		core_test t(pf, 0);
		const bool ok = t.run();
		std::cout << (ok ? "\nArithmetic core: ALL PASS\n" : "\nArithmetic core: FAILURE\n");
		return ok ? 0 : 1;
	}
	catch (const std::exception & e) { std::cerr << "error: " << e.what() << "\n"; return 2; }
}
