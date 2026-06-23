/*
Phase 0 proof-of-concept for the genefer HIP backend (src/cu.h).

Exercises, in isolation, every mechanism that differs from OpenCL:
 - NVRTC runtime compile with a prepended `#define` (the per-FFT-size config model)
 - cuMemAlloc / HtoD / DtoH buffer round-trip
 - the per-launch argument array (buffer args as CUdeviceptr + scalar args)
 - global-work-size -> grid/block conversion (localWorkSize==0 -> 256, and explicit)
 - RELAUNCH with a mutated scalar arg only (the ek_fb pattern: update one arg, relaunch)

Build (no nvcc needed; NVRTC compiles the device code at runtime):
  g++ -std=c++17 -I src -I/usr/local/hip/include dev/cu_poc.cpp \
      -o build_dev/cu_poc -lhip -L/usr/local/hip/lib64 -lnvrtc \
      -Wl,-rpath,/usr/local/hip/lib64
*/

#include "hip.h"

#include <iostream>
#include <vector>

// Minimal engine exposing the protected device interface, like an `engines`-derived class.
class poc : public hipDevice
{
public:
	poc(const hipPlatform & parent, const size_t d) : hipDevice(parent, d, true) {}

	bool run()
	{
		// Kernel: out[i] = in[i]*in[i]*scale + ADDVAL. ADDVAL is injected at compile time.
		std::ostringstream src;
		src << "#define ADDVAL 7u\n"
			<< "extern \"C\" __global__ void vsq(const unsigned * in, unsigned * out, unsigned n, unsigned scale)\n"
			<< "{\n"
			<< "    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
			<< "    if (i < n) out[i] = in[i] * in[i] * scale + ADDVAL;\n"
			<< "}\n";
		loadProgram(src.str());

		const unsigned N = 4096;
		const size_t bytes = N * sizeof(unsigned);
		std::vector<unsigned> h(N), r(N);
		for (unsigned i = 0; i < N; ++i) h[i] = i;

		hip_mem din = _createBuffer(HIP_MEM_READ_ONLY, bytes);
		hip_mem dout = _createBuffer(HIP_MEM_READ_WRITE, bytes);
		_writeBuffer(din, h.data(), bytes);

		hip_kernel k = _createKernel("vsq");
		unsigned n = N, scale = 3;
		_setKernelArg(k, 0, sizeof(hip_mem), &din);
		_setKernelArg(k, 1, sizeof(hip_mem), &dout);
		_setKernelArg(k, 2, sizeof(unsigned), &n);
		_setKernelArg(k, 3, sizeof(unsigned), &scale);

		// launch 1: localWorkSize==0 -> block 256 chosen by _executeKernel
		_executeKernel(k, N, 0);
		_readBuffer(dout, r.data(), bytes);
		bool ok1 = true;
		for (unsigned i = 0; i < N; ++i) { const unsigned e = i * i * 3u + 7u; if (r[i] != e) { ok1 = false; std::cerr << "launch1 mismatch @" << i << " got " << r[i] << " exp " << e << "\n"; break; } }
		std::cout << (ok1 ? "[PASS]" : "[FAIL]") << " launch1 (block=256 auto, args: 2 buffers + 2 scalars)\n";

		// launch 2: mutate ONLY arg 3 (scale) and relaunch with an explicit block — the ek_fb pattern
		scale = 5;
		_setKernelArg(k, 3, sizeof(unsigned), &scale);
		_executeKernel(k, N, 128);
		_readBuffer(dout, r.data(), bytes);
		bool ok2 = true;
		for (unsigned i = 0; i < N; ++i) { const unsigned e = i * i * 5u + 7u; if (r[i] != e) { ok2 = false; std::cerr << "launch2 mismatch @" << i << " got " << r[i] << " exp " << e << "\n"; break; } }
		std::cout << (ok2 ? "[PASS]" : "[FAIL]") << " launch2 (block=128, relaunch after mutating only arg 3)\n";

		_releaseKernel(k);
		_releaseBuffer(din);
		_releaseBuffer(dout);
		clearProgram();
		return ok1 && ok2;
	}
};

int main()
{
	try
	{
		hipPlatform pf;
		const size_t n = pf.getDeviceCount();
		std::cout << n << " HIP device(s):\n";
		pf.displayDevices();
		if (n == 0) { std::cerr << "no HIP device\n"; return 1; }

		poc t(pf, 0);
		const bool ok = t.run();
		std::cout << (ok ? "\nPhase 0 PoC: ALL PASS\n" : "\nPhase 0 PoC: FAILURE\n");
		return ok ? 0 : 1;
	}
	catch (const std::exception & e)
	{
		std::cerr << "error: " << e.what() << "\n";
		return 2;
	}
}
