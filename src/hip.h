#include "hip/hip_runtime.h"
/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

----
CUDA backend (Driver API + NVRTC). This header is the CUDA counterpart of ocl.h:
it mirrors the protected interface used by the GPU orchestration layer
(_createBuffer / _releaseBuffer / _readBuffer / _writeBuffer / _createKernel /
_releaseKernel / _setKernelArg / _executeKernel / loadProgram / clearProgram /
getMaxWorkGroupSize / getLocalMemSize / setProfiling ...) so the orchestration
can drive either backend.

Design notes (see CUDA port plan):
 - Kernels are compiled at runtime with NVRTC, mirroring clCreateProgramWithSource
   + clBuildProgram. The per-FFT-size `#define` block is prepended to the source
   string exactly as in the OpenCL path.
 - OpenCL binds kernel args once and updates them individually; CUDA's
   hipModuleLaunchKernel needs the full argument array every launch. We therefore cache
   args per kernel (hip_kernel holds the arg blobs) and assemble the void* array
   at launch time. Buffer args are hipDeviceptr_t (8 bytes), not cl_mem.
 - OpenCL global_work_size is TOTAL threads; CUDA needs grid + block separately.
   _executeKernel converts: block = localWorkSize (or 256 when 0), grid =
   global / block, asserting exact divisibility (OpenCL required it too).
*/

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "pio.h"

// Transform-decomposition search space for the autotuner (TUNE). Vendor-neutral; mirrors
// the identical class in ocl.h (only the localMemSize param type differs: uint64_t here).
class splitter
{
private:
	struct partition
	{
		size_t size;
		uint32_t p[64];
	};

	const bool b256, b1024;
	const size_t mMax;
	std::vector<partition> part;

private:
	void split(const size_t m, const size_t i, partition & p)
	{
		if (b1024 && (m >= 10 + 5)) { p.p[i] = 10; split(m - 10, i + 1, p); }
		if (b256 && (m >= 8 + 5)) { p.p[i] = 8; split(m - 8, i + 1, p); }
		if (m >= 6 + 5) { p.p[i] = 6; split(m - 6, i + 1, p); }

		if ((5 <= m) && (m <= mMax) && (i > 0))
		{
			partition pt;
			for (size_t k = 0; k < i; ++k) pt.p[k] = p.p[k];
			pt.p[i] = static_cast<uint32_t>(m);
			pt.size = i + 1;
			part.push_back(pt);
		}
	}

	static size_t log_2(const size_t n) { size_t r = 0; for (size_t m = 1; m < n; m *= 2) ++r; return r; }

public:
	splitter(const size_t n, const size_t chunk256, const size_t chunk1024, const size_t sizeofType, const size_t sizeofVec,
		const size_t mSquareMax, const uint64_t localMemSize, const size_t maxWorkGroupSize) :
		b256((maxWorkGroupSize >= (256 / 4) * chunk256) && (localMemSize >= 256 * chunk256 * sizeofVec * sizeofType)),
		b1024((maxWorkGroupSize >= (1024 / 4) * chunk1024) && (localMemSize >= 1024 * chunk1024 * sizeofVec * sizeofType)),
		mMax(std::min(mSquareMax, std::min(log_2(size_t(localMemSize / sizeofType)), log_2(maxWorkGroupSize * 4 * sizeofVec))))
	{
		partition p;
		split(n, 0, p);
	}

	size_t getSize() const { return part.size(); }
	size_t getPartSize(const size_t i) const { return part[i].size; }
	uint32_t getPart(const size_t i, const size_t j) const { return part[i].p[j]; }
};

// #define hip_debug		1

// ---------------------------------------------------------------------------
// Handle types (CUDA counterparts of cl_mem / cl_kernel).
// hip_mem is a device pointer. hip_kernel is a heap object holding the hipFunction_t
// plus the cached argument blobs (so _setKernelArg can stay value-semantically
// identical to the OpenCL version while feeding hipModuleLaunchKernel's arg array).
// ---------------------------------------------------------------------------

enum hip_mem_flags { HIP_MEM_READ_WRITE = 0, HIP_MEM_READ_ONLY = 1 };	// flags are advisory only on CUDA

typedef hipDeviceptr_t hip_mem;

struct hip_kernel_t
{
	static const size_t max_args = 16;
	static const size_t arg_blob = 16;	// bytes per arg slot (fits hipDeviceptr_t and any scalar used)

	hipFunction_t function = nullptr;
	std::string name;
	std::array<std::array<uint8_t, arg_blob>, max_args> argData{};
	std::array<size_t, max_args> argSize{};	// 0 => slot unset
	size_t argCount = 0;					// highest set index + 1
};

typedef hip_kernel_t * hip_kernel;

// ---------------------------------------------------------------------------

class hipObject
{
protected:
	static void hipFatal(const hipError_t res, const char * const ext = nullptr)
	{
		if (res != hipSuccess)
		{
			const char * name = nullptr; hipDrvGetErrorName(res, &name);
			const char * str = nullptr; hipDrvGetErrorString(res, &str);
			std::ostringstream ss; ss << "cuda error: " << (name ? name : "?");
			if (str != nullptr) ss << " - " << str;
			if (ext != nullptr) ss << " (" << ext << ")";
			throw std::runtime_error(ss.str());
		}
	}

	static void hiprtcFatal(const hiprtcResult res, const char * const ext = nullptr)
	{
		if (res != HIPRTC_SUCCESS)
		{
			std::ostringstream ss; ss << "nvrtc error: " << hiprtcGetErrorString(res);
			if (ext != nullptr) ss << " (" << ext << ")";
			throw std::runtime_error(ss.str());
		}
	}
};

// platform: enumerate CUDA devices. Counterpart of ocl.h::platform.
// Note: there is no platform layer in CUDA; a "device" is just an ordinal.
class hipPlatform : hipObject
{
private:
	struct deviceDesc
	{
		hipDevice_t device;
		std::string name;
	};
	std::vector<deviceDesc> _devices;

public:
	hipPlatform()
	{
		hipFatal(hipInit(0));
		int count = 0; hipFatal(hipGetDeviceCount(&count));
		for (int i = 0; i < count; ++i)
		{
			hipDevice_t dev; hipFatal(hipDeviceGet(&dev, i));
			char name[256]; hipFatal(hipDeviceGetName(name, sizeof(name), dev));
			hipDeviceProp_t prop; hipFatal(hipGetDeviceProperties(&prop, dev));
			std::ostringstream ss; ss << "device '" << name << "', vendor 'AMD', " << prop.gcnArchName;
			_devices.push_back({ dev, ss.str() });
		}
	}

	virtual ~hipPlatform() {}

	size_t getDeviceCount() const { return _devices.size(); }

	size_t displayDevices() const
	{
		const size_t n = _devices.size();
		std::ostringstream ss;
		for (size_t i = 0; i < n; ++i) ss << i << " - " << _devices[i].name << "." << std::endl;
		ss << std::endl;
		pio::print(ss.str());
		return n;
	}

	hipDevice_t getDevice(const size_t d) const { return _devices[d].device; }
};

// device: owns the context, stream, NVRTC-compiled module, and drives kernels.
// Counterpart of ocl.h::device.
class hipDevice : hipObject
{
private:
	const hipDevice_t _device;
#if defined(hip_debug)
	const size_t _d;
#endif
	bool _profile = false;
	int _ccMajor = 0, _ccMinor = 0;
	std::string _gcnArch;	// e.g. "gfx1030" / "gfx906" — the hipRTC compile target
	size_t _maxWorkGroupSize = 0;	// hipDeviceAttributeMaxThreadsPerBlock
	size_t _localMemSize = 0;		// hipDeviceAttributeMaxSharedMemoryPerBlock
	hipCtx_t _context = nullptr;
	hipStream_t _stream = nullptr;
	hipModule_t _module = nullptr;

	struct profile
	{
		std::string name;
		size_t count;
		double time;	// milliseconds (hipEventElapsedTime)

		profile() {}
		profile(const std::string & name) : name(name), count(0), time(0) {}
	};
	std::map<hipFunction_t, profile> _profileMap;
	std::vector<hip_kernel_t *> _ownedKernels;	// for cleanup

public:
	hipDevice(const hipPlatform & parent, const size_t d, const bool verbose) : _device(parent.getDevice(d))
#if defined(hip_debug)
		, _d(d)
#endif
	{
		char deviceName[256]; hipFatal(hipDeviceGetName(deviceName, sizeof(deviceName), _device));
		hipFatal(hipDeviceGetAttribute(&_ccMajor, hipDeviceAttributeComputeCapabilityMajor, _device));
		hipFatal(hipDeviceGetAttribute(&_ccMinor, hipDeviceAttributeComputeCapabilityMinor, _device));
		{ hipDeviceProp_t prop; hipFatal(hipGetDeviceProperties(&prop, _device)); _gcnArch = prop.gcnArchName;
		  const size_t c = _gcnArch.find(':'); if (c != std::string::npos) _gcnArch = _gcnArch.substr(0, c); }	// strip :xnack-/:sramecc+

		int mwgs = 0; hipFatal(hipDeviceGetAttribute(&mwgs, hipDeviceAttributeMaxThreadsPerBlock, _device));
		_maxWorkGroupSize = size_t(mwgs);
		int lms = 0; hipFatal(hipDeviceGetAttribute(&lms, hipDeviceAttributeMaxSharedMemoryPerBlock, _device));
		_localMemSize = size_t(lms);

		int driverVersion = 0; hipDriverGetVersion(&driverVersion);

		if (verbose)
		{
			int computeUnits = 0; hipDeviceGetAttribute(&computeUnits, hipDeviceAttributeMultiprocessorCount, _device);
			int clockRate = 0; hipDeviceGetAttribute(&clockRate, hipDeviceAttributeClockRate, _device);	// kHz
			size_t totalMem = 0; hipDeviceTotalMem(&totalMem, _device);
			std::ostringstream ssd;
			ssd << "Running on device '" << deviceName << "', vendor 'AMD', " << _gcnArch
				<< ", driver " << driverVersion
				<< ", " << computeUnits << " CUs @ " << clockRate / 1000 << "MHz, mem=" << (totalMem >> 20) << "MB"
				<< ", maxThreadsPerBlock=" << _maxWorkGroupSize << ", sharedMem/block=" << (_localMemSize >> 10) << "kB.";
			pio::print(ssd.str());
		}

		hipFatal(hipDevicePrimaryCtxRetain(&_context, _device));
		hipFatal(hipCtxSetCurrent(_context));
		// Non-blocking stream: ALL work (kernels AND host<->device copies via the *Async
		// variants below) runs on this one stream, so it stays in-order like OpenCL's queue,
		// AND it can be stream-captured into a CUDA graph (capture forbids a blocking stream
		// that implicitly syncs with the legacy default stream). Copies must use the Async
		// forms on _stream — a plain hipMemcpyHtoD/DtoH would run on stream 0 and race us.
		hipFatal(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
	}

	virtual ~hipDevice()
	{
		if (_module != nullptr) hipModuleUnload(_module);
		for (hip_kernel_t * k : _ownedKernels) delete k;
		if (_stream != nullptr) hipStreamDestroy(_stream);
		hipDevicePrimaryCtxRelease(_device);
	}

public:
	size_t getMaxWorkGroupSize() const { return _maxWorkGroupSize; }
	size_t getLocalMemSize() const { return _localMemSize; }
	size_t getTimerResolution() const { return 1; }	// hipEventElapsedTime resolves ~0.5us; value unused on CUDA
	bool isIntel() const { return false; }

public:
	void resetProfiles()
	{
		for (auto it : _profileMap)
		{
			profile & prof = _profileMap[it.first];
			prof.count = 0;
			prof.time = 0;
		}
	}

	double getProfileTime() const
	{
		double time = 0;
		for (auto it : _profileMap) time += it.second.time;
		return time;
	}

	void displayProfiles(const size_t count) const
	{
		double ptime = 0;
		for (auto it : _profileMap) ptime += it.second.time;
		ptime /= double(count);

		std::ostringstream ss;
		for (auto it : _profileMap)
		{
			const profile & prof = it.second;
			if (prof.count != 0)
			{
				const size_t ncount = prof.count / count;
				const double ntime = prof.time / double(count);
				ss << "- " << prof.name << ": " << ncount << ", " << std::setprecision(3)
					<< ntime * 100.0 / ptime << " %, " << ntime << " ms (" << (ntime / double(ncount)) << ")" << std::endl;
			}
		}
		pio::display(ss.str());
	}

	void setProfiling(const bool enable)
	{
		_profile = enable;
		resetProfiles();
	}

public:
	// Mirrors ocl.h::readOpenCL: if the .cu source file exists, regenerate the
	// embedded C++ header string and also stream the source into `src`.
	bool readOpenCL(const char * const clFileName, const char * const headerFileName, const char * const varName, std::ostringstream & src) const
	{
		std::ifstream clFile(clFileName);
		if (!clFile.is_open()) return false;

		std::ofstream hFile(headerFileName, std::ios::binary);	// binary: don't convert line endings to CRLF
		if (!hFile.is_open()) throw std::runtime_error("cannot write CUDA kernel header file");

		hFile << "/*" << std::endl;
		hFile << "Copyright 2022, Yves Gallot" << std::endl << std::endl;
		hFile << "genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it." << std::endl;
		hFile << "Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful." << std::endl;
		hFile << "*/" << std::endl << std::endl;
		hFile << "#pragma once" << std::endl << std::endl;
		hFile << "#include <cstdint>" << std::endl << std::endl;
		hFile << "static const char * const " << varName << " = \\" << std::endl;

		std::string line;
		while (std::getline(clFile, line))
		{
			hFile << "\"";
			for (char c : line)
			{
				if ((c == '\\') || (c == '\"')) hFile << '\\';
				hFile << c;
			}
			hFile << "\\n\" \\" << std::endl;
			src << line << std::endl;
		}
		hFile << "\"\";" << std::endl;

		hFile.close();
		clFile.close();
		return true;
	}

public:
	void loadProgram(const std::string & programSrc)
	{
		hiprtcProgram prog;
		hiprtcFatal(hiprtcCreateProgram(&prog, programSrc.c_str(), "genefer.cu", 0, nullptr, nullptr));

		std::ostringstream archOpt; archOpt << "--gpu-architecture=" << _gcnArch;
		const std::string arch = archOpt.str();
		std::vector<const char *> options;
		options.push_back(arch.c_str());
		options.push_back("--std=c++14");
		// The kernel type-puns uint_32* twiddle arrays as uint2_32*/uint4_32* (DECLARE_W*); keep
		// strict aliasing off (as OpenCL effectively does) so those vector loads stay well-defined.
		options.push_back("-fno-strict-aliasing");
		options.push_back("-fwrapv");	// defined signed-overflow (clang exploits the UB; NVRTC does not)
		// NOTE on correctness: ROCm clang's loop unroller (-O2+) miscompiles the square/mul NTT
		// helpers when the vector operands are __align__ STRUCTS (uint2_32/uint4_32) -> the residue
		// diverges after a few squarings. cuda/kernel.cu therefore defines those types as clang
		// ext_vector_type natives under __HIP__ (the same uint2/uint4 OpenCL uses), which compile
		// correctly AND ~2x faster. So no -fno-unroll-loops is needed here; full -O3 is bit-exact.
		// Diagnostic escape hatches (unset by default): force -O0, or inject an extra flag
		// (e.g. GENEFER_HIPRTC_OPT=-O1) to re-bisect if a future toolchain regresses.
		if (std::getenv("GENEFER_HIPRTC_O0") != nullptr) options.push_back("-O0");
		static std::string optFlag;
		if (const char * o = std::getenv("GENEFER_HIPRTC_OPT")) { optFlag = o; options.push_back(optFlag.c_str()); }
#if defined(hip_debug)
		options.push_back("--generate-line-info");
#endif
		const hiprtcResult cres = hiprtcCompileProgram(prog, int(options.size()), options.data());

		size_t logSize = 0; hiprtcGetProgramLogSize(prog, &logSize);
		if (logSize > 1)
		{
			std::vector<char> log(logSize);
			hiprtcGetProgramLog(prog, log.data());
#if defined(hip_debug)
			std::ofstream fileOut("pgm.log"); fileOut << log.data() << std::endl; fileOut.close();
#else
			if (cres != HIPRTC_SUCCESS) { std::ostringstream ss; ss << log.data() << std::endl; pio::print(ss.str()); }
#endif
		}
		hiprtcFatal(cres, "compile");

		size_t ptxSize = 0; hiprtcFatal(hiprtcGetCodeSize(prog, &ptxSize));
		std::vector<char> ptx(ptxSize);
		hiprtcFatal(hiprtcGetCode(prog, ptx.data()));
		hiprtcFatal(hiprtcDestroyProgram(&prog));

#if defined(hip_debug)
		std::ofstream fileOut("pgm.ptx", std::ios::binary); fileOut.write(ptx.data(), std::streamsize(ptxSize)); fileOut.close();
#endif
		hipFatal(hipModuleLoadDataEx(&_module, ptx.data(), 0, nullptr, nullptr));
	}

	void clearProgram()
	{
		if (_module != nullptr) { hipFatal(hipModuleUnload(_module)); _module = nullptr; }
		for (hip_kernel_t * k : _ownedKernels) delete k;
		_ownedKernels.clear();
		_profileMap.clear();
	}

private:
	void _sync() { hipFatal(hipStreamSynchronize(_stream)); }

protected:
	hip_mem _createBuffer(const unsigned /*flags*/, const size_t size, const bool clear = true) const
	{
		hipDeviceptr_t mem = 0;
		hipFatal(hipMalloc(&mem, size));
		if (clear) hipFatal(hipMemsetD8(mem, 0, size));
		return mem;
	}

	static void _releaseBuffer(hip_mem & mem)
	{
		if (mem != 0) { hipFatal(hipFree(mem)); mem = 0; }
	}

	void _readBuffer(hip_mem & mem, void * const ptr, const size_t size, const size_t offset = 0)
	{
		// Prefill with random bytes so a silent read failure is caught (mirrors ocl.h).
		char * const cptr = static_cast<char *>(ptr);
		for (size_t i = 0; i < size; ++i) cptr[i] = static_cast<char>(std::rand());
		_sync();	// wait for queued kernels/graph launches to finish writing `mem`
		hipFatal(hipMemcpyDtoHAsync(ptr, static_cast<char *>(mem) + offset, size, _stream));
		_sync();	// wait for the copy (host buffer is consumed by the caller)
	}

	void _writeBuffer(hip_mem & mem, const void * const ptr, const size_t size, const size_t offset = 0)
	{
		_sync();	// order after prior stream work
		hipFatal(hipMemcpyHtoDAsync(static_cast<char *>(mem) + offset, ptr, size, _stream));
		_sync();	// wait for the copy (host source may be freed by the caller)
	}

protected:
	hip_kernel _createKernel(const char * const kernelName)
	{
		hip_kernel_t * kernel = new hip_kernel_t();
		kernel->name = kernelName;
		hipFatal(hipModuleGetFunction(&kernel->function, _module, kernelName), kernelName);
		_ownedKernels.push_back(kernel);
		_profileMap[kernel->function] = profile(kernelName);
		return kernel;
	}

	static void _releaseKernel(hip_kernel & kernel)
	{
		// hipFunction_t lifetime is tied to the module; just forget the handle here.
		// The hip_kernel_t object itself is freed in clearProgram()/destructor.
		kernel = nullptr;
	}

	static void _setKernelArg(hip_kernel kernel, const unsigned arg_index, const size_t arg_size, const void * const arg_value)
	{
		std::memcpy(kernel->argData[arg_index].data(), arg_value, arg_size);
		kernel->argSize[arg_index] = arg_size;
		if (arg_index + 1 > kernel->argCount) kernel->argCount = arg_index + 1;
	}

protected:
	void _executeKernel(hip_kernel kernel, const size_t globalWorkSize, const size_t localWorkSize = 0)
	{
		size_t block = localWorkSize;
		if (block == 0)
		{
			// OpenCL used a driver-chosen local size (set/copy/normalize2/...). These kernels
			// have no reqd_work_group_size and no shared mem / barriers, but they ALSO have no
			// in-kernel bounds guard, so grid*block must equal global exactly. Pick the largest
			// power-of-two <= 256 that divides globalWorkSize.
			block = 1;
			for (size_t cand = 256; cand >= 1; cand >>= 1) { if (globalWorkSize % cand == 0) { block = cand; break; } }
		}
		if (globalWorkSize % block != 0)
		{
			std::ostringstream ss; ss << "global " << globalWorkSize << " not a multiple of block " << block << " for " << kernel->name;
			throw std::runtime_error(ss.str());
		}
		const size_t grid = globalWorkSize / block;

		void * args[hip_kernel_t::max_args];
		for (size_t i = 0; i < kernel->argCount; ++i) args[i] = kernel->argData[i].data();

		if (!_profile)
		{
			hipFatal(hipModuleLaunchKernel(kernel->function, unsigned(grid), 1, 1, unsigned(block), 1, 1, 0, _stream, args, nullptr));
		}
		else
		{
			hipEvent_t start, stop;
			hipFatal(hipEventCreateWithFlags(&start, hipEventDefault));
			hipFatal(hipEventCreateWithFlags(&stop, hipEventDefault));
			hipFatal(hipEventRecord(start, _stream));
			hipFatal(hipModuleLaunchKernel(kernel->function, unsigned(grid), 1, 1, unsigned(block), 1, 1, 0, _stream, args, nullptr));
			hipFatal(hipEventRecord(stop, _stream));
			hipFatal(hipEventSynchronize(stop));
			float ms = 0; hipEventElapsedTime(&ms, start, stop);
			hipEventDestroy(start); hipEventDestroy(stop);

			profile & prof = _profileMap[kernel->function];
			prof.count++;
			prof.time += double(ms);
		}
	}

protected:
	// --- CUDA Graphs ---
	// Capture a fixed sequence of kernel launches on _stream into a replayable graph.
	// genefer's per-squaring kernel sequence is identical every iteration (only the
	// normalize 'dup' arg changes, handled by capturing one graph per dup value), so
	// replaying a graph removes the per-launch CPU/driver overhead of millions of squarings.
	// Requires the capture region to contain only stream work (no host<->device sync); the
	// squaring loop is pure kernel launches, so this holds. _profile must be false during capture.
	void beginCapture() { hipFatal(hipStreamBeginCapture(_stream, hipStreamCaptureModeThreadLocal)); }

	hipGraphExec_t endCapture()
	{
		hipGraph_t graph; hipFatal(hipStreamEndCapture(_stream, &graph));
		hipGraphExec_t exec; hipFatal(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));	// HIP uses the 5-arg form
		hipGraphDestroy(graph);
		return exec;
	}

	void launchGraph(hipGraphExec_t exec) { hipFatal(hipGraphLaunch(exec, _stream)); }

	static void destroyGraph(hipGraphExec_t & exec) { if (exec != nullptr) { hipGraphExecDestroy(exec); exec = nullptr; } }
};
