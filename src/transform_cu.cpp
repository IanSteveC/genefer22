/*
Copyright 2022, Yves Gallot

genefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.

----
CUDA backend factory. Counterpart of transform_ocl.cpp: same RNS_SIZE/is32 NTT-limit
thresholds, but instantiates the (shared) transformGPUs template against the CUDA
device backend (cu.h, selected by -DCUDA in transformGPU.h) and takes a CUDA device
ordinal instead of (cl_platform_id, cl_device_id).
*/

#include <stdexcept>

#include "transformGPU.h"

transform * transform::create_cuda(const uint32_t b, const uint32_t n, const bool isBoinc, const size_t device, const size_t num_regs,
								   const bool verbose)
{
	transform * pTransform = nullptr;
	if (b * static_cast<uint64_t>(b) < (P1S * static_cast<uint64_t>(P2S) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, false>(b, n, isBoinc, device, num_regs, verbose);
	}
	else if (b * static_cast<uint64_t>(b) < (P1U * static_cast<uint64_t>(P2U) / 2) / (size_t(1) << n))
	{
		pTransform = new transformGPUs<2, true>(b, n, isBoinc, device, num_regs, verbose);
	}
	else if (b <= 1000000000)
	{
		pTransform = new transformGPUs<3, false>(b, n, isBoinc, device, num_regs, verbose);
	}
	else
	{
		pTransform = new transformGPUs<3, true>(b, n, isBoinc, device, num_regs, verbose);
	}
	return pTransform;
}
