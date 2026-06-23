// Build-time generator: emit the specialized kernel source for one (RNS, is32, n) config,
// for offline fatbin compilation. The source depends only on (n, RNS, is32) + the device's
// MAX_WG_SZ (1024 on every CC>=2.0 GPU), NOT on b. Run with GENEFER_DUMP_SRC=<file> and
// GENEFER_DUMP_EXIT=1 so the engine ctor writes the source and stops before NVRTC/alloc.
//   usage: dump_src <rns:2|3> <is32:0|1> <n> <device>
#include "transformGPU.h"
#include <cstdlib>
#include <cstdio>
int main(int argc, char ** argv)
{
	if (argc < 5) { std::fprintf(stderr, "usage: %s <rns:2|3> <is32:0|1> <n> <device>\n", argv[0]); return 2; }
	const int rns = std::atoi(argv[1]), is32 = std::atoi(argv[2]);
	const uint32_t n = uint32_t(std::atoi(argv[3])); const size_t dev = size_t(std::atoi(argv[4]));
	const uint32_t b = 1000000; const size_t num_regs = 7;   // b is irrelevant to the emitted source
	if      (rns == 2 && !is32) { transformGPUs<2, false> t(b, n, false, dev, num_regs, false); }
	else if (rns == 2 &&  is32) { transformGPUs<2, true>  t(b, n, false, dev, num_regs, false); }
	else if (rns == 3 && !is32) { transformGPUs<3, false> t(b, n, false, dev, num_regs, false); }
	else if (rns == 3 &&  is32) { transformGPUs<3, true>  t(b, n, false, dev, num_regs, false); }
	else { std::fprintf(stderr, "bad rns/is32\n"); return 2; }
	std::fprintf(stderr, "warning: ctor returned without dump-exit (set GENEFER_DUMP_SRC + GENEFER_DUMP_EXIT)\n");
	return 0;
}
