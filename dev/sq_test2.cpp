#include "transformGPU.h"
#include "gint.h"
#include <cstdio>
#include <cstdlib>
#if defined(USE_RNS3)
  #define RNST transformGPUs<3, false>
#else
  #define RNST transformGPUs<2, false>
#endif
int main(int argc, char ** argv) {
	const uint32_t b = 1000000; const uint32_t n = (argc > 1) ? uint32_t(atoi(argv[1])) : 20; const size_t num_regs = 7;
	RNST t(b, n, false, 0, num_regs, true);
	gint g(size_t(1) << n, b);
	t.set(b - 1);
	const size_t sz = size_t(1) << n;
	for (int k = 1; k <= 40; ++k) {
		t.squareDup(false); t.getInt(g);
		uint64_t h = 1469598103934665603ull; const int32_t * d = g.data();
		for (size_t i = 0; i < sz; ++i) { h ^= uint32_t(d[i]); h *= 1099511628211ull; }
		printf("sq%-3d hash=%016llx\n", k, (unsigned long long)h);
	}
	return 0;
}
