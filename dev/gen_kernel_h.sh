#!/usr/bin/env bash
# Regenerate src/cuda/kernel.h (a C string-literal embed) from cuda/kernel.cu so the
# runtime-compiled NTT kernel source always matches the edited .cu. Called by the build scripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CU="$ROOT/cuda/kernel.cu"; H="$ROOT/src/cuda/kernel.h"
{
  printf '/*\nCopyright 2022, Yves Gallot\n\ngenefer is free source code, under the MIT license (see LICENSE). You can redistribute, use and/or modify it.\nPlease give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.\n*/\n\n#pragma once\n\n#include <cstdint>\n\n'
  printf 'static const char * const src_cuda_kernel = \\\n'
  # escape backslash, then double-quote; wrap each line as "..\n" \
  sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$/\\n" \\/' "$CU"
  printf '"";\n'
} > "$H"
echo "regenerated $H from $CU"
