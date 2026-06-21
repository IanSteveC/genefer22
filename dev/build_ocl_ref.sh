#!/usr/bin/env bash
# Build the OpenCL reference binary (geneferg) WITHOUT BOINC, for local validation.
# Deterministic direct-compile (bypasses the makefile's BOINC=true default).
# Requires: libgmp-dev (-lgmp), NVIDIA OpenCL ICD (-lOpenCL). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"; OUT="$ROOT/build_dev"; mkdir -p "$OUT"
CXX="${CXX:-g++}"
STD="-std=c++17"
WARN="-Wall -Wextra -Wsign-conversion"
GPU="-O3 -DGPU -I$ROOT/Khronos"
echo "[1/3] maing.o";        $CXX $STD $WARN $GPU -c "$SRC/main.cpp"          -o "$OUT/maing.o"
echo "[2/3] transform_ocl.o"; $CXX $STD $WARN $GPU -c "$SRC/transform_ocl.cpp" -o "$OUT/transform_ocl.o"
echo "[3/3] link geneferg";  $CXX -O3 -DGPU -static-libgcc -static-libstdc++ \
    "$OUT/maing.o" "$OUT/transform_ocl.o" -lgmp -lOpenCL -o "$OUT/geneferg"
echo "OK -> $OUT/geneferg"
