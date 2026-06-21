#!/usr/bin/env bash
# Build the CUDA backend binary (genefercu) WITHOUT BOINC, for local validation.
# Requires: libgmp-dev, CUDA toolkit (nvrtc + driver stub). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"; OUT="$ROOT/build_dev"; mkdir -p "$OUT"
CXX="${CXX:-g++}"; STD="-std=c++17"; WARN="-Wall -Wextra"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
GPU="-O3 -DGPU -DCUDA -I$CUDA_PATH/include"
echo "[1/3] maingcu.o";       $CXX $STD $WARN $GPU -c "$SRC/main.cpp"         -o "$OUT/maingcu.o"
echo "[2/3] transform_cu.o";  $CXX $STD $WARN $GPU -c "$SRC/transform_cu.cpp" -o "$OUT/transform_cu.o"
echo "[3/3] link genefercu";  $CXX -O3 -DGPU -DCUDA -static-libgcc -static-libstdc++ \
    "$OUT/maingcu.o" "$OUT/transform_cu.o" -lgmp \
    -lcuda -L"$CUDA_PATH/lib64" -lnvrtc -Wl,-rpath,"$CUDA_PATH/lib64" -o "$OUT/genefercu"
echo "OK -> $OUT/genefercu"
