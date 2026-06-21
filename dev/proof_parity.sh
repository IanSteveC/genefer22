#!/usr/bin/env bash
# Phase 4: validate the FULL proof path (Gerbicz-Li + Pietrzak-Li) matches between backends.
# Runs -p (generate proof) then -s (server: proof->certificate + 64-bit key) on each backend
# in its own dir, and compares the res64 and the server key. Identical => full parity.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
B=${1:-1000000}; N=${2:-13}
run_backend(){ # $1=bin $2=dev $3=workdir
  local bin=$1 dev=$2 wd=$3
  rm -rf "$wd"; mkdir -p "$wd"; ( cd "$wd"
    cp -s "$ROOT/cuda/kernel.cu" . 2>/dev/null; mkdir -p cuda; cp -sf "$ROOT/cuda/kernel.cu" cuda/ 2>/dev/null
    cp -sf "$ROOT/ocl/kernel.cl" . 2>/dev/null; mkdir -p ocl; cp -sf "$ROOT/ocl/kernel.cl" ocl/ 2>/dev/null
    p=$("$ROOT/build_dev/$bin" -p -b "$B" -n "$N" -d "$dev" 2>&1)
    echo "$p" | grep -oE 'res64 = [0-9A-F]+' | tail -1
    s=$("$ROOT/build_dev/$bin" -s -b "$B" -n "$N" -d "$dev" 2>&1)
    echo "$s" | grep -oiE 'key = [0-9A-Fx]+|certificate' | tail -2
  ) }
echo "=== proof parity b=$B n=$N ==="
echo "--- OpenCL (geneferg d1) ---"; run_backend geneferg 1 /tmp/pp_ocl
echo "--- CUDA (genefercu d0) ---";  run_backend genefercu 0 /tmp/pp_cu
