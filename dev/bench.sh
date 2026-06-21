#!/usr/bin/env bash
# Benchmark CUDA (genefercu, device 0) vs OpenCL (geneferg, device 1=V100) on identical workloads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
printf "%-10s %-3s | %-12s %-12s %s\n" "b" "n" "opencl" "cuda" "speedup"
for spec in "1000000 18" "1000000 20" "1000000 22"; do
  set -- $spec; b=$1; n=$2
  to=$( { /usr/bin/time -f "%e" ./build_dev/geneferg -q -b "$b" -n "$n" -d 1 >/dev/null 2>/tmp/t_o; cat /tmp/t_o; } 2>&1 | tail -1)
  tc=$( { /usr/bin/time -f "%e" ./build_dev/genefercu -q -b "$b" -n "$n" -d 0 >/dev/null 2>/tmp/t_c; cat /tmp/t_c; } 2>&1 | tail -1)
  sp=$(awk -v o="$to" -v c="$tc" 'BEGIN{ if(c>0) printf "%.2fx", o/c; else print "n/a"}')
  printf "%-10s %-3s | %-12s %-12s %s\n" "$b" "$n" "${to}s" "${tc}s" "$sp"
done
