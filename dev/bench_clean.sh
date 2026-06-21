#!/usr/bin/env bash
# Rigorous A/B: interleave OpenCL(d1) and CUDA(d0) at full clock, take the MIN of 2 runs each
# (min = least thermally perturbed). Reports time and res64-match.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
t2s(){ awk -F: '{print ($1*3600)+($2*60)+$3}'; }
run(){ # $1=bin $2=dev $3=b $4=n  -> echoes "secs res64"
  local o; o=$(timeout 600 ./build_dev/$1 -q -b "$3" -n "$4" -d "$2" 2>&1)
  local t; t=$(echo "$o" | grep -oE 'time = [0-9:]+' | tail -1 | awk '{print $3}' | t2s)
  local r; r=$(echo "$o" | grep -oE 'res64 = [0-9A-F]+' | awk '{print $3}')
  echo "${t:-99999} ${r:-FAIL}"
}
printf "%-3s | %-8s %-8s | %-6s | %s\n" "n" "opencl" "cuda" "cu/ocl" "match"
for n in "$@"; do
  b=1000000
  read o1 r1 < <(run geneferg 1 $b $n); read c1 cr1 < <(run genefercu 0 $b $n)
  read o2 r2 < <(run geneferg 1 $b $n); read c2 cr2 < <(run genefercu 0 $b $n)
  om=$(( o1<o2 ? o1 : o2 )); cm=$(( c1<c2 ? c1 : c2 ))
  ratio=$(awk -v c="$cm" -v o="$om" 'BEGIN{ if(o>0) printf "%.2f", c/o; else print "?"}')
  m=$([[ "$r1" == "$cr1" && -n "$r1" ]] && echo OK || echo "MISMATCH($r1/$cr1)")
  printf "%-3s | %-8s %-8s | %-6s | %s\n" "$n" "${om}s" "${cm}s" "$ratio" "$m"
done
