#!/usr/bin/env bash
# Unified end-to-end build for all three GPU backends. BOINC is ON by default.
#
#   bash dev/build.sh --cuda      -> build_dev/genefercu     (AOT: every arch's SASS embedded, NO NVRTC, driver-only)
#   bash dev/build.sh --opencl    -> build_dev/geneferg      (OpenCL, runtime clBuildProgram)
#   bash dev/build.sh --hip       -> build_dev/genefer_hip   (HIP, runtime hipRTC)
#
#   --no-boinc          build a standalone binary (no libboinc)
#   FATBIN_FORCE=1      (--cuda) recompile all fatbins after editing ARCHES
set -euo pipefail

# ============================ CONFIG (edit me) ============================
# CUDA arch list (one SM per entry; f-suffix = forward-compatible family target, e.g. 120f).
ARCHES=( 50 52 60 61 70 75 80 86 89 90 100f 120f )
N_LIST=(16 17 18 19 20 21 22 23)           # GFN exponents the project ships (CUDA fatbins)
RNS_LIST=("2:0" "2:1" "3:0" "3:1")         # factory branches "<rns>:<is32>"
CUDA="${CUDA:-/usr/local/cuda-12.9}"        # CUDA toolkit (12.9 = Maxwell..Blackwell)
HIPCC="${HIPCC:-/usr/bin/hipcc}"            # HIP compiler
BOINC_DIR="${BOINC_DIR:-/home/ian/builds/boinc}"
DUMP_DEVICE="${DUMP_DEVICE:-0}"             # any installed GPU; only emits the specialized CUDA source
# =========================================================================

BACKEND=""; NOBOINC="${NOBOINC:-}"
for a in "$@"; do case "$a" in
  --cuda) BACKEND=cuda ;; --opencl|--ocl) BACKEND=opencl ;; --hip) BACKEND=hip ;;
  --no-boinc) NOBOINC=1 ;; *) echo "unknown arg: $a"; exit 2 ;;
esac; done
[ -n "$BACKEND" ] || { echo "usage: $0 --cuda|--opencl|--hip [--no-boinc]"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"; OUT="$ROOT/build_dev"; CXX="${CXX:-g++}"; mkdir -p "$OUT"

# BOINC link set (OpenCL also needs libboinc_opencl.a; CUDA/HIP select GPU via aid.gpu_device_num)
BFLAGS=""; BLIBS=""; BOCL=""
if [ -z "$NOBOINC" ]; then
  [ -f "$BOINC_DIR/api/libboinc_api.a" ] || { echo "BOINC_DIR=$BOINC_DIR has no libboinc (use --no-boinc)"; exit 1; }
  BFLAGS="-DBOINC -I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib"
  BLIBS="$BOINC_DIR/api/libboinc_api.a $BOINC_DIR/lib/libboinc.a"; BOCL="$BOINC_DIR/api/libboinc_opencl.a"
fi
GMP=/usr/lib/x86_64-linux-gnu/libgmp.a
echo "=== build $BACKEND ${NOBOINC:+(standalone)}${NOBOINC:-(+BOINC)} ==="

case "$BACKEND" in
# -------------------------------------------------------------------------
opencl)
  G="-O3 -DGPU $BFLAGS -I$ROOT/Khronos"
  $CXX -std=c++17 $G -c "$SRC/main.cpp"          -o "$OUT/maing.o"
  $CXX -std=c++17 $G -c "$SRC/transform_ocl.cpp" -o "$OUT/transform_ocl.o"
  $CXX -O3 -DGPU -static-libgcc -static-libstdc++ "$OUT/maing.o" "$OUT/transform_ocl.o" \
      "$GMP" $BLIBS $BOCL -lOpenCL -o "$OUT/geneferg"
  echo "OK -> $OUT/geneferg" ;;
# -------------------------------------------------------------------------
hip)
  bash "$ROOT/dev/gen_kernel_h.sh" >/dev/null    # embed cuda/kernel.cu for hipRTC
  G="-O3 -DGPU -DHIP $BFLAGS -I$SRC -I/usr/include/x86_64-linux-gnu"
  $HIPCC -std=c++17 $G -c "$SRC/main.cpp"          -o "$OUT/mainghip.o"
  $HIPCC -std=c++17 $G -c "$SRC/transform_hip.cpp" -o "$OUT/transform_hip.o"
  $HIPCC -O3 -DGPU -DHIP "$OUT/mainghip.o" "$OUT/transform_hip.o" "$GMP" $BLIBS -lhiprtc -o "$OUT/genefer_hip"
  echo "OK -> $OUT/genefer_hip" ;;
# -------------------------------------------------------------------------
cuda)
  FAT="$OUT/fatbins"; TMP="$OUT/_fatsrc"; DUMP="$OUT/dump_src"; mkdir -p "$FAT" "$TMP"
  # Blackwell (sm_1xx) can't be compiled directly: nvcc/NVRTC's compute_1xx codegen miscompiles the
  # square NTT kernels (shared cicc bug). Build those arches by hand -- clean compute_89 PTX -> ptxas
  # to the sm_NNNf family target -- and fatbinary-merge with the directly-compiled non-Blackwell arches.
  echo "[fatbin] arches: ${ARCHES[*]}"
  $CXX -std=c++17 -O2 -DGPU -DCUDA -I"$SRC" -I"$CUDA/include" "$ROOT/dev/dump_src.cpp" \
      -L"$CUDA/lib64" -lcuda -lnvrtc -lgmp -Wl,-rpath,"$CUDA/lib64" -o "$DUMP"
  for n in "${N_LIST[@]}"; do for rc in "${RNS_LIST[@]}"; do
    rns="${rc%:*}"; is32="${rc#*:}"; cu="$TMP/n${n}_rns${rns}_is${is32}.cu"; f="$FAT/genefer_n${n}_rns${rns}_is${is32}.fatbin"
    if [ -f "$f" ] && [ -z "${FATBIN_FORCE:-}" ]; then printf "  %-34s (cached)\n" "$(basename "$f")"; continue; fi
    GENEFER_DUMP_SRC="$cu" GENEFER_DUMP_EXIT=1 "$DUMP" "$rns" "$is32" "$n" "$DUMP_DEVICE" >/dev/null 2>&1 || true
    [ -s "$cu" ] || { echo "  !! dump failed n=$n rns=$rns is32=$is32"; continue; }
    imgs=(); ptx="$TMP/.p89.ptx"; ok=1; rm -f "$ptx" "$TMP"/.c_*.cubin
    for a in "${ARCHES[@]}"; do
      cub="$TMP/.c_${a}.cubin"
      case "$a" in
        100*|101*|103*|120*|121*)   # Blackwell: clean compute_89 PTX, then ptxas to the sm_NNNf family target
          [ -s "$ptx" ] || "$CUDA/bin/nvcc" -ptx -arch=compute_89 "$cu" -o "$ptx" 2>/tmp/fatbin_err.log || { ok=0; break; }
          "$CUDA/bin/ptxas" -arch="sm_${a}" "$ptx" -o "$cub"             2>/tmp/fatbin_err.log || { ok=0; break; } ;;
        *)                          # everyone else: compile straight to a cubin
          "$CUDA/bin/nvcc" -gencode "arch=compute_${a},code=sm_${a}" --cubin "$cu" -o "$cub" 2>/tmp/fatbin_err.log || { ok=0; break; } ;;
      esac
      imgs+=(--image3="kind=elf,sm=${a},file=${cub}")
    done
    if [ "$ok" = 1 ] && "$CUDA/bin/fatbinary" --create="$f" --64 "${imgs[@]}" 2>/tmp/fatbin_err.log
      then printf "  %-34s %5s KB\n" "$(basename "$f")" "$(( $(stat -c%s "$f")/1024 ))"
      else echo "  !! build failed n=$n rns=$rns is32=$is32 (see /tmp/fatbin_err.log)"; fi
    rm -f "$ptx" "$TMP"/.c_*.cubin
  done; done
  # embed: .incbin + (n,rns,is32)->blob lookup
  SF="$OUT/fatbins.s"; HF="$SRC/cuda/fatbins.h"; TABLE=""
  printf '/* auto-generated */\n\t.section .note.GNU-stack,"",@progbits\n\t.section .rodata\n' > "$SF"
  printf '// auto-generated\n#pragma once\n#include <cstddef>\nextern "C" {\n' > "$HF"
  for f in "$FAT"/*.fatbin; do
    b=$(basename "$f" .fatbin); t=${b#*_n}; n=${t%%_*}; t=${b#*_rns}; rns=${t%%_*}; is32=${b##*_is}; s="gf_n${n}_rns${rns}_is${is32}"
    printf '\t.global %s\n\t.global %s_end\n%s:\n\t.incbin "%s"\n%s_end:\n' "$s" "$s" "$s" "$f" "$s" >> "$SF"
    printf '  extern const unsigned char %s[], %s_end[];\n' "$s" "$s" >> "$HF"
    TABLE="${TABLE}  { ${n}, ${rns}, ${is32}, ${s}, ${s}_end },\n"
  done
  { printf '}\nstruct genefer_fatbin_t { int n, rns, is32; const unsigned char * data; const unsigned char * end; };\n'
    printf 'static const genefer_fatbin_t genefer_fatbins[] = {\n%b};\n' "$TABLE"
    printf 'static inline bool genefer_get_fatbin(int n, int rns, int is32, const void ** d, size_t * s) {\n'
    printf '  for (const genefer_fatbin_t & e : genefer_fatbins) if (e.n==n && e.rns==rns && e.is32==is32)\n'
    printf '    { *d=e.data; *s=(size_t)(e.end-e.data); return true; } return false; }\n'; } >> "$HF"
  echo "[embed] $(ls "$FAT"/*.fatbin | wc -l) fatbins"
  # link: no NVRTC anywhere; only the driver at runtime
  G="-O3 -DGPU -DCUDA -DGENEFER_EMBED_FATBINS $BFLAGS -I$CUDA/include"
  $CXX -c "$SF" -o "$OUT/fatbins.o"
  $CXX -std=c++17 $G -c "$SRC/main.cpp"         -o "$OUT/maingcu_fb.o"
  $CXX -std=c++17 $G -c "$SRC/transform_cu.cpp" -o "$OUT/transform_cu_fb.o"
  $CXX -O3 -static-libgcc -static-libstdc++ "$OUT/maingcu_fb.o" "$OUT/transform_cu_fb.o" "$OUT/fatbins.o" \
      "$GMP" $BLIBS -L"$CUDA/lib64/stubs" -lcuda -lpthread -ldl -lrt -lm -o "$OUT/genefercu"
  echo "OK -> $OUT/genefercu ($(du -h "$OUT/genefercu" | cut -f1))" ;;
esac
