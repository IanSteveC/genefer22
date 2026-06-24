#!/usr/bin/env bash
###############################################################################
# genefer-mps-sweep.sh  —  CUDA MPS concurrency / SM-partition throughput sweep
#
# Finds the most efficient way to run several concurrent genefer GFN tasks on a
# single NVIDIA GPU using CUDA MPS (Multi-Process Service). For every combo of
# GFN size N, concurrency C, and MPS active-thread-% P, it launches C tasks at
# once (each capped to P% of the SMs) and reports the AGGREGATE throughput:
#
#     effective ms/bit = mean(per-task ms/bit) / C          (lower = better)
#     gain%            = improvement of "effective" vs ONE solo task (C=1)
#
# gain% > 0 means C concurrent tasks do more total work/sec than a single task
# would — i.e. the GPU was underutilized solo and MPS filled it. This is biggest
# for small N (GPU mostly idle) and shrinks as N grows toward bandwidth-bound.
#
# No root needed: runs as the current user with its OWN private MPS server
# (started and stopped by this script) in the GPU's default compute mode.
#
# Usage:
#     ./genefer-mps-sweep.sh
#   or override any CONFIG value on the command line, e.g.:
#     GENEFER=./genefercu N_LIST="16 20 23" CONCURRENCY="1 2 4" ./genefer-mps-sweep.sh
###############################################################################
set -u

# ============================ CONFIG (edit me) ===============================
GENEFER="${GENEFER:-./genefercu}"               # path to the genefer CUDA binary
DEVICE="${DEVICE:-0}"                            # CUDA device index (see: nvidia-smi -L)
BASE="${BASE:-1000000}"                          # genefer base b  (tests b^(2^N)+1)
N_LIST="${N_LIST:-16 17 18 19 20 21 22 23}"      # GFN exponents N to test
CONCURRENCY="${CONCURRENCY:-1 2 3 4}"            # task counts to test (1 = solo baseline)
MPS_PCT="${MPS_PCT:-30 40 50 60 70 80 90 100}"   # MPS active-thread-% per task
WARMUP="${WARMUP:-14}"                            # min seconds before reading (reach steady state)
MAXWAIT="${MAXWAIT:-140}"                         # max seconds to wait per config
OUTFILE="${OUTFILE:-mps_sweep_results.txt}"       # full results (CSV) written here
WORKDIR="${WORKDIR:-/tmp/genefer_mps_run}"        # scratch dir (per-task checkpoints)
MPS_DIR="${MPS_DIR:-/tmp/genefer_mps}"            # private MPS pipe/log directory
# =============================================================================

# resolve GENEFER to an ABSOLUTE path (tasks run from per-task dirs, so a relative path would break)
case "$GENEFER" in
  */*) GENEFER=$(readlink -f "$GENEFER" 2>/dev/null || echo "$GENEFER") ;;   # given a path
  *)   GENEFER=$(command -v "$GENEFER" 2>/dev/null || echo "$GENEFER") ;;    # given a bare command name
esac
NAME=$(basename "$GENEFER")
die(){ echo "ERROR: $*" >&2; exit 1; }

# --------- prerequisite checks ----------
[ -x "$GENEFER" ] || die "genefer binary not found/executable: '$GENEFER'  (set GENEFER=/path/to/genefercu)"
command -v nvidia-cuda-mps-control >/dev/null || die "nvidia-cuda-mps-control not found (install the NVIDIA CUDA MPS tools)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
command -v awk >/dev/null || die "awk not found"

export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

# --------- robust cleanup on ANY exit (Ctrl-C, error, normal) ----------
running(){ pgrep -f "$GENEFER -q" 2>/dev/null; }
stop_tasks(){ local p; p=$(running); [ -n "$p" ] && kill -TERM $p 2>/dev/null; sleep 2;
              p=$(running); [ -n "$p" ] && kill -9 $p 2>/dev/null; }
stop_mps(){ echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; sleep 1; }
cleanup(){ stop_tasks; stop_mps; }
trap cleanup EXIT INT TERM

# --------- start a private MPS server ----------
echo "==> starting private CUDA MPS server in $MPS_DIR"
rm -rf "$MPS_DIR"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || die "could not start MPS control daemon (is the GPU in EXCLUSIVE_PROCESS mode owned by someone else?)"
sleep 2
echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null 2>&1 || die "MPS daemon not responding"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEVICE" 2>/dev/null)
echo "    OK — device $DEVICE: ${GPU:-unknown}"

# --------- functional self-check: can a client actually run through MPS? ----------
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR/_chk"
echo "==> verifying a genefer task can run through MPS..."
( cd "$WORKDIR/_chk"; export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
  exec "$GENEFER" -q -b "$BASE" -n "${N_LIST%% *}" -d "$DEVICE" >out.log 2>&1 ) &
_t=0; while [ "$_t" -lt 30 ]; do sleep 2; _t=$((_t+2)); grep -q 'ms/bit' "$WORKDIR/_chk/out.log" 2>/dev/null && break; done
stop_tasks
if ! grep -q 'ms/bit' "$WORKDIR/_chk/out.log" 2>/dev/null; then
  echo "ERROR: a genefer task produced no throughput reading through MPS." >&2
  echo "  --- last output ---" >&2; tail -4 "$WORKDIR/_chk/out.log" >&2 2>/dev/null
  echo "  Likely causes: another MPS server already owns this GPU; the GPU is in" >&2
  echo "  EXCLUSIVE_PROCESS mode owned by another user; or a wrong DEVICE/GENEFER." >&2
  exit 1
fi
rm -rf "$WORKDIR/_chk"
echo "    OK — MPS client verified"

# --------- measure ONE config: C tasks each capped at P% SMs ----------
# echoes:  "<avg per-task ms/bit>  <how many of C tasks reported>"
run_cfg(){
  local N=$1 C=$2 P=$3 i pids=()
  for i in $(seq 1 "$C"); do
    local d="$WORKDIR/n${N}_c${C}_p${P}_i${i}"; mkdir -p "$d"
    # exec => $! is the genefer PID itself (so our kill reaches it, not a dead subshell).
    # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE caps this client to P% of the SMs via MPS.
    ( cd "$d"; export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$P"
      exec "$GENEFER" -q -b "$BASE" -n "$N" -d "$DEVICE" >out.log 2>&1 ) &
    pids+=("$!")
  done
  # wait until every task has >=2 ms/bit readings (steady) or we hit MAXWAIT
  local el=0 ok
  while [ "$el" -lt "$MAXWAIT" ]; do
    sleep 4; el=$((el+4)); [ "$el" -lt "$WARMUP" ] && continue
    ok=1
    for i in $(seq 1 "$C"); do
      [ "$(tr '\r' '\n' < "$WORKDIR/n${N}_c${C}_p${P}_i${i}/out.log" 2>/dev/null | grep -c 'ms/bit')" -lt 2 ] && { ok=0; break; }
    done
    [ "$ok" = 1 ] && break
  done
  # take each task's latest ms/bit reading
  local vals=""
  for i in $(seq 1 "$C"); do
    local v=$(tr '\r' '\n' < "$WORKDIR/n${N}_c${C}_p${P}_i${i}/out.log" 2>/dev/null \
              | grep -oE '[0-9.]+ ms/bit' | grep -oE '^[0-9.]+' | tail -1)
    [ -n "$v" ] && vals="$vals $v"
  done
  # tear down THIS config; SIGTERM lets genefer exit cleanly (MPS-safe), block until all gone
  kill -TERM "${pids[@]}" 2>/dev/null
  local t=0; while [ -n "$(running)" ] && [ "$t" -lt 16 ]; do sleep 0.5; t=$((t+1)); done
  [ -n "$(running)" ] && { kill -9 $(running) 2>/dev/null; sleep 1; }
  rm -rf "$WORKDIR"/n${N}_c${C}_p${P}_i* 2>/dev/null
  echo "$vals" | awk '{s=0;n=0;for(j=1;j<=NF;j++){s+=$j;n++}} END{ if(n>0) printf "%.6f %d", s/n, n; else printf "0 0" }'
}

# --------- run the sweep ----------
mkdir -p "$WORKDIR"
{ echo "# genefer MPS sweep  $(date)"
  echo "# GPU=${GPU:-?}  binary=$NAME  base=$BASE"
  echo "# effective = mean(per-task ms/bit)/C ; gain_pct vs solo (C=1,P=100)"
  echo "N,C,P,per_task_ms,n_ok,effective,solo,gain_pct"
} > "$OUTFILE"

for N in $N_LIST; do
  read -r solo nb < <(run_cfg "$N" 1 100)
  echo "$N,1,100,$solo,$nb,$solo,$solo,0.0" >> "$OUTFILE"
  printf "\n== N=%s ==  solo baseline = %s ms/bit\n" "$N" "$solo"
  for C in $CONCURRENCY; do
    [ "$C" -le 1 ] && continue
    for P in $MPS_PCT; do
      read -r ms nok < <(run_cfg "$N" "$C" "$P")
      eff=$(awk -v m="$ms" -v c="$C" 'BEGIN{printf "%.6f", m/c}')
      gain=$(awk -v s="$solo" -v e="$eff" 'BEGIN{ if(s>0) printf "%.1f",(s-e)/s*100; else print "0" }')
      echo "$N,$C,$P,$ms,$nok,$eff,$solo,$gain" >> "$OUTFILE"
      warn=""; [ "${nok:-0}" -lt "$C" ] && warn="  (! only $nok/$C tasks reported)"
      printf "  C=%s P=%3s%%   per-task=%-9s eff=%-9s gain=%+6s%%%s\n" "$C" "$P" "$ms" "$eff" "$gain" "$warn"
    done
  done
done

# --------- summary to terminal ----------
echo ""
echo "================================  SUMMARY  ================================"
echo "(gain% = aggregate-throughput improvement of C concurrent tasks vs 1 solo)"
awk -F, 'NR>4 {
  n=$1+0; g=$8+0
  if($2==1){ if(!(n in seen)){seen[n]=1; ord[++m]=n}; solo[n]=$6; next }
  if(!(n in bg) || g>bg[n]){ bg[n]=g; bc[n]=$2; bp[n]=$3; be[n]=$6 }
}
END{
  printf "  %-4s %-13s %-16s %-12s %s\n","N","solo ms/bit","best config","eff ms/bit","gain"
  printf "  %-4s %-13s %-16s %-12s %s\n","---","-----------","-----------","----------","----"
  for(i=1;i<=m;i++){ n=ord[i]
    if(n in bg) printf "  %-4s %-13s C=%s @ P=%-4s    %-12s +%.1f%%\n", n, solo[n], bc[n], bp[n]"%", be[n], bg[n]
    else        printf "  %-4s %-13s (no concurrency tested)\n", n, solo[n]
  }
}' "$OUTFILE"
echo ""
echo "Full per-config results: $OUTFILE"
echo "Tip: small N (GPU underutilized solo) gains the most from high concurrency +"
echo "     moderate MPS%% (throttle each task so C tasks partition the GPU once,"
echo "     rather than oversubscribing it at 100%%)."
