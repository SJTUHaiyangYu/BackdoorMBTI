#!/usr/bin/env bash
set -euo pipefail

# Config
CONDA_ENV="bkdmbti"
# Timestamp for this batch run (directory name) in format: 2026_01_17-00:25
MAIN_TS=$(date '+%Y_%m_%d-%H:%M')
LOG_DIR="./gen_logs/batch_run_${MAIN_TS}"
CONTINUE_ON_FAIL=true  # true to keep going after a failure

# Parallel settings
# Example: 3 free GPUs -> set GPUS=(0 1 2) or (1 2 3) based on your machine.
GPUS=(3)

mkdir -p "$LOG_DIR"

THIS_FILE=$(basename "$0")

# Command list; each entry can set its own GPU
CMDS=(
	# your commands
	"python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name badnet     --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256 --batch_size 256 --train_benign"
  "python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name ultrasonic --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256"
  "python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name daba       --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256"
  "python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name gis        --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256"
  "python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name badnet     --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256"
  "python atk_batch_train.py --data_type audio --dataset speechcommands --attack_name blend      --model_name lstm --pratio 0.1 --num_workers 4 --epochs 100 --batch_size 256"
)

# Activate conda env
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  set +e
  conda activate "$CONDA_ENV"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "conda activate ${CONDA_ENV} failed (rc=$rc); using current environment" >&2
  fi
else
  echo "conda not found; using current environment" >&2
fi

# ---- Parallel runner (GPU pool) ----

# Strict validation: require at least one GPU and no duplicates.
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS is empty; set GPUS=(...) to the free GPU IDs." >&2
  exit 2
fi
declare -A _gpu_seen
for g in "${GPUS[@]}"; do
  if [[ -n "${_gpu_seen[$g]:-}" ]]; then
    echo "Duplicate GPU id in GPUS: $g" >&2
    exit 2
  fi
  _gpu_seen[$g]=1
done
if [[ ${#CMDS[@]} -eq 0 ]]; then
  echo "CMDS is empty; nothing to run." >&2
  exit 0
fi

free_gpus=("${GPUS[@]}")
running_pids=()
declare -A pid_to_gpu
declare -A pid_to_log
declare -A pid_to_idx
failed=0

cleanup() {
  # Best-effort: terminate any still-running children on exit/signal.
  for pid in "${running_pids[@]:-}"; do
    if [[ -z "${pid:-}" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

launch_one() {
  local job_idx="$1"
  local cmd="$2"
  local gpu="$3"

  local ts_start
  ts_start=$(date '+%Y-%m-%d %H:%M:%S')

  # Extract a stable keyword from the command for log naming.
  # Priority:
  #   1) --ckpt-dir <path>  -> basename(path)
  #   2) --attack <name>    -> name
  # Fallback: "unknown"
  local keyword
  keyword="unknown"
  local parts=()
  # shellcheck disable=SC2206
  parts=($cmd)
  for ((j=0; j<${#parts[@]}; j++)); do
    if [[ "${parts[$j]}" == "--ckpt-dir" && $((j+1)) -lt ${#parts[@]} ]]; then
      keyword=$(basename "${parts[$((j+1))]}")
      break
    fi
    if [[ "${parts[$j]}" == "--attack" && $((j+1)) -lt ${#parts[@]} ]]; then
      keyword="${parts[$((j+1))]}"
      break
    fi
  done
  # Sanitize keyword to be filename-safe.
  keyword=$(echo "$keyword" | sed -E 's#[/[:space:]]+#_#g; s#[^A-Za-z0-9._-]+#_#g')

  local job_ts
  # job timestamp in format: 2026_01_17-00:25
  job_ts=$(date '+%Y_%m_%d-%H:%M')
  local log_file
  # Log file per job: Job-{i}-{time-str}.log
  log_file="${LOG_DIR}/Job-${job_idx}-${job_ts}.log"

  echo "[$ts_start] [JOB ${job_idx}/${#CMDS[@]}] GPU=${gpu} $cmd"

  (
    set +e
    echo "==== JOB ${job_idx} START ${ts_start} ===="
    echo "GPU: ${gpu}"
    echo "CMD: $cmd"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "$cmd"
    rc=$?
    ts_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "==== JOB ${job_idx} END ${ts_end} (rc=$rc) ===="
    exit $rc
  ) &> "$log_file" &

  local pid=$!
  running_pids+=("$pid")
  pid_to_gpu["$pid"]="$gpu"
  pid_to_log["$pid"]="$log_file"
  pid_to_idx["$pid"]="$job_idx"
}

reap_finished_one() {
  # Blocks until at least one child has finished, then recycles its GPU.
  while true; do
    for idx in "${!running_pids[@]}"; do
      local pid
      pid="${running_pids[$idx]}"
      if [[ -z "${pid:-}" ]]; then
        continue
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        set +e
        wait "$pid"
        local rc=$?
        set -e

        local gpu
        gpu="${pid_to_gpu[$pid]}"
        local log_file
        log_file="${pid_to_log[$pid]}"
        local job_idx
        job_idx="${pid_to_idx[$pid]}"

        unset 'running_pids[idx]'
        running_pids=("${running_pids[@]}")

        free_gpus+=("$gpu")
        unset 'pid_to_gpu[$pid]'
        unset 'pid_to_log[$pid]'
        unset 'pid_to_idx[$pid]'

        if [[ $rc -ne 0 ]]; then
          echo "[JOB ${job_idx}] FAILED (rc=$rc). See ${log_file}"
          failed=1
        else
          echo "[JOB ${job_idx}] DONE. Log: ${log_file}"
        fi
        return 0
      fi
    done
    sleep 0.2
  done
}

job_idx=1
for cmd in "${CMDS[@]}"; do
  if [[ "$CONTINUE_ON_FAIL" == false && $failed -ne 0 ]]; then
    echo "Stop on failure: not launching remaining jobs."
    break
  fi

  while [[ ${#free_gpus[@]} -eq 0 ]]; do
    reap_finished_one
  done

  gpu="${free_gpus[0]}"
  free_gpus=("${free_gpus[@]:1}")
  launch_one "$job_idx" "$cmd" "$gpu"
  ((job_idx++))
done

while [[ ${#running_pids[@]} -gt 0 ]]; do
  reap_finished_one
done

if [[ "$CONTINUE_ON_FAIL" == false && $failed -ne 0 ]]; then
  exit 1
fi

echo "All jobs finished."
