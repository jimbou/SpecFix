#!/bin/bash
# run_specfix_suite.sh
# Runs SpecFix multiple times with/without Mnimi cache and varying c/e. and collects the resulting tokens and times
# Usage: bash run_specfix_suite.sh

set -euo pipefail

# ----- Config -----
DATASET="humaneval"
DATA_PATH="datasets/humaneval_final.jsonl"
PASSK=1
MODEL="gpt-4.1-mini"
TEMP=1.0
CACHE_DIR="$HOME/.mnimi/specfix_suite_cache_final"

# default sizes
C_DEFAULT=20
E_DEFAULT=10

# output dirs/files
RESULTS_DIR="results_suite"
LOG_DIR="logs_suite"
mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$CACHE_DIR"

# Optional: activate venv if present
if [ -d "specfix-venv" ]; then
  source specfix-venv/bin/activate
fi

run() {
  local label="$1"     # human label used in filenames
  local use_cache="$2" # "yes" or "no"
  local c_size="$3"    # cluster size (-c)
  local e_size="$4"    # evaluation size (-e)

  local stat_file="${RESULTS_DIR}/${label}.stats.txt"
  local log_file="${LOG_DIR}/${label}.log"

  echo "======================================================"
  echo " Run: $label"
  echo "   Cache: $use_cache"
  echo "   -c: $c_size | -e: $e_size"
  echo "======================================================"

  
  if [ "$use_cache" = "yes" ]; then
    CMD=(python main.py
      -d "$DATASET"
      -p "$DATA_PATH"
      -c "$c_size"
      -e "$e_size"
      -k "$PASSK"
      -m "$MODEL"
      -t "$TEMP"
      --cache-dir "$CACHE_DIR"
      --output_stat_file "$stat_file")
  else
    CMD=(python main.py
      -d "$DATASET"
      -p "$DATA_PATH"
      -c "$c_size"
      -e "$e_size"
      -k "$PASSK"
      -m "$MODEL"
      -t "$TEMP"
      --output_stat_file "$stat_file")
  fi

  # Print the full command
  echo ">>> Running command:"
  printf ' %q' "${CMD[@]}"
  echo
  echo "------------------------------------------------------"

  # Run and capture output
  "${CMD[@]}" 2>&1 | tee "$log_file"

  echo "✅ Done: $label"
  echo "   Stats: $stat_file"
  echo "   Log  : $log_file"
  echo
}

# -------------------------------------------------------
# Sequence you asked for (all cache-enabled runs share $CACHE_DIR)
# 1) no cache
# run "r1_nocache_c${C_DEFAULT}_e${E_DEFAULT}" "no"  "$C_DEFAULT" "$E_DEFAULT"

# # 2) with cache (fill it)
# run "r2_cache_fill_c${C_DEFAULT}_e${E_DEFAULT}" "yes" "$C_DEFAULT" "$E_DEFAULT"

# # 3) with cache again
# run "r3_cache_again_c${C_DEFAULT}_e${E_DEFAULT}" "yes" "$C_DEFAULT" "$E_DEFAULT"

# 4) no cache, c=40
run "r4_nocache_c40_e${E_DEFAULT}" "no"  "40" "$E_DEFAULT"

# 5) cache, c=40
run "r5_cache_c40_e${E_DEFAULT}" "yes" "40" "$E_DEFAULT"

# 6) cache again, c=40
run "r6_cache_again_c40_e${E_DEFAULT}" "yes" "40" "$E_DEFAULT"

# 7) no cache, e=20
run "r7_nocache_c${C_DEFAULT}_e20" "no"  "$C_DEFAULT" "20"

# 8) cache, e=20
run "r8_cache_c${C_DEFAULT}_e20" "yes" "$C_DEFAULT" "20"

# 9) cache again, e=20
run "r9_cache_again_c${C_DEFAULT}_e20" "yes" "$C_DEFAULT" "20"
