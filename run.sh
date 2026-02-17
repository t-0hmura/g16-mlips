#!/bin/sh
#PBS -N mlips4g16_smoke
#PBS -q default
#PBS -l nodes=1:ppn=4,mem=16GB,walltime=02:00:00
#PBS -o /dev/null
#PBS -e /dev/null

set -eu

test "${PBS_O_WORKDIR:-}" && cd "$PBS_O_WORKDIR"

# Keep a job-local log even when PBS stdout/stderr are disabled.
LOG_PATH="${PBS_O_WORKDIR:-$PWD}/run.${PBS_JOBID:-local}.log"
exec >"$LOG_PATH" 2>&1
echo "[INFO] log=${LOG_PATH}"

# Optional module initialization (path-independent).
if [ -n "${MODULESHOME:-}" ] && [ -f "${MODULESHOME}/init/profile.sh" ]; then
  . "${MODULESHOME}/init/profile.sh"
fi

if command -v module >/dev/null 2>&1; then
  module load gaussian16.C02 || true
fi

# Optional conda activation:
# - Set MLIPS_CONDA_ENV to enable.
# - Optionally override CONDA_SH (default: $HOME/miniconda3/etc/profile.d/conda.sh).
CONDA_SH_PATH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH_PATH}" ]; then
  . "${CONDA_SH_PATH}"
fi
if [ -n "${MLIPS_CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  conda activate "${MLIPS_CONDA_ENV}"
fi

echo "[INFO] Python: $(command -v python3 || true)"

python3 -m py_compile plugins/*.py

run_list_models() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"
  backend_run "$pref_cmd" "$short_cmd" "$py_script" --version || true
  backend_run "$pref_cmd" "$short_cmd" "$py_script" --list-models | head -n 20
}

resolve_backend_runner() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"

  BACKEND_RUNNER_TYPE=""
  BACKEND_RUNNER=""

  if command -v "$pref_cmd" >/dev/null 2>&1; then
    BACKEND_RUNNER_TYPE="cmd"
    BACKEND_RUNNER="$pref_cmd"
    echo "[INFO] using ${pref_cmd}"
    return 0
  fi

  if [ -f "$py_script" ]; then
    BACKEND_RUNNER_TYPE="py"
    BACKEND_RUNNER="$py_script"
    echo "[INFO] prefixed command not found; using python script ${py_script}"
    return 0
  fi

  if command -v "$short_cmd" >/dev/null 2>&1; then
    BACKEND_RUNNER_TYPE="cmd"
    BACKEND_RUNNER="$short_cmd"
    echo "[WARN] prefixed command missing; using short alias ${short_cmd} (may collide across packages)"
    return 0
  fi

  echo "[ERROR] no usable command found for ${pref_cmd}/${short_cmd}"
  return 1
}

backend_run() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"
  shift 3

  resolve_backend_runner "$pref_cmd" "$short_cmd" "$py_script" || return 1
  if [ "$BACKEND_RUNNER_TYPE" = "py" ]; then
    python3 "$BACKEND_RUNNER" "$@"
  else
    "$BACKEND_RUNNER" "$@"
  fi
}

run_list_models "mlips4g16-uma" "uma" "plugins/uma_g16.py"
run_list_models "mlips4g16-orb" "orb" "plugins/orbmol_g16.py"
run_list_models "mlips4g16-mace" "mace" "plugins/mace_g16.py"
echo "[INFO] smoke test completed"

# Full Gaussian run example (after dependency installation):
# g16 < examples/water_external.gjf > water_external.log
