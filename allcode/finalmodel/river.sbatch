#!/usr/bin/env bash
# Submit with:
#   cd /hpcdata/home/research/hzhang18/daduhe
#   sbatch submit_dadu_batch1_single_directory_array.sbatch
#
# Test one parameter combination first:
#   sbatch --array=0-0 submit_dadu_batch1_single_directory_array.sbatch

#SBATCH --job-name=dadu_rain
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
# 3 rain factors x 3 sigma-x values x 3 sigma-y values = 27 cases.
# "%2" means that at most two simulations run concurrently.
#SBATCH --array=0-26%9
#SBATCH --output=/hpcdata/home/research/hzhang18/daduhe/slurm-dadu-%A_%a.out
#SBATCH --error=/hpcdata/home/research/hzhang18/daduhe/slurm-dadu-%A_%a.err

set -euo pipefail

# This is the complete template directory. It must contain, at minimum:
#   run_model1_rake55_plus_uplift_model3_dam_capture_batch1.py
#   uplift_with_noise.py
#   fault_displacement_v4.py
#   Dadu_run.py, Daduhejif.xml, initopo.csv, and all other Badlands inputs.
TEMPLATE_DIR="/hpcdata/home/research/hzhang18/daduhe"

# Each array task receives a private copy here. Do not set this inside
# TEMPLATE_DIR, otherwise a later rsync can copy a case into another case.
RUN_ROOT="/hpcdata/home/research/hzhang18/daduhe_array_runs"
MAIN_SCRIPT="run_model1_rake55_plus_uplift_model3_dam_capture_batch1.py"

# Load the Badlands Python environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate uw-2.17

# -------------------- rainfall parameter grid --------------------
# Edit only these three lists for a new parameter sweep.
RAIN_FACTORS=(1.2 1.5 2.0)
SIGMA_X=(80 100 120)
SIGMA_Y=(100 130 160)

# ---------------- fixed tectonic / precipitation parameters ----------------
FIXED_ARGS=(
  --right-uplift-height 500
  --right-uplift-x-min 120
  --right-uplift-x-max 199
  --right-uplift-center-y 425
  --right-uplift-sigma-y 30
  --minjiang-subsidence-height -700
  --minjiang-x-min 70
  --minjiang-x-max 199
  --minjiang-center-y 455
  --minjiang-sigma-y 200
  --gongga-extra-peak-x 30
  --gongga-extra-peak-y 500
  --gongga-extra-peak-sigma-x 15
  --gongga-extra-peak-sigma-y 10
  --gongga-extra-peak-rate 0.5
  --gongga-rain-factor 0.8
)

# Convert this task's integer index into one rain-factor/sigma-x/sigma-y case.
N_X=${#SIGMA_X[@]}
N_Y=${#SIGMA_Y[@]}
N_TOTAL=$((${#RAIN_FACTORS[@]} * N_X * N_Y))

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} is outside the ${N_TOTAL} configured cases."
  exit 2
fi

I=${SLURM_ARRAY_TASK_ID}
I_RAIN=$(( I / (N_X * N_Y) ))
REMAINDER=$(( I % (N_X * N_Y) ))
I_X=$(( REMAINDER / N_Y ))
I_Y=$(( REMAINDER % N_Y ))

RAIN_FACTOR=${RAIN_FACTORS[$I_RAIN]}
SX=${SIGMA_X[$I_X]}
SY=${SIGMA_Y[$I_Y]}

# This name identifies the private task directory. The Badlands output folder
# inside it is additionally named by batch1 using brf/brx/bry/bsx/bsy tokens.
CASE_NAME="rainf${RAIN_FACTOR}_sigx${SX}_sigy${SY}"
WORK_DIR="${RUN_ROOT}/${CASE_NAME}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Case: ${CASE_NAME}"
echo "Parameters: factor=${RAIN_FACTOR}, sigma_x=${SX} km, sigma_y=${SY} km"

mkdir -p "${RUN_ROOT}"

# Copy the unified model directory. Existing completed outputs and logs are
# deliberately excluded so each task starts with clean model inputs.
rsync -a --delete \
  --exclude='h5/' \
  --exclude='batch_logs/' \
  --exclude='model1_*' \
  --exclude='slurm-dadu-*' \
  --exclude='daduhe_array_runs/' \
  "${TEMPLATE_DIR}/" "${WORK_DIR}/"

cd "${WORK_DIR}"

# No --prepare-only here: batch1 generates inputs, writes the case XML,
# launches Dadu_run.py itself, then stores the used XML in the output folder.
srun python3 "${WORK_DIR}/${MAIN_SCRIPT}" \
  --model-dir "${WORK_DIR}" \
  --settings-dir "${WORK_DIR}" \
  --uplift-dir "${WORK_DIR}" \
  "${FIXED_ARGS[@]}" \
  --bend-rain-factor "${RAIN_FACTOR}" \
  --bend-rain-sigma-x "${SX}" \
  --bend-rain-sigma-y "${SY}" \
  --overwrite

echo "End: $(date)"

