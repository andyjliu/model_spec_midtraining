#!/usr/bin/env bash
#SBATCH --job-name=multitenet_aft_ctl
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END
#SBATCH --mail-user=andyliu@andrew.cmu.edu
#SBATCH --output=/home/andyliu/value-generalization/slurm_logs/multitenet_aft_controller_%j.out

set -uo pipefail

VALUEGEN_ROOT=${SLURM_SUBMIT_DIR:-/home/andyliu/value-generalization}
SCRIPT_DIR="$VALUEGEN_ROOT/external/model_spec_midtraining/exps/multitenet_aft"
cd "$VALUEGEN_ROOT"
mkdir -p slurm_logs/multitenet_aft_n2000
source .venvs/core/bin/activate

python "$SCRIPT_DIR/controller.py" run
