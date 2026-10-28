#!/usr/bin/env bash
#SBATCH --job-name=fue-train
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Aus der Repo-Wurzel abschicken, damit slurm_logs/ gefunden wird:
#   cd ~/FuE_Masterarbeit && mkdir -p slurm_logs && sbatch wrath_of_god/scripts/train.slurm

set -euo pipefail

# ---- anpassen ----------------------------------------------------------
PROJECT="${HOME}/FuE_Masterarbeit"
ENV_NAME="fue-amp"
RUN_NAME="cluster01"
EPOCHS=50
ESM="8M"
# -----------------------------------------------------------------------

cd "$PROJECT/wrath_of_god"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT/wrath_of_god/src"
export TORCH_HOME="$PROJECT/.torch"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}"
export PYTHONUNBUFFERED=1
mkdir -p "$MPLCONFIGDIR"

echo "Knoten   : $(hostname)"
echo "Job      : ${SLURM_JOB_ID:-interaktiv}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "keine GPU sichtbar"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

srun python scripts/06_train.py \
    --name "$RUN_NAME" \
    --esm "$ESM" \
    --epochs "$EPOCHS"
