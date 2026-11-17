#!/bin/bash
#SBATCH --job-name fue-eval3
#SBATCH -w bioinf-stheno.mni.thm.de
#SBATCH --gpus 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
#SBATCH --output slurm_logs/%x_%j.out
#SBATCH --error  slurm_logs/%x_%j.err

set -euo pipefail

PROJECT="$HOME/FuE_Masterarbeit"

cd "$PROJECT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fue-amp

export PYTHONPATH="$PROJECT/wrath_of_god/src"
export TORCH_HOME="$HOME/.cache/torch"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "$PROJECT/wrath_of_god/tmp"

echo "=== Job $SLURM_JOB_ID auf $(hostname) ==="
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

command -v mmseqs >/dev/null || { echo "FEHLER: mmseqs nicht im PATH"; exit 1; }
mmseqs version

python wrath_of_god/scripts/12_eval_3_small_vs_big.py

echo "=== fertig ==="
date -Is
