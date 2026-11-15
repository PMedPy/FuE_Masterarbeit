#!/bin/bash
#SBATCH --job-name fue-tva
#SBATCH -w bioinf-stheno.mni.thm.de
#SBATCH --gpus 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH --time 0-04:00:00
#SBATCH --output slurm_logs/%x_%j.out
#SBATCH --error  slurm_logs/%x_%j.err

# ===========================================================================
# Torres' generierte Sequenzen gegen unsere. Vier Gruppen, eine Kaskade.
#
# AUS DER REPO-WURZEL abschicken, slurm_logs/ muss existieren:
#   cd ~/FuE_Masterarbeit && mkdir -p slurm_logs
#   JOB=$(sbatch --parsable wrath_of_god/scripts/eval_torres_vs_aguirre.slurm)
#   echo $JOB && tail -F slurm_logs/fue-tva_${JOB}.out
#
# Knoten: stheno (A30, 128 Kerne). Der Flaschenhals ist mmseqs auf der CPU,
# nicht die GPU. Anderen Knoten nehmen: sbatch -w bioinf-yamata... davor.
#
# Gewichte liegen in ~/.cache -- einmalig auf dem LOGIN-Knoten gefuellt,
# der Rechenknoten hat kein Internet (siehe eval_run1.slurm).
#
# Unser 50k-Satz ist aus Eval-Run 1 vollstaendig gecacht; neu gerechnet
# wird praktisch nur Torres' Satz.
# ===========================================================================

set -euo pipefail

PROJECT="$HOME/FuE_Masterarbeit"
RUN_NAME="torres_vs_aguirre"

cd "$PROJECT"

# Nicht-interaktive Shell liest ~/.bashrc nicht -- ohne das kein conda activate.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fue-amp

export PYTHONPATH="$PROJECT/wrath_of_god/src"
export TORCH_HOME="$HOME/.cache/torch"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1          # scheitert sofort statt nach Minuten im Timeout
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MPLBACKEND=Agg            # kein Display auf dem Rechenknoten

mkdir -p "$PROJECT/wrath_of_god/tmp"

echo "=== Job $SLURM_JOB_ID auf $(hostname) ==="
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# mmseqs braucht der KnownAMPSimilarityScorer und der DiversityFilter.
command -v mmseqs >/dev/null || { echo "FEHLER: mmseqs nicht im PATH"; exit 1; }
mmseqs version

# KEIN srun: scheitert auf diesem Cluster mit "Task launch ... failed".
python wrath_of_god/scripts/11_eval_run_torres_vs_aguirre.py \
    --torres  wrath_of_god/data_tracked/torres_cellbiomaterial.csv \
    --aguirre wrath_of_god/results/50k_samples/ampdiffusion_650M_50k.csv \
    --out     "wrath_of_god/runs/${RUN_NAME}_$(date +%Y-%m-%d_%H%M)" \
    --top 30

echo "=== fertig ==="
date -Is
