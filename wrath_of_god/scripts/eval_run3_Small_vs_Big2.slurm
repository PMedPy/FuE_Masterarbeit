#!/bin/bash
#SBATCH --job-name fue-eval3
#SBATCH -w bioinf-stheno.mni.thm.de
#SBATCH --gpus 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
#SBATCH --output slurm_logs/%x_%j.out
#SBATCH --error  slurm_logs/%x_%j.err

# Eval 3 -- Aguirre Small (8M) gegen Big (650M), ungefiltert und nach Kaskade.
# Aus der Repo-Wurzel abschicken, slurm_logs/ muss existieren:
#   cd ~/FuE_Masterarbeit && mkdir -p slurm_logs
#   JOB=$(sbatch --parsable wrath_of_god/scripts/eval_run3_Big_vs_Small.slurm)
#   echo $JOB && tail -F slurm_logs/fue-eval3_${JOB}.out

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
export MPLBACKEND=Agg            # kein Display auf dem Rechenknoten

mkdir -p "$PROJECT/wrath_of_god/tmp"

echo "=== Job $SLURM_JOB_ID auf $(hostname) ==="
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

command -v mmseqs >/dev/null || { echo "FEHLER: mmseqs nicht im PATH"; exit 1; }
mmseqs version

# Eingaben pruefen, bevor fuenf Modelle geladen werden.
for f in wrath_of_god/results/50k_samples/aguirre-small_50k.faa \
         wrath_of_god/results/50k_samples/ampdiffusion_650M_50k.faa \
         wrath_of_god/data_tracked/torres_train.fasta ; do
    [ -f "$f" ] || { echo "FEHLER: $f fehlt"; exit 1; }
done

python wrath_of_god/scripts/12_eval_3_small_vs_big_2.py

echo "=== fertig ==="
date -Is
ls -1d "$PROJECT"/wrath_of_god/runs/eval_3_small_vs_big_* 2>/dev/null | tail -1 || true
