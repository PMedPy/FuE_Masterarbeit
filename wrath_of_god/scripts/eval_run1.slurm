#!/bin/bash
#SBATCH --job-name fue-eval
#SBATCH -w bioinf-euryale.mni.thm.de
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64G
#SBATCH --time 0-06:00:00
#SBATCH --output slurm_logs/%x_%j.out
#SBATCH --error  slurm_logs/%x_%j.err

# ===========================================================================
# Evaluation: Trainingssatz gegen generierte Sequenzen.
#
# AUS DER REPO-WURZEL abschicken, slurm_logs/ muss existieren:
#   cd ~/FuE_Masterarbeit && mkdir -p slurm_logs
#   JOB=$(sbatch --parsable wrath_of_god/scripts/eval.slurm); echo $JOB
#
# VORHER EINMALIG auf dem LOGIN-Knoten (der Rechenknoten hat kein Internet):
#   export TORCH_HOME=~/FuE_Masterarbeit/.torch && mkdir -p $TORCH_HOME
#   export HF_HOME=~/FuE_Masterarbeit/.hf       && mkdir -p $HF_HOME
#   python -c "import esm; esm.pretrained.esm2_t6_8M_UR50D(); print('ESM ok')"
#   python -c "from transformers import AutoModelForCausalLM as M; \
#              from tokenizers import Tokenizer as T; \
#              M.from_pretrained('hugohrban/progen2-medium', trust_remote_code=True); \
#              T.from_pretrained('hugohrban/progen2-medium'); print('ProGen2 ok')"
#
# UND LOKAL vorher hochschieben (nicht in environment.yml / git):
#   rsync -avP --mkpath ~/.../wrath_of_god/data/models/apex_state_dicts.pt \
#        thm:~/FuE_Masterarbeit/wrath_of_god/data/models/
#   rsync -avP --mkpath ~/.../wrath_of_god/data_tracked/torres_train.fasta \
#        thm:~/FuE_Masterarbeit/wrath_of_god/data_tracked/
#   rsync -avP --mkpath ~/.../wrath_of_god/results/50k_samples/ \
#        thm:~/FuE_Masterarbeit/wrath_of_god/results/50k_samples/
# ===========================================================================

set -euo pipefail

PROJECT="$HOME/FuE_Masterarbeit"
RUN_NAME="eval_run1"

cd "$PROJECT"

# Ein SLURM-Job laeuft in einer nicht-interaktiven Shell, die ~/.bashrc nicht
# einliest -- ohne diese Zeile kennt er "conda activate" nicht.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fue-amp

# Macht "aguirre" auffindbar, ohne pip install -e .
export PYTHONPATH="$PROJECT/wrath_of_god/src"

# Gewichte NUR aus dem lokalen Cache -- der Rechenknoten darf nicht ins Netz.
export TORCH_HOME="$PROJECT/.torch"
export HF_HOME="$PROJECT/.hf"
export HF_HUB_OFFLINE=1          # scheitert sofort statt nach Minuten im Timeout
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$PROJECT/wrath_of_god/tmp"

echo "=== Job $SLURM_JOB_ID auf $(hostname) ==="
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# mmseqs braucht der KnownAMPSimilarityScorer und der DiversityFilter.
# Lieber hier scheitern als nach zwei Stunden Scoring.
command -v mmseqs >/dev/null || { echo "FEHLER: mmseqs nicht im PATH"; exit 1; }
mmseqs version

# KEIN srun: ein srun innerhalb von sbatch scheitert auf diesem Cluster mit
# "Task launch ... failed: Communication connection failure".
python wrath_of_god/scripts/10_eval_run1.py \
    --train     wrath_of_god/data_tracked/torres_train_sequences.csv \
    --generated wrath_of_god/results/50k_samples/ampdiffusion_650M_50k.csv \
    --out       "wrath_of_god/runs/${RUN_NAME}_$(date +%Y-%m-%d_%H%M)"

echo "=== fertig ==="
date -Is