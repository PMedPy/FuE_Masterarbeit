#!/usr/bin/env bash
#SBATCH --job-name=fue-train
#SBATCH -w bioinf-euryale.mni.thm.de
#SBATCH --gpus 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 0-12:00:00
#SBATCH --mail-user=pemd95@mni.thm.de
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Laeuft nativ auf dem Knoten mit der conda-Umgebung aus dem Home.
# Falls das scheitert: Container-Variante am Ende dieser Datei.
#
# Abschicken aus der Repo-Wurzel:
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR"

echo "Knoten   : $(hostname)"
echo "Job      : ${SLURM_JOB_ID:-interaktiv}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU      : keine"
python -c "import torch; print('torch', torch.__version__, '| build-cuda:', torch.version.cuda, '| verfuegbar:', torch.cuda.is_available())"

python scripts/06_train.py \
    --name "$RUN_NAME" \
    --esm "$ESM" \
    --epochs "$EPOCHS"

# ===========================================================================
# Container-Variante, falls der native Lauf scheitert:
# Die obigen conda-Zeilen und das srun ersetzen durch:
#
#   srun --container-image 'nvcr.io\#nvidia/pytorch:24.10-py3' \
#        --no-container-remap-root \
#        --container-mounts "$PROJECT":/workspace \
#        --container-workdir /workspace/wrath_of_god \
#        bash -lc '
#          pip install --user --no-cache-dir fair-esm pyyaml
#          export PYTHONPATH=/workspace/wrath_of_god/src
#          export TORCH_HOME=/workspace/.torch
#          export MPLCONFIGDIR=/tmp/mpl
#          python scripts/06_train.py --name cluster01 --esm 8M --epochs 50
#        '
# ===========================================================================
