#!/usr/bin/env bash
# Baut einmalig das Enroot-Image fuer das Training.
# Auf dem LOGIN-Knoten ausfuehren (dort gibt es Netz).
#
#   bash scripts/build_container.sh
#
# Ergebnis: $IMAGE_DIR/fue-amp.sqsh  -- im SLURM-Job als
#           --container-image=<pfad>.sqsh referenziert.

set -euo pipefail

# ---- anpassen ----------------------------------------------------------
BASE_IMAGE="docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
IMAGE_DIR="${HOME}/containers"          # besser: ein Scratch-/Work-Verzeichnis
SCRATCH="${SCRATCH:-${HOME}/scratch}"   # Enroot braucht viel temporaeren Platz
# -----------------------------------------------------------------------

NAME="fue-amp"
mkdir -p "$IMAGE_DIR" "$SCRATCH/enroot"

export ENROOT_CACHE_PATH="$SCRATCH/enroot/cache"
export ENROOT_DATA_PATH="$SCRATCH/enroot/data"
export ENROOT_TEMP_PATH="$SCRATCH/enroot/tmp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

echo "== 1/4 Basis-Image importieren =="
enroot import -o "$IMAGE_DIR/base.sqsh" "$BASE_IMAGE"

echo "== 2/4 Container anlegen =="
enroot remove -f "$NAME" 2>/dev/null || true
enroot create --name "$NAME" "$IMAGE_DIR/base.sqsh"

echo "== 3/4 Zusatzpakete installieren =="
enroot start --root --rw "$NAME" bash -lc '
  pip install --no-cache-dir \
    fair-esm>=2.0.0 \
    pyyaml \
    pandas \
    pyarrow \
    matplotlib \
    seaborn \
    scipy \
    tqdm
  python -c "import torch, esm, yaml, matplotlib; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)"
'

echo "== 4/4 Fertiges Image exportieren =="
enroot export -o "$IMAGE_DIR/${NAME}.sqsh" "$NAME"
enroot remove -f "$NAME"
rm -f "$IMAGE_DIR/base.sqsh"

echo
echo "Fertig: $IMAGE_DIR/${NAME}.sqsh"
ls -lh "$IMAGE_DIR/${NAME}.sqsh"
