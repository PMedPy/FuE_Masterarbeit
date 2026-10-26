"""Häufigste Helperfunctions"""

#==Importe==
import torch
import pandas as pd
from aguirre.config import Config, TRACKED_DIR
cfg = Config()

# Baut Padding oder Validmask (tensor B,L) aus Längen (siehe Data_Coordinator)
def masker(lengths: torch.Tensor, valid_mask: bool = False, pep_max_len: int = cfg.pep_max_len):
    """(B,) Längen -> (B, L) bool, True = Padding, bei valid_mask = True ist False = Padding."""

    #Guardian
    too_long = lengths > pep_max_len # Bool (B,)
    if too_long.any(): # .any() schaut, ob ein Element der Bedingung widerspricht
        raise ValueError(
            f"{int(too_long.sum())} von {lengths.numel()} Sequenzen sind laenger \n"
            f"als pep_max_len = {pep_max_len}. Groesste gefundene Laenge: \n"
            f"{int(lengths.max())}."
        )

    #Maske
    padding_mask = torch.arange(pep_max_len, device= lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)
    if valid_mask:
        return ~padding_mask
    else: 
        return padding_mask


# Laedt mean/std der Embedding-Dimensionen aus der Standards-CSV
def load_standards(cfg_in: Config | None = None):
    """mean/std pro Embedding-Dimension laden -> Einzige Quelle.

    Der Sampler liest hier. Data_coordinator liest die CSV derzeit noch
    selbst (data.py, STANDARDS_PATH) -> zwei Lesepfade auf dieselbe Datei,
    das gehoert zusammengefuehrt. Bis dahin: wer 04_standardize.py neu
    laufen laesst, aendert beide Pfade gleichzeitig.

    Args:
        cfg_in: Config. None -> Modul-Default

    Returns:
        (mean, std), beide (D,) float32.
    """
    c = cfg if cfg_in is None else cfg_in
    path = TRACKED_DIR / f"standards_{c.esm_model_name}_L{c.pep_max_len}.csv"

    #Guardian
    if not path.exists():
        raise FileNotFoundError(
            f"Standards nicht gefunden: {path}\n"
            "Erst scripts/04_standardize.py laufen lassen."
        )
    df = pd.read_csv(path, header=0)
    for col in ("mean", "std"):
        if col not in df.columns:
            raise ValueError(
                f"Spalte '{col}' fehlt in {path}. Gefunden: {list(df.columns)}"
            )

    mean = torch.tensor(df["mean"].to_numpy(), dtype=torch.float32)   # (D,)
    std = torch.tensor(df["std"].to_numpy(), dtype=torch.float32)     # (D,)

    if mean.shape[0] != c.embed_dim:
        raise ValueError(
            f"Standards haben D={mean.shape[0]}, Config sagt {c.embed_dim}. "
            "Falsches ESM-Modell oder falsche CSV."
        )
    if (std <= 0).any():
        raise ValueError("std enthaelt Werte <= 0 -- Division wuerde explodieren.")

    return mean, std