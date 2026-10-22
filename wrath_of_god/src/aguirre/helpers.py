"""Häufigste Helperfunctions"""

#==Importe==
import torch
from aguirre.config import Config
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