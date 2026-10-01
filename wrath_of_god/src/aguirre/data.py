"""Abschnitt 1 + 3 -- Rohdaten und Dataset/Dataloader.

Abschnitt 1: Sequenzen laden und filtern.
    Eingang  : CSV/FASTA
    Ausgang  : list[str], NUR AMPs (keine Non-AMPs -- ein unkonditioniertes
               Diffusionsmodell lernt die Verteilung seiner Trainingsdaten)
    Schritte : Laengenfilter, nur die 20 Standard-Aminosaeuren,
               Duplikate entfernen, Train/Val-Split

Abschnitt 3: Dataset, das vorberechnete Embeddings ausliefert.
    __getitem__ gibt zurueck:
        emb           (L, D)   float32
        padding_mask  (L,)     bool   -- True = Padding, wird maskiert
    Nach dem Collate im Dataloader:
        emb           (B, L, D)
        padding_mask  (B, L)

Wichtig: Embeddings EINMAL vorberechnen und cachen, nicht im __getitem__
berechnen. Sonst laeuft ESM in jedem Epoch neu und jeder Testlauf wird zaeh.
"""

from torch.utils.data import Dataset


def load_sequences():
    """Sequenzen laden und filtern -> list[str]."""
    raise NotImplementedError


class PeptideDataset(Dataset):
    """Liefert vorberechnete ESM-Embeddings + Padding-Maske."""

    # TODO
