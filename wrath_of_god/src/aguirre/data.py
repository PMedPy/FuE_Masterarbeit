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
import torch
from torch.utils.data import Dataset
from aguirre.config import DATA_DIR, PROJECT_ROOT
from pathlib import Path
import subprocess

#==Default Dateipfad für Fasta-Datei==
fasta_datei_pfad = DATA_DIR / "unclustered_data.faa"

def write_fasta(data: list[str], path: str | Path = fasta_datei_pfad):
    """Baut aus einer Liste von Peptidsequenzen eine Fasta-Datei"""

    with open(path, "w") as datei:
        for i, seq in enumerate(data): 
            header = f">AMP_{i}\n" # "_" Für späteren Split bei der Wiederfindung
            sequence = f"{seq}\n"
            datei.write(header)
            datei.write(sequence)
    return path

def run_mmseqs_search(
    fasta_path: str | Path,
    out_path: str | Path,
    tmp_dir: str | Path,
    sensitivity: float = 7.5,
    evalue: float = 10000,
    max_seqs: int = 2000,
) -> Path:
    """All-gegen-alle-Suche mit MMseqs2. Gibt den Pfad zur .m8-Datei zurueck."""
    #==Input in Paths ändern==
    fasta_path = Path(fasta_path)
    out_path = Path(out_path)
    tmp_dir = Path(tmp_dir)

    tmp_dir.mkdir(parents = True, exist_ok = True)
    out_path.parent

    #==Command==
    # mmseqs easy-search  <fasta>  <fasta>  <out.m8>  <tmp_dir>  -s 7.5  -e 10000  --max-seqs 2000
    cmd = [
        "mmseqs", 
        "easy-search", 
        str(fasta_path), 
        str(fasta_path), 
        str(out_path), 
        str(tmp_dir), 
        "-s",
        str(sensitivity),
        "-e",
        str(evalue),
        "--max-seqs",
        str(max_seqs)]
    
    #==run & return==
    subprocess.run(cmd, check= True)
    return out_path



def load_sequences():
    """Sequenzen laden und filtern -> list[str]."""
    raise NotImplementedError


class PeptideDataset(Dataset):
    """Liefert vorberechnete ESM-Embeddings + Padding-Maske."""

    # TODO
