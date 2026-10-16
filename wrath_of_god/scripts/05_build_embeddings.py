"""05 -- Vorberechnete ESM-2-Embeddings als Cache, ein File pro Split.

Fuenfter Schritt. Schickt jede Sequenz einmal durch ESM-2 und legt die
Embeddings als Tensor-Datei ab, damit das Training sie nur noch lesen muss.

Warum vorberechnen: ESM ist eingefroren (eval, requires_grad_(False)). Die
Embeddings sind damit eine deterministische Funktion der Sequenz und aendern
sich waehrend des gesamten Trainings nie. Sie in jeder Epoche neu zu rechnen
hiesse, dieselbe Zahl hundertmal auszurechnen -- bei 650M knapp zehn Minuten
pro Epoche fuer nichts.

Der Cache enthaelt ROHE Embeddings, nicht standardisierte. Nur so bleibt der
Schalter standardize=True/False im Data_coordinator moeglich, und mu/sigma
lassen sich austauschen, ohne ESM neu laufen zu lassen.

IN  : data_tracked/split_t075.csv
      data/all19K_unclustered.faa
OUT : data/cache/<split>-Set_<modell>_L<max_len>.pt   (3 Dateien pro Modell)

Inhalt jeder Datei (dict):
    "emb"      Tensor (N, L_max, D) float32 -- Padding ist exakt 0
    "lengths"  Tensor (N,) int64            -- echte Residuen je Sequenz
    "ids"      list[str] der Laenge N       -- AMP_x, in derselben Reihenfolge

Die drei Eintraege sind zeilenweise ausgerichtet: emb[i] gehoert zu ids[i] und
hat lengths[i] echte Residuen.

Zwei Entwurfsentscheidungen, die im Code leicht zu uebersehen sind:

1. Sortiert nach Sequenzlaenge. encode() laesst den batch_converter auf die
   laengste Sequenz IM BATCH padden -- unsortiert zieht eine 50er alle anderen
   mit hoch (gemessener Faktor 2,2 an Tokens). Sortiert wird im DataFrame,
   damit ID, Sequenz und Laenge zusammenbleiben; die Reihenfolge im Cache ist
   deshalb NICHT die der FASTA, sondern nach Laenge aufsteigend. Ueber "ids"
   bleibt die Zuordnung trotzdem eindeutig.

2. Der Zieltensor wird mit torch.empty vorbelegt und batchweise befuellt,
   statt eine Liste zu sammeln und am Ende zu stacken. torch.stack haette
   waehrend des Stapelns beide Kopien gleichzeitig im Speicher -- bei 650M
   7,4 statt 3,7 GB, was auf 16 GB geteiltem Speicher nicht mehr passt.
   Der assert pos == N prueft, dass jede Zeile befuellt wurde: torch.empty
   initialisiert nicht, eine uebersprungene Zeile enthielte Speichermuell.

Parameter (aus Config): esm_model_name, embed_dim, pep_max_len
Batchgroesse B: beeinflusst nur Tempo und Speicher, nicht das Ergebnis.

Gemessen auf einem M2 Air (16 GB, MPS) mit B = 16:
    8M   (D=320)   31 s     ->  1,2 GB fuer alle drei Splits
    650M (D=1280)  9 min 31 s -> 4,7 GB fuer alle drei Splits

Erwartete Groessen: train 15507, val 1939, test 1938 (Summe 19384 = FASTA).
Stimmen alle drei, ist die ID-Zuordnung ueber alle Splits intakt.

Der Cache liegt unter data/ und ist damit gitignored -- jede Maschine baut ihn
sich selbst. Fuer einen Trainingslauf auf einem Cluster gehoert er dorthin,
nicht uebers Netz geschoben.
"""

# %% ==0 Importe==
# Extern:

import pandas as pd
import torch
from tqdm.auto import tqdm

# Aus Projekt:
from aguirre.config import DATA_DIR, PROJECT_ROOT, Config, CACHE_DIR
from aguirre.data import read_fasta
from aguirre.esm_wrapper import ESMWrapper

cfg_8m   = Config(esm_model_name="esm2_t6_8M_UR50D",    embed_dim=320)
cfg_650m = Config(esm_model_name="esm2_t33_650M_UR50D", embed_dim=1280)

CACHE_DIR.mkdir(parents=True, exist_ok=True) # Noch nicht angelegt


# %% ==1 Pfade==
TRACKED_DIR = PROJECT_ROOT / "wrath_of_god" / "data_tracked"
SPLIT_PATH  = TRACKED_DIR / "split_t075.csv"
FASTA_PATH = DATA_DIR / "all19K_unclustered.faa"

# %%==2 Embedding-Schleife==
cfg = cfg_650m
df_split = pd.read_csv(filepath_or_buffer= SPLIT_PATH, sep= ",", names= ["ID", "Cluster", "Split"], header = 0)
df_split = df_split[["ID", "Split"]].copy()
ids, seqs = read_fasta(FASTA_PATH)
df_AMPs = pd.DataFrame({"ID": ids, "Sequenz": seqs})

splits = ["train", "val", "test"]
wrapper = ESMWrapper(cfg= cfg)

for split in splits:
    # Pfad
    EMBEDDING_PATH = CACHE_DIR / f"{split}-Set_{cfg.esm_model_name}_L{cfg.pep_max_len}.pt"

    # Filtern
    maske_0 = df_split["Split"] == f"{split}"
    df_splitting = df_split[maske_0].copy()
    # Maskierung mit AMP_IDs im Train-Set
    df_amp_s = df_AMPs[df_AMPs["ID"].isin(df_splitting["ID"])].copy()
    # Sortieren
    df_s = df_amp_s.assign(laenge = df_amp_s["Sequenz"].str.len())
    df_sorted = df_s.sort_values("laenge", kind= "stable").reset_index(drop= True)
    # Sortierte Größen:
    seqs_sorted = df_sorted["Sequenz"].to_list()
    ids_sorted = df_sorted["ID"].to_list()
    lengths_sorted = df_sorted["laenge"].to_list()
    B = 16 #Batchgröße
    start_positionen = list(range(0, len(seqs_sorted), B)) #Sortieren führt dazu, dass Sequenzen gleicher Länge zusammenkommen (encode dimension für alle Embeddings abhängig von längster Sequenz)
    batches = [seqs_sorted[i:i+B] for i in start_positionen]

    # Batchschleife:
    N = len(seqs_sorted)

    emb_lengths = []
    emb_all = torch.empty(N, cfg.pep_max_len, cfg.embed_dim, dtype = torch.float32)
    pos = 0
    for b in tqdm(batches, desc=split): 
        EMBs, len_s = wrapper.encode(b) # Shape (B,L_max, D)
        emb_all[pos : pos + len(b)] = EMBs
        pos += len(b)
        emb_lengths.extend(len_s)

    print(f"Anzahl von {split}-Set ist {N}")

    #Guardians
    assert pos == N, f"{pos} statt {N} Zeilen gefüllt"
    assert emb_lengths == lengths_sorted, "Reihenfolge verrutscht"

    #Speichern
    torch.save({ #Speichert tensoren in pt-Datei (siehe Cash in Data)
        "emb": emb_all,
        "lengths": torch.tensor(lengths_sorted),
        "ids": ids_sorted,
    }, EMBEDDING_PATH) 

