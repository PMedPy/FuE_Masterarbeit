"""04 -- Standardisierungsstatistiken der ESM-2-Embeddings.

Vierter Schritt. Berechnet Mittelwert und Standardabweichung PRO
Embedding-Dimension ueber alle echten Residuen des Trainingssplits.

Wozu: Das Diffusionsmodell addiert Standardnormalrauschen auf die Embeddings.
Damit Signal und Rauschen in derselben Groessenordnung liegen, muessen die
Embeddings auf mu = 0 und sigma = 1 gebracht werden. Dimensionsweise und nicht
global, weil die Skalen der ESM-Dimensionen stark auseinanderliegen: ein
globaler Skalar liesse einzelne "laute" Dimensionen den MSE-Loss dominieren.
Das Verhaeltnis max(sigma)/median(sigma) am Ende zeigt, wie schief die
Verteilung tatsaechlich ist.

Die Statistiken gehoeren zur DATENDEFINITION, nicht zum Modell: einmal
berechnet, abgespeichert, und beim Training wie beim Sampling als feste
Konstanten verwendet. Nur so ist die Transformation umkehrbar -- ein generiertes
Embedding muss vor dem Decoder mit genau diesen Werten zuruecktransformiert
werden (x = sigma * x_norm + mu).

Streaming: Es werden nur zwei Summen pro Dimension mitgefuehrt, nie alle
Embeddings. Der Speicherverbrauch ist damit unabhaengig von der
Datensatzgroesse -- alle 14.452 x 50 x 1280 Werte waeren rund 4 GB.

IN  : data_tracked/split_t075.csv
      data/all19K_unclustered.faa
OUT : data_tracked/standards_<modell>_L<max_len>.csv   (Spalten: dim, mean, std)
      dieselbe Datei als .json                         (Metadaten des Laufs)

Parameter (aus Config): esm_model_name, embed_dim, pep_max_len
Laufzeit: 8M etwa 40 s, 650M etwa 7 min (M2 Air, Batchgroesse 16, sortiert).

ACHTUNG: Die Statistiken gelten nur fuer genau dieses Modell und diesen
Laengenfilter. 8M-Werte sind fuer 650M-Embeddings wertlos und umgekehrt --
deshalb stehen beide im Dateinamen.
"""

# %% ==0 Importe==
# Extern:
import json
import random

import pandas as pd
import torch
from tqdm.auto import tqdm

# Aus Projekt:
from aguirre.config import DATA_DIR, TRACKED_DIR, Config
from aguirre.data import read_fasta
from aguirre.esm_wrapper import ESMWrapper

cfg = Config()

# %% ==1 Pfade==
TRACKED_DIR = PROJECT_ROOT / "wrath_of_god" / "data_tracked"
SPLIT_PATH  = TRACKED_DIR / f"split_t{int(cfg.upper_seq2_threshold * 100):03d}.csv"
FASTA_PATH  = DATA_DIR / "all19K_unclustered.faa"

# Modell UND Laengenfilter im Namen -- sonst ueberschreibt ein 8M-Lauf die
# 650M-Werte, ohne dass es auffaellt.
STANDARDS_PATH = TRACKED_DIR / f"standards_{cfg.esm_model_name}_L{cfg.pep_max_len}.csv"

print(f"Modell: {cfg.esm_model_name} (D = {cfg.embed_dim}) auf {cfg.device}")

# %% ==2 Trainingssequenzen zusammenstellen==

# header=0 ist notwendig: die Datei HAT eine Kopfzeile. Gibt man nur names= an,
# setzt pandas implizit header=None und liest die Kopfzeile als Datenzeile.
df_split = pd.read_csv(SPLIT_PATH, sep=",", names=["ID", "Cluster", "Split"], header=0)
df_split = df_split[df_split["Split"] == "train"]       # nur Train -- kein Leakage

# Die Zuordnung ID -> Sequenz MUSS aus derselben FASTA kommen, aus der der Split
# gebaut wurde: write_fasta vergibt die IDs positionsbasiert, eine anders
# gefilterte FASTA hat dieselben Namen fuer ANDERE Sequenzen.
ids, seqs = read_fasta(FASTA_PATH)
df_AMPs = pd.DataFrame({"ID": ids, "Sequenz": seqs})
df_AMPs = df_AMPs[df_AMPs["ID"].isin(df_split["ID"])].copy()

# Oberer Laengenfilter. Gehoert hierher und nicht in 01, weil pep_max_len eine
# Modelleigenschaft ist (Sequenzlaenge des Diffusionsmodells), keine der Daten.
df_AMPs = df_AMPs[df_AMPs["Sequenz"].str.len() <= cfg.pep_max_len]

AMPs = df_AMPs["Sequenz"].tolist()

# Sollwert fuer die Kontrolle unten -- aus den Daten abgeleitet, nicht
# hartcodiert. Stimmt er mit dem aufsummierten n ueberein, dann passen
# FASTA, Split, Filter und die Laengenrueckgabe von encode() alle zusammen.
n_residuen_soll = int(df_AMPs["Sequenz"].str.len().sum())
print(f"{len(AMPs)} Trainingssequenzen, {n_residuen_soll} echte Residuen")

# %% ==3 Batches bilden==

# Beeinflusst nur Tempo und Speicher, nicht das Ergebnis.
B = 16   # 8M: 256 | 650M: 16 (M2 Air, 16 GB gemeinsamer Speicher)

# Nach Laenge sortieren: encode() laesst den batch_converter auf die Laenge der
# laengsten Sequenz IM BATCH padden. Unsortiert zieht eine 50er-Sequenz alle
# anderen im Batch mit hoch -- gemessener Faktor 2,2 an verarbeiteten Tokens.
# Summen sind reihenfolgeunabhaengig, der Trick ist also gratis.
# (Fuer einen spaeteren Embedding-Cache waere das anders: dort braeuchte man
# die Zuordnung zu den IDs zurueck.)
AMPs_sortiert = sorted(AMPs, key=len)
batches = [AMPs_sortiert[i:i + B] for i in range(0, len(AMPs_sortiert), B)]
print(f"{len(batches)} Batches (erster {len(batches[0])}, letzter {len(batches[-1])})")

# %% ==4 Streaming-Schleife==

# Zwei Akkumulatoren der Form (D,). float64 auf der CPU: MPS kann kein float64,
# und encode() gibt ohnehin CPU-Tensoren zurueck.
summe_x  = torch.zeros(cfg.embed_dim, dtype=torch.float64)   # Summe der Werte
summe_x2 = torch.zeros(cfg.embed_dim, dtype=torch.float64)   # Summe der Quadrate
n = 0                                                        # Zahl echter Residuen

wrapper = ESMWrapper(cfg=cfg)

for b in tqdm(batches, desc=cfg.esm_model_name):
    EMBs, lengths = wrapper.encode(b)     # (B, L_max, D) float32 CPU, Padding = exakt 0
    x = EMBs.double()                      # ERST casten, dann summieren -- sonst
                                           # laeuft die Summe in float32 und der
                                           # float64-Akkumulator bringt nichts

    # Ueber Batch- und Positionsachse summieren, Dimensionsachse bleibt stehen.
    # Die Padding-Nullen tragen zu keiner der Summen bei (0 und 0^2 = 0), eine
    # Maske ist hier also nicht noetig.
    summe_x  += x.sum(dim=(0, 1))          # -> (D,)
    summe_x2 += (x ** 2).sum(dim=(0, 1))   # -> (D,)

    # n kommt aus den Sequenzlaengen, NICHT aus der Tensorform: sonst wuerde das
    # Padding mitgezaehlt, mu Richtung null gezogen und sigma verfaelscht.
    n += sum(lengths)

# %% ==5 Statistiken und Kontrollen==

mu = summe_x / n

# Varianz = Mittel der Quadrate minus Quadrat des Mittels. Mathematisch nie
# negativ, numerisch aber eine Differenz fast gleich grosser Zahlen: das
# Ergebnis kann bei -1e-12 landen, und sqrt() macht daraus ein stilles NaN, das
# sich ueber die Normalisierung bis in Loss und Gradienten fortpflanzt.
# clamp_min(0) setzt es auf den mathematisch korrekten Wert null -- und der
# Assert darunter macht aus einer null gewordenen Dimension einen sichtbaren
# Abbruch statt eines spaeteren inf beim Teilen.
sigma = (summe_x2 / n - mu ** 2).clamp_min(0).sqrt()

assert n == n_residuen_soll, f"Residuenzahl {n} != Soll {n_residuen_soll}"
assert (sigma > 0).all(), f"{(sigma == 0).sum().item()} Dimensionen mit sigma = 0"
assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()

# Gegenprobe an einer ZUFALLSstichprobe quer durch den Datensatz: nach der
# Normalisierung muessen Mittel ~ 0 und Std ~ 1 herauskommen. Ein nach Laenge
# sortierter Batch waere dafuer untauglich -- er ist zu homogen und liefert
# systematisch zu kleine Werte, weil er nur die Varianz INNERHALB einer
# Laengengruppe sieht und nicht die zwischen den Gruppen.
random.seed(0)
probe = random.sample(AMPs, min(128, len(AMPs)))
EMBs, lengths = wrapper.encode(probe)
x = EMBs.double()

# Maske der echten Positionen: Zeile mit Positionsnummern gegen Spalte mit
# Sequenzlaengen broadcasten -> (B, L_max), True wo Position < Laenge.
maske = torch.arange(x.shape[1])[None, :] < torch.tensor(lengths)[:, None]
z = (x[maske] - mu) / sigma                                   # (N_echt, D)
print(f"Gegenprobe: Mittel = {z.mean():+.4f}  Std = {z.std():.4f}   (erwartet 0 und 1)")

# Schiefe der Skalenverteilung -- begruendet die dimensionsweise Normalisierung.
# Nahe 1: ein globaler Skalar wuerde reichen. Deutlich darueber: nicht.
r = (torch.max(sigma) / sigma.median()).item()
print(f"sigma: min {sigma.min():.4f} | median {sigma.median():.4f} | "
      f"max {torch.max(sigma):.4f} | max/median = {r:.1f}")

# %% ==6 Speichern==

# Long-Format, eine Zeile pro Dimension. CSV statt Parquet: 1280 Zeilen sind
# winzig, und CSV ist lesbar, diffbar und git-tauglich.
df_standards = pd.DataFrame({"mean": mu.numpy(), "std": sigma.numpy()})
df_standards.to_csv(STANDARDS_PATH, index_label="dim")

# Metadaten als Sidecar: Ein CSV kann sie nicht tragen, aber ohne sie ist in
# sechs Monaten nicht mehr nachvollziehbar, zu welchem Lauf die Zahlen gehoeren.
meta = {
    "esm_model_name": cfg.esm_model_name,
    "embed_dim":      cfg.embed_dim,
    "pep_min_len":    cfg.pep_min_len,
    "pep_max_len":    cfg.pep_max_len,
    "split":          "train",
    "split_file":     SPLIT_PATH.name,
    "fasta_file":     FASTA_PATH.name,
    "n_sequences":    len(AMPs),
    "n_residues":     n,
}
STANDARDS_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2))

print(df_standards.head())
print(f"gespeichert: {STANDARDS_PATH.name} + .json")
