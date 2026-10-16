"""03 -- Clustern der Sequenzen und Aufteilung in Train/Val/Test.

Dritter Schritt. Baut aus der Paarliste einen Graphen: Knoten sind Sequenzen,
eine Kante verbindet zwei Sequenzen, deren Identitaet ueber dem oberen
Threshold liegt. Die Zusammenhangskomponenten dieses Graphen sind die Cluster.

Wozu ueberhaupt clustern: Aufgeteilt wird nach CLUSTERN, nicht nach einzelnen
Sequenzen. Sonst landen zwei fast identische Peptide in Train und Test, und die
Testmetrik misst Memorierung statt Generalisierung. Ein Cluster laesst sich
nicht aufteilen, ohne genau diese Leckage zu erzeugen.

IN  : data/all19K_hits.m8            (alternativ data/pairs_slim.parquet aus 02)
      data/all19K_unclustered.faa    (fuer die VOLLSTAENDIGE ID-Liste: Sequenzen
                                      ohne jeden Treffer stehen in keiner
                                      Paarzeile, bilden aber Einzelcluster und
                                      wuerden sonst im Split fehlen)
OUT : data_tracked/split_t075.csv    (Spalten: id, cluster, split -- klein, im Git)

Parameter: upper_seq2_threshold = 0.75  (Kantenschwelle im Graphen)
           lower_seq2_threshold = 0.3   (Vorfilter aus 02)
           ratios = (0.8, 0.1, 0.1), mode = "greedy", seed = 0

Der Dateiname wird aus upper_seq2_threshold abgeleitet, damit ein anderer
Threshold nicht die bestehende getrackte Tabelle ueberschreibt.
Laufzeit: wenige Minuten, dominiert vom Einlesen der 1-GB-m8.
"""

# %% ==0 Importe==
# Extern:
import pandas as pd

# Aus Projekt:
from aguirre.data import ClusterSplitter, read_fasta
from aguirre.config import PROJECT_ROOT, DATA_DIR, Config

cfg = Config()

# %% ==1 Pfade und Idempotenz-Schutz==
M8_PATH    = DATA_DIR / "all19K_hits.m8"
FASTA_PATH = DATA_DIR / "all19K_unclustered.faa"

# Threshold im Dateinamen: 0.75 -> "split_t075.csv"
SPLIT_PATH = (
    PROJECT_ROOT / "wrath_of_god" / "data_tracked"
    / f"split_t{int(cfg.upper_seq2_threshold * 100):03d}.csv"
)

# FORCE = True ueberschreibt eine getrackte Datei, von der 04 und das Training
# abhaengen. Danach muessen die Standardisierungsstatistiken neu berechnet werden.
FORCE = False
if SPLIT_PATH.exists() and not FORCE:
    print(f"{SPLIT_PATH.name} existiert schon -- uebersprungen (FORCE = True zum Neubauen)")
    raise SystemExit(0)

# %% ==2 Paare und IDs laden==

# Spaltenreihenfolge wie in 02 -- siehe --format-output in run_mmseqs_search
COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]

M8 = pd.read_csv(
    M8_PATH,
    sep     = "\t",
    names   = COLS,
    usecols = COLS[:8],
    dtype   = {"fident": "float32", "qcov": "float32", "tcov": "float32"},
)

pairs = M8[M8["query"] != M8["target"]].copy()                    # Selbsttreffer raus
pairs["ident"] = pairs["fident"] * pairs[["qcov", "tcov"]].max(axis=1)

# Die vollstaendige ID-Liste kommt aus der FASTA, nicht aus den Paaren.
# ids und seqs sind in Dateireihenfolge, ids[i] gehoert zu seqs[i].
ids, seqs = read_fasta(FASTA_PATH)
print(f"{len(ids)} Sequenzen, {len(pairs)} Paare")

# %% ==3 Clustern==

# min_ident aus cfg uebergeben, damit die 0.3 nur an EINER Stelle steht --
# ClusterSplitter hat sie sonst noch als eigenen Default.
cs = ClusterSplitter(pairs=pairs, ids=ids, min_ident=cfg.lower_seq2_threshold)

# Adjazenzmatrix bei der Kantenschwelle aufbauen und Zusammenhangskomponenten
# bestimmen. labels[i] ist die Clusternummer von ids[i]; sie liegt danach auch
# als cs.labels bereit, deshalb hier der Unterstrich.
n_comp, _ = cs.cluster(cfg.upper_seq2_threshold)
print(f"{n_comp} Cluster bei ident > {cfg.upper_seq2_threshold}")
# Fuer mehr Detail (Groesse des groessten Clusters etc.): print(cs.summary())

# %% ==4 Split zuweisen und speichern==

# First-Fit-Decreasing: groesste Cluster zuerst, jedes komplett in genau ein
# Set. Die Anteile stimmen dadurch nur ungefaehr -- ein grosses Cluster kann
# ein Set ueber sein Soll heben. Das ist der Preis dafuer, keine Leckage zu
# erzeugen. seed = 0 ist fix, der Split also reproduzierbar.
df_split = cs.assign_split(ratios=(0.8, 0.1, 0.1), mode="greedy", seed=0)

print(df_split["split"].value_counts())

df_split.to_csv(SPLIT_PATH, index=False)
print(f"gespeichert: {SPLIT_PATH.name}")
