"""02 -- All-gegen-alle-Suche mit MMseqs2 und Vorfilter der Paare.

Zweiter Schritt. Sucht jede Sequenz gegen jede andere und schreibt die Treffer
als .m8-Tabelle. Daraus entsteht eine schlanke Paarliste, die nur Paare
oberhalb des unteren Identitaets-Thresholds behaelt -- das spart in 03
Rechenzeit und Speicher, ohne das Ergebnis zu veraendern (Paare unterhalb
koennen bei keinem hoeheren Threshold mehr eine Kante bilden).

Braucht das externe Binary "mmseqs". Das steht bewusst NICHT in der
environment.yml: die Conda-Variante zieht llvm-openmp mit, das mit dem libomp
aus PyTorch kollidiert (OMP Error #15). Siehe Kommentar in der environment.yml.

IN  : data/all19K_unclustered.faa
OUT : data/all19K_hits.m8        (~1,0 GB -- nicht im Git)
      data/pairs_slim.parquet    (~67 MB  -- nicht im Git)

Parameter: in Zelle 2 explizit gesetzt statt als Default in data.py versteckt.
           Sie bestimmen, welche Sequenzen als aehnlich gelten, und damit den
           gesamten Split -- sie gehoeren in den Methodenteil.
           Plus lower_seq2_threshold (Config) als Vorfilter der Paare.

Laufzeit: etwa 20 Minuten, Ausgabe rund 1 GB. Der Idempotenz-Schutz steht
deshalb VOR dem Suchlauf, nicht danach.
"""

# %% ==0 Importe==
# Extern:
import shutil
import sys

import pandas as pd

# Aus Projekt:
from aguirre.data import run_mmseqs_search
from aguirre.config import DATA_DIR, TMP_DIR, Config

cfg = Config()

# %% ==1 Pfade und Vorbedingungen==
FASTA_PATH   = DATA_DIR / "all19K_unclustered.faa"
M8_PATH      = DATA_DIR / "all19K_hits.m8"
PARQUET_PATH = DATA_DIR / "pairs_slim.parquet"

FORCE = False   # True: m8 und Parquet neu bauen (~20 min, ~1 GB)

# Binary pruefen, bevor irgendetwas gerechnet wird: sonst laeuft das Skript an
# und scheitert erst im subprocess mit einer unverstaendlichen Meldung.
if shutil.which("mmseqs") is None:
    sys.exit(
        "mmseqs nicht gefunden. Installation (bewusst nicht in environment.yml,\n"
        "siehe Kommentar dort -- Conda-Variante kollidiert mit PyTorchs libomp):\n"
        "  macOS : brew install mmseqs2\n"
        "  Linux : https://github.com/soedinglab/MMseqs2/releases\n"
        "Pruefen: mmseqs version"
    )

# %% ==2 MMseqs2-Suche==

MMSEQS_PARAMS = dict(
    sensitivity      = 7.5,     # -s  hoch = empfindlicher, findet auch entfernte Verwandte
    evalue           = 10000,   # -e  absichtlich permissiv; gefiltert wird spaeter ueber ident
    max_seqs         = 10000,   # --max-seqs  maximale Treffer pro Query
    kmer_len         = 5,       # -k  kurze k-mere, weil AMPs selbst kurz sind (Median 21)
    spaced_kmer_mode = 0,
    mask             = 0,       # kein Low-Complexity-Masking: AMPs sind oft repetitiv,
                                # Masking wuerde echte Verwandtschaften verstecken
    cov_mode         = 5,
    coverage_rate    = 0.5,     # -c  Mindestabdeckung des Alignments
)

# Deaktiviert zwei Heuristiken, die bei sehr kurzen Sequenzen Treffer verwerfen,
# die wir behalten wollen.
EXTRA_ARGS = ["--comp-bias-corr", "0", "--min-ungapped-score", "0"]

if M8_PATH.exists() and not FORCE:
    print(f"{M8_PATH.name} existiert schon -- Suche uebersprungen (FORCE = True zum Neubauen)")
else:
    run_mmseqs_search(
        fasta_path = FASTA_PATH,
        out_path   = M8_PATH,
        tmp_dir    = TMP_DIR,
        extra_args = EXTRA_ARGS,
        **MMSEQS_PARAMS,
    )
    print(f"gespeichert: {M8_PATH.name}")

# %% ==3 Paare einlesen und vorfiltern==

# Die Spaltennamen muessen zum --format-output in run_mmseqs_search passen:
#   query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits
# Das ist NICHT das 12-spaltige m8-Standardformat (dort stehen mismatch,
# gapopen, qstart, qend, tstart, tend an Stelle von qlen/tlen/qcov/tcov).
# Reihenfolge und Anzahl sind fix -- aendert sich das Format in data.py, muss
# diese Liste mit.
COLS = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]

M8 = pd.read_csv(
    M8_PATH,
    sep     = "\t",                # m8 ist tab-separiert
    names   = COLS,                # die Datei hat keine Kopfzeile
    usecols = COLS[:8],            # evalue und bits brauchen wir nicht -> spart RAM
    dtype   = {"fident": "float32", "qcov": "float32", "tcov": "float32"},
)

# Jede Sequenz trifft sich selbst -> genau eine Selbstreferenz pro Sequenz.
# Ihre Anzahl muss also der Sequenzzahl entsprechen: eine billige
# Integritaetspruefung der .m8.
hits = M8[M8["query"] != M8["target"]].copy()
n_selbst = M8.shape[0] - hits.shape[0]
print(f"{n_selbst} Selbsttreffer entfernt (= Sequenzzahl), {hits.shape[0]} echte Paare uebrig")

# Identitaet = Sequenzidentitaet im Alignment x Abdeckung der schlechter
# abgedeckten Seite. fident allein genuegt nicht: ein 5-Residuen-Fragment kann
# zu 100 % identisch sein und trotzdem kaum etwas ueber die ganze Sequenz
# aussagen. max(qcov, tcov) straft solche Teiltreffer ab.
hits["ident"] = hits["fident"] * hits[["qcov", "tcov"]].max(axis=1)

# Vorfilter: alles unter dem unteren Threshold kann die Clusterung bei keinem
# spaeteren (hoeheren) Threshold mehr beeinflussen -> jetzt wegwerfen.
slim = hits[hits["ident"] > cfg.lower_seq2_threshold]
print(f"{len(slim)} Paare mit ident > {cfg.lower_seq2_threshold} behalten")

# %% ==4 Speichern==
if PARQUET_PATH.exists() and not FORCE:
    print(f"{PARQUET_PATH.name} existiert schon -- uebersprungen")
else:
    slim.to_parquet(PARQUET_PATH, engine="pyarrow")
    print(f"gespeichert: {PARQUET_PATH.name}")
