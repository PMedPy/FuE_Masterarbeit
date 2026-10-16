"""01 -- Rohsequenzen zu FASTA mit stabilen IDs.

Erster Schritt der Datenpipeline. Liest die Peptidsequenzen aus dem
Torres-Datensatz, filtert zu kurze heraus und schreibt sie als FASTA mit
fortlaufenden IDs (AMP_0, AMP_1, ...).

Wichtig: Die IDs entstehen HIER und nur hier. Alle folgenden Schritte
referenzieren sie. `write_fasta` vergibt sie positionsbasiert, also ist die
Reihenfolge der gefilterten Sequenzen Teil der Datendefinition. Jede Aenderung
am Filter verschiebt die Zuordnung ID -> Sequenz und macht alte Splittabellen
und Statistiken unbrauchbar -- dann muessen 02 bis 04 ebenfalls neu laufen.

IN  : data_tracked/torres_train_sequences.csv   (Spalte "sequence", 19.670 Zeilen)
OUT : data/all19K_unclustered.faa               (19.384 Sequenzen bei pep_min_len = 6)

Parameter (aus Config): pep_min_len
Laufzeit: Sekunden.

Ausfuehrung: `python scripts/01_build_fasta.py` oder zellenweise ueber die
"# %%"-Marker im VS-Code-Interactive-Window.
"""

# %% ==0 Importe==
# Extern:
import pandas as pd

# Aus Projekt:
from aguirre.data import write_fasta
from aguirre.config import DATA_RAW, DATA_DIR, Config

cfg = Config()

# %% ==1 Pfad und Idempotenz-Schutz==
FASTA_PATH = DATA_DIR / "all19K_unclustered.faa"

# FORCE = True vergibt ALLE IDs neu -- siehe Warnung im Docstring.
FORCE = False
if FASTA_PATH.exists() and not FORCE:
    print(f"{FASTA_PATH.name} existiert schon -- uebersprungen (FORCE = True zum Neubauen)")
    raise SystemExit(0)

# %% ==2 Laden, filtern, schreiben==

# Rohdaten aus dem Torres-Repo. Duplikate sind keine enthalten (geprueft:
# 19.670 Zeilen, 19.670 eindeutig), ein drop_duplicates() waere ein No-Op.
series = pd.read_csv(DATA_RAW)["sequence"]
n_vorher = len(series)

# Unterer Laengenfilter: unter 6 Residuen ist kein funktionales Peptid.
# Die OBERE Grenze (pep_max_len) wird hier absichtlich nicht angewandt -- sie
# gehoert zum Modell, nicht zu den Rohdaten, und greift erst in 04 bzw. im
# Training. So bleibt die FASTA fuer verschiedene Modell-Laengen nutzbar.
series = series[series.str.len() >= cfg.pep_min_len]

print(f"{n_vorher} Sequenzen -> {len(series)} nach Filter >= {cfg.pep_min_len} Residuen")

# Schreibt ">AMP_i\n<seq>\n" pro Eintrag, i = Position in dieser Liste.
write_fasta(data=series.to_list(), path=FASTA_PATH)
print(f"gespeichert: {FASTA_PATH.name}")
