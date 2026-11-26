#!/usr/bin/env python

"""Externe Daten: Datenbankauszuege, Novelty-Referenz, Bewertungsmengen.

Baut alles, was NICHT aus unserem Modell kommt. Laeuft einmal durch und wird
danach nur wiederholt, wenn eine Datenbank neu gezogen wurde.

Die Rohdaten liegen untracked unter data/cache/DBs/. Was genau geholt wurde,
mit Version, Abrufdatum und SHA256, steht in data_tracked/extern/QUELLEN.md --
ohne diese Datei ist spaeter nicht feststellbar, auf welchem Stand eine
Auswertung beruht.

Lizenzen sind nicht einheitlich: DRAMP general steht unter CC BY 4.0, der
Patentsatz verlangt eine Autorisierung, DBAASP widerspricht sich und APD
raeumt gar nichts ein. Vor einer Veroeffentlichung Abschnitt D in
notizen/veroeffentlichung_checkliste.md lesen.

Manuelle Voraussetzungen, die dieses Skript NICHT beschafft:
    data/cache/DBs/dramp/general_amps.xlsx
    data/cache/DBs/dramp/patent_amps.xlsx
    data/cache/DBs/apd/naturalAMPs_APD2024a.fasta
    data/cache/DBs/uniprot/uniprotkb_*.fasta.gz
Die Bezugsquellen stehen in QUELLEN.md und in den Modul-Docstrings.

Laufzeit: beim ersten Mal rund 80 Minuten, fast alles davon der
DBAASP-Detailabruf. Danach wenige Minuten, weil der Cache greift.

Schritte:
0. Pfade
1. Einzelauszuege je Datenbank, mit Plausibilitaetspruefungen
2. Novelty-Referenz ueber alle Quellen
3. Splits aus dem Torres-Set
4. Decke: real_amps_test
5. Boden: Fragmente aus zytoplasmatischen UniProt-Proteinen
6. Boden: Zufallssequenzen
7. Vergleichsdatensatz fuer die Praediktoren, plus FASTA fuers ANIA-Webtool
8. Laufmetadaten
"""

import json
import platform
import subprocess
from datetime import datetime

import pandas as pd

from aguirre.config import (
    DATA_DIR,
    DB_CACHE_DIR,
    DB_EXTERN_DIR,
    EVAL_SET_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    SPLIT_DIR,
    TRACKED_DIR,
)
from aguirre.extern import apd, dbaasp, dramp, referenz, tool_val, uniprot, zufall
from aguirre.panel import schreibe_split_fasta


def git_stand(projekt):
    """Commit und ob uneingecheckte Aenderungen drin sind."""
    def g(*a):
        return subprocess.run(["git", "-C", str(projekt), *a],
                              capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


#===0. Pfade===
RUN_PATH = RUNS_DIR / f"extern_{datetime.now():%Y%m%d_%H%M%S}"
RUN_PATH.mkdir(parents=True, exist_ok=True)

UNIPROT_FASTA = (DB_CACHE_DIR / "uniprot"
                 / "uniprotkb_reviewed_true_AND_cc_scl_term_2026_11_25.fasta.gz")
N_BODEN = 10000

print(f"Cache:     {DB_CACHE_DIR}")
print(f"Auszuege:  {DB_EXTERN_DIR}")
print(f"Mengen:    {SPLIT_DIR}")
print(f"Eval-Sets: {EVAL_SET_DIR}")


#===1. Einzelauszuege je Datenbank===
# DBAASP ueber die REST-API, zwei Stufen. Beide sind resumebar -- was im Cache
# liegt, wird nicht neu geholt, ein Wiederholungslauf kostet also fast nichts.
# Stufe 1: 128 Listenseiten mit allen 25 542 Metadatensaetzen, rund eine Minute.
# Stufe 2: Details der Peptide zu den Zielspezies, rund 4 560 Abrufe, 70 Minuten.
# Der Speziesfilter ist der grosse Hebel -- ohne ihn waeren es 25 542 Abrufe.
dbaasp.hole_liste()
spezies_df = dbaasp.hole_spezies_listen()
kandidaten = dbaasp.filtere_kandidaten(spezies_df)
dbaasp.hole_details(kandidaten["id"].tolist())

messungen = dbaasp.parse_messungen(
    nur_messgroesse="MIC", out_pfad=DB_EXTERN_DIR / "dbaasp_messungen.csv")

# Plausibilitaetspruefungen. Keine Sondierung, sondern Bedingungen, die bei
# jedem Neubau gelten muessen -- sie haben beim ersten Mal drei Parserfehler
# gefunden. Harter Abbruch statt Warnung: ein stiller Fehler im Datensatz ist
# teurer als ein abgebrochener Lauf.
mw_pro_rest = (messungen["mw_dbaasp"] / messungen["sequence_length"]).dropna()
if not mw_pro_rest.between(40, 220).all():
    raise ValueError(
        "Molmasse pro Rest ausserhalb 40-220 g/mol: "
        f"min {mw_pro_rest.min():.1f}, max {mw_pro_rest.max():.1f}. "
        "Die leichteste Aminosaeure (Gly) hat 57, die schwerste (Trp) 186; "
        "Werte darueber deuten auf eine falsch zugeordnete Molmasse."
    )

roh = pd.to_numeric(messungen["concentration_raw"], errors="coerce")
maske = messungen["unit_raw"].eq("µg/ml") & roh.notna()
abweichung = (messungen.loc[maske, "mic_ug_ml"] - roh[maske]).abs().max()
if abweichung > 1e-9:
    raise ValueError(
        f"mic_ug_ml weicht von concentration_raw ab, maximal {abweichung}. "
        "Bei Einheit µg/ml muessen beide identisch sein -- weicht es ab, hat "
        "die Einheitenumrechnung in parse_messungen einen Fehler."
    )
print(f"[pruefung] MW pro Rest {mw_pro_rest.min():.0f}-{mw_pro_rest.max():.0f} g/mol, "
      f"Einheitenabweichung {abweichung:g} -- beides in Ordnung")

# DRAMP und APD sind manuelle Downloads. Bei DRAMP zwei CSVs aus derselben
# Datei, weil Novelty und Praediktorenvergleich verschiedene Filter brauchen:
# fuer Novelty zaehlt jede bekannte Sequenz, fuer den Vergleich nur saubere,
# unmodifizierte mit lesbarer MIC.
general = dramp.lade_general()
dramp.baue_novelty_csv(general)
dramp.baue_vergleich_csv(general)

# Der Patentsatz hat dieselben Spaltennamen wie general, deshalb laeuft
# baue_novelty_csv unveraendert darauf. Ohne ihn deckt die Referenz nur 53,3 %
# von Torres' Trainingssequenzen ab, mit ihm 99,6 %.
dramp.baue_novelty_csv(df=dramp.lade_patent(),
                       out_pfad=DB_EXTERN_DIR / "dramp_patent_novelty.csv")

apd.baue_novelty_csv()


#===2. Novelty-Referenz===
# Vereinigt DBAASP (Listencache, alle Sequenzen -- nicht der Detailcache, der
# nur die Zielspezies hat), DRAMP general, DRAMP patent und APD.
# Speist max_ident_bekannt im Panel.
bekannt = referenz.baue_novelty_referenz(mit_patent=True)


#===3. Splits aus dem Torres-Set===
# Geht ueber die ids aus split_t075.csv, NICHT ueber Positionen:
# torres_train.fasta hat 19 670 Sequenzen, der Split nur 19 384 -- beim
# Clustern sind 286 weggefallen.
#
# train (15 507) ist die Referenz fuer TrainSimilarityScorer -> max_ident_train.
# test (1 938) ist der reine Torres-Testsplit und heisst bewusst
# real_amps_test_TORRES: real_amps_test wird in Abschnitt 4 die erweiterte
# Decke. Der reine Split bleibt erhalten, damit der Vergleich "gleiche
# Verteilung wie train" moeglich bleibt.
schreibe_split_fasta("train", TRACKED_DIR / "split_t075.csv",
                     DATA_DIR / "all19K_unclustered.faa",
                     SPLIT_DIR / "real_amps_train.fasta")
schreibe_split_fasta("test", TRACKED_DIR / "split_t075.csv",
                     DATA_DIR / "all19K_unclustered.faa",
                     SPLIT_DIR / "real_amps_test_torres.fasta")


#===4. Decke===
# real_amps_test = Torres-Testsplit + amps_bekannt ohne train und val.
# val fliegt mit raus: es hat nie Gradienten erzeugt, aber ueber den
# Validation-Loss die Modellauswahl mitbestimmt.
#
# Schreibt FASTA und CSV. Die CSV ist die Fassung von Record: sie traegt die
# Spalte `kanonisch`, und das Panel bewertet nur die markierten Zeilen. Die
# Datei bleibt vollstaendig, damit die Zahl der Ausgeschlossenen berichtbar
# bleibt statt still zu verschwinden.
decke = referenz.baue_real_amps_test(ausschluss=("train", "val"))


#===5. Boden: UniProt===
# Ein Fragment je Protein aus zytoplasmatischen Nicht-AMP-Proteinen,
# cysteinfrei, Laenge an die Decke angepasst. referenz_fasta haengt
# max_ident_bekannt als Spalte an und verwirft nichts -- wo der Schnitt liegt,
# entscheidet man nach der Verteilung, nicht vorher.
lower = uniprot.baue_uniprot_boden(
    proteine_fasta=UNIPROT_FASTA,
    n=N_BODEN,
    referenz_fasta=DB_EXTERN_DIR / "amps_bekannt.fasta",
)


#===6. Boden: Zufall===
# Unterste Sprosse der Leiter: hat das Modell ueberhaupt etwas gelernt?
# Laengen aus derselben Quelle wie der UniProt-Boden, damit sich die beiden
# nur im Inhalt unterscheiden.
rand = zufall.baue_random_len(n=N_BODEN)


#===7. Vergleichsdatensatz fuer die Praediktoren===
# Gemessene MICs aus DBAASP und DRAMP, eine Zeile je (Sequenz, Spezies).
# Die FASTA geht ins ANIA-Webtool; ohne deren Ergebnis kann Skript 15 nicht
# starten.
tv = tool_val.baue_tool_val()
ania_fasta = tool_val.schreibe_ania_fasta(tv, EVAL_SET_DIR / "tool_val_ania.fasta")
print(f"\nNaechster Schritt von Hand: {ania_fasta} ins ANIA-Webtool, "
      f"Ergebnis als {EVAL_SET_DIR / 'ania_preds.csv'} ablegen, "
      "dann scripts/15_APEX_vs_ANIA.py starten.")


#===8. Laufmetadaten===
(RUN_PATH / "lauf_metadaten.json").write_text(json.dumps({
    "zeit": datetime.now().isoformat(timespec="seconds"),
    "git": git_stand(PROJECT_ROOT),
    "python": platform.python_version(),
    "eingaben": {
        "uniprot_fasta": str(UNIPROT_FASTA),
        "split_csv": str(TRACKED_DIR / "split_t075.csv"),
    },
    "mengen": {
        "dbaasp_messungen": len(messungen),
        "amps_bekannt": len(bekannt),
        "real_amps_test": len(decke),
        "uniprot_lower": len(lower),
        "random_len": len(rand),
        "tool_val": len(tv),
    },
}, indent=2), encoding="utf-8")
print(f"Laufmetadaten: {RUN_PATH / 'lauf_metadaten.json'}")
