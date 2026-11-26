#!/usr/bin/env python

"""Praediktorenvergleich: APEX gegen ANIA gegen gemessene MICs.

Unabhaengig von unserem Modell und von Torres. Die Frage ist allein, wie gut
die beiden Werkzeuge die in Datenbanken gemessenen MICs treffen -- und damit,
wie viel Gewicht ihre Vorhersagen im Panel tragen duerfen.

VORAUSSETZUNGEN, beide aus Skript 14 bzw. von Hand:
    data_tracked/eval_sets/tool_val.csv        <- 14_eval_external_data.py
    data_tracked/eval_sets/ania_preds.csv      <- ANIA-Webtool, siehe unten

ANIA ist eine Weboberflaeche und laesst sich nicht aus dem Skript aufrufen.
Skript 14 legt dafuer eval_sets/tool_val_ania.fasta an; die wird hochgeladen
und das Ergebnis als ania_preds.csv zurueckgelegt. Erwartete Kopfzeile:

    ID,Sequence,Sequence Length,E. coli,S. aureus,P. aeruginosa

ANIA gibt LOG MIC aus, nicht MIC -- ergaenze_ania rechnet mit log_basis=10
zurueck. Dass es der Zehnerlogarithmus ist, ist an den Daten geprueft, die
Begruendung steht im Docstring der Funktion.

Zur Auswertung: der systematische Faktor wird NUR aus unzensierten Messungen
geschaetzt. Die zensierte Likelihood waere hier fehlspezifiziert, weil die
Schranke auf der Messung sitzt und die im Nenner des Residuums steht -- eine
Simulation mit 150 Replikationen zeigt dafuer 27 % statt 89 % HDI-Deckung.
Die rechtszensierten Messungen bekommen stattdessen ihre eigene, beantwortbare
Frage: liegt die Vorhersage ebenfalls oberhalb der Schranke? Einzelheiten im
Docstring von posterior_gitter.

Schritte:
0. Pfade und Vorbedingungen
1. Datensatz laden
2. APEX-Vorhersagen
3. ANIA-Vorhersagen
4. Messrauschen: die Grenze, die kein Praediktor unterbieten kann
5. Bayes-Tabellen
6. Abbildungen
7. Laufmetadaten
"""

import json
import platform
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from aguirre.config import EVAL_SET_DIR, PROJECT_ROOT, RESULTS_DIR, RUNS_DIR
from aguirre.extern import tool_val


def git_stand(projekt):
    """Commit und ob uneingecheckte Aenderungen drin sind."""
    def g(*a):
        return subprocess.run(["git", "-C", str(projekt), *a],
                              capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


#===0. Pfade und Vorbedingungen===
TOOL_VAL_CSV = EVAL_SET_DIR / "tool_val.csv"
ANIA_CSV = EVAL_SET_DIR / "ania_preds.csv"
VORHERSAGEN_CSV = EVAL_SET_DIR / "tool_val_vorhersagen.csv"

# Vor allem anderen pruefen. APEX rechnet ein paar Minuten, und die erst zu
# verbrennen, um danach an der fehlenden ANIA-Datei zu scheitern, waere
# aergerlich.
assert TOOL_VAL_CSV.is_file(), (
    f"{TOOL_VAL_CSV} fehlt. Erst scripts/14_eval_external_data.py laufen lassen.")
assert ANIA_CSV.is_file(), (
    f"{ANIA_CSV} fehlt. {EVAL_SET_DIR / 'tool_val_ania.fasta'} ins ANIA-Webtool "
    f"geben und das Ergebnis als {ANIA_CSV.name} dort ablegen.")

RUN_PATH = RUNS_DIR / f"toolval_{datetime.now():%Y%m%d_%H%M%S}"
RUN_PATH.mkdir(parents=True, exist_ok=True)
print(f"Eingaben:   {TOOL_VAL_CSV.name}, {ANIA_CSV.name}")
print(f"Ergebnisse: {RESULTS_DIR}")


#===1. Datensatz laden===
tv = pd.read_csv(TOOL_VAL_CSV)
print(f"[laden] {len(tv)} Zeilen, {tv['sequence'].nunique()} eindeutige Sequenzen")
print(f"[laden] je Spezies: {tv['species'].value_counts().to_dict()}")
print(f"[laden] Zensur: {tv['zensur'].value_counts().to_dict()}")


#===2. APEX===
# Rechnet einmal je eindeutiger Sequenz, nicht je Zeile -- dieselbe Sequenz
# steht fuer bis zu drei Spezies in der Tabelle. Die Staemme je Spezies werden
# geometrisch gemittelt, weil MIC auf einer Log-Skala lebt.
tv = tool_val.ergaenze_apex(tv)


#===3. ANIA===
tv = tool_val.ergaenze_ania(tv, ANIA_CSV)
tv.to_csv(VORHERSAGEN_CSV, index=False)
print(f"[tool_val] geschrieben: {VORHERSAGEN_CSV}")


#===4. Messrauschen===
# `spanne` ist das Verhaeltnis groesster zu kleinster unzensierter Messung
# derselben Sequenz fuer dieselbe Spezies, also die Streuung zwischen Laboren.
# Kein Praediktor kann ein sigma darunter erreichen -- er muesste die Messung
# besser vorhersagen, als sie sich selbst reproduziert. Ohne diese Zahl liest
# sich ein sigma von 0,7 wie ein schlechtes Ergebnis, mit ihr womoeglich wie
# ein fast optimales.
spanne = tv["spanne"].dropna()
rausch_grenze = float(np.log10(spanne.median()) / 2) if len(spanne) else float("nan")
print(f"[rauschen] {len(spanne)} Paare mit mehr als einer Messung")
if len(spanne):
    print(f"[rauschen] Spanne max/min: Median {spanne.median():.1f}x, "
          f"75. Perzentil {spanne.quantile(.75):.1f}x")
    print(f"[rauschen] untere Schranke fuer sigma: ca. {rausch_grenze:.2f} "
          "log10-Einheiten")


#===5. Bayes-Tabellen===
# Zweimal: einmal alles, einmal ohne die C-terminal amidierten. Die machen bei
# DRAMP zwei Drittel aus, und man will wissen, ob sie das Bild tragen.
#
# Die Spalte, auf die es ankommt, ist `faktor` mit HDI -- p_ueberschaetzt ist
# bei dieser Datenmenge fast immer 0 oder 1 und damit ohne Information.
# `r_vs_mess` prueft die Annahme, auf der die Auswertung ruht: haengt der
# Fehler vom MIC-Niveau ab, war das Weglassen der zensierten Messungen
# selektiv und der Faktor verschoben. Nahe null heisst unbedenklich.
tab = tool_val.bayes_tabelle(tv, out_csv=RESULTS_DIR / "toolval_bayes.csv")
tab_rein = tool_val.bayes_tabelle(
    tv, nur_unmodifiziert=True,
    out_csv=RESULTS_DIR / "toolval_bayes_unmodifiziert.csv")

print("\n--- alle Peptide ---")
print(tab.to_string(index=False))
print("\n--- ohne C-terminale Amidierung ---")
print(tab_rein.to_string(index=False))


#===6. Abbildungen===
# plot_baender ist die konventionelle Darstellung und hat das
# Regressions-zur-Mitte-Artefakt, das in die Bildunterschrift gehoert
# (tool_val.BILDUNTERSCHRIFT_BAENDER). plot_scatter hat es nicht und traegt
# den Faktor mit HDI direkt im Bild -- das ist die Abbildung, auf die man
# sich beruft.
fig_baender = tool_val.plot_baender(tv, out=RESULTS_DIR / "toolval_baender.png")
fig_scatter = tool_val.plot_scatter(tv, out=RESULTS_DIR / "toolval_scatter.png")


#===7. Laufmetadaten===
(RUN_PATH / "lauf_metadaten.json").write_text(json.dumps({
    "zeit": datetime.now().isoformat(timespec="seconds"),
    "git": git_stand(PROJECT_ROOT),
    "python": platform.python_version(),
    "eingaben": {"tool_val": str(TOOL_VAL_CSV), "ania": str(ANIA_CSV)},
    "mengen": {
        "zeilen": len(tv),
        "sequenzen": int(tv["sequence"].nunique()),
        "mit_apex": int(tv["mic_apex"].notna().sum()),
        "mit_ania": int(tv["mic_ania"].notna().sum()),
    },
    "rauschgrenze_sigma_log10": rausch_grenze,
    "faktoren": tab.to_dict(orient="records"),
}, indent=2), encoding="utf-8")
print(f"\nLaufmetadaten: {RUN_PATH / 'lauf_metadaten.json'}")
