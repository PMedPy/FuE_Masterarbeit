"""report.py -- Tabellen und Plots eines Panel-Runs.

Bezugsgroesse ist die ungefilterte Rohmenge. Hier wird nichts verworfen: wer
erst filtert und dann zaehlt, wie viele der Uebrigen das Panel als AMP
erkennt, misst, was die Filter moegen -- nicht, was das Modell kann.

Alle Tabellen haben Sets als Zeilen. Ein Torres-vs-Aguirre-Run hat:

    torres, aguirre_small, aguirre_big      die Kandidaten
    train_amps                              Decke (echte, validierte AMPs)
    uniprot                                 Boden (Spezifitaetskontrolle)
    random_len                              Boden (Nullmodell, laengenangepasst)
    shuffled                                Komposition ohne Ordnung

Groessenordnung aus Beierle et al. 2026 (4-Tool-Panel, Einstimmigkeit):
echte AMPs 77,8 %, generative Modelle 62-67 %, UniProt 0,79 %. Die Decke
liegt klar unter 100 %, deshalb bekommt jede Rate zusaetzlich die Spalte
"Anteil der Decke" = eigene Rate / train_amps-Rate.

Alles wird aus der gespeicherten Score-Tabelle erzeugt
(scores.parquet eines Eval-Laufs), nie direkt aus den Scorern -- nur so ist
ein Plot ohne Neuberechnung reproduzierbar. Dieses Modul liest die Tabelle
NICHT selbst: es bekommt fertige SequenceSets. Wer sie laedt, entscheidet
das aufrufende Skript.

Festlegungen dieses Moduls (alle aenderbar, ohne Funktionen anzufassen):

    SPALTEN     Alle Spaltennamen stehen an EINER Stelle. Das Schema aus
                Abschnitt 7 der Spec weicht von den Namen in evaluation.py ab;
                solange das nicht entschieden ist, bleibt hier die
                Code-Variante. Umstellen = ein Dict tauschen, siehe unten.
    DECKE       Set, gegen das "Anteil der Decke" normiert wird. Default ist
                der Holdout-Split: das Modell hat diese AMPs nie gesehen, eine
                hohe Panelrate dort kann also nicht durch Memorisierung
                erklaert werden. Jede Tabellenfunktion nimmt es als Parameter.
    BOEDEN      Spezifitaetskontrollen. Bekommen in Plots gestrichelte Linien,
                die Decke eine gepunktete -- cfg.plot_farben hat sechs Farben
                bei sieben Sets, die Linienart traegt die Unterscheidung.
    Kandidaten  sind OPTIONAL. Fehlt ein Kandidatenset (z.B. die noch nicht
                entdoppelte Torres-Referenz), fehlt nur seine Zeile. Fehlende
                BASELINES dagegen lehnt build_report ab -- ohne Decke ist T2
                nicht interpretierbar.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# _slug und _gruppenfarben sind in evaluation mit Unterstrich als privat
# markiert. Abschnitt 6 der Spec erlaubt, Fehlendes nach panel zu kopieren --
# hier wird bewusst importiert statt kopiert: Dateinamen und Gruppenfarben
# muessen zu den bestehenden zwoelf Laeufen passen, und zwei Kopien derselben
# Farbzuordnung laufen frueher oder spaeter auseinander.
from aguirre.evaluation import (
    CompositionComparison,
    PositionalComparison,
    SequenceSet,
    _gruppenfarben,
    _slug,
    build_stem,
    cfg,
    cliffs_delta,
    cliffs_label,
    mann_whitney,
)
from aguirre.panel.aggregate import (
    AgreementMatrix,
    VoteRule,
    bh_correct,
    cohens_kappa,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# Spaltenregistry -- die einzige Stelle, an der Spaltennamen stehen
# ---------------------------------------------------------------------------
#
# Hintergrund: Abschnitt 7 der Spec erklaert ein Schema <tool>_<groesse> fuer
# verbindlich, weil daran der Merge der beiden Eval-Haelften haengt. Die Namen
# in evaluation.py folgen der umgekehrten Konvention, und evaluation.py ist
# eingefroren -- ein Rename dort wuerde zusaetzlich jeden Score-Cache
# ungueltig machen (Evaluator._load_cache prueft scorer.columns gegen die
# Cache-Header, APEX und ProGen2 wuerden komplett neu rechnen).
#
# Deshalb: intern die Code-Namen, und beim Schreiben der scores.parquet einmal
# umbenennen. Faellt die Entscheidung anders aus, genuegt hier
#     SPALTEN = SPALTEN_SPEC
# -- keine Funktion unten muss angefasst werden.

SPALTEN: dict[str, tuple[str, ...] | str] = {
    "vote":         ("p_macrel", "p_amplify", "p_antibp3"),
    "gram":         "gram_antibp3",
    "apex_mittel":  "mic_apex",
    "apex_geom":    "mic_apex_geom",
    "ania":         ("mic_ania",),
    "ppl":          "ppl_progen2",
    "pppl":         "pppl_esm2",
    "shuffle_logR": "logR_shuffle",
    "shuffle_p":    "p_shuffle",
    "lowcomplex":   "flag_lowcomplex",
    "entropie":     "aa_entropy",
    "id_train":     "id_max_train",
    "id_bekannt":   "max_ident_bekannt",
    "hemo":         ("p_hemo_hemopi", "p_hemo_macrel"),
}

SPALTEN_SPEC: dict[str, tuple[str, ...] | str] = {
    "vote":         ("macrel_p", "amplify_p", "antibp3_p"),
    "gram":         "antibp3_gram",
    "apex_mittel":  "apex_mic_mean",
    "apex_geom":    "apex_mic_geom",
    "ania":         ("ania_mic_ec", "ania_mic_pa", "ania_mic_sa"),
    "ppl":          "progen2_ppl",
    "pppl":         "esm150_pppl",
    "shuffle_logR": "shuffle_logR",
    "shuffle_p":    "shuffle_p",
    "lowcomplex":   "lowcomplex_flag",
    "entropie":     "aa_entropy",
    "id_train":     "id_max_train",
    "id_bekannt":   "id_max_known",
    "hemo":         ("hemopi_p", "macrel_hemo_p"),
}


def spalten_flach(
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
) -> list[str]:
    """Alle Score-Spalten der Registry als flache Liste, Reihenfolge stabil."""
    reg = SPALTEN if registry is None else registry
    flach: list[str] = []
    for wert in reg.values():
        flach.extend([wert] if isinstance(wert, str) else list(wert))
    return flach


# ---------------------------------------------------------------------------
# Rollen der Sets
# ---------------------------------------------------------------------------

# Nenner fuer "Anteil der Decke". Der Holdout-Split (split == "test" in
# split_t075.csv, bei 0,75 Identitaet geclustert) statt des Trainingssets:
# diese AMPs hat das Modell nie gesehen, eine hohe Panelrate dort laesst sich
# also nicht mit Memorisierung wegerklaeren. Als Parameter in jeder Funktion.
DECKE = "test_amps"

# Spezifitaetskontrollen. Jede beantwortet eine andere billige Erklaerung fuer
# ein gutes Ergebnis: uniprot "sagt das Panel zu jedem Protein ja?",
# random_len "reicht die richtige Laenge?", shuffled "liest es ueberhaupt die
# Reihenfolge, oder nur die Komposition?".
BOEDEN: tuple[str, ...] = ("uniprot", "random_len", "shuffled")

# Optional -- fehlt eines, fehlt nur seine Zeile.
KANDIDATEN: tuple[str, ...] = ("torres", "aguirre_small", "aguirre_big")

# Ohne diese ist keine Rate interpretierbar; build_report lehnt ihr Fehlen ab.
PFLICHT_BASELINES: tuple[str, ...] = (DECKE,) + BOEDEN

REFERENZ_PAARE = (
    ("aguirre_small", "torres"),
    ("aguirre_big", "torres"),
    ("aguirre_big", "aguirre_small"),
)


def linienstil(set_name: str, decke: str = DECKE) -> str:
    """Linienart nach Rolle des Sets: Kandidat, Boden oder Decke.

    cfg.plot_farben hat sechs Farben, das Panel sieben Sets -- eine Farbe
    wiederholt sich zwangslaeufig. Die Linienart traegt die Unterscheidung,
    damit in P1 und P2 trotzdem sieben Kurven lesbar bleiben.
    """
    if set_name == decke:
        return ":"
    if set_name in BOEDEN:
        return "--"
    return "-"


def _kurz(spalte: str) -> str:
    """Spaltenname ohne p_-Praefix bzw. _p-Suffix, fuer Tabellenkoepfe.

    Funktioniert fuer beide Schemata: "p_macrel" und "macrel_p" werden beide
    zu "macrel". Dadurch heissen die kappa-Spalten in T3 gleich, egal welche
    Registry aktiv ist.
    """
    if spalte.startswith("p_"):
        return spalte[2:]
    if spalte.endswith("_p"):
        return spalte[:-2]
    return spalte


# Torres' Selektionsschwelle in µmol/L. Dieselbe Zahl steckt als Default in
# MICFilter und in APEXScorer.inaktiv_schwelle -- hier nur fuer die Spalte
# "% mit mittlerer MIC <= 64" in T4.
MIC_SCHWELLE = 64.0

# Schwelle, ab der ein Haemolyse-Score als "haemolytisch" zaehlt (T6).
HEMO_SCHWELLE = 0.5

# Novelty-Schwellen aus Torres' Kriterien (T6).
NOVELTY_SCHWELLEN = (0.60, 0.40)

# Signifikanzniveau fuer "% mit p_shuffle < 0,05" in T5.
SHUFFLE_ALPHA = 0.05


def _rolle(name: str, decke: str = DECKE) -> str:
    """Decke, Boden oder Kandidat -- als Spalte in jeder Tabelle.

    Ohne sie muss ein Gutachter die Rollen aus den Setnamen raten, und
    "Anteil der Decke" ist in der Deckenzeile selbst definitionsgemaess 1.
    """
    if name == decke:
        return "Decke"
    if name in BOEDEN:
        return "Boden"
    return "Kandidat"


# Laengengates der Fremdtools, Spalte -> {"min_len": int|None, "max_len": int|None}.
#
# ABSICHTLICH LEER. Die Gates gehoeren laut Abschnitt 6 in den Scorer, nicht
# in den Aufrufer -- sie stehen als Klassenattribute in panel/external.py und
# sind dort noch nicht ausgefuellt. Solange dieses Dict leer ist, berichtet T1
# die Zahl der NaN pro Scorer, aber nicht ihren Grund. Sobald die Werte in
# external.py stehen, hier eintragen (oder von dort importieren), dann
# ergaenzt T1 die Spalten nan_kurz_/nan_lang_/nan_sonstig_ von selbst.
#
# Der Grund "nicht-kanonisch" aus der Spec kann bei uns nicht auftreten:
# SequenceSet.__post_init__ lehnt Sequenzen mit Zeichen ausserhalb
# cfg.AS_alphabet schon beim Bauen ab. Was ANIA verwirft (BJOUXZ), kommt also
# gar nicht erst in ein Set -- sofern AS_alphabet die 20 Standard-AS sind.
GATES: dict[str, dict[str, int | None]] = {}


def _verteilung(werte: pd.Series, praefix: str) -> dict:
    """Median, IQR und n einer Metrik. NaN zaehlt nicht in den Nenner.

    IQR als q1/q3 statt als eine Zahl: die Spannweite allein verschweigt, ob
    die Verteilung nach oben oder unten schief liegt, und bei MIC ist genau
    das die interessante Richtung.
    """
    gueltig = werte.dropna()
    n = len(gueltig)
    return {
        f"n_{praefix}": n,
        f"median_{praefix}": float(gueltig.median()) if n else np.nan,
        f"q1_{praefix}": float(gueltig.quantile(0.25)) if n else np.nan,
        f"q3_{praefix}": float(gueltig.quantile(0.75)) if n else np.nan,
    }


def _prozent(maske: pd.Series, bewertbar: pd.Series) -> float:
    """Anteil in Prozent. Nenner sind die BEWERTBAREN Zeilen, nicht alle.

    Sonst sinkt jeder Anteil mit der Coverage, und ein Tool, das die Haelfte
    der Sequenzen nicht annehmen konnte, sieht aus wie eines, das sie
    abgelehnt hat.
    """
    n = int(bewertbar.sum())
    if not n:
        return np.nan
    return 100.0 * float((maske & bewertbar).sum()) / n


def _spalte(s: SequenceSet, name: str) -> pd.Series:
    """Spalte als float, oder eine reine NaN-Spalte, wenn sie fehlt.

    Fehlende Score-Spalten brechen eine Tabelle nicht ab -- anders als
    fehlende Vote-Spalten, die VoteRule ablehnt. Begruendung: ein Lauf ohne
    ANIA oder ohne id_max_train ist ein gueltiger Lauf mit weniger Spalten,
    ein Vote ueber zwei statt drei Tools ist eine andere Groesse.
    """
    if name in s.data.columns:
        return s.data[name].astype(float)
    return pd.Series(np.nan, index=s.data.index, dtype=float)


def panel_stem(sets: Iterable[str], datum: date | None = None) -> str:
    """Gemeinsamer Dateinamen-Stamm eines Panel-Laufs.

    Benutzt build_stem aus evaluation, damit die Dateien neben den bisherigen
    zwoelf Laeufen chronologisch einsortieren und gleich aussehen.
    """
    return build_stem("panel", sets, datum=datum)


@dataclass
class PanelReport:
    """Ergebnis eines Panel-Runs: benannte Tabellen plus benannte Figures.

    Analog zu ComparisonResult in evaluation.py, damit sich beide gleich
    speichern lassen.

    Attributes:
        tabellen: {"T2_panel": DataFrame, ...}
        figures: {"P5_shuffle": Figure, ...}
        manifest: Checkpoints, Sampler-Settings, Seeds, Toolversionen,
            Set-Groessen, Datum, Git-Commit. Ohne das ist ein Run nicht
            reproduzierbar.
    """

    tabellen: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, plt.Figure] = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)
    stem: str = "panel"

    def save(
        self,
        out_dir: Path,
        stem: str | None = None,
        dpi: int | None = None,
        pdf: bool = True,
        close: bool = True,
    ) -> list[Path]:
        """Tabellen als CSV, Figures als PNG + PDF, Manifest als JSON.

        Dateinamen folgen build_stem aus evaluation.py, damit die Dateien
        neben den bisherigen Laeufen einsortieren: <stem>__<name>.<endung>.
        Die Suffix-Konvention mit doppeltem Unterstrich ist dieselbe wie in
        ComparisonResult.save.

        Args:
            out_dir: Zielordner, wird angelegt. Fuer Eval-Laeufe cfg-seitig
                RESULTS_DIR -- NICHT runs/, dort liegen die Trainingslaeufe.
            stem: ueberschreibt self.stem, falls gesetzt.
            dpi: Aufloesung der PNGs. None -> cfg.plot_dpi.
            pdf: zusaetzlich als PDF speichern. Fuer die Ausarbeitung noetig,
                weil ein PNG beim Skalieren in LaTeX unscharf wird.
            close: Figures nach dem Speichern schliessen. Im Skript True,
                sonst laeuft bei sieben Sets der Speicher voll.

        Returns:
            Liste der geschriebenen Pfade.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamm = stem or self.stem
        dpi = cfg.plot_dpi if dpi is None else dpi
        geschrieben: list[Path] = []

        for name, tab in self.tabellen.items():
            pfad = out_dir / f"{stamm}__{_slug(name)}.csv"
            tab.to_csv(pfad, index=False)
            geschrieben.append(pfad)

        for name, fig in self.figures.items():
            basis = out_dir / f"{stamm}__{_slug(name)}"
            fig.savefig(basis.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            geschrieben.append(basis.with_suffix(".png"))
            if pdf:
                fig.savefig(basis.with_suffix(".pdf"), bbox_inches="tight")
                geschrieben.append(basis.with_suffix(".pdf"))
            if close:
                plt.close(fig)

        # default=str, damit Path und date nicht am JSON-Encoder scheitern --
        # ohne das faellt der Lauf erst nach allen Tabellen um.
        pfad = out_dir / f"{stamm}__manifest.json"
        pfad.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        geschrieben.append(pfad)
        return geschrieben


# ---------------------------------------------------------------------------
# 1. Tabellen
# ---------------------------------------------------------------------------

def tabelle_t1_sets(
    sets: Mapping[str, SequenceSet],
    train_sequences: Iterable[str] | None = None,
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    gates: Mapping[str, Mapping[str, int | None]] | None = None,
    decke: str = DECKE,
) -> pd.DataFrame:
    """T1 -- Set-Uebersicht und Coverage.

    n, n unique, % Duplikate, % exakt im Trainingsset, Laenge (Median, IQR,
    min, max), und pro Scorer die Zahl der NaN mit Grund (zu kurz / zu lang /
    nicht-kanonisch / nicht in CSV).

    Die Tabelle, die alle vergessen und nach der jeder Gutachter fragt.

    Args:
        sets: die Sets des Laufs, Name -> SequenceSet.
        train_sequences: Trainingssequenzen des MODELLS fuer die Spalte
            "% exakt im Trainingsset". None -> Spalte bleibt NaN. Bewusst ein
            eigener Parameter und nicht sets["train_amps"]: seit die Decke der
            Holdout-Split ist, muss das Trainingsset gar nicht mehr unter den
            bewerteten Sets sein.
        registry: Spaltenregistry, None -> SPALTEN.
        gates: Laengengates je Spalte fuer die Aufschluesselung der NaN,
            None -> GATES (leer, siehe dort). Ohne Gates nur Zahl und Anteil.
        decke: Name des Deckensets, nur fuer die Rollenspalte.

    Returns:
        Eine Zeile je Set. Coverage-Spalten heissen cov_<tool> (Prozent
        bewertbar) und nan_<tool> (absolute Zahl).
    """
    reg = SPALTEN if registry is None else registry
    tore = GATES if gates is None else gates
    train = set(train_sequences) if train_sequences is not None else None

    # Nur Spalten, die in mindestens einem Set vorkommen. Ein Lauf ohne ANIA
    # soll keine leeren ANIA-Spalten in der Tabelle haben.
    score_spalten = [
        c for c in spalten_flach(reg)
        if any(c in s.data.columns for s in sets.values())
    ]

    zeilen: list[dict] = []
    for name, s in sets.items():
        d = s.data
        n = len(d)
        n_uniq = int(d["sequence"].nunique()) if n else 0
        laenge = d["length"]

        zeile: dict = {
            "set": name,
            "rolle": _rolle(name, decke),
            "n": n,
            "n_unique": n_uniq,
            "dupl_prozent": 100.0 * (n - n_uniq) / n if n else np.nan,
            "in_train_prozent": (
                100.0 * float(d["sequence"].isin(train).mean())
                if train is not None and n else np.nan
            ),
            "len_median": float(laenge.median()) if n else np.nan,
            "len_q1": float(laenge.quantile(0.25)) if n else np.nan,
            "len_q3": float(laenge.quantile(0.75)) if n else np.nan,
            "len_min": int(laenge.min()) if n else np.nan,
            "len_max": int(laenge.max()) if n else np.nan,
        }

        for c in score_spalten:
            kurz = _kurz(c)
            if c not in d.columns or not n:
                zeile[f"cov_{kurz}"] = np.nan
                zeile[f"nan_{kurz}"] = np.nan
                continue
            fehlend = int(d[c].isna().sum())
            zeile[f"cov_{kurz}"] = 100.0 * (n - fehlend) / n
            zeile[f"nan_{kurz}"] = fehlend

            tor = tore.get(c)
            if tor is None:
                continue
            # Aufschluesselung nur ueber die tatsaechlich fehlenden Zeilen --
            # eine zu kurze Sequenz, die trotzdem einen Score hat, ist kein
            # Coverage-Verlust und gehoert hier nicht gezaehlt.
            fehlt_maske = d[c].isna()
            min_len, max_len = tor.get("min_len"), tor.get("max_len")
            kurz_n = int((fehlt_maske & (laenge < min_len)).sum()) if min_len else 0
            lang_n = int((fehlt_maske & (laenge > max_len)).sum()) if max_len else 0
            zeile[f"nan_kurz_{kurz}"] = kurz_n
            zeile[f"nan_lang_{kurz}"] = lang_n
            zeile[f"nan_sonstig_{kurz}"] = fehlend - kurz_n - lang_n

        zeilen.append(zeile)

    return pd.DataFrame(zeilen)


def tabelle_t2_panel(
    sets: Mapping[str, SequenceSet],
    regel: VoteRule | None = None,
    decke: str = DECKE,
) -> pd.DataFrame:
    """T2 -- die Kernaussage.

    Pro Set: % positiv je Tool bei der Schwelle, dann 3/3, >=2/3, >=1/3, jede
    Zahl mit Wilson-Intervall, plus "Anteil der Decke".

    Alle drei Konsensstufen werden IMMER berichtet. Der Abstand zwischen ihnen
    ist selbst ein Befund: ein grosser Spread heisst, die Sequenzen liegen in
    einem Bereich, in dem die Tools uneinig und damit unzuverlaessig sind.

    Zwei verschiedene Nenner, und das ist Absicht:

        rate_<tool>   geteilt durch die Zahl der von DIESEM Tool bewerteten
                      Sequenzen. Sonst sieht ein Tool mit schlechter Coverage
                      kuenstlich schwach aus, obwohl es nur weniger Sequenzen
                      annehmen konnte.
        rate_<k>von<m> geteilt durch n_bewertbar, also die Zeilen mit
                      vollstaendigem Vote. Eine Konsensstufe ueber Zeilen zu
                      zaehlen, bei denen ein Tool fehlt, waere keine Rate
                      einer Regel mehr (siehe VoteRule, nan_policy).

    Args:
        sets: die Sets des Laufs.
        regel: Vote-Regel; None -> VoteRule ueber SPALTEN["vote"]. min_votes
            der Regel spielt hier KEINE Rolle: alle Stufen werden ohnehin
            berichtet, gezaehlt wird direkt ueber n_votes.
        decke: Set, durch dessen Rate normiert wird.

    Returns:
        Eine Zeile je Set. "anteil_decke_<k>von<m>" ist ein Verhaeltnis, keine
        Prozentzahl: 0,80 heisst 80 % der Deckenrate. Fehlt das Deckenset,
        bleiben diese Spalten NaN -- build_report lehnt das ab.

    Raises:
        ValueError: wenn einem Set eine Vote-Spalte fehlt (ueber VoteRule).
    """
    regel = regel or VoteRule(spalten=tuple(SPALTEN["vote"]))
    m = len(regel.spalten)
    stufen = list(range(m, 0, -1))

    zeilen: dict[str, dict] = {}
    for name, s in sets.items():
        d = regel.apply(s).data
        n_votes = d["n_votes"]
        n_bewertbar = int(d["vote_complete"].sum())

        zeile: dict = {
            "set": name,
            "rolle": _rolle(name, decke),
            "n": len(d),
            "n_bewertbar": n_bewertbar,
        }

        for c in regel.spalten:
            werte = s.data[c]
            n_tool = int(werte.notna().sum())
            k_tool = int((werte >= regel.schwelle).sum())
            zeile[f"rate_{_kurz(c)}"] = 100.0 * k_tool / n_tool if n_tool else np.nan
            zeile[f"n_{_kurz(c)}"] = n_tool

        for stufe in stufen:
            sfx = f"{stufe}von{m}"
            if n_bewertbar:
                k = int((n_votes >= stufe).sum())
                lo, hi = wilson_interval(k, n_bewertbar)
                zeile[f"rate_{sfx}"] = 100.0 * k / n_bewertbar
                zeile[f"ci_lo_{sfx}"] = 100.0 * lo
                zeile[f"ci_hi_{sfx}"] = 100.0 * hi
            else:
                zeile[f"rate_{sfx}"] = np.nan
                zeile[f"ci_lo_{sfx}"] = np.nan
                zeile[f"ci_hi_{sfx}"] = np.nan

        zeilen[name] = zeile

    tab = pd.DataFrame(list(zeilen.values()))

    for stufe in stufen:
        sfx = f"{stufe}von{m}"
        referenz = zeilen.get(decke, {}).get(f"rate_{sfx}")
        # Deckenrate 0 ergibt keine Division, sondern NaN: "unendlich mal
        # besser als die Decke" waere keine Aussage, sondern ein Artefakt.
        if referenz and not pd.isna(referenz) and referenz > 0:
            tab[f"anteil_decke_{sfx}"] = tab[f"rate_{sfx}"] / referenz
        else:
            tab[f"anteil_decke_{sfx}"] = np.nan

    return tab


def tabelle_t3_agreement(
    sets: Mapping[str, SequenceSet],
    regel: VoteRule | None = None,
    decke: str = DECKE,
) -> pd.DataFrame:
    """T3 -- kappa-Matrix pro Set plus Verteilung der vote_pattern.

    Die Matrix wird zu Spalten flachgeklopft (kappa_<a>_<b>), weil alle
    Tabellen Sets als Zeilen haben. n wandert als echte Spalte mit: attrs
    ueberlebt kein to_csv, und kappa ohne n ist nicht vergleichbar.

    kappa ist NaN, wenn beide Tools in einem Set ausnahmslos gleich urteilen
    (entartete Randverteilungen). Das ist auf dem UniProt-Boden der Normalfall
    und heisst "nicht bestimmbar", nicht "keine Uebereinstimmung".

    Auch hier zwei Nenner, absichtlich verschieden benannt:

        muster_<bits>            Anteil an den VOLLSTAENDIGEN Votes. Dadurch
                                 passt muster_111 genau zu rate_3von3 aus T2,
                                 und die Muster summieren sich auf 100 %.
        unvollstaendig_prozent   Anteil an ALLEN Zeilen des Sets. Steht
                                 bewusst ausserhalb der Musterverteilung.

    Args:
        sets: die Sets des Laufs.
        regel: Vote-Regel; None -> VoteRule ueber SPALTEN["vote"].
        decke: Name des Deckensets, nur fuer die Rollenspalte.

    Returns:
        Eine Zeile je Set.
    """
    regel = regel or VoteRule(spalten=tuple(SPALTEN["vote"]))
    spalten = list(regel.spalten)
    matrix = AgreementMatrix(spalten=regel.spalten, schwelle=regel.schwelle)

    paare = [
        (spalten[i], spalten[j])
        for i in range(len(spalten))
        for j in range(i + 1, len(spalten))
    ]
    # Alle moeglichen Muster vorgeben, auch die nie vorkommenden: eine Spalte,
    # die in einem Lauf fehlt und im naechsten da ist, macht die Tabellen
    # sonst nicht mehr vergleichbar. Sortiert nach Stimmenzahl absteigend.
    alle_muster = sorted(
        ("".join(bits) for bits in itertools.product("01", repeat=len(spalten))),
        key=lambda muster: (-muster.count("1"), muster),
    )

    zeilen: list[dict] = []
    for name, s in sets.items():
        kappa = matrix.compute(s)
        d = regel.apply(s).data
        komplett = d["vote_complete"]

        zeile: dict = {
            "set": name,
            "rolle": _rolle(name, decke),
            "n_vollstaendig": kappa.attrs["n"],
            "n_unvollstaendig": kappa.attrs["n_unvollstaendig"],
        }
        for a, b in paare:
            zeile[f"kappa_{_kurz(a)}_{_kurz(b)}"] = float(kappa.loc[a, b])

        anteile = d.loc[komplett, "vote_pattern"].value_counts(normalize=True)
        for muster in alle_muster:
            zeile[f"muster_{muster}"] = 100.0 * float(anteile.get(muster, 0.0))
        zeile["unvollstaendig_prozent"] = (
            100.0 * float((~komplett).mean()) if len(d) else np.nan
        )

        zeilen.append(zeile)

    return pd.DataFrame(zeilen)


def tabelle_t4_aktivitaet(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    decke: str = DECKE,
    mic_schwelle: float = MIC_SCHWELLE,
) -> pd.DataFrame:
    """T4 -- Aktivitaet.

    Median und IQR von mic_apex und den ANIA-Spalten, % mit mittlerer
    MIC <= 64 µmol/L (Torres-Schwelle), und Spearman rho zwischen APEX und
    ANIA INNERHALB jedes Sets.

    Keine Mittelung ueber Praediktoren hinweg: MIC-Modelle koennen ranken,
    absolute Werte aber nicht zuverlaessig vorhersagen.

    ACHTUNG, Torres-Falle: torres_cellbiomaterial.csv bringt elf eigene
    Erregerspalten und average_MIC mit, gerechnet von Torres mit SEINEM
    APEX-Lauf. Die duerfen hier nicht einfliessen -- "Torres' average_MIC"
    gegen "unser mic_apex" zu stellen waere genau die Mittelung ueber
    Praediktoren hinweg, die Abschnitt 10 verbietet. Fuer alle Sets, Torres
    eingeschlossen, zaehlt der Wert aus UNSEREM APEX-Lauf. Torres' Spalten
    sind ein Abgleich (siehe 11_eval_run_torres_vs_aguirre.py), kein
    Panelwert.

    Args:
        sets: die Sets des Laufs.
        registry: Spaltenregistry, None -> SPALTEN.
        decke: Name des Deckensets, nur fuer die Rollenspalte.
        mic_schwelle: Grenze in µmol/L fuer die Anteilsspalte.

    Returns:
        Eine Zeile je Set. rho_apex_<ania> ist Spearman INNERHALB des Sets,
        ueber die Zeilen mit beiden Werten -- nie ueber Sets hinweg.
    """
    reg = SPALTEN if registry is None else registry
    apex = str(reg["apex_mittel"])
    roh_ania = reg["ania"]
    ania_spalten = [roh_ania] if isinstance(roh_ania, str) else list(roh_ania)

    zeilen: list[dict] = []
    for name, s in sets.items():
        werte_apex = _spalte(s, apex)
        zeile: dict = {"set": name, "rolle": _rolle(name, decke)}
        zeile.update(_verteilung(werte_apex, "apex"))
        zeile[f"anteil_mic{mic_schwelle:g}_apex"] = _prozent(
            werte_apex <= mic_schwelle, werte_apex.notna()
        )

        for spalte in ania_spalten:
            werte_ania = _spalte(s, spalte)
            kurz = _kurz(spalte)
            zeile.update(_verteilung(werte_ania, kurz))
            zeile[f"anteil_mic{mic_schwelle:g}_{kurz}"] = _prozent(
                werte_ania <= mic_schwelle, werte_ania.notna()
            )

            # Spearman nur ueber Zeilen mit BEIDEN Werten. Rangbasiert, weil
            # MIC log-verteilt ist und ein Pearson-rho von den
            # Saettigungswerten dominiert wuerde.
            paar = pd.concat([werte_apex, werte_ania], axis=1).dropna()
            genug = (
                len(paar) >= 3
                and paar.iloc[:, 0].nunique() > 1
                and paar.iloc[:, 1].nunique() > 1
            )
            if genug:
                rho, p_wert = stats.spearmanr(paar.iloc[:, 0], paar.iloc[:, 1])
                zeile[f"rho_apex_{kurz}"] = float(rho)
                zeile[f"rho_p_apex_{kurz}"] = float(p_wert)
            else:
                zeile[f"rho_apex_{kurz}"] = np.nan
                zeile[f"rho_p_apex_{kurz}"] = np.nan
            zeile[f"n_paare_apex_{kurz}"] = len(paar)

        zeilen.append(zeile)

    return pd.DataFrame(zeilen)


def tabelle_t5_plausibilitaet(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    decke: str = DECKE,
    alpha: float = SHUFFLE_ALPHA,
) -> pd.DataFrame:
    """T5 -- Median ppl_progen2, pppl_esm2, logR_shuffle,
    % p_shuffle < 0,05, % flag_lowcomplex.

    Anker ist die Deckenzeile. Torres berichtet 17,90 generiert / 15,83
    gefiltert / 17,93 Trainingsset -- das ist die Groessenordnung.
    pppl_esm2 ist Decoder-Diagnostik, kein Natuerlichkeitsbeleg.

    ACHTUNG, Torres-Falle: torres_cellbiomaterial.csv bringt eine eigene
    ppl-Spalte mit, die Torres selbst gerechnet hat. Sie gehoert NICHT in
    diese Tabelle. Abschnitt 4.1 macht ProGen2 zum Natuerlichkeits-Scorer of
    record, und das heisst UNSER ProGen2-Lauf ueber alle Sets -- sonst steht
    in der Torres-Zeile eine fremde Rechnung neben unseren. Der Abgleich der
    beiden Zahlen ist eine eigene Pruefung und steht in
    11_eval_run_torres_vs_aguirre.py.

    Args:
        sets: die Sets des Laufs.
        registry: Spaltenregistry, None -> SPALTEN.
        decke: Name des Deckensets, nur fuer die Rollenspalte.
        alpha: Schwelle fuer "% mit p_shuffle < alpha".

    Returns:
        Eine Zeile je Set.
    """
    reg = SPALTEN if registry is None else registry
    zeilen: list[dict] = []
    for name, s in sets.items():
        ppl = _spalte(s, str(reg["ppl"]))
        pppl = _spalte(s, str(reg["pppl"]))
        logr = _spalte(s, str(reg["shuffle_logR"]))
        p_shuffle = _spalte(s, str(reg["shuffle_p"]))
        lowcomplex = _spalte(s, str(reg["lowcomplex"]))
        entropie = _spalte(s, str(reg["entropie"]))

        zeile: dict = {"set": name, "rolle": _rolle(name, decke)}
        zeile.update(_verteilung(ppl, "ppl"))
        zeile.update(_verteilung(pppl, "pppl"))
        zeile.update(_verteilung(logr, "shuffle_logR"))
        zeile.update(_verteilung(entropie, "entropie"))

        zeile[f"anteil_p_unter_{alpha:g}"] = _prozent(
            p_shuffle < alpha, p_shuffle.notna()
        )
        # flag_lowcomplex ist bool, kommt aus der Parquet-Tabelle aber als
        # 0/1 zurueck -- deshalb ueber > 0.5 statt ueber == True.
        zeile["anteil_lowcomplex"] = _prozent(lowcomplex > 0.5, lowcomplex.notna())
        zeilen.append(zeile)

    return pd.DataFrame(zeilen)




def tabelle_t6_novelty_sicherheit(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    decke: str = DECKE,
    schwellen: tuple[float, ...] = NOVELTY_SCHWELLEN,
    hemo_schwelle: float = HEMO_SCHWELLE,
) -> pd.DataFrame:
    """T6 -- Median ident_train und ident_bekannt, % > 0,60, % > 0,40;
    % haemolytisch nach HemoPI und Macrel plus deren Uebereinstimmung.

    Zwei Novelty-Spalten, weil sie verschiedene Fragen beantworten:
    ident_train ist die Memorisierungspruefung, ident_bekannt die
    Novelty-Aussage. Torres macht nur Zweiteres.

    Stand jetzt gibt es nur Zweiteres auch bei uns: KnownAMPSimilarityScorer
    liefert die Identitaet gegen DRAMP/DBAASP/APD3, ein Scorer gegen das
    EIGENE Trainingsset existiert nicht. Die id_train-Spalten bleiben deshalb
    NaN, bis jemand ihn baut -- sie fallen nicht weg, damit in der Tabelle
    sichtbar bleibt, dass die Memorisierungspruefung aussteht.

    Die Haemolyse-Uebereinstimmung ist Cohens kappa ueber die Zeilen, in denen
    BEIDE Tools geurteilt haben. NaN heisst auch hier "nicht bestimmbar" --
    auf einem Set, auf dem beide ausnahmslos "nicht haemolytisch" sagen, sind
    sie sich perfekt einig, aber kappa ist dort nicht definiert.

    Args:
        sets: die Sets des Laufs.
        registry: Spaltenregistry, None -> SPALTEN.
        decke: Name des Deckensets, nur fuer die Rollenspalte.
        schwellen: Identitaetsschwellen fuer die Anteilsspalten.
        hemo_schwelle: ab wann ein Score als haemolytisch zaehlt.

    Returns:
        Eine Zeile je Set.
    """
    reg = SPALTEN if registry is None else registry
    roh_hemo = reg["hemo"]
    hemo_spalten = [roh_hemo] if isinstance(roh_hemo, str) else list(roh_hemo)

    zeilen: list[dict] = []
    for name, s in sets.items():
        zeile: dict = {"set": name, "rolle": _rolle(name, decke)}

        for schluessel, praefix in (
            ("id_train", "id_train"), ("id_bekannt", "id_bekannt"),
        ):
            werte = _spalte(s, str(reg[schluessel]))
            zeile.update(_verteilung(werte, praefix))
            for schwelle in schwellen:
                zeile[f"anteil_{praefix}_ueber_{schwelle:g}"] = _prozent(
                    werte > schwelle, werte.notna()
                )

        for spalte in hemo_spalten:
            werte = _spalte(s, spalte)
            kurz = _kurz(spalte)
            zeile[f"anteil_{kurz}"] = _prozent(werte >= hemo_schwelle, werte.notna())
            zeile[f"n_{kurz}"] = int(werte.notna().sum())

        if len(hemo_spalten) == 2:
            erst = _spalte(s, hemo_spalten[0])
            zweit = _spalte(s, hemo_spalten[1])
            beide = erst.notna() & zweit.notna()
            zeile["n_hemo_paare"] = int(beide.sum())
            zeile["kappa_hemo"] = (
                cohens_kappa(
                    (erst[beide] >= hemo_schwelle).to_numpy(),
                    (zweit[beide] >= hemo_schwelle).to_numpy(),
                )
                if beide.any() else np.nan
            )

        zeilen.append(zeile)

    return pd.DataFrame(zeilen)


def metriken_standard(
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
) -> tuple[str, ...]:
    """Die acht Metriken, ueber die T7 standardmaessig testet.

    Als Registry-Schluessel gefuehrt, nicht als Spaltennamen -- sonst muss
    diese Liste beim Schemawechsel mitgepflegt werden.
    """
    reg = SPALTEN if registry is None else registry
    schluessel = (
        "apex_mittel", "apex_geom", "ppl", "pppl",
        "shuffle_logR", "entropie", "id_train", "id_bekannt",
    )
    return tuple(str(reg[k]) for k in schluessel if k in reg)


def tabelle_t7_tests(
    sets: Mapping[str, SequenceSet],
    metriken: Sequence[str] | None = None,
    paare: Sequence[tuple[str, str]] = REFERENZ_PAARE,
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
) -> pd.DataFrame:
    """T7 -- Mann-Whitney-U p, Cliffs delta mit Label, n je Gruppe, BH-q.

    Effektstaerke immer neben dem p-Wert: bei n = 50 000 ist jeder Unterschied
    signifikant, erst Cliffs delta sagt, ob er relevant ist. Die
    BH-Korrektur laeuft ueber das GANZE Gitter (Paare x Metriken), nicht pro
    Metrik.

    Einzige Tabelle, die NICHT Sets als Zeilen hat: eine Zeile ist ein
    Paarvergleich einer Metrik. Sets als Zeilen gibt es hier nicht, weil eine
    Zeile zwei Sets betrifft.

    Das Vorzeichen von delta sagt die Richtung: positiv heisst, set_a hat die
    hoeheren Werte. cliffs_label bewertet nur den Betrag, deshalb stehen
    beide Spalten nebeneinander.

    Ein Paar, dessen Set fehlt, wird uebersprungen statt den Lauf
    abzubrechen -- Kandidatensets sind optional. Ein Test, der nicht laufen
    konnte (leere Gruppe, konstante Werte), bekommt NaN und zaehlt damit auch
    nicht in die BH-Korrektur: man korrigiert nicht gegen Tests, die es nicht
    gab.

    Args:
        sets: die Sets des Laufs.
        metriken: Spaltennamen. None -> metriken_standard(registry), gefiltert
            auf die Spalten, die in beiden Sets des Paares vorkommen.
        paare: zu testende Paare (set_a, set_b).
        registry: Spaltenregistry, None -> SPALTEN.

    Returns:
        Eine Zeile je (Paar, Metrik). Leer, wenn kein Paar vollstaendig ist.
    """
    reg = SPALTEN if registry is None else registry
    spalten = tuple(metriken) if metriken is not None else metriken_standard(reg)

    zeilen: list[dict] = []
    for set_a, set_b in paare:
        if set_a not in sets or set_b not in sets:
            continue
        for metrik in spalten:
            werte_a = _spalte(sets[set_a], metrik).dropna()
            werte_b = _spalte(sets[set_b], metrik).dropna()
            zeile: dict = {
                "paar": f"{set_a} vs {set_b}",
                "set_a": set_a,
                "set_b": set_b,
                "metrik": metrik,
                "n_a": len(werte_a),
                "n_b": len(werte_b),
                "median_a": float(werte_a.median()) if len(werte_a) else np.nan,
                "median_b": float(werte_b.median()) if len(werte_b) else np.nan,
            }
            try:
                u_wert, p_wert = mann_whitney(werte_a, werte_b)
                delta = cliffs_delta(werte_a, werte_b)
                zeile.update({
                    "U": u_wert,
                    "p": p_wert,
                    "delta": delta,
                    "delta_label": cliffs_label(delta),
                })
            except ValueError:
                # Leere Gruppe oder durchgehend identische Werte. Kein Test,
                # also auch kein p -- NaN statt einer erfundenen 1,0.
                zeile.update({"U": np.nan, "p": np.nan,
                              "delta": np.nan, "delta_label": ""})
            zeilen.append(zeile)

    tab = pd.DataFrame(zeilen)
    if tab.empty:
        return pd.DataFrame(columns=[
            "paar", "set_a", "set_b", "metrik", "n_a", "n_b",
            "median_a", "median_b", "U", "p", "delta", "delta_label", "q_bh",
        ])
    tab["q_bh"] = bh_correct(tab["p"].to_numpy())
    return tab




# ---------------------------------------------------------------------------
# 2. Plots (Pflicht)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. Plots (Pflicht)
# ---------------------------------------------------------------------------
#
# Durchgehend: Farbe kommt aus _gruppenfarben (dasselbe Set hat in jeder
# Abbildung dieselbe Farbe), Linienart aus linienstil (Kandidat durchgezogen,
# Boden gestrichelt, Decke gepunktet). cfg.plot_farben hat sechs Farben bei
# sieben Sets -- ohne die Linienart waeren zwei Kurven nicht zu trennen.
#
# Fehlt einer Abbildung ihre Spalte, gibt sie None zurueck statt eine leere
# Achse zu zeichnen. build_report sammelt diese Faelle mit Begruendung ins
# Manifest, damit die Luecke dokumentiert ist und nicht nur fehlt.

FIGSIZE = (5.5, 4.2)


def _stil(ax, titel: str, x: str, y: str) -> None:
    """Gemeinsamer Achsenstil, damit die Abbildungen zu den bisherigen passen."""
    ax.set_title(titel)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=cfg.plot_grid_alpha)


def _hat_werte(sets: Mapping[str, SequenceSet], spalte: str) -> bool:
    """True, wenn mindestens ein Set in dieser Spalte etwas anderes als NaN hat."""
    return any(_spalte(s, spalte).notna().any() for s in sets.values())


def plot_p1_laengen(
    sets: Mapping[str, SequenceSet],
    decke: str = DECKE,
    marken: Sequence[float] = (),
) -> plt.Figure:
    """P1 -- Laengenverteilung aller Sets ueberlagert.

    Sanity-Check und Erklaerung fuer die Coverage-Luecken in T1.

    Relative Haeufigkeit statt absoluter Zahlen, weil die Sets verschieden
    gross sind (50 000 gegen 1 938). Als Stufenkurve, weil Laenge ganzzahlig
    ist -- geglaettete Kurven erfinden dort Zwischenwerte.

    Args:
        sets: die Sets des Laufs.
        decke: Name des Deckensets, bestimmt die Linienart.
        marken: senkrechte Linien, z.B. Laengengates der Tools (10, 100).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    farben = _gruppenfarben(sets)
    for name, s in sets.items():
        laengen = s.data["length"].to_numpy()
        if not len(laengen):
            continue
        kanten = np.arange(laengen.min(), laengen.max() + 2) - 0.5
        anteile, _ = np.histogram(laengen, bins=kanten, density=False)
        ax.step(kanten[:-1] + 0.5, anteile / len(laengen), where="mid",
                color=farben[name], linestyle=linienstil(name, decke),
                linewidth=1.8, label=f"{name} (n={len(laengen)})")
    for marke in marken:
        ax.axvline(marke, color="0.4", linewidth=0.8, linestyle=(0, (1, 3)))
    _stil(ax, "Laengenverteilung", "Laenge (Residuen)", "Anteil der Sequenzen")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def standard_schwelle(
    metrik: str,
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    decke: str = DECKE,
) -> float | None:
    """Senkrechte Schwelle fuer P2, je nach Metrik.

    MIC-Spalten bekommen Torres' 64 µmol/L. Perplexitaeten bekommen den
    Median des Deckensets -- eine absolute PPL-Schwelle gibt es nicht, die
    Decke IST der Massstab. Alles andere bekommt keine Linie, statt eine
    erfundene zu zeichnen.
    """
    reg = SPALTEN if registry is None else registry
    roh_ania = reg["ania"]
    mic_spalten = {str(reg["apex_mittel"]), str(reg["apex_geom"])}
    mic_spalten |= {roh_ania} if isinstance(roh_ania, str) else set(roh_ania)
    if metrik in mic_spalten:
        return MIC_SCHWELLE
    if metrik in {str(reg["ppl"]), str(reg["pppl"])} and decke in sets:
        werte = _spalte(sets[decke], metrik).dropna()
        return float(werte.median()) if len(werte) else None
    return None


def plot_p2_ecdf(
    sets: Mapping[str, SequenceSet],
    metrik: str,
    schwelle: float | None = None,
    log: bool | None = None,
    decke: str = DECKE,
) -> plt.Figure | None:
    """P2 -- ECDF einer Metrik, alle Sets, mit Schwelle als Vertikale.

    ECDF statt Violin: Schwellenanteile direkt ablesbar, keine
    Glaettungsartefakte, und Verteilungen mit sehr verschiedenen n bleiben
    vergleichbar.

    Args:
        sets: die Sets des Laufs.
        metrik: Spaltenname.
        schwelle: Position der Vertikalen. None -> keine Linie.
        log: logarithmische x-Achse. None -> automatisch fuer MIC-Spalten,
            die ueber Groessenordnungen streuen.
        decke: bestimmt die Linienart.

    Returns:
        None, wenn kein Set die Spalte mit Werten hat.
    """
    if not _hat_werte(sets, metrik):
        return None
    if log is None:
        log = metrik.startswith("mic_") or "_mic_" in metrik

    fig, ax = plt.subplots(figsize=FIGSIZE)
    farben = _gruppenfarben(sets)
    for name, s in sets.items():
        werte = _spalte(s, metrik).dropna().to_numpy()
        if log:
            werte = werte[werte > 0]     # log-Achse vertraegt keine Nullen
        if not len(werte):
            continue
        x = np.sort(werte)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=farben[name],
                linestyle=linienstil(name, decke), linewidth=1.8,
                label=f"{name} (n={len(x)})")
    if schwelle is not None:
        ax.axvline(schwelle, color="0.3", linewidth=1.0, linestyle=(0, (4, 2)))
    if log:
        ax.set_xscale("log")
    _stil(ax, f"ECDF: {metrik}", metrik, "Anteil <= x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_p3_vote_pattern(
    sets: Mapping[str, SequenceSet],
    regel: VoteRule | None = None,
    decke: str = DECKE,
) -> plt.Figure:
    """P3 -- gestapelter Balken der Stimmenzahl (0/1/2/3) pro Set.

    Zeigt die Konsensstruktur, nicht nur die Kopfzahl.

    Die Anteile gehen ueber ALLE Zeilen und summieren sich auf 100 %, die
    unvollstaendigen Votes als eigenes Segment. Nur so sieht man an der
    Abbildung, dass ein Set schlechte Coverage hat -- in T2 steht das in
    einer eigenen Spalte, hier waere es sonst unsichtbar.

    Sequenzielle Farben statt Gruppenfarben: die Stimmenzahl ist eine
    Ordnung, keine Kategorie.
    """
    regel = regel or VoteRule(spalten=tuple(SPALTEN["vote"]))
    m = len(regel.spalten)
    stufen = list(range(m + 1))

    namen = list(sets)
    anteile = np.zeros((len(namen), m + 2))
    for i, name in enumerate(namen):
        d = regel.apply(sets[name]).data
        n = len(d)
        if not n:
            continue
        for stufe in stufen:
            anteile[i, stufe] = 100.0 * float((d["n_votes"] == stufe).sum()) / n
        anteile[i, m + 1] = 100.0 * float((~d["vote_complete"]).sum()) / n

    cmap = plt.get_cmap(cfg.plot_cmap)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    unten = np.zeros(len(namen))
    for stufe in stufen:
        ax.bar(namen, anteile[:, stufe], bottom=unten,
               color=cmap(stufe / max(m, 1)), label=f"{stufe} Stimmen")
        unten += anteile[:, stufe]
    if anteile[:, m + 1].any():
        ax.bar(namen, anteile[:, m + 1], bottom=unten, color="0.75",
               hatch="//", label="unvollstaendig")
    _stil(ax, "Verteilung der Stimmenzahl", "", "Anteil der Sequenzen (%)")
    ax.tick_params(axis="x", rotation=45)
    for beschriftung in ax.get_xticklabels():
        beschriftung.set_ha("right")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_p4_apex_vs_ania(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
) -> plt.Figure | None:
    """P4 -- Streudiagramm APEX gegen ANIA pro Set, log-log, mit
    Identitaetslinie und Spearman rho. Macht die APEX-Konservativitaet
    unmittelbar sichtbar.

    Returns:
        None, wenn keine ANIA-Spalte Werte hat -- der CsvScorer dafuer ist
        noch nicht gebaut. build_report vermerkt das im Manifest.
    """
    reg = SPALTEN if registry is None else registry
    apex = str(reg["apex_mittel"])
    roh_ania = reg["ania"]
    ania_spalten = [roh_ania] if isinstance(roh_ania, str) else list(roh_ania)
    ania = next((c for c in ania_spalten if _hat_werte(sets, c)), None)
    if ania is None or not _hat_werte(sets, apex):
        return None

    namen = [n for n in sets if _spalte(sets[n], ania).notna().any()]
    spalten_n = min(3, max(1, len(namen)))
    zeilen_n = int(np.ceil(len(namen) / spalten_n))
    fig, achsen = plt.subplots(zeilen_n, spalten_n,
                               figsize=(3.2 * spalten_n, 3.0 * zeilen_n),
                               squeeze=False)
    farben = _gruppenfarben(sets)

    for achse, name in zip(achsen.flat, namen):
        paar = pd.concat([_spalte(sets[name], apex), _spalte(sets[name], ania)],
                         axis=1).dropna()
        paar = paar[(paar.iloc[:, 0] > 0) & (paar.iloc[:, 1] > 0)]
        if len(paar):
            achse.scatter(paar.iloc[:, 0], paar.iloc[:, 1], s=4, alpha=0.25,
                          color=farben[name], edgecolors="none")
            grenzen = [min(paar.min()), max(paar.max())]
            achse.plot(grenzen, grenzen, color="0.3", linewidth=0.8,
                       linestyle=(0, (4, 2)))
            achse.set_xscale("log")
            achse.set_yscale("log")
        rho = np.nan
        genug = (len(paar) >= 3 and paar.iloc[:, 0].nunique() > 1
                 and paar.iloc[:, 1].nunique() > 1)
        if genug:
            rho = float(stats.spearmanr(paar.iloc[:, 0], paar.iloc[:, 1])[0])
        achse.set_title(f"{name}  rho={rho:.2f}  n={len(paar)}", fontsize=9)
        achse.set_xlabel(apex, fontsize=8)
        achse.set_ylabel(ania, fontsize=8)
        achse.grid(alpha=cfg.plot_grid_alpha)
    for achse in achsen.flat[len(namen):]:
        achse.set_visible(False)
    fig.tight_layout()
    return fig


def plot_p5_shuffle(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    decke: str = DECKE,
) -> plt.Figure | None:
    """P5 -- x = PPL original, y = mittlere PPL der Permutationen, ein Punkt
    pro Sequenz, Identitaetslinie.

    Punkte oberhalb der Linie haben Ordnungsinformation. Diese eine Abbildung
    erklaert den Shuffle-Score vollstaendig -- deutlich besser als ein
    Histogramm von R. Geht in den Vortrag.

    Die y-Groesse steht in KEINER Spalte: ShuffleScorer liefert R, logR, z, p,
    sd und das Flag, nicht die Permutations-PPL selbst. Sie ist aber exakt
    rekonstruierbar. logR ist definitionsgemaess mean_k L(pi_k) - L(s), und
    PPL = exp(L), also

        PPL_shuffle = PPL_original * exp(logR)

    Das ist genau das geometrische Mittel der Permutations-PPL -- also die
    Groesse, die Abschnitt 4.3 verlangt (ueber Permutationen das Log mitteln,
    nicht die PPL). Eine zusaetzliche Spalte im Scorer ist nicht noetig.

    Returns:
        None, wenn PPL oder logR fehlen.
    """
    reg = SPALTEN if registry is None else registry
    ppl_spalte, logr_spalte = str(reg["ppl"]), str(reg["shuffle_logR"])
    if not (_hat_werte(sets, ppl_spalte) and _hat_werte(sets, logr_spalte)):
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    farben = _gruppenfarben(sets)
    grenzen: list[float] = []
    for name, s in sets.items():
        paar = pd.concat([_spalte(s, ppl_spalte), _spalte(s, logr_spalte)],
                         axis=1).dropna()
        paar = paar[paar.iloc[:, 0] > 0]
        if not len(paar):
            continue
        original = paar.iloc[:, 0].to_numpy()
        permutiert = original * np.exp(paar.iloc[:, 1].to_numpy())
        ax.scatter(original, permutiert, s=4, alpha=0.25, color=farben[name],
                   edgecolors="none", label=f"{name} (n={len(original)})")
        grenzen += [original.min(), original.max(),
                    permutiert.min(), permutiert.max()]
    if grenzen:
        spanne = [min(grenzen), max(grenzen)]
        ax.plot(spanne, spanne, color="0.3", linewidth=0.8, linestyle=(0, (4, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
    _stil(ax, "Shuffle: Ordnung gegen Komposition",
          f"{ppl_spalte} (original)", "PPL der Permutationen (geom. Mittel)")
    legende = ax.legend(fontsize=8)
    for griff in legende.legend_handles:
        griff.set_alpha(1.0)
    fig.tight_layout()
    return fig


def plot_p6_novelty_vs_aktivitaet(
    sets: Mapping[str, SequenceSet],
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
) -> plt.Figure | None:
    """P6 -- x = ident_train, y = mic_apex.

    Die Memorisierungspruefung: sitzen die potenten Sequenzen alle bei hoher
    Identitaet, hat das Modell kopiert.

    Returns:
        None, solange es keinen Scorer gegen das eigene Trainingsset gibt.
        Bewusst KEIN Ersatz durch id_max_known: das waere eine Novelty-
        Abbildung mit der Beschriftung einer Memorisierungspruefung.
    """
    reg = SPALTEN if registry is None else registry
    id_train, apex = str(reg["id_train"]), str(reg["apex_mittel"])
    if not (_hat_werte(sets, id_train) and _hat_werte(sets, apex)):
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    farben = _gruppenfarben(sets)
    for name, s in sets.items():
        paar = pd.concat([_spalte(s, id_train), _spalte(s, apex)], axis=1).dropna()
        paar = paar[paar.iloc[:, 1] > 0]
        if not len(paar):
            continue
        ax.scatter(paar.iloc[:, 0], paar.iloc[:, 1], s=4, alpha=0.25,
                   color=farben[name], edgecolors="none",
                   label=f"{name} (n={len(paar)})")
    ax.set_yscale("log")
    ax.axhline(MIC_SCHWELLE, color="0.3", linewidth=1.0, linestyle=(0, (4, 2)))
    _stil(ax, "Memorisierung: Identitaet gegen Aktivitaet",
          id_train, f"{apex} (µmol/L)")
    legende = ax.legend(fontsize=8)
    for griff in legende.legend_handles:
        griff.set_alpha(1.0)
    fig.tight_layout()
    return fig


def plot_p7_komposition(
    sets: Mapping[str, SequenceSet],
    decke: str = DECKE,
) -> plt.Figure | None:
    """P7 -- Heatmap Sets x 20 AS als z-Scores gegen die Decke.

    Rechnet die Anteile NICHT selbst, sondern nimmt die Tabelle von
    CompositionComparison aus evaluation.py (Spalten anteil,
    anteil_referenz, n_residuen) und bildet daraus

        z = (p_set - p_decke) / sqrt(p_decke * (1 - p_decke) / n_residuen)

    also den binomialen Standardfehler. Der Unterschied zum log2-Enrichment,
    das CompositionComparison zeichnet: z beruecksichtigt n. Derselbe
    Anteilsunterschied ist bei 50 000 Sequenzen ein starker Befund und bei
    1 938 Rauschen -- das Enrichment sieht in beiden Faellen gleich aus.

    Divergierende Farbskala statt cfg.plot_cmap: z hat ein Vorzeichen, und
    eine sequenzielle Skala wie viridis macht "deutlich weniger" und
    "deutlich mehr" optisch gleich auffaellig, ohne die Richtung zu zeigen.

    Returns:
        None, wenn das Deckenset fehlt -- ohne Referenz gibt es keinen
        z-Score.
    """
    if decke not in sets:
        return None

    ergebnis = CompositionComparison(reference=decke).compare(dict(sets))
    tab = ergebnis.table
    for figur in ergebnis.figures.values():
        plt.close(figur)          # nur die Tabelle wird gebraucht

    standardfehler = np.sqrt(
        tab["anteil_referenz"] * (1.0 - tab["anteil_referenz"]) / tab["n_residuen"]
    )
    tab = tab.assign(z=(tab["anteil"] - tab["anteil_referenz"]) / standardfehler)
    matrix = tab.pivot(index="gruppe", columns="aminosaeure", values="z")
    matrix = matrix.reindex([n for n in sets if n in matrix.index])

    grenze = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * matrix.shape[1] + 2),
                                    0.45 * matrix.shape[0] + 2))
    bild = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", aspect="auto",
                     vmin=-grenze, vmax=grenze)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title(f"Aminosaeure-Komposition als z-Score gegen {decke}")
    fig.colorbar(bild, ax=ax, label="z (Standardfehler)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Plots (nice to have)
# ---------------------------------------------------------------------------

def plot_p8_position(
    sets: Mapping[str, SequenceSet],
    top_k: int = 3,
    max_length: int | None = None,
) -> dict[str, plt.Figure]:
    """P8 -- Positionsprofil ueber PositionalComparison.

    ABWEICHUNG VON DER SPEC, bewusst: Abschnitt 9 beschreibt P8 als Ladung
    und Hydrophobizitaet ueber die relative Position. PositionalComparison
    zeichnet etwas anderes -- die haeufigsten Aminosaeuren je Position. Eine
    Ladungs- oder Hydrophobizitaetskurve braeuchte eine Aminosaeureskala
    (Kyte-Doolittle o.ae.), die es im Paket nirgends gibt, also waere das
    keine Wiederverwendung mehr, sondern ein neuer Scorer. Fuer ein
    "nice to have" ist das zu viel -- deshalb hier die vorhandene Abbildung,
    und die Spec-Variante bleibt offen.

    Gibt ein DICT zurueck, nicht eine Figure: PositionalComparison liefert
    eine Heatmap PRO Set, sieben Sets passen nicht in eine Achse.

    Returns:
        {Setname: Figure}. Leer, wenn keine Sequenzen da sind.
    """
    laenge = max_length or int(max(
        (int(s.data["length"].max()) for s in sets.values() if len(s.data)),
        default=50,
    ))
    ergebnis = PositionalComparison(top_k=top_k, max_length=laenge).compare(dict(sets))
    return dict(ergebnis.figures)


def plot_p9_konsens_abfall(
    sets: Mapping[str, SequenceSet],
    regel: VoteRule | None = None,
    decke: str = DECKE,
) -> plt.Figure:
    """P9 -- % verbleibend gegen Konsensschwelle (1/3, 2/3, 3/3), eine Linie
    pro Set. Kompakte Variante von T2 fuer den Vortrag.

    Nenner ist n_bewertbar, genau wie in T2 -- sonst stehen in Tabelle und
    Abbildung zwei verschiedene Zahlen fuer dieselbe Aussage.
    """
    regel = regel or VoteRule(spalten=tuple(SPALTEN["vote"]))
    m = len(regel.spalten)
    stufen = np.arange(1, m + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    farben = _gruppenfarben(sets)
    for name, s in sets.items():
        d = regel.apply(s).data
        n_bewertbar = int(d["vote_complete"].sum())
        if not n_bewertbar:
            continue
        anteile = [100.0 * float((d["n_votes"] >= stufe).sum()) / n_bewertbar
                   for stufe in stufen]
        ax.plot(stufen, anteile, marker="o", color=farben[name],
                linestyle=linienstil(name, decke), linewidth=1.8,
                label=f"{name} (n={n_bewertbar})")
    ax.set_xticks(stufen, [f"{int(stufe)}/{m}" for stufe in stufen])
    ax.set_ylim(0, 100)
    _stil(ax, "Konsens-Abfallkurve", "Konsensschwelle", "Anteil positiv (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Orchestrierung
# ---------------------------------------------------------------------------

def build_report(
    sets: Mapping[str, SequenceSet],
    metriken: Sequence[str] | None = None,
    regel: VoteRule | None = None,
    manifest: dict | None = None,
    decke: str = DECKE,
    registry: Mapping[str, tuple[str, ...] | str] | None = None,
    train_sequences: Iterable[str] | None = None,
    paare: Sequence[tuple[str, str]] = REFERENZ_PAARE,
    ecdf_metriken: Sequence[str] | None = None,
    optional: bool = True,
    datum: date | None = None,
) -> PanelReport:
    """Baut alle Tabellen und Pflicht-Plots und fuellt einen PanelReport.

    Fehlende Pflicht-Baselines werden hier abgelehnt: ohne Decke gibt es
    keinen Nenner fuer "Anteil der Decke", und ohne Boeden ist jede Rate
    bedeutungslos. Fehlende KANDIDATEN sind dagegen in Ordnung, ihre Zeile
    faellt einfach weg.

    Nicht gezeichnete Abbildungen landen mit Begruendung unter
    manifest["uebersprungen"]. Eine fehlende Abbildung, die nirgends vermerkt
    ist, sieht beim Schreiben aus wie eine vergessene.

    Args:
        sets: alle Sets des Laufs.
        metriken: Spalten fuer T7. None -> metriken_standard(registry).
        regel: Vote-Regel. None -> VoteRule ueber der Registry.
        manifest: Grunddaten des Laufs (Checkpoint, Sampler, Seeds,
            Toolversionen, Git-Commit). Wird ergaenzt, nicht ersetzt. Den
            Git-Commit muss das aufrufende Skript liefern -- eine Bibliothek
            sollte dafuer kein git aufrufen.
        decke: Set, gegen das normiert wird.
        registry: Spaltenregistry. None -> SPALTEN.
        train_sequences: Trainingssequenzen des Modells fuer T1.
        paare: Vergleichspaare fuer T7.
        ecdf_metriken: Metriken, fuer die P2 gezeichnet wird. None -> die vier
            aussagekraeftigsten. Alle acht waeren 16 Dateien allein fuer P2.
        optional: auch P8 und P9 bauen.
        datum: fuer den Dateinamen-Stamm. None -> heute.

    Returns:
        PanelReport mit T1-T7, P1-P7 (und P8, P9) sowie gefuelltem Manifest.

    Raises:
        ValueError: wenn eine Pflicht-Baseline fehlt.
    """
    reg = SPALTEN if registry is None else registry
    regel = regel or VoteRule(spalten=tuple(reg["vote"]))

    fehlend = [name for name in (decke,) + BOEDEN if name not in sets]
    if fehlend:
        raise ValueError(
            f"Pflicht-Baselines fehlen: {fehlend}. Vorhanden: {list(sets)}. "
            f"Ohne Decke gibt es keinen Nenner fuer 'Anteil der Decke', ohne "
            f"Boeden keine Spezifitaetskontrolle -- eine Konsensrate ohne "
            f"Referenz ist bedeutungslos (Abschnitt 5)."
        )

    tabellen = {
        "T1_sets": tabelle_t1_sets(sets, train_sequences=train_sequences,
                                   registry=reg, decke=decke),
        "T2_panel": tabelle_t2_panel(sets, regel=regel, decke=decke),
        "T3_agreement": tabelle_t3_agreement(sets, regel=regel, decke=decke),
        "T4_aktivitaet": tabelle_t4_aktivitaet(sets, registry=reg, decke=decke),
        "T5_plausibilitaet": tabelle_t5_plausibilitaet(sets, registry=reg, decke=decke),
        "T6_novelty_sicherheit": tabelle_t6_novelty_sicherheit(sets, registry=reg,
                                                               decke=decke),
        "T7_tests": tabelle_t7_tests(sets, metriken=metriken, paare=paare,
                                     registry=reg),
    }

    figures: dict[str, plt.Figure] = {}
    uebersprungen: dict[str, str] = {}

    figures["P1_laengen"] = plot_p1_laengen(sets, decke=decke)

    if ecdf_metriken is None:
        ecdf_metriken = [str(reg[k]) for k in
                         ("apex_mittel", "ppl", "shuffle_logR", "id_bekannt")
                         if k in reg]
    for metrik in ecdf_metriken:
        figur = plot_p2_ecdf(sets, metrik, decke=decke,
                             schwelle=standard_schwelle(metrik, sets, reg, decke))
        if figur is None:
            uebersprungen[f"P2_ecdf_{_slug(metrik)}"] = (
                f"Spalte {metrik} hat keine Werte"
            )
        else:
            figures[f"P2_ecdf_{_slug(metrik)}"] = figur

    figures["P3_vote_pattern"] = plot_p3_vote_pattern(sets, regel=regel, decke=decke)

    for schluessel, bauer, grund in (
        ("P4_apex_vs_ania", lambda: plot_p4_apex_vs_ania(sets, registry=reg),
         "keine ANIA-Spalte mit Werten (CsvScorer noch nicht gebaut)"),
        ("P5_shuffle", lambda: plot_p5_shuffle(sets, registry=reg, decke=decke),
         "ProGen2-PPL oder logR_shuffle fehlen"),
        ("P6_novelty_vs_aktivitaet",
         lambda: plot_p6_novelty_vs_aktivitaet(sets, registry=reg),
         "kein Scorer gegen das eigene Trainingsset (id_max_train fehlt)"),
        ("P7_komposition", lambda: plot_p7_komposition(sets, decke=decke),
         f"Deckenset {decke!r} fehlt"),
    ):
        figur = bauer()
        if figur is None:
            uebersprungen[schluessel] = grund
        else:
            figures[schluessel] = figur

    if optional:
        for name, figur in plot_p8_position(sets).items():
            figures[f"P8_position_{_slug(name)}"] = figur
        figures["P9_konsens_abfall"] = plot_p9_konsens_abfall(sets, regel=regel,
                                                              decke=decke)

    voll = dict(manifest or {})
    voll.update({
        "datum": (datum or date.today()).isoformat(),
        "vote_regel": (
            f"{regel.min_votes}/{len(regel.spalten)} bei p>={regel.schwelle:g} "
            f"ueber {', '.join(regel.spalten)} (nan={regel.nan_policy})"
        ),
        "decke": decke,
        "sets": {name: len(s.data) for name, s in sets.items()},
        "registry": {k: list(v) if not isinstance(v, str) else v
                     for k, v in reg.items()},
        "uebersprungen": uebersprungen,
    })

    return PanelReport(tabellen=tabellen, figures=figures, manifest=voll,
                       stem=panel_stem(sets, datum=datum))
