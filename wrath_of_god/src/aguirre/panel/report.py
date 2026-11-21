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
(runs/<run>/scores.parquet), nie direkt aus den Scorern -- nur so ist ein
Plot ohne Neuberechnung reproduzierbar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from aguirre.evaluation import SequenceSet, build_stem, cliffs_delta, mann_whitney
from aguirre.panel.aggregate import AgreementMatrix, VoteRule, bh_correct, wilson_interval

# Set, gegen das normalisiert und getestet wird.
DECKE = "train_amps"
REFERENZ_PAARE = (
    ("aguirre_small", "torres"),
    ("aguirre_big", "torres"),
    ("aguirre_big", "aguirre_small"),
)


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

    def save(self, out_dir: Path, stem: str) -> None:
        """Tabellen als CSV, Figures als PNG + PDF, Manifest als JSON.

        TODO(paul): build_stem() aus evaluation.py benutzen, damit die
        Dateinamen zu den bisherigen Laeufen passen.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Tabellen
# ---------------------------------------------------------------------------

def tabelle_t1_sets(sets: Mapping[str, SequenceSet]) -> pd.DataFrame:
    """T1 -- Set-Uebersicht und Coverage.

    n, n unique, % Duplikate, % exakt im Trainingsset, Laenge (Median, IQR,
    min, max), und pro Scorer die Zahl der NaN mit Grund (zu kurz / zu lang /
    nicht-kanonisch / nicht in CSV).

    Die Tabelle, die alle vergessen und nach der jeder Gutachter fragt.
    """
    raise NotImplementedError


def tabelle_t2_panel(
    sets: Mapping[str, SequenceSet],
    regel: VoteRule | None = None,
) -> pd.DataFrame:
    """T2 -- die Kernaussage.

    Pro Set: % positiv je Tool bei der Schwelle, dann 3/3, >=2/3, >=1/3, jede
    Zahl mit Wilson-Intervall, plus "Anteil der Decke".

    Alle drei Konsensstufen werden IMMER berichtet. Der Abstand zwischen ihnen
    ist selbst ein Befund: ein grosser Spread heisst, die Sequenzen liegen in
    einem Bereich, in dem die Tools uneinig und damit unzuverlaessig sind.
    """
    raise NotImplementedError


def tabelle_t3_agreement(sets: Mapping[str, SequenceSet]) -> pd.DataFrame:
    """T3 -- kappa-Matrix pro Set plus Verteilung der vote_pattern."""
    raise NotImplementedError


def tabelle_t4_aktivitaet(sets: Mapping[str, SequenceSet]) -> pd.DataFrame:
    """T4 -- Aktivitaet.

    Median und IQR von mic_apex und den ANIA-Spalten, % mit mittlerer
    MIC <= 64 µmol/L (Torres-Schwelle), und Spearman rho zwischen APEX und
    ANIA INNERHALB jedes Sets.

    Keine Mittelung ueber Praediktoren hinweg: MIC-Modelle koennen ranken,
    absolute Werte aber nicht zuverlaessig vorhersagen.
    """
    raise NotImplementedError


def tabelle_t5_plausibilitaet(sets: Mapping[str, SequenceSet]) -> pd.DataFrame:
    """T5 -- Median ppl_progen2, pppl_esm2, logR_shuffle,
    % p_shuffle < 0,05, % flag_lowcomplex.

    Anker ist die train_amps-Zeile. Torres berichtet 17,90 generiert / 15,83
    gefiltert / 17,93 Trainingsset -- das ist die Groessenordnung.
    pppl_esm2 ist Decoder-Diagnostik, kein Natuerlichkeitsbeleg.
    """
    raise NotImplementedError


def tabelle_t6_novelty_sicherheit(sets: Mapping[str, SequenceSet]) -> pd.DataFrame:
    """T6 -- Median ident_train und ident_bekannt, % > 0,60, % > 0,40;
    % haemolytisch nach HemoPI und Macrel plus deren Uebereinstimmung.

    Zwei Novelty-Spalten, weil sie verschiedene Fragen beantworten:
    ident_train ist die Memorisierungspruefung, ident_bekannt die
    Novelty-Aussage. Torres macht nur Zweiteres.
    """
    raise NotImplementedError


def tabelle_t7_tests(
    sets: Mapping[str, SequenceSet],
    metriken: Sequence[str],
    paare: Sequence[tuple[str, str]] = REFERENZ_PAARE,
) -> pd.DataFrame:
    """T7 -- Mann-Whitney-U p, Cliffs delta mit Label, n je Gruppe, BH-q.

    Effektstaerke immer neben dem p-Wert: bei n = 50 000 ist jeder Unterschied
    signifikant, erst Cliffs delta sagt, ob er relevant ist. Die
    BH-Korrektur laeuft ueber das GANZE Gitter (Paare x Metriken), nicht pro
    Metrik.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Plots (Pflicht)
# ---------------------------------------------------------------------------

def plot_p1_laengen(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P1 -- Laengenverteilung aller Sets ueberlagert.

    Sanity-Check und Erklaerung fuer die Coverage-Luecken in T1.
    """
    raise NotImplementedError


def plot_p2_ecdf(sets: Mapping[str, SequenceSet], metrik: str) -> plt.Figure:
    """P2 -- ECDF einer Metrik, alle Sets, mit Schwelle als Vertikale.

    ECDF statt Violin: Schwellenanteile direkt ablesbar, keine
    Glaettungsartefakte, und Verteilungen mit sehr verschiedenen n bleiben
    vergleichbar.
    """
    raise NotImplementedError


def plot_p3_vote_pattern(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P3 -- gestapelter Balken der Stimmenzahl (0/1/2/3) pro Set.

    Zeigt die Konsensstruktur, nicht nur die Kopfzahl.
    """
    raise NotImplementedError


def plot_p4_apex_vs_ania(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P4 -- Streudiagramm APEX gegen ANIA pro Set, log-log, mit
    Identitaetslinie und Spearman rho. Macht die APEX-Konservativitaet
    unmittelbar sichtbar.
    """
    raise NotImplementedError


def plot_p5_shuffle(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P5 -- x = PPL original, y = mittlere PPL der Permutationen, ein Punkt
    pro Sequenz, Identitaetslinie.

    Punkte oberhalb der Linie haben Ordnungsinformation. Diese eine Abbildung
    erklaert den Shuffle-Score vollstaendig -- deutlich besser als ein
    Histogramm von R. Geht in den Vortrag.
    """
    raise NotImplementedError


def plot_p6_novelty_vs_aktivitaet(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P6 -- x = ident_train, y = mic_apex.

    Die Memorisierungspruefung: sitzen die potenten Sequenzen alle bei hoher
    Identitaet, hat das Modell kopiert.
    """
    raise NotImplementedError


def plot_p7_komposition(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P7 -- Heatmap Sets x 20 AS als z-Scores gegen train_amps.

    TODO(david): CompositionComparison aus evaluation.py wiederverwenden
    statt neu zu rechnen.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Plots (nice to have)
# ---------------------------------------------------------------------------

def plot_p8_position(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P8 -- Ladung und Hydrophobizitaet ueber die relative Position.
    PositionalComparison existiert schon.
    """
    raise NotImplementedError


def plot_p9_konsens_abfall(sets: Mapping[str, SequenceSet]) -> plt.Figure:
    """P9 -- % verbleibend gegen Konsensschwelle (1/3, 2/3, 3/3), eine Linie
    pro Set. Kompakte Variante von T2 fuer den Vortrag.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Orchestrierung
# ---------------------------------------------------------------------------

def build_report(
    sets: Mapping[str, SequenceSet],
    metriken: Sequence[str],
    regel: VoteRule | None = None,
    manifest: dict | None = None,
) -> PanelReport:
    """Baut alle Tabellen und Pflicht-Plots und fuellt einen PanelReport.

    TODO(paul): fehlende Baselines hier ablehnen, nicht still weglaufen --
    ohne train_amps gibt es keine Decke und T2 ist nicht interpretierbar.
    """
    raise NotImplementedError
