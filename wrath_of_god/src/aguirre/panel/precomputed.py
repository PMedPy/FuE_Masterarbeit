"""precomputed.py -- Scores aus einer CSV statt aus einem CLI-Aufruf.

Fuer Tools, die im Rechenjob nicht leben sollen. Hauptfall ANIA: das braucht
drei Conda-Umgebungen, eine davon mit R und dem Paket `kaos` fuer die
CGR-Kodierung. Das will niemand live in einem SLURM-Job haben.

Workflow:

    1. ANIA einmal separat laufen lassen (eigenes Env, eigener Job).
    2. Ergebnis-CSV nach data_tracked/ legen, gekeyt ueber "sequence".
    3. Ueber AniaScorer einhaengen -- fuer den Evaluator ist das ein
       gewoehnlicher Scorer.

Taugt genauso fuer alles, was auf einer anderen Maschine gerechnet wurde.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

from aguirre.evaluation import Scorer


class CsvScorer(Scorer):
    """Liest Score-Spalten aus einer CSV, die ueber "sequence" gekeyt ist.

    Args:
        name: Kurzname, geht in den fingerprint.
        csv_path: CSV mit Spalte "sequence" plus den Score-Spalten.
        spalten: Zuordnung {CSV-Spalte: Zielspalte im SequenceSet}. Explizit,
            damit Umbenennungen in der CSV nicht still das Schema aendern.

    Sequenzen, die nicht in der CSV stehen, bekommen eine NaN-Zeile. Das ist
    der Normalfall und kein Fehler: die CSV kann aelter sein als das Set.
    Die Coverage landet in Tabelle T1 und wird dort berichtet.
    """

    def __init__(
        self,
        name: str,
        csv_path: Path,
        spalten: dict[str, str],
    ) -> None:
        self.name = name
        self.csv_path = Path(csv_path)
        self.spalten = dict(spalten)
        self.columns = list(spalten.values())

    @property
    def fingerprint(self) -> str:
        """Name PLUS Inhaltshash der CSV.

        Mtime reicht nicht: eine neu erzeugte, inhaltlich gleiche CSV wuerde
        den Cache sonst grundlos invalidieren, und eine ueberschriebene mit
        gleicher Mtime wuerde ihn nicht invalidieren.
        """
        raise NotImplementedError

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """CSV laden, auf die Eingabereihenfolge mappen, fehlende -> NaN.

        TODO(paul): ueber ein dict {sequence: zeile} mappen, nicht ueber
        pd.merge -- merge auf Duplikaten in sequences vervielfacht Zeilen und
        bricht damit den Scorer-Vertrag.
        """
        raise NotImplementedError


class AniaScorer(CsvScorer):
    """ANIA-MIC-Vorhersagen (E. coli, P. aeruginosa, S. aureus) aus CSV.

    ANIA ist der zweite, unabhaengige MIC-Regressor neben APEX: andere
    Architektur (Chaos Game Representation + Inception/Transformer), andere
    Trainingsquellen (DBAASP, dbAMP, DRAMP). Nur deshalb ist der Vergleich
    APEX gegen ANIA aussagekraeftig.

    Einheit ist µmol/L wie bei APEX -- beim Export aus ANIA pruefen, ob die
    CSV log-MIC oder MIC enthaelt, und HIER umrechnen, nicht spaeter im Plot.
    Referenz: https://github.com/SilverGojo4/ANIA

    Spaltenschema analog zu APEX in evaluation.py: mic_ania ist das Aggregat
    (arithmetisches Mittel, damit mit Torres' Schwelle von 64 vergleichbar),
    mic_ania_geom das geometrische (MIC ist log-verteilt, fuer
    Verteilungsvergleiche die richtige Groesse), mic_ania__<stamm> die
    Rohwerte. Nur so laesst sich mic_apex gegen mic_ania direkt stellen.
    """

    STRAIN_SPALTEN = {
        "EC": "mic_ania__ec",
        "PA": "mic_ania__pa",
        "SA": "mic_ania__sa",
    }

    def __init__(self, csv_path: Path) -> None:
        super().__init__(
            name="ania",
            csv_path=csv_path,
            spalten=dict(self.STRAIN_SPALTEN),  # TODO(paul): echte CSV-Header
        )
        self.columns = [*self.STRAIN_SPALTEN.values(), "mic_ania", "mic_ania_geom"]

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Wie CsvScorer, plus die beiden Aggregate.

        TODO(paul): geometrisches_mittel() aus evaluation.py benutzen, nicht
        neu schreiben -- dann sind APEX- und ANIA-Aggregat identisch definiert.
        """
        raise NotImplementedError
