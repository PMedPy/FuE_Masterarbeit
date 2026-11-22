"""aggregate.py -- vom Score zur Aussage.

Die Vote-Auswertung ist ein vierter Typ neben Scorer, Filter und Comparison:

    Scorer      Sequenzen -> Spalten
    Filter      SequenceSet -> kleineres SequenceSet
    Comparison  mehrere SequenceSets -> Tabelle + Figure
    Aggregator  SequenceSet -> SequenceSet mit abgeleiteten Spalten

Harte Anforderung: ein Aggregator muss auf einer GESPEICHERTEN Score-Tabelle
laufen, ohne einen einzigen Scorer. Nur dann laesst sich die Vote-Regel beim
Schreiben beliebig oft aendern, ohne APEX neu zu rechnen.

AgreementMatrix ist absichtlich KEIN Aggregator: kappa ist keine
Pro-Sequenz-Groesse. Es liegt hier, weil es inhaltlich zum Vote gehoert, hat
aber eine eigene Signatur (compute -> DataFrame).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from aguirre.evaluation import SequenceSet

VOTE_SPALTEN = ("p_macrel", "p_amplify", "p_antibp3")


# ---------------------------------------------------------------------------
# 1. Basisklasse
# ---------------------------------------------------------------------------

class Aggregator(ABC):
    """Haengt abgeleitete Spalten an, verwirft nie Zeilen."""

    @abstractmethod
    def apply(self, s: SequenceSet) -> SequenceSet:
        """Returns: neues SequenceSet, gleiche Zeilen, zusaetzliche Spalten."""


# ---------------------------------------------------------------------------
# 2. Vote
# ---------------------------------------------------------------------------

@dataclass
class VoteRule(Aggregator):
    """Zaehlt Stimmen ueber die Vote-Spalten.

    Drei Tools mit Absicht: bei ungerader Anzahl ist "2 von 3" eine eindeutige
    Mehrheit. Bei vier Tools ist das Wort "Mehrheit" ambig, und genau daran
    scheitert die Lesbarkeit der Literaturtabellen.

    Attributes:
        spalten: Vote-Spalten, in fester Reihenfolge (bestimmt vote_pattern).
        schwelle: Score ab dem eine Stimme als positiv zaehlt.
        min_votes: ab wie vielen Stimmen is_amp True ist. 2 von 3 = Mehrheit,
            3 von 3 = Einstimmigkeit.
        nan_policy: was bei fehlenden Scores passiert.
            "incomplete" (Standard): n_votes bleibt NaN und vote_complete
                False. Richtig, aber die Coverage MUSS dann berichtet werden.
            "as_negative": NaN zaehlt als Nein. Verzerrt systematisch nach
                unten -- nur mit ausdruecklicher Begruendung benutzen.

    Erzeugte Spalten:
        n_votes         Anzahl positiver Stimmen (float, wegen NaN)
        is_amp          bool: n_votes >= min_votes
        vote_pattern    z.B. "110" in der Reihenfolge von spalten; daraus
                        laesst sich spaeter sehen, welches Tool der Ausreisser
                        ist -- das zeigt die kappa-Matrix allein nicht
        vote_complete   bool: lagen alle Scores vor
    """

    spalten: tuple[str, ...] = VOTE_SPALTEN
    schwelle: float = 0.5
    min_votes: int = 2
    nan_policy: Literal["incomplete", "as_negative"] = "incomplete"

    def apply(self, s: SequenceSet) -> SequenceSet:
        """TODO(paul): fehlende Spalten mit klarer Meldung ablehnen, nicht
        still ignorieren -- ein Vote ueber zwei statt drei Tools sieht in der
        Tabelle genauso aus und ist falsch.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Uebereinstimmung
# ---------------------------------------------------------------------------

@dataclass
class AgreementMatrix:
    """Paarweise Cohens kappa zwischen den Vote-Tools.

    Warum das mitlaeuft: der Gewinn eines Panels haengt daran, dass die Fehler
    der Tools dekorreliert sind (Condorcet -- bei unabhaengigen Fehlern und
    p > 0,5 wird die Mehrheit mit jedem Tool besser, bei perfekt korrelierten
    Fehlern bringt sie exakt null). AMP-Klassifikatoren teilen aber ihre
    Positivdaten, meist APD3-abgeleitet. Bei kappa ~ 0,9 ist unser Panel
    effektiv ein Klassifikator mit drei Namen. Das ist dann ein Befund, den
    wir berichten -- kein Scheitern.
    """

    spalten: tuple[str, ...] = VOTE_SPALTEN
    schwelle: float = 0.5

    def compute(self, s: SequenceSet) -> pd.DataFrame:
        """Returns: quadratische Matrix, Index == Spalten == self.spalten.

        Nur Zeilen mit vollstaendigen Scores benutzen und n dazu berichten --
        kappa auf unterschiedlichen Teilmengen ist nicht vergleichbar.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Statistik-Helfer (von report.py mitbenutzt)
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95-%-Konfidenzintervall fuer einen Binomialanteil.

    Fuer Anteile das richtige Intervall -- die Normalapproximation laeuft bei
    Raten nahe 0 oder 1 (UniProt: 0,79 %) aus dem Einheitsintervall heraus.
    Jede Prozentzahl in T2 bekommt eines.
    """
    raise NotImplementedError


def bh_correct(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg: q-Werte zu einer Liste von p-Werten.

    Drei Paarvergleiche x rund acht Metriken sind ~24 Tests. Ohne Korrektur
    ist jede "Signifikanz" in T7 geschenkt. Rohe p-Werte UND q-Werte
    berichten, nicht ersetzen.
    """
    raise NotImplementedError
