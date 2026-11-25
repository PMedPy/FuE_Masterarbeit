"""aggregate.py -- vom Score zur Aussage.

Die Vote-Auswertung ist ein vierter Typ neben Scorer, Filter und Comparison:

    Scorer      Sequenzen -> Spalten
    Filter      SequenceSet -> kleineres SequenceSet
    Comparison  mehrere SequenceSets -> Tabelle + Figure
    Aggregator  SequenceSet -> SequenceSet mit abgeleiteten Spalten

Harte Anforderung: ein Aggregator muss auf einer GESPEICHERTEN Score-Tabelle laufen, ohne einen einzigen Scorer.
Nur dann laesst sich die Vote-Regel beim Schreiben beliebig oft aendern, ohne APEX neu zu rechnen.
Deshalb benutzt dieses Modul aus evaluation nur SequenceSet -> keinen Scorer, kein Backend, kein Modell.

Einschraenkung, die man kennen muss:
`import aguirre.panel.aggregate` zieht ueber evaluation -> config trotzdem torch mit herein.
Gerechnet wird damit nichts, aber das Modul laeuft nicht in einer schlanken Umgebung ohne torch.
Wer die Vote-Regel am Schreibtisch auf einer scores.parquet variieren will,
braucht also fue-amp -> oder config muss seinen torch-Import irgendwann lazy machen.

AgreementMatrix ist absichtlich KEIN Aggregator:
kappa ist keine Pro-Sequenz-Groesse. Es liegt hier, weil es inhaltlich zum Vote gehoert,
hat aber eine eigene Signatur (compute -> DataFrame).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from aguirre.evaluation import SequenceSet

VOTE_SPALTEN = ("p_macrel", "p_amplify", "p_antibp3")

# Zeichen fuer eine fehlende Stimme in vote_pattern. Absichtlich nicht "0":
# "Tool sagt nein" und "Tool konnte nicht bewerten" sind verschiedene Befunde, 
# und genau deshalb gibt es die die Coverage-Spalte in T1.
FEHLT = "."


# ---------------------------------------------------------------------------
# 0. Spaltenpruefung (von VoteRule und AgreementMatrix geteilt)
# ---------------------------------------------------------------------------

def _vote_scores(
    s: SequenceSet,
    spalten: tuple[str, ...],
    wer: str,
) -> pd.DataFrame:
    """Holt die Vote-Spalten und lehnt alles ab, was still falsch waere.

    Drei Fehlerklassen, die in der fertigen Tabelle identisch aussehen und
    deshalb hier und nicht spaeter auffallen muessen:

        1. Spalte fehlt -> ein Vote ueber zwei statt drei Tools.
           Sieht aus wieein gueltiger Vote, ist aber eine andere Groesse.
        2. Spalte ist nicht numerisch -> ein Parserfehler im CsvScorer liefert Strings;
           ">=" auf Strings vergleicht lexikografisch und schweigt.
        3. Werte ausserhalb [0, 1] -> AMPlify liefert neben der Wahrscheinlichkeit
           auch einen log-skalierten Score. Mit dem in der Spalte ist bei Schwelle 0,5 alles positiv,
           und die Rate sieht hervorragend aus.

    Args:
        s: Set mit den Score-Spalten.
        spalten: erwartete Spalten, in fester Reihenfolge.
        wer: Klassenname fuer die Fehlermeldung.

    Returns:
        DataFrame mit genau diesen Spalten als float, Reihenfolge wie uebergeben.

    Raises:
        ValueError: bei jeder der drei Fehlerklassen, mit Nennung der Spalte.
    """
    fehlt = [c for c in spalten if c not in s.data.columns]
    if fehlt:
        raise ValueError(
            f"{wer}: Vote-Spalten fehlen in Set {s.name!r}: {fehlt}. "
            f"Vorhanden: {list(s.data.columns)}. "
            f"Ein Vote ueber {len(spalten) - len(fehlt)} statt {len(spalten)} "
            f"Tools steht am Ende in derselben Tabellenspalte und ist falsch -- "
            f"deshalb Abbruch statt stillem Weglassen."
        )

    scores = s.data[list(spalten)]

    nicht_zahl = [c for c in spalten if not pd.api.types.is_numeric_dtype(scores[c])]
    if nicht_zahl:
        raise ValueError(
            f"{wer}: Spalten {nicht_zahl} in Set {s.name!r} sind nicht "
            f"numerisch (dtype "
            f"{ {c: str(scores[c].dtype) for c in nicht_zahl} }). "
            f"Vermutlich ein Parserfehler im zustaendigen Scorer. Ein "
            f"Vergleich mit der Schwelle waere lexikografisch und falsch."
        )

    scores = scores.astype(float)

    ausser = {
        c: (float(scores[c].min()), float(scores[c].max()))
        for c in spalten
        if scores[c].notna().any()
        and (scores[c].min() < 0.0 or scores[c].max() > 1.0)
    }
    if ausser:
        raise ValueError(
            f"{wer}: Spalten ausserhalb [0, 1] in Set {s.name!r}: "
            f"{ {c: f'{lo:g}..{hi:g}' for c, (lo, hi) in ausser.items()} }. "
            f"Die Schwelle ist fuer Wahrscheinlichkeiten definiert. "
            f"Steht hier ein roher oder log-skalierter Toolscore, ist bei 0,5 alles positiv und die Rate sieht hervorragend aus."
        )

    return scores


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

    Drei Tools mit Absicht: bei ungerader Anzahl ist "2 von 3" eine eindeutige Mehrheit.
    Bei vier Tools ist das Wort "Mehrheit" ambig, und genau daran scheitert die Lesbarkeit der Literaturtabellen.

    Attributes:
        spalten: Vote-Spalten, in fester Reihenfolge (bestimmt vote_pattern).
        schwelle: Score ab dem eine Stimme als positiv zaehlt.
        min_votes: ab wie vielen Stimmen is_amp True ist. 2 von 3 = Mehrheit, 3 von 3 = Einstimmigkeit.
        nan_policy: was bei fehlenden Scores passiert.
            "incomplete" (Standard): n_votes bleibt NaN und vote_complete False.
                Richtig, aber die Coverage MUSS dann berichtet werden.
            "as_negative": NaN zaehlt als Nein. Verzerrt systematisch nach unten:
                -> nur mit ausdruecklicher Begruendung benutzen.

    Erzeugte Spalten:
        n_votes         Anzahl positiver Stimmen (float, wegen NaN)
        is_amp          bool: n_votes >= min_votes
        vote_pattern    z.B. "110" in der Reihenfolge von spalten;
                        daraus laesst sich spaeter sehen, welches Tool der Ausreisser ist.
                        Das zeigt die kappa-Matrix allein nicht.
                        Eine fehlende Stimme steht als ".", nicht als "0".
        vote_complete   bool: lagen alle Scores vor

    Bewusste Entscheidung zu "incomplete": eine Zeile mit zwei positiven und
    einem fehlenden Score hat min_votes=2 bereits erreicht, egal was das dritte Tool gesagt haette
    Trotzdem ist n_votes dort NaN und is_amp False.
    Grund: sonst stuende in derselben Spalte eine Mischung aus Dreier- und
    Zweier-Votes, und die Rate waere nicht mehr die Rate einer Regel.
    Wer die bestimmbaren Faelle mitnehmen will, nimmt sie ueber vote_pattern
    aus der gespeicherten Tabelle -> dafuer ist das Muster da.
    """

    spalten: tuple[str, ...] = VOTE_SPALTEN
    schwelle: float = 0.5
    min_votes: int = 2
    nan_policy: Literal["incomplete", "as_negative"] = "incomplete"

    def __post_init__(self) -> None:
        self.spalten = tuple(self.spalten)
        if not self.spalten:
            raise ValueError("Mindestens eine Vote-Spalte noetig.")
        doppelt = [c for c in set(self.spalten) if self.spalten.count(c) > 1]
        if doppelt:
            raise ValueError(
                f"Spalte {doppelt} mehrfach in spalten -> das Tool haette zwei Stimmen."
            )
        if not 0.0 <= self.schwelle <= 1.0:
            raise ValueError(
                f"schwelle muss in [0, 1] liegen, war {self.schwelle}."
            )
        if not 1 <= self.min_votes <= len(self.spalten):
            raise ValueError(
                f"min_votes muss zwischen 1 und {len(self.spalten)} "
                f"liegen, war {self.min_votes}."
            )
        if self.nan_policy not in ("incomplete", "as_negative"):
            raise ValueError(f"Unbekannte nan_policy {self.nan_policy!r}.")

    def apply(self, s: SequenceSet) -> SequenceSet:
        """Haengt n_votes, is_amp, vote_pattern und vote_complete an.

        Args:
            s: Set mit den Vote-Spalten. Darf aus einer gespeicherten Score-Tabelle stammen;
               kein Scorer wird angefasst.

        Returns:
            Neues SequenceSet mit vier zusaetzlichen Spalten.

        Raises:
            ValueError: wenn eine Vote-Spalte fehlt, nicht numerisch ist oder
                ausserhalb [0, 1] liegt (siehe _vote_scores).
        """
        scores = _vote_scores(s, self.spalten, wer=type(self).__name__)

        # NaN >= schwelle ist False -> fuer "as_negative" also schon richtig.
        ja = scores.ge(self.schwelle)
        vorhanden = scores.notna()
        vote_complete = vorhanden.all(axis=1)

        n_votes = ja.sum(axis=1).astype(float)
        if self.nan_policy == "incomplete":
            n_votes = n_votes.where(vote_complete)

        # Vergleich mit NaN ergibt False -> unvollstaendige Zeilen sind damit automatisch kein AMP, ohne eigenen Zweig.
        is_amp = (n_votes >= self.min_votes)

        zeichen = ja.astype(int).astype(str).where(vorhanden, FEHLT)
        vote_pattern = reduce(
            lambda a, b: a + b, (zeichen[c] for c in self.spalten)
        )

        neu = pd.DataFrame(
            {
                "n_votes": n_votes,
                "is_amp": is_amp.to_numpy(dtype=bool),
                "vote_pattern": vote_pattern,
                "vote_complete": vote_complete.to_numpy(dtype=bool),
            },
            index=s.data.index,
        )
        return s.with_columns(neu)


# ---------------------------------------------------------------------------
# 3. Uebereinstimmung
# ---------------------------------------------------------------------------

def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohens kappa fuer zwei binaere Urteile ueber dieselben Objekte.

    kappa = (po - pe) / (1 - pe), po = beobachtete Uebereinstimmung,
    pe = die bei gleichen Randhaeufigkeiten zufaellig erwartete.

    Selbst gerechnet statt ueber sklearn: das Paket steht nicht in den Abhaenigkeiten
    und fuer den binaeren Fall sind es vier Zeilen.

    Wichtig -> pe = 1 ist kein Fehler, sondern ein haeufiger Fall:
    wenn beide Tools auf dem UniProt-Boden ausnahmslos "nein" sagen,
    ist die Uebereinstimmung perfekt, aber die zufaellig erwartete auch.
    kappa ist dort nicht definiert und wird NaN. Das heisst "nicht bestimmbar",
    NICHT "keine Uebereinstimmung". Eine 0 an dieser Stelle waere die glatte Falschaussage,
    dass sich die Tools nicht einig sind.

    Args:
        a: bool-Array der Urteile von Tool A.
        b: bool-Array der Urteile von Tool B, gleiche Laenge und Reihenfolge.

    Returns:
        kappa in [-1, 1], oder NaN bei n = 0 oder entarteten Randverteilungen.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Laengen verschieden: {a.shape} vs {b.shape}.")
    n = a.size
    if n == 0:
        return float("nan")

    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    if np.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


@dataclass
class AgreementMatrix:
    """Paarweise Cohens kappa zwischen den Vote-Tools.

    Warum das mitlaeuft: der Gewinn eines Panels haengt daran,
    dass die Fehler der Tools dekorreliert sind (Condorcet -> bei unabhaengigen Fehlern und
    p > 0,5 wird die Mehrheit mit jedem Tool besser, bei perfekt korrelierten
    Fehlern bringt sie exakt null). AMP-Klassifikatoren teilen aber ihre Positivdaten
    meist APD3-abgeleitet. Bei kappa ~ 0,9 ist unser Panel effektiv "nur" ein Klassifikator mit drei Namen.
    """

    spalten: tuple[str, ...] = VOTE_SPALTEN
    schwelle: float = 0.5

    def __post_init__(self) -> None:
        self.spalten = tuple(self.spalten)

    def compute(self, s: SequenceSet) -> pd.DataFrame:
        """Returns: quadratische Matrix, Index == Spalten == self.spalten.

        Nur Zeilen mit vollstaendigen Scores benutzen und n dazu berichten.
        Kappa auf unterschiedlichen Teilmengen ist nicht vergleichbar.

        n steht in df.attrs["n"] (und n_unvollstaendig).
        attrs ueberlebt kein to_csv: T3 in report.py muss n als eigene Spalte
        in die gespeicherte Tabelle schreiben.
        """
        scores = _vote_scores(s, self.spalten, wer=type(self).__name__)
        vollstaendig = scores.notna().all(axis=1)
        ja = scores.loc[vollstaendig.to_numpy()].ge(self.schwelle)
        n = int(vollstaendig.sum())

        namen = list(self.spalten)
        m = pd.DataFrame(np.nan, index=namen, columns=namen, dtype=float)
        if n > 0:
            for i, sp_a in enumerate(namen):
                m.loc[sp_a, sp_a] = 1.0
                for sp_b in namen[i + 1:]:
                    k = cohens_kappa(ja[sp_a].to_numpy(), ja[sp_b].to_numpy())
                    m.loc[sp_a, sp_b] = k
                    m.loc[sp_b, sp_a] = k

        m.attrs["n"] = n
        m.attrs["n_unvollstaendig"] = int(len(scores) - n)
        m.attrs["set"] = s.name
        return m


# ---------------------------------------------------------------------------
# 4. Statistik-Helfer (von report.py mitbenutzt)
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95-%-Konfidenzintervall fuer einen Binomialanteil.

    Fuer Anteile das richtige Intervall, die Normalapproximation laeuft bei
    Raten nahe 0 oder 1 (UniProt: 0,79 %) aus dem Einheitsintervall heraus.
    Jede Prozentzahl in T2 bekommt eines.

    Args:
        k: Zahl der Treffer.
        n: Zahl der Versuche. Das ist die Zahl der BEWERTBAREN Sequenzen,
            nicht die Set-Groesse, sonst ist das Intervall zu eng.
        alpha: Irrtumswahrscheinlichkeit, 0,05 -> 95 %.

    Returns:
        (untere, obere) Grenze, auf [0, 1] beschnitten. Bei n = 0 (nan, nan).
    """
    if n < 0 or k < 0:
        raise ValueError(f"k und n muessen >= 0 sein, waren k={k}, n={n}.")
    if k > n:
        raise ValueError(f"k ({k}) groesser als n ({n}).")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha muss in (0, 1) liegen, war {alpha}.")
    if n == 0:
        return (float("nan"), float("nan"))

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    nenner = 1.0 + z**2 / n
    mitte = (p + z**2 / (2.0 * n)) / nenner
    halb = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / nenner

    # Bei k = 0 bzw. k = n ist die Grenze analytisch exakt 0 bzw. 1.
    # Ohne Sonderfall bleibt durch die Gleichung ein Rundungsrest von 3e-18 stehen,
    # der in der gespeicherten CSV ausgeschrieben wird und dort unschoen oder wie ein Fehler aussieht.
    unten = 0.0 if k == 0 else float(max(0.0, mitte - halb))
    oben = 1.0 if k == n else float(min(1.0, mitte + halb))
    return (unten, oben)


def bh_correct(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg: q-Werte zu einer Liste von p-Werten.

    Drei Paarvergleiche x rund acht Metriken sind ~24 Tests.
    Ohne Korrektur ist jede "Signifikanz" in T7 geschenkt.
    Rohe p-Werte UND q-Werte berichten, nicht ersetzen.

    Verfahren:
        p sortieren, mit m/Rang multiplizieren,
        dann von hinten nach vorne das laufende Minimum bilden.
        Das Minimum ist noetig, damit die q-Werte monoton bleiben,
        ohne es kann ein kleinerer p-Wert ein groesseres q bekommen.

    Args:
        pvals: p-Werte in beliebiger Reihenfolge.
               NaN ist erlaubt (ein Test, der nicht lief) und zaehlt NICHT zu m,
               sonst korrigiert man gegen Tests, die es nicht gab.

    Returns:
        Array mit q-Werten in der Reihenfolge der Eingabe; NaN bleibt NaN.
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"Eindimensional erwartet, war shape {p.shape}.")

    q = np.full(p.shape, np.nan, dtype=float)
    gueltig = ~np.isnan(p)
    if np.any(p[gueltig] < 0.0) or np.any(p[gueltig] > 1.0):
        raise ValueError("p-Werte muessen in [0, 1] liegen.")

    m = int(gueltig.sum())
    if m == 0:
        return q

    idx = np.flatnonzero(gueltig)
    ordnung = idx[np.argsort(p[idx], kind="stable")]
    roh = p[ordnung] * m / np.arange(1, m + 1, dtype=float)
    monoton = np.minimum.accumulate(roh[::-1])[::-1]
    q[ordnung] = np.minimum(monoton, 1.0)
    return q
