"""shuffle.py -- kompositionskontrollierte Natuerlichkeit.

Die rohe Perplexitaet verwechselt "typische Aminosaeuren" mit "typischem
Muster". AMPs sind kompositionell extrem (K/R-reich), eine K-reiche Sequenz
hat also per se niedrige PPL. Shuffling erhaelt die Komposition EXAKT und
zerstoert nur die Ordnung -- der Quotient isoliert damit den
ordnungsabhaengigen Anteil:

    R(s) = PPL_shuf / PPL_orig = exp(L_shuf - L_orig) = exp(dL)

Lesart: PPL ist exp(L) mit L = mittlere negative Log-Likelihood pro Residuum.
Ein Quotient zweier PPLs ist also eine Differenz zweier Log-Likelihoods --
die log-Regel (Produkt -> Summe) rueckwaerts. dL hat die Einheit nats pro
Residuum und ist der Log-Likelihood-Vorsprung, den die REIHENFOLGE bringt.
R ist das geometrische Mittel des Likelihood-Verhaeltnisses pro Rest; fuer
die ganze Sequenz gilt der Faktor R**N.

    R ~ 1   natuerlich nur wegen der Zusammensetzung, kein Muster
    R >> 1  echte Ordnungsinformation
    R < 1   die echte Reihenfolge ist unwahrscheinlicher als eine zufaellige

Umhuellt wird ProGen2 (autoregressiv, UniRef90/UniProtKB/BFD30, unabhaengig
von beiden Generatorvarianten). Um ESM gewickelt kuerzt der Quotient die
KOMPOSITIONELLE Modellvorliebe weg, aber nicht die ordnungsbezogene -- und
genau die hat der Generator im Latentraum optimiert.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from aguirre.evaluation import Scorer


def n_distinct_permutations(seq: str) -> int:
    """Multinomialkoeffizient N! / prod(n_a!).

    Bei KKKWKKKK gibt es kaum unterscheidbare Permutationen -- R ~ 1 ist dort
    konstruktionsbedingt und nicht inhaltlich. Diese Zahl setzt das Flag.
    """
    raise NotImplementedError


class ShuffleScorer(Scorer):
    """Dekorator: umhuellt einen Scorer, der eine Perplexitaet liefert.

    Berechnet selbst keine PPL. Dadurch laeuft er ohne eine Zeile Mehrarbeit
    ueber ProGen2Scorer UND ESMPseudoPerplexityScorer, und der
    Cache-Fingerprint komponiert sich korrekt mit.

    Args:
        inner: Scorer, der ppl_column liefert (z.B. ProGen2Scorer()).
        ppl_column: Name der PPL-Spalte in inner.columns.
        n_perm: Zahl der Permutationen pro Sequenz. 50 ist ein brauchbarer
            Kompromiss; unter 20 wird z zu verrauscht.
        seed: fuer Reproduzierbarkeit, gehoert in den fingerprint.
        min_distinct: unterhalb dieser Zahl distinkter Permutationen wird
            flag_lowcomplex gesetzt.
    """

    name = "shuffle"
    columns = [
        "R_shuffle",
        "logR_shuffle",
        "z_shuffle",
        "p_shuffle",
        "sd_shuffle",
        "flag_lowcomplex",
    ]

    def __init__(
        self,
        inner: Scorer,
        ppl_column: str,
        n_perm: int = 50,
        seed: int = 0,
        min_distinct: int = 1000,
    ) -> None:
        if ppl_column not in inner.columns:
            raise ValueError(
                f"{ppl_column!r} nicht in inner.columns ({inner.columns})."
            )
        self.inner = inner
        self.ppl_column = ppl_column
        self.n_perm = n_perm
        self.seed = seed
        self.min_distinct = min_distinct

    @property
    def fingerprint(self) -> str:
        return (
            f"shuffle-{self.inner.fingerprint}-{self.ppl_column}"
            f"-K{self.n_perm}-s{self.seed}"
        )

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Ablauf pro Sequenz:

        1. L(s) = log(PPL) des Originals aus inner.score().
        2. n_perm Permutationen ziehen (rng aus seed + Sequenzindex, damit
           reproduzierbar und trotzdem pro Sequenz verschieden).
        3. inner.score() auf ALLE Permutationen aller Sequenzen in EINEM Aufruf
           -- nicht pro Sequenz, sonst zahlst du den Modell-Overhead n_perm-mal.
        4. dL = mean_k L(pi_k) - L(s). Ueber die Permutationen das LOG mitteln,
           nicht die PPL: Mitteln auf der PPL-Skala und dann teilen ergibt
           wegen Jensen einen systematisch groesseren Wert.
        5. R = exp(dL), logR = dL, sd = std_k L(pi_k),
           z = dL / sd,
           p = (1 + #{k: L(pi_k) <= L(s)}) / (n_perm + 1).
           Das +1 verhindert p = 0 -- mit K Permutationen ist 1/(K+1) die
           feinste behauptbare Aufloesung.
        6. flag_lowcomplex, wenn n_distinct_permutations(s) < min_distinct
           oder sd nahe null. Dort ist R bedeutungslos.

        Hinweis zur Laufzeit: (n_perm + 1) Forward-Passes pro Sequenz. Bei
        ProGen2 billig, aber 50 000 x 51 ist unnoetig -- fuer einen
        Verteilungsvergleich reichen rund 2 000 Sequenzen pro Set. Die
        Stichprobengroesse gehoert in das Run-Manifest.
        """
        raise NotImplementedError
