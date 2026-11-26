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

Statistisch ist das ein Permutationstest mit Teststatistik L(s) und
Nullhypothese: das Modell haelt DIESE Anordnung fuer nicht wahrscheinlicher
als eine zufaellige Anordnung derselben Reste. Austauschbarkeit unter der
Null ist hier keine Annahme, sondern per Konstruktion gegeben -- jede
Sequenz ist ihre eigene, gepaarte Kontrolle. Das ist strenger als jede
Baseline gegen eine Population, weil Komposition, Laenge, Nettoladung und
mittlere Hydrophobizitaet nicht statistisch adjustiert, sondern konstruktiv
konstant gehalten werden.

Umhuellt wird ProGen2 (autoregressiv, UniRef90/UniProtKB/BFD30, unabhaengig
von beiden Generatorvarianten). Um ESM gewickelt kuerzt der Quotient die
KOMPOSITIONELLE Modellvorliebe weg, aber nicht die ordnungsbezogene -- und
genau die hat der Generator im Latentraum optimiert. Fuer die Hauptaussage
wird darum ProGen2 umhuellt, nicht ESM.

CAVEAT POSITIONSPRIOR (gehoert in die Diskussion, ist kein Implementierungs-
fehler): ProGen2 ist kausal, die Verteilung an Position 1 ist durch das
BOS-Token "1" konditioniert und in natuerlichen Proteinen dort stark auf M
konzentriert. Eine Permutation veraendert also auch, WELCHES Residuum am
N-Terminus steht. dL mischt damit zwei Beitraege: interne
Motivinformation und N-terminale Residuenidentitaet. Das ist nicht falsch --
"Reihenfolge" schliesst Position ein --, aber R darf nicht als reines Mass
fuer internes Muster gelesen werden.
"""

from __future__ import annotations

import hashlib
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

    Beispiel: KKKWKKKK hat N = 8 mit n_K = 7, n_W = 1, also
    8! / (7! * 1!) = 8 unterscheidbare Permutationen. Zieht man davon 50,
    sind fast alle Dubletten, die Nullverteilung kollabiert und sd geht
    gegen null.

    Parameters
    ----------
    seq : str
        Aminosaeuresequenz.

    Returns
    -------
    int
        Exakte Zahl unterscheidbarer Permutationen. Ganzzahlig gerechnet,
        kein float -- bei N = 50 ist N! jenseits der double-Genauigkeit,
        Pythons ints sind beliebig gross.
    """
    teiler = 1
    for n_a in Counter(seq).values():
        teiler *= math.factorial(n_a)
    return math.factorial(len(seq)) // teiler


class ShuffleScorer(Scorer):
    """Dekorator: umhuellt einen Scorer, der eine Perplexitaet liefert.

    Berechnet selbst keine PPL. Dadurch laeuft er ohne eine Zeile Mehrarbeit
    ueber ProGen2Scorer UND ESMPseudoPerplexityScorer, und der
    Cache-Fingerprint komponiert sich korrekt mit.

    Args:
        inner: Scorer, der ppl_column liefert (z.B. ProGen2Scorer()).
        ppl_column: Name der PPL-Spalte in inner.columns.
        n_perm: Zahl der Permutationen pro Sequenz. 50 ist ein brauchbarer
            Kompromiss; unter 20 wird z zu verrauscht. Bei n_perm = 1 ist
            sd per Definition (ddof=1) undefiniert -- dann sind sd und z NaN.
        seed: fuer Reproduzierbarkeit, gehoert in den fingerprint.
        min_distinct: unterhalb dieser Zahl distinkter Permutationen wird
            flag_lowcomplex gesetzt.

            Herleitung des Defaults: zieht man K Permutationen aus M
            unterscheidbaren, ist die erwartete Zahl von Dubletten nach dem
            Geburtstagsargument etwa K^2 / (2M). Fuer K = 50 heisst
            "weniger als 0,1 erwartete Dubletten" also M > K^2 / 0,2 = 12500.
            Darunter ist die gezogene Nullverteilung kein Abbild der echten
            mehr, sd ist zu klein und z dadurch zu gross. Bei anderem n_perm
            skaliert die Schwelle mit n_perm**2 -- 12500 passt zu K = 50.
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
        min_distinct: int = 12_500,
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
        2. n_perm Permutationen ziehen (rng aus sha1(sequence) + seed, NICHT
           aus dem Listenindex -- siehe Kommentar im Code).
        3. inner.score() auf ALLE Permutationen aller Sequenzen in EINEM Aufruf
           -- nicht pro Sequenz, sonst zahlst du den Modell-Overhead n_perm-mal.
        4. dL = mean_k L(pi_k) - L(s). Ueber die Permutationen das LOG mitteln,
           nicht die PPL: Mitteln auf der PPL-Skala und dann teilen ergibt
           wegen Jensen einen systematisch groesseren Wert.
        5. R = exp(dL), logR = dL, sd = std_k L(pi_k) mit ddof=1,
           z = dL / sd,
           p = (1 + #{k: L(pi_k) <= L(s)}) / (n_perm + 1).
           Das +1 verhindert p = 0 -- mit K Permutationen ist 1/(K+1) die
           feinste behauptbare Aufloesung.
        6. flag_lowcomplex, wenn n_distinct_permutations(s) < min_distinct.
           Dort ist R bedeutungslos.

        Hinweis zur Laufzeit: (n_perm + 1) Forward-Passes pro Sequenz. Bei
        ProGen2 billig, aber 50 000 x 51 ist unnoetig -- fuer einen
        Verteilungsvergleich reichen rund 2 000 Sequenzen pro Set. Die
        Stichprobengroesse gehoert in das Run-Manifest.

        Parameters
        ----------
        sequences : Sequence[str]
            Sequenzen in beliebiger Reihenfolge.

        Returns
        -------
        pd.DataFrame
            len(sequences) Zeilen in Eingabereihenfolge, Spalten wie
            self.columns. Nicht bewertbare Sequenzen ergeben eine NaN-Zeile.
        """
        seqs = list(sequences)
        n = len(seqs)
        K = self.n_perm

        # Permutationen ziehen. Seed aus sha1(sequence) + self.seed, bewusst
        # NICHT aus dem Listenindex: der Evaluator dedupliziert und uebergibt
        # Bloecke von flush_every Sequenzen, die Position einer Sequenz ist
        # also nicht stabil. So ist die Permutationsmenge eine Funktion der
        # Sequenz allein -- reihenfolgeunabhaengig und damit cachefest.
        perms: list[str] = []                                   # n * K Strings
        for seq in seqs:
            h = int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(h + self.seed)
            reste = list(seq)                                   # N Zeichen
            for _ in range(K):
                perms.append("".join(rng.permutation(reste)))

        # EIN Aufruf fuer Originale UND Permutationen. Layout der Eingabe:
        # [orig_0 .. orig_{n-1}, perm_0_0 .. perm_0_{K-1}, perm_1_0, ...] --
        # die Permutationen liegen sequenzweise hintereinander, deshalb
        # rekonstruiert ein einzelnes reshape(n, K) die Zuordnung.
        ppl = self.inner.score(seqs + perms)[self.ppl_column].to_numpy(float)
        L = np.log(ppl)                                         # (n + n*K,)
        L_orig = L[:n]                                          # (n,)
        L_perm = L[n:].reshape(n, K)                            # (n, K)

        dL = L_perm.mean(axis=1) - L_orig                       # (n,)
        sd = L_perm.std(axis=1, ddof=1)                         # (n,)

        # z nur dort, wo die Nullverteilung ueberhaupt streut. sd = 0 heisst,
        # alle Permutationen hatten denselben Verlust -- dann ist der Abstand
        # in Streuungseinheiten nicht definiert.
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 0, dL / sd, np.nan)               # (n,)

        # "<=" ist bewusst konservativ: zieht ein Shuffle zufaellig das
        # Original, zaehlt es als Treffer GEGEN uns, statt nachgezogen zu
        # werden. Vergleich gegen die Spalte L_orig[:, None] -> (n, 1) wird
        # gegen (n, K) gebroadcastet, Summe entlang der Permutationsachse.
        p = (1 + (L_perm <= L_orig[:, None]).sum(axis=1)) / (K + 1)   # (n,)

        # Ohne diese Maske waere p bei nicht bewertbaren Sequenzen 1/(K+1):
        # NaN <= NaN ist False, es wuerde also "keine Permutation war besser"
        # gezaehlt und die Sequenz saehe maximal signifikant aus. R, logR, sd
        # und z sind durch die NaN-Arithmetik schon NaN.
        unbewertbar = np.isnan(L_orig) | np.isnan(L_perm).any(axis=1)  # (n,)
        p = np.where(unbewertbar, np.nan, p)                    # (n,)

        # Reine Zeichenstatistik, braucht kein Modell -- deshalb auch fuer
        # nicht bewertbare Sequenzen gueltig und nie NaN.
        flag = np.array(
            [n_distinct_permutations(s) < self.min_distinct for s in seqs]
        )                                                       # (n,) bool

        werte = pd.DataFrame(
            {
                "R_shuffle": np.exp(dL),
                "logR_shuffle": dL,
                "z_shuffle": z,
                "p_shuffle": p,
                "sd_shuffle": sd,
                "flag_lowcomplex": flag,
            }
        )
        # Der Evaluator prueft die Spaltenreihenfolge streng gegen columns.
        return werte[self.columns]
