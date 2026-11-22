"""similarity.py -- zwei Identitaetsmasse mit zwei verschiedenen Referenzen.

KnownAMPSimilarityScorer in evaluation.py kann beides, aber name und columns
sind dort KLASSENattribute: zwei Instanzen mit verschiedenen Referenzen
hiessen beide "ident_bekannt" und schrieben beide in max_ident_bekannt --
der Evaluator wuerde die eine mit der anderen ueberschreiben. Deshalb hier
eine Unterklasse, die nur Name, Spalte und Fingerprint umbiegt. score()
benutzt durchgaengig self.columns[0], der Rest traegt also unveraendert.

    max_ident_train     Referenz = NUR unser Train-Split (15 507)
                        -> Memorisierungstest. Hat das Modell abgeschrieben?
    max_ident_bekannt   Referenz = breite AMP-DB (DRAMP + DBAASP + APD3 + neu)
                        -> Novelty. Ist das Peptid neu fuer das Feld?

Bisher zeigte cfg.amp_referenz_fasta auf torres_train.fasta (19 670 Sequenzen,
also train + val + test und etwas mehr). Damit war max_ident_bekannt faktisch
ein unscharfer Memorisierungstest und keine Novelty-Aussage. Die beiden Rollen
brauchen zwei Dateien.

Gemessen wird in beiden Faellen fident * qcov -- der Anteil der QUERY, der
identisch zu etwas in der Referenz ist. Kein Treffer ergibt 0.0, nicht NaN:
"nichts gefunden" heisst maximal unaehnlich, also bestanden.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aguirre.config import Config
from aguirre.data import read_fasta, write_fasta
from aguirre.evaluation import KnownAMPSimilarityScorer


class TrainSimilarityScorer(KnownAMPSimilarityScorer):
    """Maximale Identitaet zum eigenen Train-Split -- der Memorisierungstest.

    Der Fehlermodus, den weder Perplexitaet noch Entropie noch APEX sehen: ein
    Modell, das Trainingsdaten reproduziert, gewinnt in allen dreien. Nur
    gegen den TRAIN-Split rechnen, nicht gegen alle 19 384 -- sonst sieht eine
    Sequenz, die einer Test-Sequenz gleicht, memoriert aus, obwohl das Modell
    sie nie gesehen hat, und die beiden Faelle sind nicht mehr trennbar.
    """

    name = "ident_train"
    columns = ["max_ident_train"]

    @property
    def fingerprint(self) -> str:
        # Die Eltern-Property hat "ident_bekannt" hart drin. Ohne das Umbiegen
        # sehen beide Varianten im Cache gleich aus, falls jemand versehentlich
        # dieselbe Referenz-FASTA einhaengt.
        return super().fingerprint.replace("ident_bekannt", self.name, 1)

    @classmethod
    def from_config(cls, cfg_in: Config) -> "TrainSimilarityScorer":
        """Braucht cfg.train_referenz_fasta.

        TODO(paul): Config-Eintrag anlegen, z.B.
            train_referenz_fasta: Path = TRACKED_DIR / "train_split.fasta"
        und mit schreibe_split_fasta("train", ...) einmalig erzeugen.
        """
        return cls(
            referenz=cfg_in.train_referenz_fasta,
            cov_mode=cfg_in.similarity_cov_mode,
            coverage=cfg_in.similarity_coverage,
            evalue=cfg_in.similarity_evalue,
        )


def baue_similarity_scorer(
    namen: tuple[str, ...], cfg_in: Config
) -> list[KnownAMPSimilarityScorer]:
    """Wie baue_scorer in evaluation.py, nur fuer die beiden Identitaetsmasse.

    Erlaubt: "ident_train", "ident_bekannt". Lambdas als Bauplaene, damit
    nichts instanziiert wird, was nicht angefordert war -- der Konstruktor
    prueft die Referenzdatei und wirft sonst FileNotFoundError.
    """
    raise NotImplementedError


def schreibe_split_fasta(
    split: str,
    split_csv: Path,
    alle_fasta: Path,
    out: Path,
) -> Path:
    """Schreibt die Sequenzen EINES Splits als FASTA.

    Gebraucht an zwei Stellen: als Referenz fuer TrainSimilarityScorer
    (split="train") und als Bewertungsmenge fuer die Decke im Panel
    (split="test", echte AMPs, die das Modell nie gesehen hat).

    Args:
        split: "train" | "val" | "test".
        split_csv: data_tracked/split_t075.csv, Spalten id, cluster, split.
        alle_fasta: data/all19K_unclustered.faa, enthaelt alle 19 384 mit ids.
        out: Zielpfad.

    Achtung: torres_train.fasta hat 19 670 Sequenzen, der Split nur 19 384 --
    beim Clustern sind 286 weggefallen. Ueber die ids aus split_csv gehen,
    nicht ueber Positionen.

    Returns:
        out, damit der Aufruf direkt in einen Scorer-Konstruktor passt.
    """
    raise NotImplementedError
