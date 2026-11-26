"""panel -- Konsens-Panel fuer die Modellbewertung (Branch A).

Ergaenzt evaluation.py um das, was dort fehlt: Fremdtool-Scorer ueber CLI,
den Shuffle-Score, das zweite Identitaetsmass und die Vote-Aggregation.
Spezifikation siehe Projektdoku claude/eval_panel_spec.md.

Abhaengigkeit zeigt NUR in eine Richtung:

    panel  ---->  evaluation        (Scorer, SequenceSet, Filter, Statistik)
    panel  --X-->                   evaluation importiert nie aus panel

evaluation.py ist eingefroren (Stand der Laeufe 1-12, Grundlage der
FuE-Pruefung). Fehlt dort etwas, wird es hierher kopiert -- nicht refactort.

Module:

    external      SubprocessScorer + Macrel, AMPlify, AntiBP3, HemoPI
    precomputed   CsvScorer: Scores aus einer CSV statt aus einem CLI (ANIA)
    shuffle       ShuffleScorer: Dekorator ueber einen PPL-Scorer
    similarity    TrainSimilarityScorer (max_ident_train) neben dem
                  KnownAMPSimilarityScorer aus evaluation (max_ident_bekannt)
    aggregate     Aggregator, VoteRule, AgreementMatrix
    report        Tabellen T1-T7 und Plots P1-P9

Datenfluss (Branch A, ungefiltert -- hier wird nichts verworfen):

    SequenceSet --Evaluator(Scorer... + panel-Scorer)--> SequenceSet
                --VoteRule--> SequenceSet (+ n_votes, is_amp, vote_pattern)
                --build_report({...Sets inkl. Baselines})--> PanelReport
"""

from __future__ import annotations

from aguirre.evaluation import KnownAMPSimilarityScorer
from aguirre.panel.aggregate import AgreementMatrix, Aggregator, VoteRule
from aguirre.panel.external import (
    AMPlifyScorer,
    AntiBP3Scorer,
    DummyVoteScorer,
    HemoPIScorer,
    MacrelScorer,
    SubprocessScorer,
)
from aguirre.panel.precomputed import AniaScorer, CsvScorer
from aguirre.panel.report import PanelReport, build_report
from aguirre.panel.shuffle import ShuffleScorer
from aguirre.panel.similarity import (
    TrainSimilarityScorer,
    baue_similarity_scorer,
    schreibe_split_fasta,
)

__all__ = [
    "AMPlifyScorer",
    "AgreementMatrix",
    "Aggregator",
    "AniaScorer",
    "AntiBP3Scorer",
    "CsvScorer",
    "DummyVoteScorer",
    "HemoPIScorer",
    "KnownAMPSimilarityScorer",
    "MacrelScorer",
    "PanelReport",
    "ShuffleScorer",
    "SubprocessScorer",
    "TrainSimilarityScorer",
    "VoteRule",
    "baue_similarity_scorer",
    "build_report",
    "schreibe_split_fasta",
]
