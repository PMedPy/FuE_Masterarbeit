#!/usr/bin/env python

"""Panel-Run (Branch A): Modellbewertung von Aguirre Small, Big und Torres.

Bezugsgröße ist die ungefilterte Rohmenge — hier wird NICHTS gefiltert.
Die Torres-Kaskade (MICFilter/SimilarityFilter/DiversityFilter) gehört in
Branch B und läuft in einem eigenen Skript.

Spezifikation: Projektdoku claude/eval_panel_spec.md

Dateinamen-Falle: results/50k_samples/ampdiffusion_650M_50k.faa enthält die
AGUIRRE-BIG-Sequenzen, nicht Torres. Torres' generierter 50k-Satz liegt als
data_tracked/torres_cellbiomaterial.csv, mit ihren eigenen APEX-MICs und
ppl-Werten (siehe Quercheck in Abschnitt 2).

Schritte:
0. Config anpassen (ESM-Scorer auf 150M!), Pfade
1. Bewertungsmengen in SequenceSets, inklusive Decke und Böden
2. Evaluation: bestehende Scorer + Panel-Scorer
3. Vote-Aggregation
4. Tabellen und Plots
5. Laufmetadaten
"""

import json
import os
import platform
import subprocess
from datetime import datetime

from aguirre.config import (
    CACHE_DIR,
    Config,
    DATA_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RUNS_DIR,
    TRACKED_DIR,
)
from aguirre.evaluation import Evaluator, SequenceSet, baue_scorer
from aguirre.panel import (
    AMPlifyScorer,
    AgreementMatrix,
    AniaScorer,
    AntiBP3Scorer,
    HemoPIScorer,
    MacrelScorer,
    ShuffleScorer,
    TrainSimilarityScorer,
    VoteRule,
    build_report,
    schreibe_split_fasta,
)


def git_stand(projekt):
    """Commit und ob uneingecheckte Aenderungen drin sind."""
    def g(*a):
        return subprocess.run(["git", "-C", str(projekt), *a],
                              capture_output=True, text=True).stdout.strip()
    commit = g("rev-parse", "--short", "HEAD")
    return {"commit": commit or None,
            "dirty": bool(g("status", "--porcelain")) if commit else None}


#===0. Config anpassen, Pfade + Definitionen===
laufname = f"eval_panel_{datetime.now():%Y-%m-%d_%H%M}"
RUN_PATH = RUNS_DIR / laufname
RUN_PATH.mkdir(parents=True, exist_ok=False)
cfg = Config()

# Neutraler Checkpoint, weder Smalls (8M) noch Bigs (650M) Encoder. Sonst
# würde Small von seinem eigenen Encoder bewertet und Big nicht — der
# Vergleich Small gegen Big wäre systematisch verzerrt.
cfg.esm_scorer_model_name = "esm2_t30_150M_UR50D"
cfg.esm_scorer_embed_dim = 640

SAMPLES_DIR = RESULTS_DIR / "50k_samples"
SMALL_PATH = SAMPLES_DIR / "aguirre-small_50k.faa"
BIG_PATH = SAMPLES_DIR / "ampdiffusion_650M_50k.faa"      # = Aguirre Big
TORRES_PATH = TRACKED_DIR / "torres_cellbiomaterial.csv"  # = Torres 50k

SPLIT_CSV = TRACKED_DIR / "split_t075.csv"
ALLE_FASTA = DATA_DIR / "all19K_unclustered.faa"
TRAIN_FASTA = TRACKED_DIR / "train_split.fasta"           # Referenz + Zeile
TEST_FASTA = TRACKED_DIR / "test_split.fasta"             # die Decke

# TODO(paul): Böden erzeugen (eigenes Skript 14_build_eval_sets.py)
UNIPROT_PATH = SAMPLES_DIR / "uniprot_nonamp.faa"
RANDOM_PATH = SAMPLES_DIR / "random_len_matched.faa"
SHUFFLED_PATH = SAMPLES_DIR / "shuffled_own.faa"

ANIA_CSV = TRACKED_DIR / "ania_preds.csv"                 # TODO: Batch-Lauf

#===1. Bewertungsmengen===
# Einmalig, danach idempotent: die Split-FASTAs werden an zwei Stellen
# gebraucht — der Train-Split als Referenz für max_ident_train, der
# Test-Split als Decke im Panel (echte AMPs, die das Modell nie gesehen hat).
for split, ziel in (("train", TRAIN_FASTA), ("test", TEST_FASTA)):
    if not ziel.is_file():
        schreibe_split_fasta(split=split, split_csv=SPLIT_CSV,
                             alle_fasta=ALLE_FASTA, out=ziel)

# Torres' CSV bringt 20 eigene Spalten mit (ihre APEX-MICs, ihre ppl). Die
# gehören nicht in unsere Score-Tabelle — sonst stehen zwei verschiedene
# MIC-Quellen unter ähnlichen Namen nebeneinander. Nur die Sequenz laden,
# der Quercheck gegen ihre Werte läuft getrennt.
torres_roh = SequenceSet.from_csv(path=TORRES_PATH, name="Torres")
torres = SequenceSet(name="Torres", data=torres_roh.data[["sequence"]].copy())

sets = {
    "aguirre_small": SequenceSet.from_fasta(path=SMALL_PATH, name="Aguirre Small"),
    "aguirre_big": SequenceSet.from_fasta(path=BIG_PATH, name="Aguirre Big"),
    "torres": torres,
    "real_amps_test": SequenceSet.from_fasta(path=TEST_FASTA, name="Echte AMPs (Test)"),
    "real_amps_train": SequenceSet.from_fasta(path=TRAIN_FASTA, name="Echte AMPs (Train)"),
    # TODO(paul): einkommentieren, sobald die Böden stehen.
    # "uniprot": SequenceSet.from_fasta(path=UNIPROT_PATH, name="UniProt Nicht-AMPs"),
    # "random_len": SequenceSet.from_fasta(path=RANDOM_PATH, name="Zufall (längenangepasst)"),
    # "shuffled": SequenceSet.from_fasta(path=SHUFFLED_PATH, name="Permutiert"),
}

#===2. Evaluation: bestehende + Panel-Scorer===
# ident_bekannt bleibt aus baue_scorer, zeigt aber jetzt auf die BREITE
# AMP-DB — cfg.amp_referenz_fasta umstellen, bisher war das torres_train.fasta
# und damit faktisch ein zweiter Memorisierungstest.
basis = ["aa_entropy", "progen2", "esm2_pppl", "apex", "ident_bekannt"]
scorers = baue_scorer(basis, cfg)

progen2 = next(sc for sc in scorers if sc.name == "progen2")
panel_scorers = [
    TrainSimilarityScorer.from_config(cfg),   # max_ident_train
    MacrelScorer(),
    AMPlifyScorer(),
    AntiBP3Scorer(),
    HemoPIScorer(),
    AniaScorer(csv_path=ANIA_CSV),
    # Dekorator über ProGen2, nicht über ESM: der Quotient kürzt dort nur die
    # kompositionelle Modellvorliebe, nicht die ordnungsbezogene.
    ShuffleScorer(inner=progen2, ppl_column="ppl_progen2", n_perm=50, seed=0),
]

cfg.scorer_auswahl = tuple(basis) + tuple(sc.name for sc in panel_scorers)
cfg.to_yaml(RUN_PATH / "config.yaml")

Eval = Evaluator(
    scorers=scorers + panel_scorers,
    cache_dir=CACHE_DIR / "scores",
)
sets = {k: Eval.run(seqset=v) for k, v in sets.items()}

# Quercheck APEX-Port: torres_roh enthält Torres' eigene average_MIC und ppl
# für genau diese 50 000 Sequenzen. Unsere mic_apex gegen ihre average_MIC
# validiert den portierten Inferenzpfad aus apex_net.py auf fremden Zahlen.
# TODO(paul): eigener kleiner Plot, gehört nicht ins Panel.

#===3. Vote-Aggregation===
regel = VoteRule(min_votes=2)   # Mehrheit; T2 berichtet alle drei Stufen
sets = {k: regel.apply(v) for k, v in sets.items()}
kappa = {k: AgreementMatrix().compute(v) for k, v in sets.items()}

#===4. Tabellen und Plots===
metriken = [
    "mic_apex", "mic_ania", "ppl_progen2", "logR_shuffle", "pppl_esm2",
    "aa_entropy", "max_ident_train", "max_ident_bekannt", "p_hemo_hemopi",
]
report = build_report(sets=sets, metriken=metriken, regel=regel)
report.save(out_dir=RUN_PATH, stem=laufname)

#===5. Laufmetadaten===
(RUN_PATH / "lauf_metadaten.json").write_text(json.dumps({
    "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
    "git": git_stand(PROJECT_ROOT),
    "host": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "eingaben": {k: str(v) for k, v in {
        "aguirre_small": SMALL_PATH, "aguirre_big": BIG_PATH,
        "torres": TORRES_PATH, "train_split": TRAIN_FASTA,
        "test_split": TEST_FASTA, "ania_csv": ANIA_CSV,
    }.items()},
    "esm_scorer": cfg.esm_scorer_model_name,
    "referenzen": {
        "max_ident_train": str(cfg.train_referenz_fasta),
        "max_ident_bekannt": str(cfg.amp_referenz_fasta),
    },
    "scorer": list(cfg.scorer_auswahl),
    "fingerabdruecke": {sc.name: sc.fingerprint for sc in scorers + panel_scorers},
    "vote_regel": {"spalten": list(regel.spalten), "schwelle": regel.schwelle,
                   "min_votes": regel.min_votes, "nan_policy": regel.nan_policy},
    "n_sequenzen": {k: len(v.data) for k, v in sets.items()},
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Panel-Run geschrieben nach {RUN_PATH}")
for name, m in kappa.items():
    print(f"\n=== kappa {name} ===\n{m}")
