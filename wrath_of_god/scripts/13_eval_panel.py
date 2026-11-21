#!/usr/bin/env python

"""Panel-Run (Branch A): Modellbewertung von Torres, Aguirre Small und Big.

Bezugsgröße ist die ungefilterte Rohmenge — hier wird NICHTS gefiltert.
Die Torres-Kaskade (MICFilter/SimilarityFilter/DiversityFilter) gehört in
Branch B und läuft in einem eigenen Skript.

Spezifikation: Projektdoku claude/eval_panel_spec.md

Schritte:
0. Config anpassen (ESM-Scorer auf 150M!), Pfade
1. Daten in SequenceSets, inklusive der vier Baselines
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

from aguirre.config import Config, PROJECT_ROOT, RESULTS_DIR, RUNS_DIR, CACHE_DIR, TRACKED_DIR, DATA_RAW
from aguirre.evaluation import Evaluator, SequenceSet, baue_scorer
from aguirre.panel import (
    AMPlifyScorer,
    AgreementMatrix,
    AniaScorer,
    AntiBP3Scorer,
    HemoPIScorer,
    MacrelScorer,
    ShuffleScorer,
    VoteRule,
    build_report,
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

# WICHTIG: neutraler Checkpoint, weder Smalls (8M) noch Bigs (650M) Encoder.
# Sonst würde Small von seinem eigenen Encoder bewertet und Big nicht — der
# Vergleich Small gegen Big wäre systematisch verzerrt.
cfg.esm_scorer_model_name = "esm2_t30_150M_UR50D"
cfg.esm_scorer_embed_dim = 640

SAMPLES_DIR = RESULTS_DIR / "50k_samples"
TORRES_PATH = SAMPLES_DIR / "ampdiffusion_650M_50k.faa"
SMALL_PATH = SAMPLES_DIR / "aguirre-small_50k.faa"
BIG_PATH = SAMPLES_DIR / "aguirre-big_50k.faa"

# TODO(paul): Baselines erzeugen und hier eintragen. Ohne train_amps gibt es
# keine Decke und die Panel-Tabelle ist nicht interpretierbar.
TRAIN_PATH = DATA_RAW
UNIPROT_PATH = SAMPLES_DIR / "uniprot_nonamp.faa"       # TODO
RANDOM_PATH = SAMPLES_DIR / "random_len_matched.faa"    # TODO
SHUFFLED_PATH = SAMPLES_DIR / "shuffled_own.faa"        # TODO

ANIA_CSV = TRACKED_DIR / "ania_preds.csv"          # TODO: Batch-Lauf

#===1. Daten in SequenceSets===
# TODO(paul): Baselines ergänzen, sobald die FASTAs stehen.
sets = {
    "torres": SequenceSet.from_fasta(path=TORRES_PATH, name="Torres"),
    "aguirre_small": SequenceSet.from_fasta(path=SMALL_PATH, name="Aguirre Small"),
    "aguirre_big": SequenceSet.from_fasta(path=BIG_PATH, name="Aguirre Big"),
    "train_amps": SequenceSet.from_csv(path=TRAIN_PATH, name="Train-AMPs"),
}

#===2. Evaluation: bestehende + Panel-Scorer===
basis = ["aa_entropy", "progen2", "esm2_pppl", "apex", "ident_bekannt"]
scorers = baue_scorer(basis, cfg)

progen2 = next(sc for sc in scorers if sc.name == "progen2")
panel_scorers = [
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

#===3. Vote-Aggregation===
# Mehrheit (2 von 3) als Standardregel; T2 berichtet alle drei Stufen.
regel = VoteRule(min_votes=2)
sets = {k: regel.apply(v) for k, v in sets.items()}
kappa = {k: AgreementMatrix().compute(v) for k, v in sets.items()}

#===4. Tabellen und Plots===
metriken = [
    "mic_apex", "mic_ania", "ppl_progen2", "logR_shuffle",
    "pppl_esm2", "aa_entropy", "ident_bekannt", "p_hemo_hemopi",
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
        "torres": TORRES_PATH, "small": SMALL_PATH, "big": BIG_PATH,
        "train": TRAIN_PATH, "ania_csv": ANIA_CSV,
    }.items()},
    "esm_scorer": cfg.esm_scorer_model_name,
    "scorer": list(cfg.scorer_auswahl),
    "fingerabdruecke": {sc.name: sc.fingerprint for sc in scorers + panel_scorers},
    "vote_regel": {"spalten": list(regel.spalten), "schwelle": regel.schwelle,
                   "min_votes": regel.min_votes, "nan_policy": regel.nan_policy},
    "n_sequenzen": {k: len(v.data) for k, v in sets.items()},
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Panel-Run geschrieben nach {RUN_PATH}")
for name, m in kappa.items():
    print(f"\n=== kappa {name} ===\n{m}")
