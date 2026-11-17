#!/usr/bin/env python

"""Vergleich von großem und kleinen Aguirremodell (Big: 821,5 M Params, Small 18,2 M Params)
Schritte:
0. Config anpassen
1. Daten Sammeln, Aufbereiten und in SequenceSets übertragen
"""

from datetime import datetime

from aguirre.config import Config, RESULTS_DIR, RUNS_DIR, CACHE_DIR
from aguirre.evaluation import (
    SequenceSet,
    Evaluator,
    MICFilter,
    SimilarityFilter,
    DiversityFilter,
    baue_scorer,
    kaskade,
    compare,
)

#===0. Config anpassen, Pfade + Definitionen===
laufname = f"eval_3_small_vs_big_{datetime.now():%Y-%m-%d_%H%M}"
RUN_PATH = RUNS_DIR / laufname
RUN_PATH.mkdir(parents= True, exist_ok= False)
cfg = Config()

cfg.esm_scorer_model_name = "esm2_t6_8M_UR50D"
cfg.esm_scorer_embed_dim = 320

cfg.to_yaml(RUN_PATH / "config.yaml")

SAMPLES_DIR = RESULTS_DIR / "50k_samples"
SMALL_PATH = SAMPLES_DIR / "aguirre-small_50k.faa"
BIG_PATH = SAMPLES_DIR / "ampdiffusion_650M_50k.faa"

scorer = ["aa_entropy", "progen2", "esm2_pppl", "apex", "ident_bekannt"]
filters = [
    MICFilter.from_config(cfg_in= cfg),
    SimilarityFilter.from_config(cfg_in= cfg),
    DiversityFilter.from_config(cfg_in= cfg)
]

#===1. Daten in SequenceSets
Small_set = SequenceSet.from_fasta(path= SMALL_PATH, name= "Aguirre Small")
Big_set = SequenceSet.from_fasta(path= BIG_PATH, name= "Aguirre Big")

#==2. Evaluation mit Evaluatorklasse==
Eval = Evaluator(
    scorers= baue_scorer(scorer, cfg),
    cache_dir= CACHE_DIR / "scores",
)

Small_set_eval = Eval.run(seqset= Small_set)
Big_set_eval = Eval.run(seqset= Big_set)

#==3. Filter==

Small_set_eval_filtered, small_stufen = kaskade(seqset= Small_set_eval, filters= filters)
Big_set_eval_filtered, big_stufen = kaskade(seqset= Big_set_eval, filters= filters)

#==4. Vergleicher==
gruppe1 = {"Aguirre Small": Small_set_eval, "Aguirre Big": Big_set_eval}
gruppe2 = {"Aguirre Small (gefiltert)": Small_set_eval_filtered, "Aguirre Big (gefiltert)": Big_set_eval_filtered}

vergleich_1 = compare(
    groups= gruppe1,
    out_dir= RUN_PATH,
)
vergleich_2 = compare(
    groups= gruppe2,
    out_dir= RUN_PATH,
)
print(f"{len(vergleich_1) + len(vergleich_2)} Vergleiche geschrieben nach {RUN_PATH}")
