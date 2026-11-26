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
    DB_EXTERN_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RUNS_DIR,
    SPLIT_DIR,
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

# Alle Bewertungsmengen kommen aus Skript 14. Dieses Skript baut keine mehr
# selbst — sonst gäbe es zwei Konventionen im Repo und die Decke wäre je nach
# Aufrufreihenfolge eine andere Menge.
TRAIN_FASTA = SPLIT_DIR / "real_amps_train.fasta"         # Referenz + Zeile
DECKE_CSV = SPLIT_DIR / "real_amps_test.csv"              # die Decke
TEST_TORRES_FASTA = SPLIT_DIR / "real_amps_test_torres.fasta"
BEKANNT_FASTA = DB_EXTERN_DIR / "amps_bekannt.fasta"

# TODO(paul): Böden erzeugen (Abschnitt in 14_eval_external_data.py)
UNIPROT_PATH = SAMPLES_DIR / "uniprot_nonamp.faa"
RANDOM_PATH = SAMPLES_DIR / "random_len_matched.faa"
SHUFFLED_PATH = SAMPLES_DIR / "shuffled_own.faa"

# Die zwei Ähnlichkeitsreferenzen. Sie beantworten verschiedene Fragen:
# max_ident_train fragt "memoriert?", max_ident_bekannt fragt "neu?".
# Default von amp_referenz_fasta ist torres_train.fasta — damit wäre das
# zweimal dieselbe Frage. Hier auf die vereinigte Datenbankmenge umstellen.
cfg.train_referenz_fasta = TRAIN_FASTA
cfg.amp_referenz_fasta = BEKANNT_FASTA

ANIA_CSV = TRACKED_DIR / "ania_preds.csv"                 # TODO: Batch-Lauf

#===1. Bewertungsmengen===
fehlend = [q for q in (TRAIN_FASTA, DECKE_CSV, BEKANNT_FASTA) if not q.is_file()]
if fehlend:
    raise FileNotFoundError(
        "Erst scripts/14_eval_external_data.py laufen lassen. Es fehlt: "
        + ", ".join(str(q) for q in fehlend))

# Die Decke kommt als CSV, nicht als FASTA, wegen der Spalte `kanonisch`.
# Bewertet wird nur, was aus den 20 proteinogenen Aminosäuren besteht: die
# generierten Mengen sind zu 100 % kanonisch, die Decke ohne Filter zu rund
# 80 %. Beides mit denselben Tools zu messen und als Vergleich auszugeben
# wäre zweierlei Messung — und die Fehlerrichtung die ungünstige, weil
# unplausible Werte auf der Decke sie nach unten ziehen und unser Modell
# besser aussehen lassen. Die Datei bleibt vollständig, die Zahl der
# Übersprungenen steht unten in den Laufmetadaten.
decke_alle = SequenceSet.from_csv(path=DECKE_CSV, name="Echte AMPs (Test)")
decke = decke_alle.subset(decke_alle.data["kanonisch"], name="Echte AMPs (Test)")
n_uebersprungen = len(decke_alle.data) - len(decke.data)
print(f"[decke] {len(decke.data)} von {len(decke_alle.data)} bewertet, "
      f"{n_uebersprungen} nicht-kanonisch übersprungen")

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
    "real_amps_test": decke,
    "real_amps_train": SequenceSet.from_fasta(path=TRAIN_FASTA, name="Echte AMPs (Train)"),
    # Der reine Torres-Testsplit (1 938), gleiche Verteilung wie train. Nur
    # nötig, wenn du zeigen willst, dass die erweiterte Decke sich anders
    # verhält als Torres' eigene Testmenge. Kostet kaum etwas.
    # "real_amps_test_torres": SequenceSet.from_fasta(
    #     path=TEST_TORRES_FASTA, name="Echte AMPs (Test, nur Torres)"),
    # TODO(paul): einkommentieren, sobald die Böden stehen.
    # "uniprot": SequenceSet.from_fasta(path=UNIPROT_PATH, name="UniProt Nicht-AMPs"),
    # "random_len": SequenceSet.from_fasta(path=RANDOM_PATH, name="Zufall (längenangepasst)"),
    # "shuffled": SequenceSet.from_fasta(path=SHUFFLED_PATH, name="Permutiert"),
}

#===2. Evaluation: bestehende + Panel-Scorer===
# ident_bekannt bleibt aus baue_scorer und zeigt jetzt auf amps_bekannt.fasta
# (in Abschnitt 0 gesetzt): DBAASP, DRAMP general, DRAMP patent und APD,
# 41 791 Sequenzen. Ohne den Patentsatz wären es 26 853 und die Neuheit
# systematisch überschätzt — die Patente decken 90 % Sequenzen ab, die in
# keiner der anderen drei Quellen stehen.
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
        "torres": TORRES_PATH, "real_amps_train": TRAIN_FASTA,
        "real_amps_test": DECKE_CSV, "ania_csv": ANIA_CSV,
    }.items()},
    "decke": {"gesamt": len(decke_alle.data), "bewertet": len(decke.data),
              "nicht_kanonisch": n_uebersprungen},
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
