#!/usr/bin/env python
"""10_eval_run1.py -- Evaluationslauf: Trainingssatz gegen generierte Sequenzen.

Verdrahtung, sonst nichts. Alles Inhaltliche steht in aguirre.evaluation --
sonst laege die Logik an zwei Stellen und waere nicht testbar.

Ablauf:
    1. CLI parsen, Config laden, Overrides anwenden
    2. Eingabedateien pruefen            <- scheitern BEVOR Modelle laden
    3. Filter/Scorer-Abhaengigkeiten pruefen
    4. Lauf-Ordner anlegen, config.yaml SOFORT schreiben
    5. Daten laden
    6. LengthFilter auf die generierten (APEX ueberspringt > 50 Residuen still)
    7. Scorer bauen, Evaluator mit Cache
    8. Scoren
    9. Filterkaskade MIC -> Similarity -> Diversity (Torres' Reihenfolge)
    10. Vergleiche
    11. lauf_metadaten.json
    12. Zusammenfassung

Beispiele:
    # Probelauf: 500 Sequenzen, nur die billigen Scorer
    python wrath_of_god/scripts/10_eval_run1.py \\
        --train wrath_of_god/data_tracked/torres_train_sequences.csv \\
        --generated wrath_of_god/results/50k_samples/ampdiffusion_650M_50k.csv \\
        --out wrath_of_god/runs/probe \\
        --limit 500 --scorer aa_entropy ident_bekannt

    # Voller Lauf (siehe wrath_of_god/scripts/eval.slurm)
    python wrath_of_god/scripts/10_eval_run1.py --train ... --generated ... --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from aguirre.config import Config, CACHE_DIR, PROJECT_ROOT
from aguirre.evaluation import (
    DiversityFilter,
    Evaluator,
    Filter,
    LengthFilter,
    MICFilter,
    SequenceSet,
    SimilarityFilter,
    baue_scorer,
    run_evaluation,
)

# Welcher Filter welche Spalte braucht und welcher Scorer sie liefert.
# None = der Filter kommt ohne Scorer aus.
FILTER_BEDARF: dict[str, tuple[str | None, str | None]] = {
    "LengthFilter":     (None, None),              # "length" kommt vom SequenceSet
    "MICFilter":        ("mic_apex", "apex"),
    "SimilarityFilter": ("max_ident_bekannt", "ident_bekannt"),
    "DiversityFilter":  (None, None),              # braucht mmseqs, keinen Scorer
}


# ---------------------------------------------------------------------------
# Helfer
# ---------------------------------------------------------------------------

def git_stand(projekt: Path) -> dict:
    """Code-Stand dieses Laufs.

    "dirty" ist der wichtige Teil: Mit uneingecheckten Aenderungen zeigt der
    Hash auf einen Stand, der so nie gelaufen ist.
    """
    def lauf(*args: str) -> str:
        return subprocess.run(
            args, cwd=projekt, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": lauf("git", "rev-parse", "--short", "HEAD"),
            "branch": lauf("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(lauf("git", "status", "--porcelain")),
        }
    except Exception as fehler:                      # kein git, kein Repo, egal
        return {"commit": None, "branch": None, "dirty": None,
                "fehler": str(fehler)}


def lade_set(pfad: Path, name: str, limit: int | None, seed: int) -> SequenceSet:
    """CSV oder FASTA -> SequenceSet.

    --limit zieht eine ZUFALLSSTICHPROBE, nicht die ersten N: Die Reihenfolge
    einer Generierungsdatei ist nicht zufaellig, und head(N) verzerrt jede
    daraus geschaetzte Rate.
    """
    pfad = Path(pfad)
    if pfad.suffix.lower() in (".fasta", ".faa", ".fa"):
        seqset = SequenceSet.from_fasta(pfad, name)
        df = seqset.data
    else:
        df = pd.read_csv(pfad)
        if "sequence" not in df.columns:
            raise ValueError(
                f"{pfad} hat keine Spalte 'sequence'. Vorhanden: "
                f"{list(df.columns)}"
            )
        behalten = [c for c in ("name", "id", "sequence") if c in df.columns]
        df = df[behalten].copy()

    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
        print(f"[{name}] auf {limit} Sequenzen begrenzt (Zufallsstichprobe).")

    return SequenceSet(name=name, data=df.reset_index(drop=True))


def pruefe_abhaengigkeiten(filters: list[Filter], scorer_namen: list[str]) -> None:
    """Bricht ab, wenn ein Filter eine Spalte braucht, die kein Scorer liefert.

    Lieber hier als nach zwei Stunden ProGen2 -- oder, schlimmer, als ein
    stillschweigend uebersprungener Filter.
    """
    fehlt = []
    for f in filters:
        spalte, scorer = FILTER_BEDARF.get(type(f).__name__, (None, None))
        if scorer and scorer not in scorer_namen:
            fehlt.append(f"{type(f).__name__} braucht '{spalte}' aus Scorer '{scorer}'")
    if fehlt:
        raise SystemExit(
            "Filter und Scorer-Auswahl passen nicht zusammen:\n  "
            + "\n  ".join(fehlt)
            + f"\nGewaehlte Scorer: {scorer_namen}"
            + "\nEntweder --scorer erweitern oder --no-filter benutzen."
        )


def baue_filter(cfg: Config) -> list[Filter]:
    """Torres' Kriterien (1)-(3), in seiner Reihenfolge.

    MIC zuerst, weil billig und drastisch: aus 50.000 werden einige hundert.
    Diversitaet zuletzt, weil All-gegen-All quadratisch ist und auf 50.000
    Kandidaten unbezahlbar waere.
    """
    return [
        MICFilter.from_config(cfg),
        SimilarityFilter.from_config(cfg),
        DiversityFilter.from_config(cfg),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluation: Trainingssatz gegen generierte Sequenzen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train", required=True, type=Path,
                   help="CSV oder FASTA des Trainingssatzes")
    p.add_argument("--generated", required=True, type=Path,
                   help="CSV oder FASTA der generierten Sequenzen")
    p.add_argument("--out", required=True, type=Path,
                   help="Zielordner des Laufs")
    p.add_argument("--scorer", nargs="+", default=None, metavar="NAME",
                   help="ueberschreibt cfg.scorer_auswahl")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="nur N generierte Sequenzen (Zufallsstichprobe) -- fuer Probelaeufe")
    p.add_argument("--no-cache", action="store_true",
                   help="Score-Cache umgehen (alles neu rechnen)")
    p.add_argument("--no-filter", action="store_true",
                   help="nur scoren und vergleichen, nicht filtern")
    p.add_argument("--no-length-filter", action="store_true",
                   help="Laengenfilter vor dem Scoren ueberspringen")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()

    # ---- 1. Config ------------------------------------------------------
    cfg = Config()
    if args.scorer:
        cfg.scorer_auswahl = tuple(args.scorer)
    scorer_namen = list(cfg.scorer_auswahl)

    # ---- 2. Eingaben pruefen, BEVOR irgendein Modell laedt ---------------
    for bez, pfad in (("--train", args.train), ("--generated", args.generated)):
        if not Path(pfad).is_file():
            raise SystemExit(f"{bez}: Datei nicht gefunden: {pfad}")

    # ---- 3. Abhaengigkeiten ---------------------------------------------
    filters = [] if args.no_filter else baue_filter(cfg)
    pruefe_abhaengigkeiten(filters, scorer_namen)

    # ---- 4. Lauf-Ordner, Config SOFORT schreiben ------------------------
    # Ein abgebrochener Lauf ohne config.yaml ist nicht rekonstruierbar --
    # und genau die brechen ab.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "config.yaml")

    print(f"=== Evaluationslauf -> {out_dir} ===")
    print(f"Scorer : {scorer_namen}")
    print(f"Filter : {[type(f).__name__ for f in filters] or 'keine'}")

    # ---- 5. Daten --------------------------------------------------------
    train = lade_set(args.train, "train", None, cfg.seed)
    generiert = lade_set(args.generated, "generiert", args.limit, cfg.seed)
    print(f"train {len(train.data)} | generiert {len(generiert.data)}")

    # ---- 6. Laenge zuerst ------------------------------------------------
    # APEX ueberspringt Sequenzen ueber 50 Residuen STILL. Besser vorher
    # wegfiltern als sie als NaN durch die Auswertung schleppen.
    if not args.no_length_filter:
        generiert = LengthFilter.from_config(cfg).apply(generiert)

    # ---- 7. Scorer und Evaluator ----------------------------------------
    scorers = baue_scorer(scorer_namen, cfg)
    evaluator = Evaluator(
        scorers=scorers,
        cache_dir=None if args.no_cache else CACHE_DIR / "scores",
    )
    fingerabdruecke = {sc.name: sc.fingerprint for sc in scorers}
    for name, fp in fingerabdruecke.items():
        print(f"  {name:16s} {fp}")

    # ---- 8.-10. Scoren, filtern, vergleichen ----------------------------
    ergebnisse, stufen = run_evaluation(
        train=train,
        generated=generiert,
        evaluator=evaluator,
        filters=filters,
        out_dir=out_dir,
    )

    # ---- 11. Metadaten ---------------------------------------------------
    dauer = time.perf_counter() - start
    metadaten = {
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        "dauer_s": round(dauer, 1),
        "git": git_stand(PROJECT_ROOT),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version.split()[0],
        "eingaben": {
            "train": str(args.train),
            "generated": str(args.generated),
            "limit": args.limit,
        },
        "scorer": scorer_namen,
        "fingerabdruecke": fingerabdruecke,
        "filter": [type(f).__name__ for f in filters],
        "filterstufen": [{"stufe": s, "n": n} for s, n in stufen],
        "vergleiche": sorted(ergebnisse),
    }
    (out_dir / "lauf_metadaten.json").write_text(
        json.dumps(metadaten, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 12. Zusammenfassung --------------------------------------------
    print("\n=== Filterkaskade ===")
    for stufe, n in stufen:
        print(f"  {stufe:20s} {n:>8d}")
    print(f"\n{len(ergebnisse)} Vergleiche, {dauer / 60:.1f} min")
    if metadaten["git"]["dirty"]:
        print("WARNUNG: uneingecheckte Aenderungen -- der Commit-Hash in den "
              "Metadaten beschreibt diesen Lauf NICHT vollstaendig.")
    print(f"fertig: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())