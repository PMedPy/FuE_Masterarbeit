#!/usr/bin/env python
"""11_eval_run_torres_vs_aguirre.py -- Torres' generierte Sequenzen gegen unsere.

Vier Gruppen, beide Saetze durch dieselbe Kaskade:

    torres              Torres' 50.000 (Cell Biomaterials 2025)
    aguirre             unsere 50.000 (AMP-Diffusion 650M)
    torres_gefiltert    nach MIC -> Similarity -> Diversity
    aguirre_gefiltert   dieselbe Kaskade, dieselben Schwellen

Verglichen wird in drei Runden, damit kein Plot ueberfuellt ist:

    roh         torres gegen aguirre -- Verteilungen, Komposition, Positionen
    gefiltert   torres_gefiltert gegen aguirre_gefiltert -- dasselbe

Zwei Gruppen je Abbildung, nie vier: ein Violinplot mit vier Gruppen zeigt
nichts, was diese beiden Runden nicht lesbarer zeigen.

Zusaetzlich zwei Tabellen:

    top30_...csv   je 30 Sequenzen aus den gefilterten Saetzen, zwei Spalten,
                   sortiert nach vorhergesagtem MIC (bestes zuerst)
    ppl_...csv     unsere ProGen2-Perplexitaet gegen Torres' mitgelieferte
                   ppl-Spalte, auf identischen Sequenzen -- prueft, ob unsere
                   Perplexitaetsrechnung seiner entspricht

Alles Inhaltliche steht in aguirre.evaluation. Hier nur Verdrahtung.

Beispiele:
    # Probelauf, billige Scorer
    python wrath_of_god/scripts/11_eval_run_torres_vs_aguirre.py \\
        --out wrath_of_god/runs/probe_tva --limit 500 \\
        --scorer aa_entropy ident_bekannt apex

    # Voller Lauf: siehe wrath_of_god/scripts/eval_torres_vs_aguirre.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from aguirre.config import Config, CACHE_DIR
from aguirre.evaluation import (
    STANDARD_SCORE_SPALTEN,
    LOG_SPALTEN,
    CompositionComparison,
    DistributionComparison,
    Evaluator,
    Filter,
    LengthFilter,
    MICFilter,
    PositionalComparison,
    SequenceSet,
    SimilarityFilter,
    DiversityFilter,
    baue_scorer,
    build_stem,
)

TORRES = "wrath_of_god/data_tracked/torres_cellbiomaterial.csv"
AGUIRRE = "wrath_of_god/results/50k_samples/ampdiffusion_650M_50k.csv"


def lade(pfad: Path, name: str, limit: int | None, seed: int,
         zusatz: tuple[str, ...] = ()) -> SequenceSet:
    """CSV -> SequenceSet. Behaelt nur "sequence" und was in zusatz steht.

    Torres' CSV bringt 19 weitere Spalten mit (elf Erregerstaemme,
    average_MIC, physikochemische Werte). Die wollen wir nicht mitschleppen,
    nur "ppl" fuer den Abgleich.

    --limit zieht eine ZUFALLSSTICHPROBE, nicht die ersten N: die Reihenfolge
    einer Generierungsdatei ist nicht zufaellig.
    """
    df = pd.read_csv(pfad)
    if "sequence" not in df.columns:
        raise SystemExit(f"{pfad}: keine Spalte 'sequence'. Vorhanden: {list(df.columns)}")

    behalten = ["sequence"] + [c for c in zusatz if c in df.columns]
    df = df[behalten].copy()

    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
        print(f"[{name}] auf {limit} Sequenzen begrenzt (Zufallsstichprobe).")

    return SequenceSet(name=name, data=df.reset_index(drop=True))


def kaskade(seqset: SequenceSet, filters: list[Filter]) -> tuple[SequenceSet, list[dict]]:
    """Wendet die Filter in Reihenfolge an und protokolliert jede Stufe."""
    stufen = [{"satz": seqset.name, "stufe": "ungefiltert", "n": len(seqset.data)}]
    aktuell = seqset
    for f in filters:
        aktuell = f.apply(aktuell)
        stufen.append({"satz": seqset.name, "stufe": type(f).__name__, "n": len(aktuell.data)})
        if len(aktuell.data) == 0:
            print(f"[{seqset.name}] nach {type(f).__name__} nichts mehr uebrig.")
            break
    return aktuell, stufen


def top_tabelle(links: SequenceSet, rechts: SequenceSet, n: int = 30) -> pd.DataFrame:
    """Zwei Spalten, je n Sequenzen, sortiert nach mic_apex (bestes zuerst).

    Der kuerzere Satz wird mit Leerstrings aufgefuellt, damit die Tabelle
    rechteckig bleibt.
    """
    def beste(s: SequenceSet) -> list[str]:
        df = s.data
        if "mic_apex" in df.columns:
            df = df.sort_values("mic_apex")
        return df["sequence"].head(n).tolist()

    a, b = beste(links), beste(rechts)
    breite = max(len(a), len(b))
    a += [""] * (breite - len(a))
    b += [""] * (breite - len(b))
    return pd.DataFrame({links.name: a, rechts.name: b})


def ppl_abgleich(torres: SequenceSet) -> pd.DataFrame | None:
    """Unsere ProGen2-Perplexitaet gegen Torres' eigene ppl-Spalte.

    Gleiche Sequenzen, gleiches Modell (ProGen2-medium laut Paper). Weichen
    die Werte systematisch ab, liegt es an der Implementierung und nicht am
    Generator -- das entscheidet, ob ein Perplexitaetsvergleich zwischen den
    Saetzen ueberhaupt aussagekraeftig ist.
    """
    df = torres.data
    if not {"ppl", "ppl_progen2"} <= set(df.columns):
        return None

    paar = df[["sequence", "ppl", "ppl_progen2"]].dropna()
    paar = paar.assign(differenz=paar["ppl_progen2"] - paar["ppl"])
    rho = paar["ppl"].corr(paar["ppl_progen2"], method="spearman")
    print(
        f"[ppl-Abgleich] n={len(paar)}  Spearman rho={rho:.4f}  "
        f"Median Differenz={paar['differenz'].median():+.3f}  "
        f"(Torres {paar['ppl'].median():.2f} vs wir {paar['ppl_progen2'].median():.2f})"
    )
    return paar


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Torres' generierte Sequenzen gegen unsere.")
    p.add_argument("--torres", type=Path, default=Path(TORRES))
    p.add_argument("--aguirre", type=Path, default=Path(AGUIRRE))
    p.add_argument("--out", required=True, type=Path, help="Zielordner des Laufs")
    p.add_argument("--scorer", nargs="+", default=None, help="ueberschreibt cfg.scorer_auswahl")
    p.add_argument("--limit", type=int, default=None, help="nur N Sequenzen JE SATZ (Probelauf)")
    p.add_argument("--top", type=int, default=30, help="Sequenzen je Spalte in der Top-Tabelle")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    start = time.perf_counter()
    konfig = Config()
    if args.scorer:
        konfig.scorer_auswahl = tuple(args.scorer)
    scorer_namen = list(konfig.scorer_auswahl)

    for bez, pfad in (("--torres", args.torres), ("--aguirre", args.aguirre)):
        if not Path(pfad).is_file():
            raise SystemExit(f"{bez}: Datei nicht gefunden: {pfad}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    konfig.to_yaml(out_dir / "config.yaml")

    print(f"=== Torres vs Aguirre -> {out_dir} ===")
    print(f"Scorer : {scorer_namen}")

    # ---- Laden. Laengenfilter ZUERST: APEX ueberspringt > 50 Residuen still,
    #      und Torres' Satz enthaelt Sequenzen bis hinunter zu 1 Residuum.
    laengen = LengthFilter.from_config(konfig)
    saetze = [
        laengen.apply(lade(args.torres, "torres", args.limit, konfig.seed, zusatz=("ppl",))),
        laengen.apply(lade(args.aguirre, "aguirre", args.limit, konfig.seed)),
    ]

    # ---- Scoren
    scorers = baue_scorer(scorer_namen, konfig)
    evaluator = Evaluator(scorers, cache_dir=None if args.no_cache else CACHE_DIR / "scores")
    fingerabdruecke = {sc.name: sc.fingerprint for sc in scorers}
    for name, fp in fingerabdruecke.items():
        print(f"  {name:16s} {fp}")

    bewertet = []
    for s in saetze:
        print(f"[scoren] {s.name} ({len(s.data)} Sequenzen)")
        bewertet.append(evaluator.run(s))

    # ---- Kaskade, fuer jeden Satz getrennt, gleiche Schwellen
    filters: list[Filter] = [
        MICFilter.from_config(konfig),
        SimilarityFilter.from_config(konfig),
        DiversityFilter.from_config(konfig),
    ]
    gefiltert, stufen = [], []
    for s in bewertet:
        g, st = kaskade(s, filters)
        gefiltert.append(SequenceSet(name=f"{s.name}_gefiltert", data=g.data))
        stufen += st

    # ---- Vergleiche in drei Runden. Zwei Gruppen je Bild bleiben lesbar;
    #      vier Violinen nebeneinander sind der Ausnahmefall, nicht die Regel.
    roh = {s.name: s for s in bewertet}
    fein = {g.name: g for g in gefiltert if len(g.data) > 0}
    roh_ref = bewertet[0].name                    # "torres"
    fein_ref = f"{bewertet[0].name}_gefiltert"

    def spalten_von(groups: dict) -> list[str]:
        """Score-Spalten, die in ALLEN Gruppen dieser Runde vorliegen."""
        da = [c for c in STANDARD_SCORE_SPALTEN
              if all(c in s.data.columns for s in groups.values())]
        fehlt = [c for c in STANDARD_SCORE_SPALTEN if c not in da]
        if fehlt:
            print(f"[Vergleiche] uebersprungen (fehlt in einer Gruppe): {fehlt}")
        return da

    ergebnisse = {}
    for runde, groups, referenz in (("roh", roh, roh_ref),
                                    ("gefiltert", fein, fein_ref)):
        if len(groups) < 2 or referenz not in groups:
            print(f"[Vergleiche] Runde '{runde}' uebersprungen ({len(groups)} Gruppe(n)).")
            continue
        print(f"[Runde {runde}] {[(n, len(s.data)) for n, s in groups.items()]}")
        for c in spalten_von(groups):
            ergebnisse[f"{runde}_verteilung_{c}"] = DistributionComparison(
                column=c, log_scale=c in LOG_SPALTEN).compare(groups)
        ergebnisse[f"{runde}_komposition"] = CompositionComparison(
            reference=referenz).compare(groups)
        ergebnisse[f"{runde}_positionen"] = PositionalComparison(
            max_length=konfig.pep_max_len).compare(groups)

    dateien = []
    for res in ergebnisse.values():
        dateien += res.save(out_dir, close=True)

    # ---- Zwei Zusatztabellen
    if len(gefiltert) == 2 and all(len(g.data) > 0 for g in gefiltert):
        stamm = build_stem("top", [g.name for g in gefiltert])
        pfad = out_dir / f"{stamm}.csv"
        top_tabelle(*gefiltert, n=args.top).to_csv(pfad, index=False)
        dateien.append(pfad)

    paar = ppl_abgleich(bewertet[0])
    if paar is not None:
        pfad = out_dir / f"{build_stem('ppl-abgleich', ['torres'])}.csv"
        paar.to_csv(pfad, index=False)
        dateien.append(pfad)

    print(f"[Vergleiche] {len(dateien)} Dateien nach {out_dir} geschrieben.")

    # ---- Metadaten
    dauer = time.perf_counter() - start
    (out_dir / "lauf_metadaten.json").write_text(json.dumps({
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        "dauer_s": round(dauer, 1),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "eingaben": {"torres": str(args.torres), "aguirre": str(args.aguirre),
                     "limit": args.limit},
        "scorer": scorer_namen,
        "fingerabdruecke": fingerabdruecke,
        "filter": [type(f).__name__ for f in filters],
        "filterstufen": stufen,
        "gruppen": {n: len(s.data) for n, s in groups.items()},
        "vergleiche": sorted(ergebnisse),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Kaskaden ===")
    for st in stufen:
        print(f"  {st['satz']:10s} {st['stufe']:18s} {st['n']:>8d}")
    print(f"\n{len(ergebnisse)} Vergleiche, {dauer / 60:.1f} min")
    print(f"fertig: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
