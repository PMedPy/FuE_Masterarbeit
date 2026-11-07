"""08 -- Generative Inferenz: Sequenzen ziehen und als FASTA schreiben.

Achter Schritt. Laedt einen Trainingslauf ueber Sampler.load() und schreibt
die generierten Sequenzen nach FASTA und CSV.

IN  : runs/train_<name>_<datum>/final.pt        (oder checkpoints/checkpoint_<tag>.pt)
      data/cache/train-Set_<modell>_L<max_len>.pt   (fuer die Laengenverteilung)
      data_tracked/standards_<modell>_L<max_len>.csv
OUT : runs/train_<name>_<datum>/samples/<tag>_n<N>.faa
      runs/train_<name>_<datum>/samples/<tag>_n<N>.csv

Aufruf:
    python scripts/08_sample.py --run runs/train_cluster650_2026-11-04_1615 -n 50000
    python scripts/08_sample.py --run runs/... -n 1024 --batch-size 256 --chunk 256

Warum in Bloecken statt in einem Rutsch
---------------------------------------
Sampler.sample() macht alles auf einmal und haelt dabei (N, L, D) float32 im
RAM. Bei N = 50000, L = 50, D = 1280 sind das 12,8 GB, nur um sie danach zu
dekodieren. Dieses Skript ruft deshalb sample_embeddings() und
decode_embeddings() blockweise auf und haengt jeden Block sofort an die
Ausgabedateien an. Drei Vorteile:

1. Der Speicher bleibt bei --chunk 1000 unter 300 MB statt 12,8 GB.
2. Ein Abbruch (Zeitlimit, Knotenfehler) kostet nur den letzten Block --
   alles davor steht schon in der Datei. Es gibt kein --resume im Projekt.
3. Jeder Block schreibt eine Fortschrittszeile mit gemessener Rate und
   Restzeit. Bei einem Lauf ueber Stunden ist das der einzige Weg zu wissen,
   ob er in die --time-Grenze laeuft.

Reproduzierbarkeit
------------------
Die Laengen werden EINMAL mit --seed gezogen und dann blockweise
aufgeteilt -- gleiche Seed-Angabe, gleiche Laengen, unabhaengig von
--chunk. Fuer die Embeddings bekommt jeder Block seed + Blocknummer:
sample_embeddings() setzt den globalen torch-RNG, derselbe Seed in jedem
Block wuerde identische Sequenzen erzeugen.

FASTA-Header
------------
">gen_00001" mit fuenf Stellen, um verwechslung mit ">AMP_i" aus data.write_fasta() zu verhindern
"""

import argparse
import time
from pathlib import Path

from aguirre.sampler import Sampler


def parse_args():
    p = argparse.ArgumentParser(description="Sequenzen aus einem Trainingslauf generieren.")
    p.add_argument("--run", required=True,
                   help="Lauf-Ordner, z.B. runs/train_cluster650_2026-11-04_1615")
    p.add_argument("--tag", default="final",
                   help='Checkpoint: "final" | "best" | "last" (Default: final)')
    p.add_argument("-n", type=int, required=True, dest="n",
                   help="Anzahl zu generierender Sequenzen")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Sequenzen pro Reverse-Kette (Default 256)")
    p.add_argument("--chunk", type=int, default=1000,
                   help="Sequenzen pro Schreibblock (Default 1000)")
    p.add_argument("--length-source", default="train",
                   choices=["train", "val", "test", "uniform"],
                   help="Laengenverteilung (Default train = empirisch)")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--out", default=None,
                   help="Basisname ohne Endung (Default: <run>/samples/<tag>_n<N>)")
    p.add_argument("--no-standardize", action="store_true",
                   help="nur fuer Laeufe, die mit --no-standardize trainiert wurden")
    p.add_argument("--no-ema", action="store_true",
                   help="rohe Gewichte statt EMA -- fast immer falsch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="vorhandene Ausgabedateien ueberschreiben")
    return p.parse_args()


def main():
    args = parse_args()
    run = Path(args.run)

    out_base = Path(args.out) if args.out else run / "samples" / f"{args.tag}_n{args.n}"
    fasta_path = out_base.with_suffix(".faa")
    csv_path = out_base.with_suffix(".csv")

    # Vor dem Laden pruefen: ein Stunden-Lauf soll nicht an einer
    # vorhandenen Datei sterben, nachdem er 6 GB Checkpoint gelesen hat.
    if not args.force:
        for pfad in (fasta_path, csv_path):
            if pfad.exists():
                raise SystemExit(
                    f"{pfad} existiert schon. Mit --force ueberschreiben "
                    "oder --out anders setzen."
                )
    out_base.parent.mkdir(parents=True, exist_ok=True)

    sampler = Sampler.load(
        run,
        tag=args.tag,
        standardize=not args.no_standardize,
        use_ema=not args.no_ema,
        device=args.device,
    )
    print(sampler, flush=True)
    h = sampler.herkunft
    print(f"Quelle   : {h['quelle']}  (EMA: {h['use_ema']}, "
          f"Epoche: {h['epoch']}, best_val: {h['best_val']})", flush=True)

    lengths = sampler.draw_lengths(args.n, source=args.length_source, seed=args.seed)
    print(f"Laengen  : {args.length_source}, min {int(lengths.min())}, "
          f"max {int(lengths.max())}, Mittel {float(lengths.float().mean()):.1f}", flush=True)
    print(f"Plan     : {args.n} Sequenzen, batch {args.batch_size}, "
          f"Bloecke von {args.chunk}, T={sampler.diffusion.num_timesteps}", flush=True)

    fasta_path.write_text("")
    csv_path.write_text("name,length,sequence\n")

    t0 = time.time()
    for block, start in enumerate(range(0, args.n, args.chunk)):
        stop = min(start + args.chunk, args.n)
        namen = [f"gen_{i:05d}" for i in range(start, stop)]

        emb, lens = sampler.sample_embeddings(
            lengths[start:stop],
            batch_size=args.batch_size,
            progress=False,
            seed=args.seed + block,
        )
        df = sampler.decode_embeddings(emb, lens, names=namen,
                                       batch_size=args.batch_size)
        del emb

        with fasta_path.open("a") as fh:
            for name, seq in zip(df["name"], df["sequence"]):
                fh.write(f">{name}\n{seq}\n")
        df.to_csv(csv_path, mode="a", header=False, index=False)

        dt = time.time() - t0
        rate = stop / dt
        rest = (args.n - stop) / rate
        print(f"[{stop:>6}/{args.n}]  {dt / 60:7.1f} min  |  "
              f"{rate * 3600:7.0f} Seq/h  |  Restzeit ~{rest / 3600:5.2f} h",
              flush=True)

    print(f"\nFertig in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"FASTA    : {fasta_path}", flush=True)
    print(f"CSV      : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
