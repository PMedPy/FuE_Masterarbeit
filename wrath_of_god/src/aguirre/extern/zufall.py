"""zufall.py -- der Nullmodell-Boden aus Zufallssequenzen.

Aminosaeuren unabhaengig gezogen, Laengen an die Decke angepasst.
Beantwortet im Panel die unterste Frage: hat das Modell ueberhaupt etwas
gelernt? Liegen die generierten Peptide hier, ist nichts passiert.

Der Boden ist eine Leiter, und das hier ist die unterste Sprosse:

    zufall      unabhaengige Ziehung    -- irgendetwas gelernt?
    permutiert  eigene Sequenz gemischt -- Reihenfolge oder nur Komposition?
    uniprot     echte Nicht-AMP-Fragmente -- spezifisch?

Die mittlere Sprosse steckt schon im ShuffleScorer und wird hier nicht
gebaut. Dieses Modul liegt trotz fehlender externer Quelle in extern/, weil
extern/ der Ort fuer Datensatzbau ist; panel/ bewertet nur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aguirre.config import Config, SPLIT_DIR
from aguirre.extern.apd import lese_fasta
from aguirre.extern.uniprot import laengen_pool, schreibe


def baue_random_len(
    n: int = 5000,
    komposition: str | Path = "uniform",
    laengen_quelle: Path | None = None,
    min_len: int | None = 5,
    max_len: int | None = 50,
    seed: int | None = None,
    out_fasta: Path | None = None,
) -> pd.DataFrame:
    """Zufallssequenzen mit angepasster Laengenverteilung.

    Args:
        komposition: "uniform" zieht gleichverteilt aus cfg.AS_alphabet.
            Ein Pfad zu einer Protein-FASTA zieht stattdessen nach deren
            Aminosaeurehaeufigkeiten.

            "uniform" ist der schwaechere Gegner: echte Proteine sind weit
            von Gleichverteilung entfernt, AMPs zusaetzlich K/R/L-reich. Ein
            Modell schlaegt Gleichverteilung also schon, wenn es nur die
            Komposition gelernt hat. Mit Hintergrundhaeufigkeit misst man,
            was darueber hinausgeht. Beide zu bauen kostet fast nichts und
            ergibt eine Leiter statt eines Punkts -- als Hintergrundquelle
            bietet sich der UniProt-Download an, der fuer den anderen Boden
            ohnehin heruntergeladen wird.
        min_len, max_len: Laengenfenster des Pools, Default 5 bis 50 --
            siehe uniprot.laengen_pool fuer die Begruendung.
        seed: Default Config().seed.
    """
    cfg = Config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    pool = laengen_pool(laengen_quelle, min_len=min_len, max_len=max_len)
    alphabet = list(cfg.AS_alphabet)

    if komposition == "uniform":
        p = np.full(len(alphabet), 1 / len(alphabet))
        quelle = "uniform"
    else:
        zaehler = pd.Series(
            list("".join(s.upper() for _, s in lese_fasta(Path(komposition))))
        ).value_counts()
        p = np.array([zaehler.get(a, 0) for a in alphabet], dtype=float)
        if p.sum() == 0:
            raise ValueError(f"Keine Aminosaeuren in {komposition} gezaehlt.")
        p /= p.sum()
        quelle = Path(komposition).name

    laengen = rng.choice(pool, size=n)
    seqs = ["".join(rng.choice(alphabet, size=int(L), p=p)) for L in laengen]

    df = pd.DataFrame({"sequence": seqs}).drop_duplicates().reset_index(drop=True)
    df["sequence_length"] = df["sequence"].str.len()
    df.insert(0, "id", [f"RANDOM_{i:05d}" for i in range(len(df))])

    print(f"[zufall] Komposition: {quelle}")
    print(f"[zufall] {n} gezogen -> {len(df)} nach Deduplizierung")
    print(f"[zufall] Laenge: Median {df['sequence_length'].median():.0f}, "
          f"min {df['sequence_length'].min()}, max {df['sequence_length'].max()}")
    schreibe(df, out_fasta or SPLIT_DIR / "random_len.fasta")
    return df
