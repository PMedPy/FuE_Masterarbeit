"""uniprot.py -- der Spezifitaets-Boden aus zytoplasmatischen Nicht-AMPs.

Fragmente aus Proteinen, die sicher keine AMPs sind. Beantwortet im Panel die
Frage: ist das Panel spezifisch, oder nennt es alles Proteinartige ein AMP?

Konstruktion nach der ueblichen Praxis fuer AMP-Negativsets (Veltri et al.
2018 und Folgearbeiten): zytoplasmatische Lokalisation, keine
AMP-Schluesselwoerter, cysteinfreie Fragmente, Laengen an das Positivset
angepasst. Dass die Wahl der Negativ-Sampling-Methode Benchmarks verzerrt,
ist belegt und gehoert in die Limitationen -- Sidorczuk et al., Brief
Bioinform 23(5):bbac343 (2022),
https://academic.oup.com/bib/article/23/5/bbac343/6703531

Download (UniProt-Suchfeld, Format FASTA canonical, unkomprimiert, "all"):

    (reviewed:true) AND (cc_scl_term:SL-0086) AND (length:[150 TO 2000])
    NOT (cc_scl_term:SL-0243)
    NOT (keyword:antimicrobial) NOT (keyword:antibiotic)
    NOT (keyword:fungicide)

SL-0086 ist Cytoplasm, SL-0243 ist Secreted. Ablegen unter
data/cache/DBs/uniprot/, unveraendert, und in QUELLEN.md eintragen.
Den vollen Treffersatz laden, nicht die ersten n -- UniProt liefert in
Accession-Reihenfolge, und die korreliert mit Organismus und
Einreichungsjahrzehnt. Die Zufallsauswahl macht dieses Modul mit festem Seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aguirre.config import Config, SPLIT_DIR
from aguirre.evaluation import KnownAMPSimilarityScorer
from aguirre.extern.apd import lese_fasta

KANONISCH = set("ACDEFGHIKLMNPQRSTVWY")
VERBOTEN_IM_FRAGMENT = set("C")


def laengen_pool(
    quelle: Path | None = None,
    nur_kanonisch: bool = True,
    min_len: int | None = 5,
    max_len: int | None = 50,
) -> np.ndarray:
    """Die empirische Laengenverteilung, aus der gezogen wird.

    Quelle ist real_amps_test, also die Decke -- nicht der Trainingssplit.
    Boden und Decke sollen sich im INHALT unterscheiden, nicht in der Laenge.

    Gezogen wird mit Zuruecklegen aus den TATSAECHLICHEN Laengen, nicht aus
    einer angepassten Verteilung. Eine Normal- oder Gammanaeherung wuerde die
    Schiefe und die Haeufung auf runden Werten glattbuegeln, und genau die
    macht die Laengenverteilung von AMPs aus.

    min_len/max_len begrenzen den Pool, Default 5 bis 50. Grund:
    real_amps_test reicht von 1 bis 190 Resten, die generierten Mengen gehen
    nur bis 50, und cfg.apex_max_len steht ebenfalls auf 50. Ohne Begrenzung
    enthielte der Boden Fragmente, die kein Modell je erzeugen koennte und
    die APEX ohnehin als NaN zurueckgibt -- womit jedes Tool wieder auf einer
    anderen Teilmenge rechnete. Dasselbe Fenster wendet filtere_kandidaten in
    extern/dbaasp.py schon an. Mit None laesst sich die Begrenzung abschalten.
    """
    quelle = Path(quelle or SPLIT_DIR / "real_amps_test.csv")
    df = pd.read_csv(quelle)
    if nur_kanonisch and "kanonisch" in df.columns:
        df = df[df["kanonisch"]]
    L = df["sequence_length"]
    if min_len is not None:
        L = L[L >= min_len]
    if max_len is not None:
        L = L[L <= max_len]
    if L.empty:
        raise ValueError(f"Laengenpool leer nach Filter [{min_len}, {max_len}].")
    return L.to_numpy()


def _akzession(header: str) -> str:
    """sp|P12345|NAME_HUMAN ... -> P12345. Faellt auf den ersten Token zurueck."""
    teile = header.split("|")
    return teile[1] if len(teile) >= 2 else header.split()[0]


def _freie_laeufe(seq: str) -> list[tuple[int, int]]:
    """Maximale Teilstuecke ohne Cystein und ohne nicht-kanonische Zeichen.

    Gibt [(start, ende)) zurueck. In einem Durchgang laesst sich damit sagen,
    wo ueberhaupt ein Fragment der Laenge L liegen kann, statt fuer jedes L
    alle Startpunkte einzeln zu pruefen.
    """
    laeufe, start = [], 0
    for i, c in enumerate(seq):
        if c in VERBOTEN_IM_FRAGMENT or c not in KANONISCH:
            if i > start:
                laeufe.append((start, i))
            start = i + 1
    if len(seq) > start:
        laeufe.append((start, len(seq)))
    return laeufe


def baue_uniprot_boden(
    proteine_fasta: Path,
    n: int = 5000,
    rand: int = 25,
    laengen_quelle: Path | None = None,
    min_len: int | None = 5,
    max_len: int | None = 50,
    referenz_fasta: Path | None = None,
    seed: int | None = None,
    out_fasta: Path | None = None,
) -> pd.DataFrame:
    """Ein Fragment je Protein aus zytoplasmatischen Nicht-AMP-Proteinen.

    EIN Fragment pro Protein, nicht das Protein durchkacheln. Fragmente aus
    demselben Protein teilen Zusammensetzung, Faltungstyp und Organismus --
    sie sind nicht unabhaengig. Die effektive Stichprobengroesse waere die
    Zahl der Proteine, nicht die der Fragmente, und jedes Konfidenzintervall
    ueber den Boden waere zu eng.

    `rand` schneidet beide Enden ab. Am N-Terminus sitzen Signal- und
    Transitpeptide -- ausgerechnet die membraninteragierenden, haeufig
    kationischen Abschnitte, die in einem Negativset nichts verloren haben.
    Plus das Start-Methionin.

    Cystein faellt raus, weil viele AMPs disulfidstabilisiert sind
    (Defensine); ein cysteinreiches Fragment kann dem Panel AMP-artig
    vorkommen, und dann misst der Boden nicht Spezifitaet.

    Passt kein cysteinfreies Fenster der gezogenen Laenge in den Kern, wird
    das PROTEIN uebersprungen und nicht die Laenge verkuerzt. Verkuerzen
    verzerrt genau die Groesse, die kontrolliert werden soll, und zwar
    systematisch bei cysteinreichen Proteinen. Proteine sind reichlich
    vorhanden, Laengentreue nicht.

    Args:
        proteine_fasta: UniProt-Download, Query im Modul-Docstring.
        n: gewuenschte Fragmentzahl. Reichen die Proteine nicht, kommt
            weniger heraus und die Ausgabe sagt es.
        rand: abgeschnittene Reste je Ende.
        min_len, max_len: Laengenfenster des Pools, siehe laengen_pool.
        referenz_fasta: wenn gesetzt, wird max_ident_bekannt gegen diese
            Referenz berechnet und als SPALTE ergaenzt. Default None, damit
            der Bau auch ohne MMseqs2 durchlaeuft.
        seed: Default Config().seed.

    Returns:
        DataFrame mit id, sequence, sequence_length, protein, start
        (und max_ident_bekannt, falls referenz_fasta gesetzt).
    """
    cfg = Config()
    rng = np.random.default_rng(cfg.seed if seed is None else seed)

    proteine = [(k, s.upper()) for k, s in lese_fasta(Path(proteine_fasta))]
    if not proteine:
        raise ValueError(f"{proteine_fasta} enthaelt keine Sequenzen.")
    pool = laengen_pool(laengen_quelle, min_len=min_len, max_len=max_len)

    treffer: list[dict] = []
    zu_kurz = kein_fenster = 0

    for idx in rng.permutation(len(proteine)):
        if len(treffer) >= n:
            break
        header, seq = proteine[idx]
        kern = seq[rand:len(seq) - rand]
        laenge = int(rng.choice(pool))
        if len(kern) < laenge:
            zu_kurz += 1
            continue

        moeglich = [(a, b) for a, b in _freie_laeufe(kern) if b - a >= laenge]
        if not moeglich:
            kein_fenster += 1
            continue

        # Startpunkt gleichverteilt ueber ALLE gueltigen Positionen, nicht
        # erst einen Lauf und dann darin -- sonst waeren kurze Laeufe
        # ueberrepraesentiert.
        gewichte = np.array([b - a - laenge + 1 for a, b in moeglich], dtype=float)
        lauf = moeglich[int(rng.choice(len(moeglich), p=gewichte / gewichte.sum()))]
        start = int(rng.integers(lauf[0], lauf[1] - laenge + 1))

        treffer.append({"protein": _akzession(header), "start": rand + start,
                        "sequence": kern[start:start + laenge]})

    df = pd.DataFrame(treffer)
    vorher = len(df)
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    df["sequence_length"] = df["sequence"].str.len()
    df.insert(0, "id", [f"UNIPROT_{i:05d}" for i in range(len(df))])
    df = df.loc[:, ["id", "sequence", "sequence_length", "protein", "start"]]

    print(f"[uniprot] {len(proteine)} Proteine gelesen")
    print(f"[uniprot] uebersprungen: {zu_kurz} zu kurz, {kein_fenster} ohne "
          "cysteinfreies Fenster der gezogenen Laenge")
    print(f"[uniprot] {vorher} Fragmente -> {len(df)} nach Deduplizierung")
    if len(df) < n:
        print(f"[uniprot] WARNUNG: nur {len(df)} von {n} erreicht. Mehr "
              "Proteine laden oder rand verkleinern.")
    print(f"[uniprot] Laenge: Median {df['sequence_length'].median():.0f}, "
          f"min {df['sequence_length'].min()}, max {df['sequence_length'].max()}")
    if referenz_fasta is not None:
        df = ergaenze_identitaet(df, Path(referenz_fasta), cfg)

    schreibe(df, out_fasta or SPLIT_DIR / "uniprot_lower.fasta")
    return df


def ergaenze_identitaet(df: pd.DataFrame, referenz: Path,
                        cfg: Config | None = None) -> pd.DataFrame:
    """Haengt max_ident_bekannt an -- als Zahl, nicht als Urteil.

    Kryptische AMPs sind real: antimikrobielle Abschnitte stecken auch in
    gewoehnlichen Proteinen. Ein paar Treffer im Boden sind zu erwarten und
    kein Konstruktionsfehler. Wo der Schnitt liegt, entscheidet man besser,
    wenn man die Verteilung gesehen hat -- deshalb wird hier nichts
    verworfen, nur gemessen und ausgezaehlt.

    Zur Einordnung der Schwellen: das Mass ist fident * qcov, 0.5 heisst
    also etwa 100 % Identitaet ueber die halbe Sequenz oder 71 % ueber 71 %.
    Torres' eigenes Neuheitskriterium liegt bei 0.6
    (cfg.similarity_max_ident).
    """
    cfg = cfg or Config()
    scorer = KnownAMPSimilarityScorer(
        referenz=referenz,
        cov_mode=cfg.similarity_cov_mode,
        coverage=cfg.similarity_coverage,
        evalue=cfg.similarity_evalue,
    )
    # score() nimmt eine Liste von Strings und gibt ein DataFrame mit
    # RangeIndex 0..n-1 in Eingabereihenfolge zurueck -- den SequenceSet
    # packt sonst der Evaluator aus. Index vorher gleichziehen, sonst
    # verschiebt join() die Zuordnung still.
    df = df.reset_index(drop=True)
    werte = scorer.score(df["sequence"].tolist())
    if len(werte) != len(df):
        raise ValueError(f"score() gab {len(werte)} Zeilen fuer {len(df)} Sequenzen.")
    df = df.join(werte.reset_index(drop=True))

    spalte = scorer.columns[0]
    # Kein Treffer ist 0.0, nicht NaN -- siehe die leere .m8 in score().
    print(f"[uniprot] ueberhaupt ein Treffer gegen {Path(referenz).name}: "
          f"{int((df[spalte] > 0).sum())} von {len(df)}")
    for schwelle in (0.4, 0.5, 0.6):
        print(f"[uniprot]   max_ident_bekannt > {schwelle}: "
              f"{int((df[spalte] > schwelle).sum())}")
    return df


def schreibe(df: pd.DataFrame, out_fasta: Path) -> None:
    """FASTA plus CSV danebenlegen. Die CSV traegt die Herkunftsspalten."""
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with out_fasta.open("w", encoding="utf-8") as f:
        for kennung, seq in zip(df["id"], df["sequence"]):
            f.write(f">{kennung}\n{seq}\n")
    df.to_csv(out_fasta.with_suffix(".csv"), index=False)
    print(f"[schreiben] {out_fasta} (+ .csv)")
