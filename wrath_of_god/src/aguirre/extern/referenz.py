"""referenz.py -- vereinigte Referenz aller bekannten AMPs.

Speist `max_ident_bekannt` im Panel: "aehnelt dieses generierte Peptid
irgendeinem bekannten AMP?" Das ist die weitere Frage als `max_ident_train`,
die nur gegen unseren eigenen Trainingssplit prueft.

Vier Quellen, alle schon lokal:

    DBAASP        Listencache, alle Seiten        20 178 eindeutig von 25 542
    DRAMP general dramp_novelty.csv               11 485
    DRAMP patent  dramp_patent_novelty.csv        16 586
    APD natural   apd_novelty.csv                  3 306

Der Patentsatz ist nachtraeglich dazugekommen, und zwar nicht aus
Vollstaendigkeitsdrang: ohne ihn deckt die Referenz nur 53,3 % von Torres'
19 670 Trainingssequenzen ab, mit ihm 99,6 %. Torres hat die vollen
DRAMP-Downloads benutzt. Eine Referenz ohne Patente wuerde die Neuheit
generierter Peptide systematisch ueberschaetzen -- und das ist die Richtung,
in die man bei einer Neuheitsaussage gerade nicht irren will.

Aus DBAASP wird bewusst der LISTENcache genommen, nicht der Detailcache: fuer
Details wurden nur die 4 560 Peptide der drei Zielspezies geholt. Eine
Novelty-Referenz soll aber vollstaendig sein -- ein generiertes Peptid, das
einem AMP ohne gemessene MIC gleicht, ist genauso wenig neu.

Nicht gefiltert wird nach Modifikationen, Laenge oder kanonischen Zeichen.
Alles wird grossgeschrieben, weil MMseqs2 Gross- und Kleinschreibung nicht
unterscheidet -- damit fallen D-Aminosaeuren (klein geschrieben in DBAASP) mit
ihren L-Varianten zusammen, was hier richtig ist: dieselbe Sequenz ist
dieselbe Sequenz.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aguirre.config import DB_EXTERN_DIR, SPLIT_DIR, TRACKED_DIR, DATA_DIR
from aguirre.extern import dbaasp

# Die 20 proteinogenen Aminosaeuren. Alles andere -- X, B, Z, U, O, aber
# auch Parsing-Rueckstaende wie "-", "," und das Konsensus-Zeichen Phi --
# faellt hier raus.
KANONISCH = r"^[ACDEFGHIKLMNPQRSTVWY]+$"


def _aus_dbaasp() -> pd.Series:
    df = dbaasp.lade_liste()
    return df["sequence"].dropna()


def _aus_csv(pfad: Path) -> pd.Series:
    if not pfad.is_file():
        raise FileNotFoundError(f"{pfad} fehlt. Erst das passende baue_novelty_csv().")
    return pd.read_csv(pfad)["sequence"].dropna()


def baue_novelty_referenz(
    out_csv: Path | None = None,
    out_fasta: Path | None = None,
    mit_patent: bool = True,
) -> pd.DataFrame:
    """Vereinigt die Datenbanken zu einer deduplizierten Referenz.

    Args:
        mit_patent: DRAMP-Patentsequenzen einbeziehen. Default True, siehe
            Modul-Docstring. Auf False nur fuer den Vergleich "wie saehe die
            Neuheit ohne Patente aus" -- dann aber eine andere out_csv
            angeben, sonst wird die gute Referenz ueberschrieben.

    Returns:
        DataFrame mit id, sequence, sequence_length, quellen, n_quellen.
        `quellen` ist alphabetisch und mit "|" verbunden, z.B. "apd|patent".
        "patent" statt "dramp_patent", damit eine Suche nach "dramp" nicht
        versehentlich die Patente mittrifft.
    """
    roh = {
        "dbaasp": _aus_dbaasp(),
        "dramp": _aus_csv(DB_EXTERN_DIR / "dramp_novelty.csv"),
        "apd": _aus_csv(DB_EXTERN_DIR / "apd_novelty.csv"),
    }
    if mit_patent:
        roh["patent"] = _aus_csv(DB_EXTERN_DIR / "dramp_patent_novelty.csv")

    teile = []
    for name, serie in roh.items():
        s = serie.astype(str).str.strip().str.upper()
        s = s[s.str.len() > 0].drop_duplicates()
        print(f"[referenz] {name}: {len(serie)} roh -> {len(s)} eindeutig")
        teile.append(pd.DataFrame({"sequence": s, "quelle": name}))

    lang = pd.concat(teile, ignore_index=True)
    df = (lang.groupby("sequence")["quelle"]
              .agg(lambda q: "|".join(sorted(set(q))))
              .reset_index()
              .rename(columns={"quelle": "quellen"}))
    df["n_quellen"] = df["quellen"].str.count(r"\|") + 1
    df["sequence_length"] = df["sequence"].str.len()
    df = df.sort_values("sequence").reset_index(drop=True)
    df.insert(0, "id", [f"BEKANNT_{i:05d}" for i in range(len(df))])

    ziel_csv = Path(out_csv) if out_csv else DB_EXTERN_DIR / "amps_bekannt.csv"
    ziel_fa = Path(out_fasta) if out_fasta else DB_EXTERN_DIR / "amps_bekannt.fasta"
    ziel_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ziel_csv, index=False)
    with ziel_fa.open("w", encoding="utf-8") as f:
        for kennung, seq in zip(df["id"], df["sequence"]):
            f.write(f">{kennung}\n{seq}\n")

    gesamt_roh = sum(len(s) for s in roh.values())
    print(f"[referenz] {gesamt_roh} Zeilen aus {len(roh)} Quellen -> {len(df)} eindeutige "
          f"Sequenzen ({gesamt_roh - len(df)} Ueberschneidungen entfernt)")
    print("[referenz] Quellenverteilung:")
    print(df["quellen"].value_counts().to_string())
    print(f"[referenz] Laenge: Median {df['sequence_length'].median():.0f}, "
          f"min {df['sequence_length'].min()}, max {df['sequence_length'].max()}")
    print(f"[referenz] geschrieben: {ziel_csv}")
    print(f"[referenz] geschrieben: {ziel_fa}")
    return df


def _lies_fasta_map(pfad: Path) -> dict[str, str]:
    """FASTA -> {id: sequence}, grossgeschrieben, umbruchtolerant."""
    aus: dict[str, str] = {}
    kennung, teile = None, []
    with Path(pfad).open(encoding="utf-8") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            if zeile.startswith(">"):
                if kennung and teile:
                    aus[kennung] = "".join(teile).upper()
                kennung, teile = zeile[1:].strip(), []
            else:
                teile.append(zeile)
    if kennung and teile:
        aus[kennung] = "".join(teile).upper()
    return aus


def baue_real_amps_test(
    split_csv: Path | None = None,
    alle_fasta: Path | None = None,
    bekannt_csv: Path | None = None,
    out_fasta: Path | None = None,
    ausschluss: tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    """Decke des Panels: alle echten AMPs, die das Modell nie gesehen hat.

    real_amps_test = Torres-Testsplit + amps_bekannt ohne alles, was in den
    ausgeschlossenen Splits steht. Dedupliziert wird EXAKT, nicht ueber
    Aehnlichkeit -- so hat Torres es bei seinen generierten Sequenzen auch
    gemacht. Nahe Verwandte des Trainingssets bleiben also drin; wie viel
    Restverwandtschaft das ist, zeigt `max_ident_train` auf genau diesem Set.

    `val` ist per Default mit ausgeschlossen: es hat nie Gradienten erzeugt,
    aber ueber den Validation-Loss die Modellauswahl mitbestimmt. Als Decke
    "nie gesehen" waere es damit nicht sauber.

    Die Spalte `kanonisch` markiert Sequenzen aus den 20 proteinogenen
    Aminosaeuren. Geloescht wird nichts -- die Datei bleibt vollstaendig --,
    aber das Panel bewertet nur `kanonisch == True`. Grund: die generierten
    Mengen sind zu 100 % kanonisch, die Decke ohne Filter zu rund 80 %.
    Beides mit denselben Tools zu messen und als Vergleich auszugeben, waere
    zweierlei Messung. Dazu kaeme, dass ein Tool auf einer Sequenz NaN liefert
    und das naechste nicht -- dann rechnet die Vote-Aggregation je Tool auf
    einer anderen Teilmenge, und Cohens Kappa misst teils Abwesenheit statt
    Uneinigkeit. Und die Fehlerrichtung waere die unguenstige: unplausible
    Werte auf der Decke ziehen sie nach unten, das Modell saehe besser aus.

    Returns:
        DataFrame mit id, sequence, sequence_length, kanonisch, herkunft,
        quellen. `herkunft` ist "torres_test" oder "datenbank".
    """
    split_csv = Path(split_csv or TRACKED_DIR / "split_t075.csv")
    alle_fasta = Path(alle_fasta or DATA_DIR / "all19K_unclustered.faa")
    bekannt_csv = Path(bekannt_csv or DB_EXTERN_DIR / "amps_bekannt.csv")
    out_fasta = Path(out_fasta or SPLIT_DIR / "real_amps_test.fasta")

    tabelle = pd.read_csv(split_csv)
    nach_id = _lies_fasta_map(alle_fasta)

    def seqs(name: str) -> set[str]:
        ids = tabelle.loc[tabelle["split"] == name, "id"]
        return {nach_id[i] for i in ids if i in nach_id}

    test = seqs("test")
    raus = set().union(*(seqs(s) for s in ausschluss)) if ausschluss else set()

    bekannt = pd.read_csv(bekannt_csv)
    bekannt["sequence"] = bekannt["sequence"].astype(str).str.upper()
    neu = bekannt[~bekannt["sequence"].isin(raus | test)]

    test_ids = {nach_id[i]: i for i in tabelle.loc[tabelle["split"] == "test", "id"]
                if i in nach_id}
    teil_torres = pd.DataFrame({
        "id": [test_ids[s] for s in sorted(test)],
        "sequence": sorted(test),
        "herkunft": "torres_test",
        "quellen": "",
    })
    teil_db = pd.DataFrame({
        "id": neu["id"].to_numpy(),
        "sequence": neu["sequence"].to_numpy(),
        "herkunft": "datenbank",
        "quellen": neu["quellen"].to_numpy(),
    })

    df = pd.concat([teil_torres, teil_db], ignore_index=True)
    if df["sequence"].duplicated().any():
        raise ValueError("Duplikate nach dem Zusammenfuegen -- Logikfehler.")
    df["sequence_length"] = df["sequence"].str.len()
    # fillna/astype, damit die Spalte sicher bool ist: SequenceSet.subset
    # lehnt alles ab, was nicht bool-dtype hat, und das faellt sonst erst
    # im Panel auf.
    df["kanonisch"] = df["sequence"].str.fullmatch(KANONISCH).fillna(False).astype(bool)
    df = df.loc[:, ["id", "sequence", "sequence_length", "kanonisch",
                    "herkunft", "quellen"]]

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    with out_fasta.open("w", encoding="utf-8") as f:
        for kennung, seq in zip(df["id"], df["sequence"]):
            f.write(f">{kennung}\n{seq}\n")
    df.to_csv(out_fasta.with_suffix(".csv"), index=False)

    print(f"[decke] Torres-Testsplit: {len(teil_torres)}")
    print(f"[decke] ausgeschlossen ({', '.join(ausschluss)}): {len(raus)} Sequenzen")
    print(f"[decke] aus amps_bekannt neu dazu: {len(teil_db)} von {len(bekannt)}")
    kan = int(df["kanonisch"].sum())
    print(f"[decke] gesamt: {len(df)} eindeutige Sequenzen")
    print(f"[decke] davon kanonisch: {kan} ({kan/len(df):.1%}) -- "
          f"{len(df) - kan} nicht-kanonisch, bleiben in der Datei, werden "
          "aber im Panel nicht bewertet")
    print(f"[decke] Laenge: Median {df['sequence_length'].median():.0f}, "
          f"min {df['sequence_length'].min()}, max {df['sequence_length'].max()}")
    print(f"[decke] geschrieben: {out_fasta} (+ .csv)")
    return df
