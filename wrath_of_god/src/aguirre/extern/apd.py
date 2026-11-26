"""apd.py -- APD-Sequenzen als Novelty-Referenz.

APD traegt zum mic_predictor_testset nichts bei: die Downloads sind reine
Sequenzlisten ohne quantitative MIC-Werte je Zielspezies. Der Beitrag ist die
Novelty-Referenz fuer max_ident_bekannt -- kuratierte NATUERLICHE AMPs, und
davon hat APD den besten Bestand.

Die Downloads-Seite (https://aps.unmc.edu/downloads) bietet sieben FASTA-
Dateien, die geschachtelt sind:

    naturalAMPs_APD2024a.fasta    3306   <- Obermenge, nur diese wird gebraucht
      animalAMPs_APD2024a.fasta   2580
        amphibianAMPs_APD2024     1110
        insectAMPs_APD2024         397
        humanAMPs_APD2024          154
      plantAMPs_APD2024            268
      bacteriaAMPs_APD2024         410

Animal + Plant + Bacteria ergeben 3258 von 3306; der Rest sind Herkuenfte ohne
eigene Datei. Synthetische und vorhergesagte AMPs (2290 bzw. 373 in APD6) gibt
es nicht als Datei, nur ueber Suche mit FASTA-Export -- weggelassen, weil
DBAASP und DRAMP synthetische Peptide breit abdecken.

Das Modul heisst apd, nicht apd6: die Version ist eine Eigenschaft des
Downloads, nicht des Parsers. dbaasp.py und dramp.py sind ebenso versionierte
Datenbanken und tragen die Nummer auch nicht im Namen. Die genaue Version
gehoert nach data_tracked/extern/QUELLEN.md -- Stand hier ist APD6
(01.01.2026, 6309 Peptide).

Nutzungsbedingung: APD raeumt keine Lizenz ein, im Seitenfuss steht
"All Rights Reserved", und um Zitation der APD6-Publikation wird gebeten
(Wang et al. 2026, NAR 54, D363-D374). Fuer ein oeffentliches Repo heisst das:
Datei und abgeleitete CSV bleiben draussen. Siehe Abschnitt D in
notizen/veroeffentlichung_checkliste.md.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

from aguirre.config import DB_CACHE_DIR, DB_EXTERN_DIR

# Der Ordner heisst apd, ohne Versionsnummer -- wie dbaasp/ und dramp/.
# Die Version steht in data_tracked/extern/QUELLEN.md, festgenagelt am SHA256.
APD_ORDNER = (DB_CACHE_DIR / "apd",)
# Das Muster ist an beiden Enden locker, weil der Dateiname auf dem Weg vom
# Server in den Cache leidet: das "r" fehlte in einer von Hand benannten
# Fassung, und ein Browser-Download haengt gern ".txt" an. Server-Name ist
# "naturalAMPs_APD2024a.fasta".
FASTA_MUSTER = "nat*AMPs_APD*.fasta*"

APD_ID = re.compile(r"^AP\d{4,6}$")
KANONISCH = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def finde_fasta() -> Path:
    """Sucht die Natural-AMPs-FASTA im Cache.

    Returns:
        Pfad der gefundenen Datei.

    Raises:
        FileNotFoundError: wenn in keinem Kandidatenordner eine passende
            Datei liegt. Die Meldung nennt, was stattdessen da ist.
    """
    for ordner in APD_ORDNER:
        if not ordner.is_dir():
            continue
        treffer = sorted(ordner.glob(FASTA_MUSTER))
        if treffer:
            return treffer[0]

    vorhanden = sorted(
        str(p.relative_to(DB_CACHE_DIR))
        for o in APD_ORDNER if o.is_dir() for p in o.iterdir()
    )
    raise FileNotFoundError(
        f"Keine Datei nach Muster {FASTA_MUSTER!r} in "
        f"{[str(o) for o in APD_ORDNER]}. Gefunden: {vorhanden or 'nichts'}. "
        "Von https://aps.unmc.edu/downloads laden (Natural AMPs) und "
        "unveraendert ablegen."
    )


def lese_fasta(pfad: Path) -> list[tuple[str, str]]:
    """Umbruchtoleranter FASTA-Leser.

    aguirre.data.read_fasta erwartet EINE Zeile pro Sequenz und wirft bei
    umgebrochenen Dateien einen Fehler. Fremd-FASTAs sind haeufig auf 60 oder
    70 Zeichen umgebrochen, deshalb hier ein eigener Leser, der die Zeilen
    eines Eintrags zusammensetzt. aguirre.data bleibt unangetastet, weil daran
    die Trainingspipeline haengt.

    Liest auch .gz direkt -- UniProt liefert seine Downloads gepackt aus,
    und die Datei soll so liegen bleiben, wie der Server sie geschickt hat.

    Returns:
        Liste von (header, sequence) in Dateireihenfolge. Der Header ist ohne
        das fuehrende ">". Leerzeilen werden uebersprungen.
    """
    pfad = Path(pfad)
    if not pfad.is_file():
        raise FileNotFoundError(f"{pfad} fehlt.")
    oeffne = gzip.open if pfad.suffix == ".gz" else open

    eintraege: list[tuple[str, str]] = []
    kopf: str | None = None
    teile: list[str] = []
    with oeffne(pfad, "rt", encoding="utf-8", errors="replace") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            if zeile.startswith(">"):
                if kopf is not None:
                    eintraege.append((kopf, "".join(teile)))
                kopf = zeile[1:].strip()
                teile = []
            else:
                teile.append(zeile)
    if kopf is not None:
        eintraege.append((kopf, "".join(teile)))
    return eintraege


def baue_novelty_csv(
    pfad: Path | None = None,
    out_pfad: Path | None = None,
) -> pd.DataFrame:
    """APD-Sequenzen als CSV fuer die Novelty-Referenz.

    Wie bei DRAMP wird NICHT nach Modifikationen gefiltert: ein generiertes
    Peptid, das einem bekannten AMP gleicht, ist nicht neu, unabhaengig davon
    wie dieses AMP modifiziert ist. Sequenzen werden grossgeschrieben, weil
    MMseqs2 Klein- und Grossbuchstaben nicht unterscheidet.

    Nicht-kanonische Zeichen werden nur GEZAEHLT, nicht entfernt -- fuer eine
    Aehnlichkeitssuche sind sie unschaedlich, und ein stiller Verlust von
    Referenzsequenzen waere schlimmer als ein X im Alignment.
    """
    quelle = Path(pfad) if pfad else finde_fasta()
    eintraege = lese_fasta(quelle)

    df = pd.DataFrame(eintraege, columns=["header", "sequence"])
    gelesen = len(df)

    # Die Header sind blanke ">AP00001" ohne Beschreibung -- der erste Token
    # ist alles, was drinsteht. Eine name-Spalte waere in jeder Zeile leer.
    df["apd_id"] = df["header"].str.split(n=1).str[0]

    # Die Datei von APD beginnt mit einem Pseudo-Header
    # ">Your search led to 3306 peptides" -- offenbar wird die FASTA serverseitig
    # ueber dieselbe Suchfunktion erzeugt, und deren Trefferzeile landet als
    # erster Header in der Datei. Nachgeprueft am Direktdownload von
    # https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta:
    # 3307 Header fuer 3306 Peptide. Er hat keine Sequenz, wuerde aber als
    # Eintrag mit apd_id "Your" durchlaufen und die Zaehlungen verfaelschen,
    # darum explizit auf das Kennungsmuster filtern.
    ist_amp = df["apd_id"].str.fullmatch(APD_ID).fillna(False)
    kein_amp = int((~ist_amp).sum())
    df = df[ist_amp].copy()

    df["sequence"] = df["sequence"].str.strip().str.upper()
    df["sequence_length"] = df["sequence"].str.len()

    df = df[df["sequence_length"] > 0]
    mit_seq = len(df)
    df = (df.drop_duplicates(subset="sequence")
            .loc[:, ["apd_id", "sequence", "sequence_length"]]
            .reset_index(drop=True))

    nicht_kanonisch = int((~df["sequence"].str.fullmatch(KANONISCH)).sum())

    ziel = Path(out_pfad) if out_pfad else DB_EXTERN_DIR / "apd_novelty.csv"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ziel, index=False)

    print(f"[apd] Quelle: {quelle}")
    print(f"[apd] {gelesen} Header gelesen -> {kein_amp} ohne APD-Kennung "
          f"verworfen -> {mit_seq} mit Sequenz -> {len(df)} eindeutig")
    print(f"[apd] nicht-kanonische Sequenzen: {nicht_kanonisch} (bleiben drin)")
    print(f"[apd] Laenge: Median {df['sequence_length'].median():.0f}, "
          f"min {df['sequence_length'].min()}, max {df['sequence_length'].max()}")
    print(f"[apd] geschrieben: {ziel}")
    return df
