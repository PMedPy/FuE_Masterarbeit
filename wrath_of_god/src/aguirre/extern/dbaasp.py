"""dbaasp.py -- Rohabruf der DBAASP-API.

Die API hat zwei Stufen, weil die Listenantwort KEINE Aktivitaetsdaten
enthaelt (geprueft 2026-11-24):

    Stufe 1  /peptides?limit=&offset=   -> Metadaten, 25 542 Eintraege
    Stufe 2  /peptides/<id>             -> Details inkl. targetActivities

Zwischen den Stufen wird gefiltert: complexity, Terminus-Modifikationen und
sequenceLength stehen schon in Stufe 1, also muss Stufe 2 nur fuer die
Ueberlebenden laufen. Zusaetzlich kann Stufe 1 serverseitig auf Zielspezies
eingeschraenkt werden (siehe ZIELSPEZIES) -- das ist der groesste Hebel, weil
Stufe 2 ein Request pro Peptid kostet.

Dieses Modul laedt und urteilt nicht. Das Parsen von Einheiten, zensierten
Werten (">128") und verschachtelten monomers-Arrays wird mehrfach scheitern,
und kein Fehlschlag darf einen neuen Download kosten.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from aguirre.config import DB_CACHE_DIR, DB_EXTERN_DIR

BASIS_URL = "https://dbaasp.org/peptides"
DBAASP_DIR = DB_CACHE_DIR / "dbaasp"
DETAIL_DIR = DBAASP_DIR / "detail"

# Kontaktadresse aus der Umgebung, damit sie nicht im Repo landet:
#   export AGUIRRE_CONTACT="..."
KONTAKT = os.environ.get("AGUIRRE_CONTACT", "")
USER_AGENT = f"aguirre-fue/0.1 ({KONTAKT})" if KONTAKT else "aguirre-fue/0.1"

TIMEOUT_S = 30
VERSUCHE = 3

# Die Spezies, die APEX (11 Staemme) und ANIA (3 Spezies) abdecken. Nur fuer
# diese ist ein Vergleich gemessener gegen vorhergesagter MIC ueberhaupt
# moeglich -- alles andere ist Ballast in Stufe 2.
ZIELSPEZIES = (
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",
    "Klebsiella pneumoniae",
    "Enterococcus faecalis",
    "Enterococcus faecium",
)


# ---------------------------------------------------------------------------
# 1. Transport
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET mit Retry -- aber nur bei Fehlern, die sich von selbst erledigen.

    Netzfehler und 5xx: nochmal, mit wachsender Wartezeit. Das ist Schluckauf.
    4xx: sofort durchfallen lassen. Ein 404 oder 400 ist ein Fehler in der
    Anfrage, den Wiederholen nicht heilt -- es verschleiert ihn nur.
    """
    for versuch in range(1, VERSUCHE + 1):
        try:
            antwort = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=TIMEOUT_S,
            )
        except (requests.Timeout, requests.ConnectionError) as fehler:
            if versuch == VERSUCHE:
                raise
            wartezeit = 2 ** versuch
            print(f"    {type(fehler).__name__}, warte {wartezeit}s "
                  f"(Versuch {versuch}/{VERSUCHE})")
            time.sleep(wartezeit)
            continue

        if antwort.status_code >= 500 and versuch < VERSUCHE:
            wartezeit = 2 ** versuch
            print(f"    HTTP {antwort.status_code}, warte {wartezeit}s "
                  f"(Versuch {versuch}/{VERSUCHE})")
            time.sleep(wartezeit)
            continue

        antwort.raise_for_status()
        return antwort

    raise RuntimeError("unerreichbar")  # nur damit der Rueckgabetyp stimmt


def _schreibe_roh(pfad: Path, inhalt: bytes) -> None:
    """Antwort-Bytes UNVERAENDERT ablegen.

    Nicht json.dumps auf das geparste dict: re-serialisiertes JSON ist nicht
    mehr dieselbe Datei, und der Fetcher soll konservieren statt urteilen.
    """
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes(inhalt)


# ---------------------------------------------------------------------------
# 2. Einzelabrufe
# ---------------------------------------------------------------------------

def hole_seite(
    limit: int = 200,
    offset: int = 0,
    zusatz: dict | None = None,
    roh_pfad: Path | None = None,
) -> dict:
    """Eine Seite der Peptidliste (Stufe 1).

    Args:
        limit: Eintraege pro Seite.
        offset: Startposition in der Gesamtliste.
        zusatz: Weitere Query-Parameter, z.B.
            {"targetSpecies.value": "Escherichia coli"}.
        roh_pfad: Wenn gesetzt, Antwort-Bytes vor dem Parsen dorthin schreiben.

    Returns:
        dict mit den Keys "data" (Liste der Eintraege) und "totalCount".
    """
    params: dict = {"limit": limit, "offset": offset}
    if zusatz:
        params.update(zusatz)
    antwort = _get(BASIS_URL, params=params)
    if roh_pfad is not None:
        _schreibe_roh(roh_pfad, antwort.content)
    return antwort.json()


def hole_detail(
    peptid_id: int | str,
    roh_pfad: Path | None = None,
) -> dict:
    """Den Detaildatensatz eines Peptids (Stufe 2).

    Hier liegen targetActivities (MIC) und hemoliticCytotoxicActivities --
    in der Listenantwort stehen sie nicht.
    """
    antwort = _get(f"{BASIS_URL}/{peptid_id}")
    if roh_pfad is not None:
        _schreibe_roh(roh_pfad, antwort.content)
    return antwort.json()


# ---------------------------------------------------------------------------
# 3. Stufe 1: Listen in den Cache
# ---------------------------------------------------------------------------

def hole_liste(
    limit: int = 200,
    pause_s: float = 0.3,
    neu: bool = False,
    zusatz: dict | None = None,
    unterordner: str = "liste",
) -> int:
    """Holt alle Seiten einer Listenabfrage und legt jede einzeln im Cache ab.

    Eine Datei pro Seite unter DBAASP_DIR/<unterordner>/offset_<offset>.json.
    Damit ist Fortsetzen trivial: eine fehlende Datei ist eine fehlende Seite.

    Die Gesamtzahl steht IN der Antwort, ist also erst nach dem ersten Abruf
    bekannt -- die Schleife laesst sich nicht vorher planen.

    Args:
        limit: Eintraege pro Seite. Liefert der Server weniger als
            angefordert, wird gewarnt: dann kappt er still.
        pause_s: Pause zwischen echten Abrufen, nicht beim Cache-Treffer.
        neu: True laedt auch vorhandene Seiten neu.
        zusatz: Zusaetzliche Query-Parameter (Filter).
        unterordner: Eigener Cache-Ordner pro Abfrage. WICHTIG: bei einer
            gefilterten Abfrage einen anderen Namen als "liste" waehlen, sonst
            mischen sich zwei verschiedene Ergebnismengen in einem Ordner.

    Returns:
        totalCount laut Server fuer diese Abfrage.
    """
    verzeichnis = DBAASP_DIR / unterordner
    verzeichnis.mkdir(parents=True, exist_ok=True)
    offset = 0
    total: int | None = None
    geholt = 0

    while True:
        pfad = verzeichnis / f"offset_{offset:06d}.json"

        if pfad.is_file() and not neu:
            seite = json.loads(pfad.read_text(encoding="utf-8"))
        else:
            antwort = _get(BASIS_URL, params={**(zusatz or {}),
                                              "limit": limit, "offset": offset})
            _schreibe_roh(pfad, antwort.content)
            seite = antwort.json()
            geholt += 1
            time.sleep(pause_s)

        if total is None:
            total = int(seite["totalCount"])
            seiten = -(-total // limit)          # aufrunden ohne math.ceil
            print(f"[{unterordner}] totalCount {total} -> {seiten} Seiten "
                  f"bei limit={limit}")
            if total == 0:
                break
        elif int(seite["totalCount"]) != total:
            # Wird die DB mitten im Lauf aktualisiert, verschiebt sich die
            # Zuordnung Offset -> Eintrag. Nicht stillschweigend weiterlaufen.
            print(f"WARNUNG: totalCount hat sich geaendert ({total} -> "
                  f"{seite['totalCount']}). Offsets koennen verschoben sein.")

        n = len(seite["data"])
        if n == 0:
            break
        if n < limit and offset + n < total:
            print(f"WARNUNG: Seite bei Offset {offset} hat nur {n} von {limit} "
                  "Eintraegen. Kappt der Server limit?")

        offset += limit
        if offset >= total:
            break
        if offset % (limit * 10) == 0:
            print(f"  ... Offset {offset}/{total}")

    im_cache = len(list(verzeichnis.glob("offset_*.json")))
    print(f"[{unterordner}] fertig: {geholt} Seiten neu geholt, "
          f"{im_cache} im Cache")
    return total


def lade_liste(unterordner: str = "liste") -> pd.DataFrame:
    """Liest die gecachten Listenseiten und gibt die Metadatentabelle.

    Vollstaendigkeit wird an EINDEUTIGEN ids geprueft, nicht an Seitenzahlen:
    ist die Serversortierung nicht stabil, kann ein Eintrag zweimal kommen und
    ein anderer nie. "128 Seiten x 200" beweist dann nichts.
    """
    verzeichnis = DBAASP_DIR / unterordner
    dateien = sorted(verzeichnis.glob("offset_*.json"))
    if not dateien:
        raise FileNotFoundError(
            f"Kein Cache in {verzeichnis}. Erst hole_liste() laufen lassen."
        )

    eintraege: list[dict] = []
    total: int | None = None
    for pfad in dateien:
        seite = json.loads(pfad.read_text(encoding="utf-8"))
        if total is None:
            total = int(seite["totalCount"])
        eintraege.extend(seite["data"])

    df = pd.DataFrame(eintraege)
    n_eindeutig = df["id"].nunique()
    print(f"[{unterordner}] {len(dateien)} Seiten, {len(df)} Zeilen, "
          f"{n_eindeutig} eindeutige ids, totalCount {total}")
    if total is not None and n_eindeutig != total:
        print(f"WARNUNG: Differenz {abs(total - n_eindeutig)}. Fehlende Offsets "
              "pruefen oder hole_liste(neu=True) fuer die Lueckenseiten.")

    return df.drop_duplicates(subset="id").reset_index(drop=True)


def hole_spezies_listen(
    spezies: Sequence[str] = ZIELSPEZIES,
    limit: int = 200,
    pause_s: float = 0.3,
) -> pd.DataFrame:
    """Je Zielspezies eine gefilterte Liste holen und die ids vereinigen.

    Spart in Stufe 2 den Grossteil der Requests: nur Peptide, die gegen eine
    der relevanten Spezies getestet wurden, brauchen ueberhaupt einen
    Detailabruf.

    Jede Spezies bekommt ihren eigenen Cache-Unterordner, damit die
    Ergebnismengen getrennt bleiben und einzeln nachgeholt werden koennen.

    Returns:
        DataFrame der vereinigten Treffer (dedupliziert ueber id), mit einer
        Zusatzspalte "gefunden_fuer" -- den Spezies, unter denen das Peptid
        auftauchte, als komma-getrennter String.
    """
    teile: list[pd.DataFrame] = []
    for art in spezies:
        ordner = "liste_" + art.lower().replace(" ", "_")
        total = hole_liste(limit=limit, pause_s=pause_s,
                           zusatz={"targetSpecies.value": art},
                           unterordner=ordner)
        if total == 0:
            print(f"  {art}: 0 Treffer -- stimmt der Parametername?")
            continue
        teil = lade_liste(unterordner=ordner)
        teil["gefunden_fuer"] = art
        teile.append(teil)

    if not teile:
        raise RuntimeError(
            "Keine Spezies lieferte Treffer. Vermutlich heisst der "
            "Filterparameter anders als 'targetSpecies.value'."
        )

    zusammen = pd.concat(teile, ignore_index=True)
    gruppiert = (zusammen.groupby("id")["gefunden_fuer"]
                 .apply(lambda s: ",".join(sorted(set(s)))))
    df = (zusammen.drop(columns="gefunden_fuer")
          .drop_duplicates(subset="id")
          .set_index("id"))
    df["gefunden_fuer"] = gruppiert
    print(f"\nVereinigt: {len(df)} eindeutige Peptide ueber "
          f"{len(teile)} Spezies")
    return df.reset_index()


def _terminus_maske(spalte: pd.Series, erlaubt: Iterable[str | None]) -> pd.Series:
    """Maske fuer erlaubte Terminus-Modifikationen.

    None in der Erlaubnisliste meint "unmodifiziert", und das steht in der
    Spalte als NaN. Pandas-Falle: Series.isin([None]) trifft NaN NICHT --
    die beiden muessen getrennt geprueft und verodert werden.
    """
    erlaubt = list(erlaubt)
    maske = spalte.isin([w for w in erlaubt if w is not None])
    if None in erlaubt:
        maske = maske | spalte.isna()
    return maske


# ---------------------------------------------------------------------------
# 4. Vorfilter zwischen den Stufen
# ---------------------------------------------------------------------------

def filtere_kandidaten(
    df: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 50,
    nur_monomere: bool = True,
    erlaubte_c_termini: Iterable[str | None] = (None, "AMD"),
    erlaubte_n_termini: Iterable[str | None] | None = None,
) -> pd.DataFrame:
    """Vorfilter vor Stufe 2, mit Bericht ueber jede Stufe.

    Begruendung der Voreinstellungen:

    nur_monomere
        Multimere und Multi-Peptide haben mehrere Ketten; APEX und ANIA sehen
        eine Sequenz. 675 von 25 542 Eintraegen.

    erlaubte_c_termini
        C-terminale Amidierung (AMD) betrifft rund 11 400 Eintraege, also 44 %.
        Sie wird NICHT verworfen, sondern bleibt als Schichtungsvariable
        erhalten: APEX und ANIA sehen die Modifikation nicht, waren aber selbst
        auf DBAASP-Daten mit genau dieser Modifikation trainiert. Ob ihr Fehler
        bei amidierten Peptiden groesser ist, ist eine Frage an die Daten --
        nicht etwas, das man vorher wegfiltert. Exotische Modifikationen (PEA,
        EN, PUT, je rund 36 Eintraege) fliegen raus.

    erlaubte_n_termini
        None heisst: nicht filtern. Erst die value_counts ansehen.

    min_len / max_len
        sequenceLength fehlt bei rund 200 Eintraegen und ist bei einigen 0 --
        beides muss weg. Die Obergrenze 50 entspricht dem Bereich, in dem
        Aguirre generiert.

    Returns:
        Gefilterte Kopie, Index zurueckgesetzt.
    """
    bericht: list[tuple[str, int]] = [("Eingang", len(df))]
    out = df.copy()

    if nur_monomere and "complexity" in out.columns:
        out = out[out["complexity"] == "monomer"]
        bericht.append(("nur Monomere", len(out)))

    laenge = pd.to_numeric(out["sequenceLength"], errors="coerce")
    out = out[laenge.between(min_len, max_len)]
    bericht.append((f"Laenge {min_len}-{max_len}", len(out)))

    if erlaubte_c_termini is not None:
        out = out[_terminus_maske(out["cTerminus"], erlaubte_c_termini)]
        bericht.append(("C-Terminus erlaubt", len(out)))

    if erlaubte_n_termini is not None:
        out = out[_terminus_maske(out["nTerminus"], erlaubte_n_termini)]
        bericht.append(("N-Terminus erlaubt", len(out)))

    vorher: int | None = None
    for name, n in bericht:
        delta = "" if vorher is None else f"   -{vorher - n}"
        print(f"  {name:<22} {n:>7}{delta}")
        vorher = n

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Stufe 2: Details in den Cache
# ---------------------------------------------------------------------------

def hole_details(
    ids: Sequence[int | str],
    pause_s: float = 0.3,
    neu: bool = False,
) -> dict[str, int]:
    """Holt die Detaildatensaetze der uebergebenen ids, eine Datei je id.

    DETAIL_DIR/<id>.json. Fortsetzen ueber Dateiexistenz, wie in Stufe 1.
    Ein 404 auf eine einzelne id bricht den Lauf NICHT ab -- bei
    zehntausenden Abrufen ist ein fehlender Einzeldatensatz zu erwarten. Er
    wird gezaehlt und am Ende berichtet.

    Laufzeit: pause_s plus Antwortzeit je Peptid. Bei 8 000 ids und 0,3 s sind
    das rund 40 Minuten, bei 25 000 ids rund zwei Stunden. Deshalb vorher
    filtern.

    Returns:
        {"geholt": n, "aus_cache": n, "fehlend": n}
    """
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    ids = list(ids)
    n = len(ids)
    zaehler = {"geholt": 0, "aus_cache": 0, "fehlend": 0}
    start = time.time()

    for i, pid in enumerate(ids, 1):
        pfad = DETAIL_DIR / f"{pid}.json"

        if pfad.is_file() and not neu:
            zaehler["aus_cache"] += 1
        else:
            try:
                antwort = _get(f"{BASIS_URL}/{pid}")
            except requests.HTTPError as fehler:
                # Ein einzelner fehlender Datensatz darf 8000 Abrufe nicht
                # abbrechen. Zaehlen, melden, weiter.
                code = getattr(fehler.response, "status_code", "?")
                zaehler["fehlend"] += 1
                print(f"  id {pid}: HTTP {code}, uebersprungen")
                time.sleep(pause_s)
                continue
            _schreibe_roh(pfad, antwort.content)
            zaehler["geholt"] += 1
            time.sleep(pause_s)

        if i % 250 == 0 or i == n:
            verbraucht = time.time() - start
            rest = (verbraucht / i) * (n - i)
            print(f"  {i}/{n}  ({verbraucht / 60:.1f} min verbraucht, "
                  f"ca. {rest / 60:.1f} min offen)")

    print(f"fertig: {zaehler}")
    return zaehler


# ---------------------------------------------------------------------------
# 6. Parsen: Messungen in lange Form
# ---------------------------------------------------------------------------

_RE_ZAHL = re.compile(r"^\d+(?:\.\d+)?$")
_RE_BEREICH = re.compile(r"^\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?$")
_RE_ERSTE_ZAHL = re.compile(r"\d+(?:\.\d+)?")


def _nam(v):
    """name aus einem verschachtelten Feld ziehen, sonst den Wert selbst."""
    return v.get("name") if isinstance(v, dict) else v


def _namen(v) -> str:
    """Verschachtelte Liste zu einem String -- tolerant gegen die Form."""
    if not v:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(_nam(v) or "")
    teile = []
    for x in v:
        if isinstance(x, dict):
            teile.append(str(x.get("name") or x.get("value") or ""))
        else:
            teile.append(str(x))
    return "|".join(t for t in teile if t)


def _zahl(roh: str) -> float | None:
    """Erste Zahl aus einem String, egal was davor oder dahinter steht."""
    m = _RE_ERSTE_ZAHL.search(roh or "")
    return float(m.group()) if m else None


def _zensur(roh: str) -> str:
    """Welche ART von Wert in concentration steht.

    "13.0"    -> "="  exakt gemessen
    ">200"    -> ">"  rechtszensiert: bis 200 getestet, keine Hemmung. Die
                      echte MIC liegt DARUEBER, nicht bei 200
    "<1"      -> "<"  linkszensiert, unterhalb der getesteten Spanne
    "4-8"     -> "~"  Bereich berichtet, DBAASP hat in activity einen Wert
                      daraus gewaehlt
    "2.5+-0.0"-> "="  Mittelwert mit Streuung
    "NA", ""  -> "?"  unbrauchbar
    """
    c = (roh or "").strip()
    if c.startswith(">"):
        return ">"
    if c.startswith("<"):
        return "<"
    if _RE_ZAHL.match(c):
        return "="
    if _RE_BEREICH.match(c):
        return "~"
    if "\u00b1" in c:
        return "="
    return "?"


# Mittlere Restmassen in Da (Rest = Aminosaeure minus ein Wasser).
_RESTMASSE = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "E": 129.1155, "Q": 128.1307, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
_WASSER = 18.0153


def mw_aus_sequenz(seq: str | None) -> float | None:
    """Mittleres Molekulargewicht aus der Sequenz, ohne Zusatzpaket.

    Summe der Restmassen plus EIN Wasser: bei jeder Peptidbindung wird ein
    Wasser abgespalten, bei n Resten also n-1 mal -- uebrig bleibt genau eines.
    Nicht-kanonische Buchstaben (B, J, O, U, X, Z) -> None, dort ist das MW aus
    der Sequenz nicht bestimmbar.

    Terminale Modifikationen sind NICHT beruecksichtigt: eine C-terminale
    Amidierung aendert das MW um -0,98 Da, bei einem 2000er Peptid also 0,05 %
    -- unterhalb jeder Relevanz fuer eine MIC-Umrechnung.
    """
    if not seq:
        return None
    summe = 0.0
    for z in str(seq).strip().upper():
        if z not in _RESTMASSE:
            return None
        summe += _RESTMASSE[z]
    return summe + _WASSER


SPALTEN_REIHENFOLGE = [
    "peptide_id", "dbaasp_id", "sequence", "sequence_length", "complexity",
    "synthesis_type", "n_terminus", "c_terminus", "unusual_aa",
    "species", "strain", "target_species_raw",
    "measure", "medium", "cfu_group", "ph", "ionic_strength", "salt_type",
    "mic_ug_ml", "mic_um", "zensur", "concentration_raw", "unit_raw",
    "mw", "mw_dbaasp", "mw_sequenz", "reference", "activity_id",
]


def parse_messungen(
    nur_messgroesse: str | None = "MIC",
    out_pfad: Path | None = None,
) -> pd.DataFrame:
    """Flacht die gecachten Detaildatensaetze in eine lange Tabelle auf.

    Eine Zeile pro Messung. Hier wird NICHTS aggregiert und nichts geglaettet:
    aus dieser Datei sind alle weiteren Auswertungen ableitbar, ohne einen
    einzigen Request.

    Einheiten: DBAASP fuehrt in "activity" durchgaengig µg/ml. Geprueft ueber
    alle 4 560 Peptide -- bei unit=µg/ml ist activity/concentration exakt
    1,0000 in 18 648 Faellen, bei unit=µM liegt das Verhaeltnis zwischen 0,22
    und 6,58, also genau MW/1000. Daraus folgt mw_dbaasp je Peptid, und damit
    mic_um auch fuer Zeilen, die in µg/ml berichtet wurden.

    Zensierung: "activity" verliert sie -- ">200" steht dort als 200.0. Die
    Spalte zensur haelt sie fest. Wer nur mic_ug_ml liest, bekommt rund 9 200
    nach unten verfaelschte Werte, und zwar ausgerechnet bei den inaktiven
    Peptiden.

    Args:
        nur_messgroesse: Nur diese activityMeasureGroup behalten. "MIC" ist
            der Standard; MBC ist bakterizid, IC50 ein anderer Endpunkt,
            MIC50/MIC90 sind Populationsstatistiken -- nicht poolbar.
            None behaelt alles und fuehrt measure als Spalte.
        out_pfad: Wohin die CSV. None -> DB_EXTERN_DIR/dbaasp_messungen.csv.

    Returns:
        DataFrame in langer Form, eine Zeile je Messung.
    """
    dateien = sorted(DETAIL_DIR.glob("*.json"))
    if not dateien:
        raise FileNotFoundError(
            f"Keine Detaildateien in {DETAIL_DIR}. Erst hole_details() laufen "
            "lassen."
        )

    zeilen: list[dict] = []
    for i, pfad in enumerate(dateien, 1):
        d = json.loads(pfad.read_text(encoding="utf-8"))

        # Peptid-Ebene: gilt fuer alle Messungen dieses Datensatzes.
        kopf = {
            "peptide_id": d.get("id"),
            "dbaasp_id": d.get("dbaaspId"),
            "sequence": d.get("sequence"),
            "sequence_length": d.get("sequenceLength"),
            "complexity": _nam(d.get("complexity")),
            "synthesis_type": _nam(d.get("synthesisType")),
            "n_terminus": _nam(d.get("nTerminus")),
            "c_terminus": _nam(d.get("cTerminus")),
            "unusual_aa": _namen(d.get("unusualAminoAcids")),
        }

        for a in d.get("targetActivities") or []:
            messgroesse = _nam(a.get("activityMeasureGroup")) or a.get(
                "activityMeasureValue")
            if nur_messgroesse is not None and messgroesse != nur_messgroesse:
                continue

            # targetSpecies ist inkonsistent: manchmal nur die Spezies,
            # manchmal Spezies plus Stamm im selben String.
            roh_art = str(_nam(a.get("targetSpecies")) or "")
            teile = roh_art.split()

            konz_roh = str(a.get("concentration") or "")
            zeilen.append({
                **kopf,
                "species": " ".join(teile[:2]),
                "strain": " ".join(teile[2:]),
                "target_species_raw": roh_art,
                "measure": messgroesse,
                "medium": _nam(a.get("medium")),
                "cfu_group": _nam(a.get("cfuGroup")),
                "ph": a.get("ph"),
                "ionic_strength": a.get("ionicStrength"),
                "salt_type": a.get("saltType"),
                "mic_ug_ml": a.get("activity"),
                "zensur": _zensur(konz_roh),
                "concentration_raw": konz_roh,
                "unit_raw": _nam(a.get("unit")),
                "reference": a.get("reference"),
                "activity_id": a.get("id"),
            })

        if i % 1000 == 0:
            print(f"  ... {i}/{len(dateien)} Dateien, {len(zeilen)} Zeilen")

    df = pd.DataFrame(zeilen)

    konz_zahl = df["concentration_raw"].map(_zahl)

    # MW aus DBAASPs eigenen Zahlen: bei µM-Zeilen ist
    # activity/concentration = MW/1000. NUR exakte Werte verwenden -- bei
    # einem Bereich wie "11-15" nimmt _zahl die 11, activity entspricht aber
    # der 15, und das Verhaeltnis waere frei erfunden.
    sauber = df["zensur"].eq("=") & df["unit_raw"].eq("µM") & konz_zahl.gt(0)
    verhaeltnis = (df["mic_ug_ml"] / konz_zahl).where(sauber) * 1000.0
    df["mw_dbaasp"] = df["peptide_id"].map(
        verhaeltnis.groupby(df["peptide_id"]).median())

    # Sequenz-MW schliesst die Luecken UND kontrolliert die DBAASP-Ableitung.
    df["mw_sequenz"] = df["sequence"].map(mw_aus_sequenz)
    df["mw"] = df["mw_dbaasp"].fillna(df["mw_sequenz"])

    # µM einheitlich aus activity. Nicht aus concentration: sonst stammen
    # mic_ug_ml und mic_um derselben Zeile aus verschiedenen Quellen und
    # koennen sich widersprechen.
    df["mic_um"] = df["mic_ug_ml"] * 1000.0 / df["mw"]

    df = df.reindex(columns=SPALTEN_REIHENFOLGE)

    ziel = out_pfad or (DB_EXTERN_DIR / "dbaasp_messungen.csv")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ziel, index=False)

    print(f"\n{len(df)} Messungen, {df['peptide_id'].nunique()} Peptide")
    print(f"zensur:        {df['zensur'].value_counts().to_dict()}")
    print(f"mic_ug_ml da:  {df['mic_ug_ml'].notna().sum()}")
    print(f"mic_um da:     {df['mic_um'].notna().sum()}")
    print(f"MW bekannt:    {df['mw'].notna().sum()} Zeilen, "
          f"{df.loc[df['mw'].notna(), 'peptide_id'].nunique()} Peptide "
          f"(davon aus DBAASP: "
          f"{df.loc[df['mw_dbaasp'].notna(), 'peptide_id'].nunique()})")
    beide = df["mw_dbaasp"].notna() & df["mw_sequenz"].notna()
    if beide.any():
        rel = ((df.loc[beide, "mw_dbaasp"] - df.loc[beide, "mw_sequenz"]).abs()
               / df.loc[beide, "mw_sequenz"])
        print(f"MW-Kontrolle:  relative Abweichung DBAASP gegen Sequenz, "
              f"Median {rel.median():.2%}, 90%-Quantil {rel.quantile(0.9):.2%}")
    print(f"geschrieben:   {ziel}")
    return df
