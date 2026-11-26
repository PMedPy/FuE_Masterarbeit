"""dramp.py -- DRAMP general_amps.xlsx in zwei CSVs.

DRAMP liefert keine strukturierten Messungen, sondern Freitext. Deshalb zwei
Ausgaben mit verschiedenem Anspruch:

    dramp_novelty.csv     ALLE Sequenzen, moeglichst inklusiv. Referenz fuer
                          max_ident_bekannt -- ein generiertes Peptid, das
                          irgendeinem bekannten AMP gleicht, ist nicht neu,
                          auch wenn dieses AMP modifiziert oder zyklisch ist.
                          Hier wird NICHT gefiltert.

    dramp_vergleich.csv   Zweitquelle fuer den Praediktorenvergleich. Streng
                          gefiltert und streng geparst: was sich nicht
                          eindeutig lesen laesst, wird verworfen und gezaehlt.
                          Das ist bei einer Zweitquelle legitim -- die
                          Hauptauswertung laeuft auf DBAASP, wo nach Medium
                          geschichtet werden kann. DRAMP hat kein Medium,
                          kein Inokulum und keine Referenz je Messung.

Das Freitextformat von Target_Organism:

    Gram-positive bacteria: Staphylococcus simulans 22 (MIC=0.304 µg/ml), ...
    Human pathogens: L100 Staphylococcus aureus ATCC6538P (MIC<=0.13 µg/ml), ...
    Bifidobacterium longum (strain: ATCC 15707) (MIC=75 µg/ml), ...

Kategorie-Praefixe, Stamm-IDs vor und hinter dem Artnamen, abgekuerzte Genera,
Kommata innerhalb von Klammern -- und elf Schreibweisen fuer drei Einheiten,
darunter zwei verschiedene Unicode-Mu-Zeichen (U+03BC und U+00B5).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from aguirre.config import DB_CACHE_DIR, DB_EXTERN_DIR
# mw_aus_sequenz wird geteilt, damit DBAASP und DRAMP dasselbe
# Molekulargewicht liefern. Bewusst importiert statt kopiert.
from aguirre.extern.dbaasp import mw_aus_sequenz

GENERAL_XLSX = DB_CACHE_DIR / "dramp" / "general_amps.xlsx"
PATENT_XLSX = DB_CACHE_DIR / "dramp" / "patent_amps.xlsx"

KANONISCH = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# Nur die drei Spezies, die APEX und ANIA gemeinsam abdecken. Abkuerzungen
# mitgedacht: DRAMP schreibt mal "Staphylococcus aureus", mal "S. aureus".
ZIEL_RE = {
    "Escherichia coli": re.compile(
        r"(?:Escherichia\s+coli|\bE\.\s*coli)", re.I),
    "Staphylococcus aureus": re.compile(
        r"(?:Staphylococcus\s+aureus|\bS\.\s*aureus)", re.I),
    "Pseudomonas aeruginosa": re.compile(
        r"(?:Pseudomonas\s+aeruginosa|\bP\.\s*aeruginosa)", re.I),
}

# MIC(?!\d) schliesst MIC50 und MIC90 aus -- das sind Populationsstatistiken
# und kein MIC. Ohne das Negativ-Lookahead wuerde "MIC90=4" als Wert 90
# gelesen.
MIC_RE = re.compile(
    r"MIC(?!\d)\s*(?P<rel>[=<>≤≥~]{0,2})\s*"
    r"(?P<wert>\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*"
    r"(?P<einheit>[A-Za-zµμ][A-Za-zµμ/]*)",
    re.I,
)

# Elf Schreibweisen, drei echte Einheiten. mg/L ist numerisch identisch mit
# µg/ml, µmol/L identisch mit µM, 1 pmol/ml entspricht 1 nM.
EINHEITEN = {
    "ug/ml": ("ug_ml", 1.0), "ug/l": ("ug_ml", 0.001),
    "mg/l": ("ug_ml", 1.0), "mg/ml": ("ug_ml", 1000.0),
    "um": ("uM", 1.0), "umol/l": ("uM", 1.0), "microm": ("uM", 1.0),
    "nmol/ml": ("uM", 1.0), "mm": ("uM", 1000.0),
    "nm": ("uM", 0.001), "pmol/ml": ("uM", 0.001),
}

ZENSUR = {"=": "=", "": "=", "<": "<", "≤": "<", "<=": "<",
          ">": ">", "≥": ">", ">=": ">", "~": "~"}


def _norm(v) -> str:
    """Zellwert zu getrimmtem String. DRAMP hat viele Werte mit Leerzeichen
    am Ende -- "Linear " und "Linear" sind dort zwei verschiedene Kategorien."""
    return str(v).strip() if pd.notna(v) else ""


def _einheit(roh: str) -> tuple[str | None, float]:
    """Rohschreibweise auf (Zieleinheit, Faktor) abbilden.

    Beide Mu-Zeichen werden zu "u" normalisiert: U+00B5 MICRO SIGN und
    U+03BC GREEK SMALL LETTER MU sehen identisch aus und kommen beide vor.
    """
    k = roh.strip().rstrip(".").lower().replace("µ", "u").replace("μ", "u")
    return EINHEITEN.get(k, (None, 1.0))


def _wert(roh: str) -> float | None:
    """Zahl aus dem Rohstring. Bei einem Bereich das GEOMETRISCHE Mittel --
    MIC ist log-verteilt, das arithmetische Mittel waere dort verzerrt."""
    t = roh.replace(",", ".").replace("–", "-")
    teile = [p.strip() for p in t.split("-") if p.strip()]
    try:
        zahlen = [float(p) for p in teile]
    except ValueError:
        return None
    if not zahlen:
        return None
    if len(zahlen) == 1:
        return zahlen[0]
    return (zahlen[0] * zahlen[-1]) ** 0.5


def _spezies(text: str) -> str | None:
    """Welche Zielspezies in diesem Textstueck steht.

    Zwei Regeln, die Fehlzuordnungen verhindern:
    - stehen MEHRERE Zielspezies darin, ist nicht entscheidbar, zu welcher der
      Messwert gehoert -> None, Zeile wird verworfen
    - genommen wird das LETZTE Vorkommen, weil der Messwert dahinter steht
    """
    gefunden = [(m.start(), art)
                for art, rx in ZIEL_RE.items()
                for m in rx.finditer(text)]
    if not gefunden:
        return None
    arten = {art for _, art in gefunden}
    if len(arten) > 1:
        return None
    return max(gefunden)[1]


def parse_target_organism(text: str) -> list[dict]:
    """Zerlegt ein Target_Organism-Feld in einzelne Messungen.

    Vorgehen: alle MIC-Vorkommen suchen, und fuer jedes den Text SEIT dem
    vorigen Vorkommen als Organismusangabe nehmen. Dadurch kann kein
    Messwert einem Organismus zugeordnet werden, zwischen denen ein anderer
    Messwert steht.

    Returns:
        Liste von dicts mit species, organism_raw, zensur, wert, einheit.
        Nur Treffer fuer die drei Zielspezies mit bekannter Einheit.
    """
    if not text:
        return []

    treffer: list[dict] = []
    pos = 0
    for m in MIC_RE.finditer(text):
        davor = text[pos:m.start()]
        pos = m.end()

        art = _spezies(davor)
        if art is None:
            continue
        ziel, faktor = _einheit(m.group("einheit"))
        wert = _wert(m.group("wert"))
        if ziel is None or wert is None:
            continue

        # Ein berichteter Bereich ist kein exakter Wert: eigenes Zeichen,
        # auch wenn kein Relationssymbol dasteht.
        zensur = ZENSUR.get(m.group("rel"), "?")
        if zensur == "=" and ("-" in m.group("wert") or "\u2013" in m.group("wert")):
            zensur = "~"

        # Stamm ist, was hinter dem Artnamen steht.
        letzte = list(ZIEL_RE[art].finditer(davor))[-1]
        treffer.append({
            "species": art,
            "strain": davor[letzte.end():].strip(" ,;:()-")[:60],
            "organism_raw": davor.strip(" ,;:")[-120:],
            "zensur": zensur,
            "wert": wert * faktor,
            "zieleinheit": ziel,
            "einheit_roh": m.group("einheit"),
            "wert_roh": m.group("wert"),
        })
    return treffer


# ---------------------------------------------------------------------------
# Einlesen
# ---------------------------------------------------------------------------

def lade_general(pfad: Path | None = None) -> pd.DataFrame:
    """Liest general_amps.xlsx und trimmt Spaltennamen."""
    p = Path(pfad) if pfad else GENERAL_XLSX
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} fehlt. Von https://dramp.cpu-bioinfor.org/downloads/ oder "
            "dem Figshare-Spiegel laden und unveraendert ablegen."
        )
    df = pd.read_excel(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def lade_patent(pfad: Path | None = None) -> pd.DataFrame:
    """Liest patent_amps.xlsx und trimmt Spaltennamen.

    Der Patentsatz ist der groesste Block von DRAMP und traegt Sequenzen, die
    in keiner der anderen Quellen stehen: 18 715 Eintraege, 16 586 eindeutige
    Sequenzen (Stand DRAMP 5.0). Fuer eine NEUHEITSaussage ist er der
    wichtigste Teil -- ein generiertes Peptid, das in einer Patentschrift
    steht, ist nicht neu, auch wenn es in keiner Aktivitaetsdatenbank auftaucht.

    Die Spaltennamen sind dieselben wie in general_amps.xlsx (DRAMP_ID,
    Sequence, Sequence_Length, Name, Source), deshalb laeuft baue_novelty_csv
    unveraendert auch auf diesem DataFrame.

    Achtung Lizenz: fuer die Patentdaten verlangt DRAMP eine Autorisierung,
    die CC-BY-4.0-Freigabe gilt nur fuer general und clinical. Die Sequenzen
    duerfen hier als Vergleichsreferenz benutzt, aber nicht mit
    weiterveroeffentlicht werden. Siehe Abschnitt D in
    notizen/veroeffentlichung_checkliste.md.
    """
    p = Path(pfad) if pfad else PATENT_XLSX
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} fehlt. Von https://dramp.cpu-bioinfor.org/downloads/ als "
            "patent_amps (xlsx) laden und unveraendert ablegen."
        )
    df = pd.read_excel(p)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# CSV 1: Novelty-Referenz
# ---------------------------------------------------------------------------

def baue_novelty_csv(
    df: pd.DataFrame | None = None,
    out_pfad: Path | None = None,
) -> pd.DataFrame:
    """Alle Sequenzen, dedupliziert. Bewusst OHNE Filter.

    Begruendung: die Novelty-Referenz soll maximal inklusiv sein. Ein
    generiertes Peptid, das einem zyklischen oder modifizierten bekannten AMP
    gleicht, ist nicht neu -- die Modifikation steht nicht in der Sequenz und
    aendert an der Aehnlichkeit nichts. Gefiltert wird nur Leeres.

    Sequenzen werden grossgeschrieben, weil Kleinbuchstaben in DRAMP
    D-Aminosaeuren bezeichnen und MMseqs2 das nicht unterscheidet.
    """
    df = lade_general() if df is None else df
    seq = df["Sequence"].map(_norm).str.upper()
    out = pd.DataFrame({
        "dramp_id": df["DRAMP_ID"].map(_norm),
        "sequence": seq,
        "sequence_length": pd.to_numeric(df["Sequence_Length"], errors="coerce"),
        "name": df["Name"].map(_norm),
        "source": df["Source"].map(_norm),
    })
    out = out[out["sequence"].str.len() > 0]
    vorher = len(out)
    out = out.drop_duplicates(subset="sequence").reset_index(drop=True)

    ziel = Path(out_pfad) if out_pfad else DB_EXTERN_DIR / "dramp_novelty.csv"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ziel, index=False)
    print(f"[novelty] {len(df)} Eintraege -> {vorher} mit Sequenz -> "
          f"{len(out)} eindeutig")
    print(f"[novelty] geschrieben: {ziel}")
    return out


# ---------------------------------------------------------------------------
# CSV 2: Vergleichssatz
# ---------------------------------------------------------------------------

def baue_vergleich_csv(
    df: pd.DataFrame | None = None,
    out_pfad: Path | None = None,
) -> pd.DataFrame:
    """Streng gefilterte Messungen fuer die drei Zielspezies, lange Form.

    Filter, jeder mit Grund:
        Linear          zyklische Peptide sind eine andere Strukturklasse,
                        sequenzbasierte Praediktoren koennen sie nicht erfassen
        Stereochemie L  Kleinbuchstaben bzw. "D"/"Mix" bedeuten
                        D-Aminosaeuren; Praediktoren nehmen L an
        N-Term frei     Acylierung macht Lipopeptide, der Fettsaeurerest
                        dominiert die Aktivitaet
        C-Term frei
          oder Amidation
        Sequenz kanonisch   X-Platzhalter sind nicht kodierbar

    Rund 4 200 DRAMP-Eintraege tragen in diesen Spalten "Not included yet" --
    ohne Annotation ist nicht belegbar, dass ein Peptid linear und
    unmodifiziert ist, also fallen sie heraus.
    """
    df = lade_general() if df is None else df

    topo = df["Linear/Cyclic/Branched"].map(_norm)
    stereo = df["Stereochemistry"].map(_norm)
    nterm = df["N-terminal_Modification"].map(_norm)
    cterm = df["C-terminal_Modification"].map(_norm)
    seq = df["Sequence"].map(_norm).str.upper()

    masken = {
        "linear": topo.eq("Linear"),
        "stereo_L": stereo.eq("L"),
        "nterm_frei": nterm.eq("Free"),
        "cterm_frei_amid": cterm.isin(["Free", "Amidation"]),
        "kanonisch": seq.str.fullmatch(KANONISCH),
    }
    ok = pd.Series(True, index=df.index)
    print(f"[vergleich] {len(df)} Eintraege")
    for name, m in masken.items():
        ok &= m
        print(f"  nach {name:<16} {ok.sum():>6}")

    zeilen: list[dict] = []
    ohne_treffer = 0
    for i in df.index[ok]:
        mw = mw_aus_sequenz(seq[i])
        kopf = {
            "dramp_id": _norm(df.at[i, "DRAMP_ID"]),
            "sequence": seq[i],
            "sequence_length": pd.to_numeric(df.at[i, "Sequence_Length"],
                                             errors="coerce"),
            "n_terminus": nterm[i],
            "c_terminus": cterm[i],
            "stereochemistry": stereo[i],
            "topologie": topo[i],
            "mw_sequenz": mw,
            "pubmed_id": _norm(df.at[i, "Pubmed_ID"]),
        }
        funde = parse_target_organism(_norm(df.at[i, "Target_Organism"]))
        if not funde:
            ohne_treffer += 1
            continue
        for f in funde:
            ug = f["wert"] if f["zieleinheit"] == "ug_ml" else (
                f["wert"] * mw / 1000.0 if mw else None)
            um = f["wert"] if f["zieleinheit"] == "uM" else (
                f["wert"] * 1000.0 / mw if mw else None)
            zeilen.append({
                **kopf,
                "species": f["species"],
                "strain": f["strain"],
                "organism_raw": f["organism_raw"],
                "zensur": f["zensur"],
                "mic_ug_ml": ug,
                "mic_um": um,
                "einheit_roh": f["einheit_roh"],
                "wert_roh": f["wert_roh"],
            })

    out = pd.DataFrame(zeilen)
    ziel = Path(out_pfad) if out_pfad else DB_EXTERN_DIR / "dramp_vergleich.csv"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ziel, index=False)

    print(f"  {ohne_treffer} gefilterte Eintraege ohne verwertbare Messung "
          "(keine Zielspezies, unbekannte Einheit, oder mehrdeutig)")
    if len(out):
        print(f"\n[vergleich] {len(out)} Messungen, "
              f"{out['dramp_id'].nunique()} Peptide")
        print(f"  je Spezies:  {out['species'].value_counts().to_dict()}")
        print(f"  zensur:      {out['zensur'].value_counts().to_dict()}")
        print(f"  mic_um da:   {out['mic_um'].notna().sum()}")
    print(f"[vergleich] geschrieben: {ziel}")
    return out
