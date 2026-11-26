"""tool_val.py -- gemessene MICs gegen APEX und ANIA.

Der Datensatz fuer den Praediktorenvergleich: Peptide mit GEMESSENER MIC aus
den Datenbanken, dazu die Vorhersagen beider Werkzeuge. Unabhaengig von
unseren generierten Daten und von Torres -- er beantwortet nur die Frage, wie
gut APEX und ANIA ueberhaupt treffen.

Quellen sind DBAASP und DRAMP; APD und UniProt liefern keine MICs. Es ist
eine VEREINIGUNG beider Messtabellen, keine Schnittmenge: jede Messung zaehlt,
Duplikate fallen bei der Aggregation je (Sequenz, Spezies) zusammen.

Ablauf, drei Schritte mit einem manuellen dazwischen:

    1. baue_tool_val()          Zusammenfuehren, Filtern, Aggregieren
       schreibe_ania_fasta()    Sequenzen exportieren
    2. [manuell] FASTA ins ANIA-Webtool, Ergebnis-CSV ablegen
    3. ergaenze_apex() + ergaenze_ania()

Zensierte Messungen bleiben drin und behalten ihre Richtung in `zensur`.
Sie als Zahl zu behandeln waere der haeufigste Fehler in solchen Vergleichen:
ein ">200" ist keine Messung, sondern eine Schranke, und beide Praediktoren
wuerden davon in dieselbe Richtung verzerrt. Wie sie ausgewertet werden,
steht in bayes.py-Teil dieses Moduls (posterior_gitter).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from aguirre.config import DB_EXTERN_DIR, EVAL_SET_DIR, TRACKED_DIR

# Die drei Spezies, die ANIA und APEX gemeinsam abdecken.
ZIELSPEZIES = ("Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa")

# APEX sagt je Stamm vorher, die Datenbanken messen je Spezies. Zusammengefasst
# wird geometrisch: MIC wird in Zweierverduennungen gemessen, lebt also auf
# einer Log-Skala, und das arithmetische Mittel liesse sich vom schlechtesten
# Stamm dominieren.
APEX_STAEMME = {
    "Escherichia coli": ["E. coli ATCC 11775", "E. coli AIC221", "E. coli AIC222"],
    "Pseudomonas aeruginosa": ["P. aeruginosa PA01", "P. aeruginosa PA14"],
    "Staphylococcus aureus": ["S. aureus ATCC 12600",
                              "S. aureus (ATCC BAA-1556) - MRSA"],
}

# Feste Grenzen statt Terzilen: ueber Spezies hinweg vergleichbar und
# interpretierbar. Terzile waeren je Spezies andere Schnitte.
MIC_BAENDER = ((0.0, 8.0, "niedrig (<=8 uM)"),
               (8.0, 64.0, "mittel (8-64 uM)"),
               (64.0, math.inf, "hoch (>64 uM)"))

KANONISCH = r"^[ACDEFGHIKLMNPQRSTVWY]+$"

# Fertige Bildunterschrift zu plot_baender. Steht hier und nicht nur im
# LaTeX, damit sie mit der Abbildung zusammen gepflegt wird -- aendert sich
# die Konstruktion, faellt die veraltete Unterschrift direkt daneben auf.
BILDUNTERSCHRIFT_BAENDER = """\
Vorhergesagte gegen gemessene MIC, getrennt nach Spezies. Aufgetragen ist
der Zehnerlogarithmus der MIC in µM; die Boxplots sind in dieser
logarithmischen Groesse berechnet, nicht nur so dargestellt, weil MIC in
Zweierverduennungen gemessen wird und damit multiplikativ skaliert.

Je Spezies sind die Peptid-Spezies-Paare in drei Baender geteilt, und zwar
nach ihrer GEMESSENEN MIC: niedrig (hoechstens 8 µM), mittel (8 bis 64 µM)
und hoch (ueber 64 µM). Innerhalb jedes Bandes stehen nebeneinander die
gemessene MIC (rot) sowie die Vorhersagen von APEX (tuerkis) und ANIA
(hellblau) fuer dieselben Peptide. Kasten: Quartile, Linie: Median,
Antennen: bis 1,5 Interquartilsabstaende; Ausreisser sind nicht gezeichnet.

Gezeigt sind ausschliesslich Messungen mit einem bestimmten Zahlenwert.
Rechtszensierte Messungen, also jene, die im Labor nur als "groesser als die
hoechste getestete Konzentration" bestimmt wurden, sind nicht enthalten --
fuer sie laesst sich kein Wert auftragen. Ihre Anzahl und die Frage, ob die
Vorhersage ebenfalls oberhalb der Schranke liegt, stehen in Tabelle X und in
der unteren Zeile von Abbildung Y.

Zur Vorsicht bei der Interpretation: die Baender sind nach der Messung
gebildet, aufgetragen ist die Vorhersage. Dadurch entsteht Regression zur
Mitte. Das niedrige Band enthaelt bevorzugt Messungen, deren Messrauschen
nach unten zeigt, weshalb dort jedes Verfahren zu hoch zu liegen scheint,
und im hohen Band umgekehrt. Dieser Effekt tritt auch bei einem
fehlerfreien, lediglich verrauschten Verfahren auf und darf nicht als
Verzerrung des Werkzeugs gelesen werden. Die systematische Abweichung wird
deshalb nicht aus dieser Abbildung abgelesen, sondern aus Abbildung Y
(Streudiagramm) und Tabelle X."""


def _aus_dbaasp(pfad: Path | None = None) -> pd.DataFrame:
    """DBAASP-Messungen auf das gemeinsame Schema bringen."""
    d = pd.read_csv(Path(pfad or DB_EXTERN_DIR / "dbaasp_messungen.csv"))
    return pd.DataFrame({
        "quelle": "dbaasp",
        "quell_id": d["dbaasp_id"],
        "sequence": d["sequence"].astype(str),
        "species": d["species"],
        "zensur": d["zensur"],
        "mic_um": d["mic_um"],
        # NaN bedeutet in DBAASP "frei"; AMD ist C-terminale Amidierung.
        "n_frei": d["n_terminus"].isna(),
        "c_amidiert": d["c_terminus"].eq("AMD"),
        "c_frei": d["c_terminus"].isna(),
        "medium": d["medium"],
    })


def _aus_dramp(pfad: Path | None = None) -> pd.DataFrame:
    """DRAMP-Messungen auf das gemeinsame Schema bringen.

    baue_vergleich_csv hat schon auf linear, L-Stereochemie, N-Terminus frei
    und C-Terminus frei/amidiert gefiltert -- die Spalten sind hier konstant
    und werden trotzdem explizit gesetzt, damit das Schema stimmt.
    """
    d = pd.read_csv(Path(pfad or DB_EXTERN_DIR / "dramp_vergleich.csv"))
    return pd.DataFrame({
        "quelle": "dramp",
        "quell_id": d["dramp_id"],
        "sequence": d["sequence"].astype(str),
        "species": d["species"],
        "zensur": d["zensur"],
        "mic_um": d["mic_um"],
        "n_frei": d["n_terminus"].eq("Free"),
        "c_amidiert": d["c_terminus"].eq("Amidation"),
        "c_frei": d["c_terminus"].eq("Free"),
        "medium": pd.NA,   # DRAMP gibt kein Medium an
    })


def _aggregiere(gruppe: pd.DataFrame) -> pd.Series:
    """Eine Zeile je (Sequenz, Spezies) aus mehreren Messungen.

    Median ueber log10(MIC) der UNZENSIERTEN Messungen. Median statt Mittel,
    weil MIC-Werte aus verschiedenen Laboren weit streuen; log, weil die
    Messung in Zweierverduennungen laeuft.

    Gibt es keine unzensierte Messung, wird die zensierte uebernommen und die
    Richtung in `zensur` behalten -- das Peptid faellt also nicht raus, nur
    weil nur eine Schranke bekannt ist.

    `spanne` ist das Verhaeltnis groesster zu kleinster unzensierter Messung.
    Das ist das Messrauschen derselben Sequenz zwischen Laboren und damit die
    Obergrenze dessen, was ein Praediktor ueberhaupt treffen kann.
    """
    unz = gruppe[(gruppe["zensur"] == "=") & gruppe["mic_um"].notna()]
    if len(unz):
        logs = np.log10(unz["mic_um"].to_numpy())
        wert, zensur = float(10 ** np.median(logs)), "="
        spanne = float(unz["mic_um"].max() / unz["mic_um"].min())
    else:
        rest = gruppe[gruppe["mic_um"].notna()]
        if rest.empty:
            return pd.Series(dtype=object)
        logs = np.log10(rest["mic_um"].to_numpy())
        wert = float(10 ** np.median(logs))
        # Haeufigste Richtung; bei Gleichstand die konservativere (">").
        zensur = rest["zensur"].mode().iat[0]
        spanne = float("nan")
    return pd.Series({
        "mic_mess": wert,
        "zensur": zensur,
        "n_mess": len(gruppe),
        "n_unzensiert": len(unz),
        "spanne": spanne,
        "quellen": "|".join(sorted(set(gruppe["quelle"]))),
        "c_amidiert": bool(gruppe["c_amidiert"].any()),
    })


def baue_tool_val(
    min_len: int = 5,
    max_len: int = 50,
    torres_csv: Path | None = None,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Der Vergleichsdatensatz, eine Zeile je (Sequenz, Spezies).

    Filter, der Reihe nach:
        1. nur die drei Spezies, die ANIA und APEX gemeinsam haben
        2. nur kanonische Grossbuchstaben -- das schliesst X und die
           klein geschriebenen D-Aminosaeuren aus DBAASP automatisch aus,
           deshalb VOR dem Grossschreiben pruefen
        3. N-Terminus frei; C-Terminus frei oder amidiert
        4. Laenge min_len..max_len (APEX kann nichts darueber)

    Amidierung bleibt DRIN und bekommt die Spalte `c_amidiert`. Sie macht
    bei DRAMP zwei Drittel der Daten aus; sie wegzuwerfen hiesse, den
    Datensatz zu halbieren. Mit der Spalte laesst sich der Vergleich einmal
    gesamt und einmal nur unmodifiziert berichten.

    Returns:
        DataFrame mit sequence, species, mic_mess, zensur, n_mess,
        n_unzensiert, spanne, quellen, c_amidiert, sequence_length,
        mic_band, in_torres_train.
    """
    roh = pd.concat([_aus_dbaasp(), _aus_dramp()], ignore_index=True)
    n0 = len(roh)

    roh = roh[roh["species"].isin(ZIELSPEZIES)]
    n1 = len(roh)
    roh = roh[roh["sequence"].str.fullmatch(KANONISCH)]
    n2 = len(roh)
    roh = roh[roh["n_frei"] & (roh["c_frei"] | roh["c_amidiert"])]
    n3 = len(roh)
    laenge = roh["sequence"].str.len()
    roh = roh[laenge.between(min_len, max_len)]
    n4 = len(roh)
    roh = roh[roh["mic_um"].notna() & (roh["mic_um"] > 0)]
    n5 = len(roh)

    # Spalten VOR dem apply auswaehlen: so stellt sich die Frage nach
    # include_groups gar nicht erst, die je nach pandas-Version anders heisst.
    df = (roh.groupby(["sequence", "species"], sort=True)
             [["zensur", "mic_um", "quelle", "c_amidiert"]]
             .apply(_aggregiere)
             .reset_index())
    df["sequence_length"] = df["sequence"].str.len()
    df["mic_band"] = pd.cut(
        df["mic_mess"],
        bins=[g[0] for g in MIC_BAENDER] + [MIC_BAENDER[-1][1]],
        labels=[g[2] for g in MIC_BAENDER], right=True, include_lowest=True)

    torres = pd.read_csv(Path(torres_csv or TRACKED_DIR / "torres_train_sequences.csv"))
    df["in_torres_train"] = df["sequence"].isin(
        set(torres["sequence"].astype(str).str.upper()))

    print(f"[tool_val] {n0} Messungen roh")
    print(f"[tool_val]   -> {n1} nach Speziesfilter")
    print(f"[tool_val]   -> {n2} nach kanonisch")
    print(f"[tool_val]   -> {n3} nach Terminifilter")
    print(f"[tool_val]   -> {n4} nach Laenge {min_len}-{max_len}")
    print(f"[tool_val]   -> {n5} mit MIC-Wert")
    print(f"[tool_val] aggregiert: {len(df)} Zeilen, "
          f"{df['sequence'].nunique()} eindeutige Sequenzen")
    print(f"[tool_val] je Spezies: {df['species'].value_counts().to_dict()}")
    print(f"[tool_val] Zensur: {df['zensur'].value_counts().to_dict()}")
    print(f"[tool_val] amidiert: {int(df['c_amidiert'].sum())} | "
          f"in Torres' Trainingsdaten: {int(df['in_torres_train'].sum())}")
    print(f"[tool_val] Baender: {df['mic_band'].value_counts().to_dict()}")
    s = df["spanne"].dropna()
    if len(s):
        print(f"[tool_val] Messrauschen (max/min je Peptid-Spezies, n={len(s)}): "
              f"Median {s.median():.1f}x, 90. Perzentil {s.quantile(.9):.1f}x")

    # Nach eval_sets, nicht nach extern: extern haelt die Auszuege aus den
    # Datenbanken, hier liegt ein fertiger Bewertungsdatensatz.
    ziel = Path(out_csv) if out_csv else EVAL_SET_DIR / "tool_val.csv"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ziel, index=False)
    print(f"[tool_val] geschrieben: {ziel}")
    return df


def schreibe_ania_fasta(df: pd.DataFrame, out_fasta: Path) -> Path:
    """Eindeutige Sequenzen als FASTA fuer das ANIA-Webtool.

    Eine Sequenz kann fuer mehrere Spezies gemessen sein; ANIA sagt alle drei
    auf einmal vorher, deshalb reicht jede Sequenz EINMAL.
    """
    out_fasta = Path(out_fasta)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    seqs = sorted(set(df["sequence"]))
    with out_fasta.open("w", encoding="utf-8") as f:
        for i, s in enumerate(seqs):
            f.write(f">TV_{i:05d}\n{s}\n")
    print(f"[tool_val] {len(seqs)} eindeutige Sequenzen -> {out_fasta}")
    return out_fasta


# ---------------------------------------------------------------------------
# Vorhersagen anhaengen
# ---------------------------------------------------------------------------

def ergaenze_apex(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """APEX ueber die eindeutigen Sequenzen, je Spezies geometrisch gemittelt.

    APEX sagt je STAMM vorher, gemessen wird je SPEZIES. Zusammengefasst wird
    geometrisch (siehe APEX_STAEMME). Gerechnet wird einmal je eindeutiger
    Sequenz, nicht je Zeile -- dieselbe Sequenz steht fuer bis zu drei
    Spezies in der Tabelle.
    """
    from aguirre.config import Config
    from aguirre.evaluation import baue_scorer

    cfg = cfg or Config()
    seqs = sorted(set(df["sequence"]))
    scorer = next(sc for sc in baue_scorer(["apex"], cfg) if sc.name == "apex")
    werte = scorer.score(seqs)
    werte.index = seqs

    lang = []
    for spezies, staemme in APEX_STAEMME.items():
        spalten = [f"mic_apex__{s}" for s in staemme]
        fehlend = [s for s in spalten if s not in werte.columns]
        if fehlend:
            raise KeyError(f"APEX lieferte {fehlend} nicht. Vorhanden: "
                           f"{[c for c in werte.columns if c.startswith('mic_apex__')]}")
        geom = np.exp(np.log(werte[spalten].to_numpy()).mean(axis=1))
        lang.append(pd.DataFrame({"sequence": seqs, "species": spezies,
                                  "mic_apex": geom}))
    aus = df.merge(pd.concat(lang, ignore_index=True),
                   on=["sequence", "species"], how="left")
    print(f"[tool_val] APEX: {int(aus['mic_apex'].notna().sum())} von {len(aus)} Zeilen")
    return aus


def ergaenze_ania(df: pd.DataFrame, ania_csv: Path,
                  seq_spalte: str | None = None,
                  log_basis: float | None = 10.0) -> pd.DataFrame:
    """ANIA-Ergebnisse des Webtools anhaengen.

    Erwartete Kopfzeile des Webtools:

        ID,Sequence,Sequence Length,E. coli,S. aureus,P. aeruginosa

    Die Spezies werden ueber Teilstrings zugeordnet (coli, aureus,
    aeruginosa), damit kleine Formatwechsel nichts brechen. Passt es nicht,
    meldet die Funktion die gefundenen Spalten, statt still zu verschmelzen.

    EINHEIT: ANIA gibt LOG MIC aus, nicht MIC -- die Spaltenkoepfe der
    Tabelle verschweigen es, die Achsenbeschriftung der Weboberflaeche nennt
    "(log MIC)". log_basis=10 rechnet zurueck. Dass es der Zehnerlogarithmus
    ist und nicht der natuerliche, ist an den Daten geprueft: ueber 6 442
    unzensierte Paare liegt der Median des log10-Residuums bei -0,07
    (Faktor 0,85) fuer die Basis 10, aber bei -0,55 (Faktor 0,28) fuer e.
    Mit der falschen Basis haette ANIA eine systematische Unterschaetzung um
    Faktor 3,6, die reines Artefakt waere.

    log_basis=None nimmt die Spalten unveraendert, falls das Webtool sein
    Format aendert.
    """
    a = pd.read_csv(Path(ania_csv))
    if seq_spalte is None:
        # ANIA schreibt "Sequence" gross; andere Werkzeuge anders.
        treffer = [c for c in a.columns if c.strip().lower() == "sequence"]
        if len(treffer) != 1:
            raise KeyError(f"Keine eindeutige Sequenzspalte in {ania_csv}. "
                           f"Gefunden: {list(a.columns)}")
        seq_spalte = treffer[0]
    elif seq_spalte not in a.columns:
        raise KeyError(f"Spalte {seq_spalte!r} fehlt in {ania_csv}. "
                       f"Gefunden: {list(a.columns)}")
    spalten = {}
    for spezies in ZIELSPEZIES:
        kurz = spezies.split()[1].lower()        # coli, aureus, aeruginosa
        treffer = [c for c in a.columns
                   if kurz in c.lower() and c != seq_spalte]
        if len(treffer) != 1:
            raise KeyError(f"Fuer {spezies} nicht genau eine Spalte gefunden, "
                           f"sondern {treffer}. Spalten: {list(a.columns)}")
        spalten[spezies] = treffer[0]
    def zurueck(roh: pd.Series) -> pd.Series:
        werte = pd.to_numeric(roh, errors="coerce")
        return werte if log_basis is None else log_basis ** werte

    lang = pd.concat(
        [pd.DataFrame({"sequence": a[seq_spalte].astype(str).str.upper(),
                       "species": sp, "mic_ania": zurueck(a[c])})
         for sp, c in spalten.items()], ignore_index=True)
    aus = df.merge(lang.drop_duplicates(subset=["sequence", "species"]),
                   on=["sequence", "species"], how="left")
    basis = "roh" if log_basis is None else f"{log_basis:g}**x zurueckgerechnet"
    print(f"[tool_val] ANIA: {int(aus['mic_ania'].notna().sum())} von {len(aus)} "
          f"Zeilen, {basis}")
    if log_basis is not None:
        g = aus["mic_ania"].dropna()
        print(f"[tool_val]   MIC [µM]: Median {g.median():.2f}, "
              f"min {g.min():.3f}, max {g.max():.0f}")
    return aus


# ---------------------------------------------------------------------------
# Bayes: wie stark weicht ein Praediktor systematisch ab?
# ---------------------------------------------------------------------------

def residuen(df: pd.DataFrame, pred_spalte: str) -> pd.DataFrame:
    """log10(Vorhersage) - log10(Messung), mit Zensurrichtung.

    Wie die Zensur sich auf das RESIDUUM uebertraegt, ist der Punkt, an dem
    man sich leicht vertut:

        ">200"  wahre MIC groesser  -> wahres Residuum KLEINER als beobachtet
        "<1"    wahre MIC kleiner   -> wahres Residuum GROESSER als beobachtet

    Die Richtung dreht sich also um, weil die Messung im Nenner steht.
    "~" ist der Mittelwert einer angegebenen Spanne und zaehlt als beobachtet.
    "?" faellt raus.
    """
    d = df[df[pred_spalte].notna() & df["mic_mess"].notna()
           & (df[pred_spalte] > 0) & (df["mic_mess"] > 0)].copy()
    d["r"] = np.log10(d[pred_spalte]) - np.log10(d["mic_mess"])
    d["art"] = d["zensur"].map({"=": "beob", "~": "beob",
                                ">": "obergrenze", "<": "untergrenze"})
    return d[d["art"].notna()]


def posterior_gitter(r: np.ndarray, art: np.ndarray, nu: int = 4,
                     mu_bereich: tuple[float, float] | None = None,
                     n_mu: int = 401, n_sigma: int = 241) -> dict:
    """Posterior fuer Lage mu und Streuung sigma der Residuen, auf einem Gitter.

    Zwei Parameter, also ist ein Gitter exakt -- kein Sampler, keine
    Konvergenzdiagnostik, bei gleichem Gitter reproduzierbar bis aufs Bit.

    Likelihood ist Student-t mit nu=4 statt Normal: MIC-Residuen haben
    schwere Raender (ein Praediktor liegt gelegentlich um Faktor 1000
    daneben), und eine Normalverteilung liesse ihre Lage von solchen
    Ausreissern ziehen.

    ACHTUNG, hart erkaufte Einsicht: die Zensurbehandlung ueber CDF/SF ist
    NUR gueltig, wenn die Schranke fest und vom latenten Wert unabhaengig
    ist -- der Lehrbuchfall der Survival-Analyse. Hier ist sie das NICHT.
    Zensiert wird die MESSUNG, und die steht im Nenner des Residuums:

        r_obs = log(pred) - log(c)  und  r_wahr = r_obs - log(c_wahr / c)

    Die "Schranke" r_obs haengt also selbst von r ab. Die uebliche
    zensierte Likelihood ist damit fehlspezifiziert. Eine Simulation mit
    150 Replikationen, 22 % rechtszensiert, zeigt die Folge deutlich:

        nur unzensiert   Fehler +0,002   HDI-Deckung 87 %   <- richtig
        Zensur als CDF   Fehler -0,075   HDI-Deckung 27 %
        Zensur als Wert  Fehler +0,090   HDI-Deckung 15 %

    Sauber waere ein gemeinsames Modell fuer Messung UND Fehler, das die
    unbekannte wahre MIC herausintegriert -- dafuer braeuchte man eine
    Verteilungsannahme ueber die MICs selbst. Der Aufwand lohnt nicht:
    zensierte Punkte werden stattdessen separat als Trefferquote berichtet
    (siehe bayes_tabelle). Die Parameter art="obergrenze"/"untergrenze"
    bleiben, weil sie fuer echte feste Schranken korrekt sind -- fuer
    MIC-Residuen aber bitte nicht benutzen.

    Prior: flach in mu, flach in log(sigma) -- letzteres ist das uebliche
    schwach informative fuer einen Skalenparameter und steckt darin, dass
    das sigma-Gitter logarithmisch gelegt ist.

    Das Gitter wird an die Daten gelegt und ist NICHT fest. Ein festes
    Fenster von -3 bis 3 klingt grosszuegig, ist es aber nicht: bei n = 400
    ist der Posterior etwa 0,07 breit, also drei Gitterzellen. Das HDI wird
    dann quantisiert und systematisch zu schmal -- und zwar umso mehr, je
    mehr Daten man hat, was die Richtung ist, in der man es am wenigsten
    erwartet. Zentrum und Breite kommen deshalb aus Median und MAD der
    Residuen, mit reichlich Luft. mu_bereich ueberschreibt das von Hand.

    Returns:
        dict mit mu (Gitter), post (Randposterior von mu), p_ueberschaetzt
        (P(mu>0)), median, hdi89, faktor (10**median) und faktor_hdi.
    """
    from scipy import stats

    r = np.asarray(r, dtype=float)
    art = np.asarray(art)
    if len(r) < 3:
        raise ValueError("Zu wenige Punkte fuer einen Posterior.")

    # Gitter an die Daten legen. MAD statt Standardabweichung, damit einzelne
    # Ausreisser das Fenster nicht aufblasen.
    zentrum = float(np.median(r))
    streu = float(1.4826 * np.median(np.abs(r - zentrum))) or float(np.std(r)) or 0.1
    se = streu / np.sqrt(len(r))
    halb = max(12.0 * se, 0.25)
    mu = (np.linspace(*mu_bereich, n_mu) if mu_bereich
          else np.linspace(zentrum - halb, zentrum + halb, n_mu))
    sigma = np.logspace(np.log10(max(streu / 10, 1e-3)),
                        np.log10(streu * 10), n_sigma)
    logpost = np.zeros((n_mu, n_sigma))
    gruppen = [(r[art == k], z) for k, z in
               (("beob", None), ("obergrenze", "cdf"), ("untergrenze", "sf"))]

    # Ueber sigma schleifen statt ein (n, n_mu, n_sigma)-Array zu bauen.
    # Bei 3000 Punkten und diesem Gitter waeren das 2,3 GB; so bleibt es bei
    # n x n_mu je Schritt. Kostet etwas Zeit, dafuer laeuft es ueberhaupt.
    for j, s in enumerate(sigma):
        spalte = 0.0
        for werte, z in gruppen:
            if not len(werte):
                continue
            standardisiert = (werte[:, None] - mu[None, :]) / s
            if z is None:
                beitrag = stats.t.logpdf(standardisiert, df=nu) - np.log(s)
            elif z == "cdf":
                beitrag = stats.t.logcdf(standardisiert, df=nu)
            else:
                beitrag = stats.t.logsf(standardisiert, df=nu)
            spalte = spalte + beitrag.sum(axis=0)
        logpost[:, j] = spalte

    logpost -= logpost.max()
    post2d = np.exp(logpost)
    post = post2d.sum(axis=1)
    post /= post.sum()

    kum = np.cumsum(post)
    median = float(np.interp(0.5, kum, mu))
    p_ueber = float(post[mu > 0].sum())

    # HDI aus dem Gitter: Dichten absteigend aufsummieren bis 89 %.
    ordnung = np.argsort(post)[::-1]
    drin = np.zeros_like(post, dtype=bool)
    drin[ordnung[np.cumsum(post[ordnung]) <= 0.89]] = True
    drin[ordnung[0]] = True
    hdi = (float(mu[drin].min()), float(mu[drin].max()))

    return {"mu": mu, "post": post, "p_ueberschaetzt": p_ueber,
            "median": median, "hdi89": hdi,
            "faktor": 10 ** median,
            "faktor_hdi": (10 ** hdi[0], 10 ** hdi[1])}


def _spearman(teil: pd.DataFrame, pred_spalte: str) -> float:
    from scipy import stats
    if len(teil) < 3:
        return float("nan")
    return float(stats.spearmanr(teil[pred_spalte], teil["mic_mess"]).statistic)


def _spearman_r(teil: pd.DataFrame) -> float:
    from scipy import stats
    if len(teil) < 3:
        return float("nan")
    return float(stats.spearmanr(teil["r"], teil["mic_mess"]).statistic)


def bayes_tabelle(df: pd.DataFrame,
                  pred_spalten: tuple[str, ...] = ("mic_apex", "mic_ania"),
                  nur_unmodifiziert: bool = False,
                  out_csv: Path | None = None) -> pd.DataFrame:
    """Je Spezies und Werkzeug: um welchen Faktor wird systematisch verfehlt?

    Der Faktor wird NUR aus unzensierten Messungen geschaetzt; die
    rechtszensierten stehen als `treffer_zensiert` daneben -- der Anteil,
    bei dem die Vorhersage ebenfalls oberhalb der Schranke liegt. Das ist
    fuer sie die einzige beantwortbare Frage, und fuer APEX bei inaktiven
    Peptiden eine aussagekraeftige.

    Die Zahl, die in die Arbeit gehoert, ist `faktor` mit HDI, nicht
    p_ueberschaetzt: bei einigen hundert Punkten ist P(mu>0) fast immer 0
    oder 1, also keine Information. "APEX ueberschaetzt um Faktor 2,3
    (89 %-HDI 1,9-2,8)" sagt dagegen etwas.
    """
    basis = df[~df["c_amidiert"]] if nur_unmodifiziert else df
    zeilen = []
    for spalte in pred_spalten:
        if spalte not in basis.columns:
            continue
        d = residuen(basis, spalte)
        for spezies in ZIELSPEZIES:
            alle = d[d["species"] == spezies]
            teil = alle[alle["art"] == "beob"]
            if len(teil) < 10:
                continue
            # NUR unzensierte Punkte -- Begruendung im Docstring von
            # posterior_gitter. Die zensierten kommen als Trefferquote
            # daneben, nicht in den Faktor.
            p = posterior_gitter(teil["r"].to_numpy(), teil["art"].to_numpy())
            zens = alle[alle["art"] == "obergrenze"]
            quote = float((zens[spalte] > zens["mic_mess"]).mean()) if len(zens) else float("nan")
            zeilen.append({
                "werkzeug": spalte.replace("mic_", ""),
                "species": spezies,
                "n": len(teil),
                "n_zensiert": len(zens),
                "treffer_zensiert": round(quote, 3),
                "faktor": round(p["faktor"], 2),
                "hdi89_unten": round(p["faktor_hdi"][0], 2),
                "hdi89_oben": round(p["faktor_hdi"][1], 2),
                "p_ueberschaetzt": round(p["p_ueberschaetzt"], 3),
                # Spearman zwischen Vorhersage und Messung, nur auf den
                # UNZENSIERTEN Punkten -- bei zensierten ist der wahre Wert
                # unbekannt, ein Rang also nicht bestimmbar.
                "spearman": round(_spearman(teil, spalte), 2),
                # Haengt der Fehler vom MIC-Niveau ab? Wenn ja, ist das
                # Weglassen der zensierten Punkte selektiv und der Faktor
                # verschoben. Nahe 0 heisst: Weglassen ist unbedenklich.
                "r_vs_mess": round(_spearman_r(teil), 2),
            })
    tab = pd.DataFrame(zeilen)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(out_csv, index=False)
        print(f"[tool_val] geschrieben: {out_csv}")
    return tab


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _speichere(fig, out: Path, dpi: int, cfg) -> None:
    """Abbildung sichern, aber keinen Verzeichnisbaum erfinden.

    parents=False mit Absicht: ein vertippter Pfad soll abbrechen und nicht
    stillschweigend einen zweiten "Ausarbeitung"-Ordner an der falschen
    Stelle anlegen. Eine Ebene wird angelegt, das ist Bequemlichkeit; zwei
    waeren schon ein Hinweis darauf, dass der Pfad nicht stimmt.
    """
    if not out.parent.exists():
        if not out.parent.parent.exists():
            raise FileNotFoundError(
                f"{out.parent.parent} gibt es nicht -- Pfad pruefen. "
                "Die Abbildungen gehoeren nach PROJECT_ROOT/Ausarbeitung/"
                "abbildungen, nicht unter wrath_of_god.")
        out.parent.mkdir()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[tool_val] geschrieben: {out} ({dpi} dpi)")


def _waehle_farben(farben, n: int, standard_index: tuple[int, ...], cfg) -> list[str]:
    """Farben aufloesen: entweder die uebergebenen oder die Vorgabe.

    Die Vorgabe zeigt auf Indizes in cfg.plot_farben statt auf Hexwerte --
    aenderst du dort die Palette, ziehen die Abbildungen mit, und die
    Farbzuordnung bleibt ueber alle Abbildungen des Projekts dieselbe.
    """
    if farben is None:
        return [cfg.plot_farben[i % len(cfg.plot_farben)]
                for i in standard_index[:n]]
    farben = list(farben)
    if len(farben) < n:
        raise ValueError(f"{n} Farben noetig, {len(farben)} uebergeben.")
    return farben[:n]


def plot_baender(df: pd.DataFrame,
                 pred_spalten: tuple[str, ...] = ("mic_apex", "mic_ania"),
                 farben=None, zensur_kasten: bool = False,
                 out: Path | None = None, dpi: int = 400):
    """Drei Achsen (Spezies), je drei MIC-Baender, je Band ein Kasten pro Quelle.

    Das gemessene MIC ist als erster Kasten je Band mit drin -- ohne ihn
    sieht man die Abweichung nicht, sondern muss sie sich denken.

    Args:
        farben: eine Farbe je Reihe, Messung zuerst. Default sind die ersten
            Eintraege aus cfg.plot_farben, also Rot fuer die Messung und die
            beiden Gruentoene fuer die Werkzeuge.
        zensur_kasten: Angaben zu den rechtszensierten Messungen unter jede
            Achse setzen. Default aus -- dieselbe Information steht in
            bayes_tabelle (Spalten n_zensiert, treffer_zensiert) und in der
            unteren Zeile von plot_scatter, und dreimal dasselbe macht die
            Abbildung nur unruhig.

    WARNUNG, die in die Bildunterschrift gehoert: die Baender sind nach der
    GEMESSENEN MIC gebildet und aufgetragen wird die VORHERGESAGTE. Damit
    bekommt man Regression zur Mitte geschenkt -- das niedrige Band enthaelt
    bevorzugt Messungen, deren Rauschen nach unten zeigt, also wirkt jeder
    Praediktor dort zu hoch und im hohen Band zu niedrig. Das passiert auch
    bei einem perfekten, nur verrauschten Praediktor. Der Scatter in
    plot_scatter hat dieses Artefakt nicht.
    """
    import matplotlib.pyplot as plt
    from aguirre.config import Config

    cfg = Config()
    # In log10 RECHNEN, nicht bloss die Achse log skalieren. Median und Box
    # waeren identisch -- Quartile sind rangbasiert --, die Whisker nicht:
    # sie reichen bis 1,5 x IQR, und der IQR wird im Datenraum gebildet. Bei
    # Rohwerten auf log-Achse waere das ein linear bestimmtes Kriterium, nur
    # log gezeichnet. MIC lebt multiplikativ, also ist log der Datenraum.
    reihen = ["mic_mess", *[s for s in pred_spalten if s in df.columns]]
    # Eine Farbe je Reihe, gemessen zuerst. Vorgabe: Rot fuer die Messung,
    # dann die beiden Gruentoene fuer die Werkzeuge.
    farben = _waehle_farben(farben, len(reihen), (0, 1, 2, 3, 4, 5), cfg)
    labels = [g[2] for g in MIC_BAENDER]
    # Tickbeschriftung aus den Zahlen bauen, nicht aus dem Labeltext: so
    # stehen die Grenzen in der Abbildung und niemand muss raten, was
    # "niedrig" heisst.
    ticks = []
    for lo, hi, name in MIC_BAENDER:
        kurz = name.split(" ")[0]
        if lo <= 0:
            ticks.append(f"{kurz}\n$\\leq$ {hi:g} µM")
        elif math.isinf(hi):
            ticks.append(f"{kurz}\n> {lo:g} µM")
        else:
            ticks.append(f"{kurz}\n{lo:g}–{hi:g} µM")

    # Nur Messungen mit echtem Zahlenwert. Bei einer rechtszensierten Zeile
    # ist mic_mess die Schranke, nicht die Messung -- sie als Zahl in einen
    # Boxplot zu legen, waere derselbe Fehler, den wir beim Faktor vermeiden,
    # und er traefe die rote Reihe und die Baendereinteilung gleichzeitig.
    unzensiert = df[df["zensur"].isin(["=", "~"])]
    weg = len(df) - len(unzensiert)
    if weg:
        print(f"[tool_val] Boxplot: {weg} zensierte von {len(df)} Zeilen "
              "nicht dargestellt (Schranke statt Messwert)")

    fig, achsen = plt.subplots(1, 3, figsize=(15, 6.6 if zensur_kasten else 5.4),
                               sharey=True)
    for ax, spezies in zip(achsen, ZIELSPEZIES):
        teil = unzensiert[unzensiert["species"] == spezies]
        breite = 0.8 / len(reihen)
        for j, spalte in enumerate(reihen):
            daten, pos = [], []
            for i, band in enumerate(labels):
                werte = teil.loc[teil["mic_band"] == band, spalte].dropna()
                werte = werte[werte > 0]
                daten.append(np.log10(werte.to_numpy()) if len(werte)
                             else np.array([np.nan]))
                pos.append(i + (j - (len(reihen) - 1) / 2) * breite)
            kasten = ax.boxplot(daten, positions=pos, widths=breite * 0.85,
                                patch_artist=True, showfliers=False,
                                medianprops={"color": "black"})
            for stueck in kasten["boxes"]:
                stueck.set_facecolor(farben[j])
                stueck.set_alpha(0.75)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(ticks, fontsize=9.5)
        ax.set_title(spezies, style="italic", fontsize=12)
        ax.grid(axis="y", alpha=cfg.plot_grid_alpha)

        if not zensur_kasten:
            continue

        zeilen = ["rechtszensierte Messungen (>Schranke):",
                  "Anteil mit Vorhersage ebenfalls darüber"]
        for (lo, hi, band), kurz in zip(MIC_BAENDER, ticks):
            voll = df[df["species"] == spezies]
            z = voll[(voll["mic_band"] == band) & (voll["zensur"] == ">")]
            name = kurz.split("\n")[0]
            if not len(z):
                zeilen.append(f"  {name:<8s} n = 0")
                continue
            anteile = "  ".join(
                f"{s.replace('mic_','')} {(z[s] > z['mic_mess']).mean():.0%}"
                for s in pred_spalten if s in z.columns)
            zeilen.append(f"  {name:<8s} n = {len(z):<5d} {anteile}")
        # UNTER die Achse, nicht hinein: im Feld verdeckte der Kasten die
        # Antennen der unteren Baender. Der Platz dafuer kommt aus dem
        # rect-Argument von tight_layout weiter unten.
        ax.text(0.0, -0.17, "\n".join(zeilen), transform=ax.transAxes,
                va="top", ha="left", fontsize=8, family="monospace",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8",
                      "boxstyle": "round,pad=0.4"})

    achsen[0].set_ylabel(r"$\log_{10}$(MIC [µM])")
    fig.legend([plt.Rectangle((0, 0), 1, 1, facecolor=f, alpha=0.75) for f in farben],
               [r.replace("mic_", "") for r in reihen],
               loc="upper right", ncol=len(reihen), frameon=False)
    fig.suptitle("Vorhergesagte gegen gemessene MIC, nach Bändern der "
                 "gemessenen MIC", fontsize=12)
    # Unten Platz lassen fuer die zweizeiligen Ticks und die Zensurkaesten.
    fig.tight_layout(rect=(0, 0.19 if zensur_kasten else 0.0, 1, 0.96))
    if out:
        _speichere(fig, Path(out), dpi, cfg)
    return fig


def plot_scatter(df: pd.DataFrame,
                 pred_spalten: tuple[str, ...] = ("mic_apex", "mic_ania"),
                 farben=None,
                 out: Path | None = None, dpi: int = 400):
    """Vorhergesagt gegen gemessen, 2 x 3: oben unzensiert, unten zensiert.

    Die Abbildung ohne das Regressions-zur-Mitte-Artefakt der Baender.

    Getrennte Zeilen, weil die beiden Faelle verschiedene Fragen stellen.
    Oben steht ein Punkt fuer ein Wertepaar, und der Abstand zur
    Winkelhalbierenden IST der Fehler -- daraus kommt der Faktor. Unten ist
    die x-Koordinate nur eine untere Schranke: die wahre MIC liegt
    irgendwo rechts davon. Ein Fehler laesst sich daraus nicht berechnen,
    nur eine Vertraeglichkeit ablesen -- liegt der Punkt OBERHALB der
    Winkelhalbierenden, ist die Vorhersage mit der Schranke vertraeglich.
    Zusammen in einem Feld waere beides nicht unterscheidbar.

    Args:
        farben: eine Farbe je Eintrag in pred_spalten. Default sind die
            beiden Gruentoene aus cfg.plot_farben (Index 1 und 2), dieselbe
            Zuordnung wie in plot_baender.
    """
    import matplotlib.pyplot as plt
    from aguirre.config import Config

    cfg = Config()
    spalten = [s for s in pred_spalten if s in df.columns]
    # Vorgabe: die beiden Gruentoene, dieselbe Zuordnung wie in plot_baender,
    # damit ein Werkzeug ueber alle Abbildungen seine Farbe behaelt.
    farben = dict(zip(spalten, _waehle_farben(farben, len(spalten), (1, 2), cfg)))
    tab = bayes_tabelle(df, tuple(spalten))

    g = df[["mic_mess", *spalten]].to_numpy(dtype=float)
    g = g[np.isfinite(g) & (g > 0)]
    grenzen = [g.min() * 0.6, g.max() * 1.6]

    fig, achsen = plt.subplots(2, 3, figsize=(15, 9.5),
                               sharex=True, sharey=True)
    for k, spezies in enumerate(ZIELSPEZIES):
        teil = df[df["species"] == spezies]
        frei = teil[teil["zensur"].isin(["=", "~"])]
        zens = teil[teil["zensur"] == ">"]

        for zeile, (daten, marker, groesse) in enumerate(
                ((frei, "o", 9), (zens, ">", 26))):
            ax = achsen[zeile, k]
            for spalte in spalten:
                ax.scatter(daten["mic_mess"], daten[spalte],
                           s=groesse, alpha=0.45, linewidths=0,
                           color=farben[spalte], label=spalte.replace("mic_", ""))
            ax.plot(grenzen, grenzen, "k--", lw=1, alpha=0.7)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*grenzen); ax.set_ylim(*grenzen)
            ax.grid(alpha=cfg.plot_grid_alpha)

        achsen[0, k].set_title(spezies, style="italic", fontsize=12)
        achsen[1, k].set_xlabel("gemessene MIC [µM]", fontsize=10)

        # Oben: der Faktor. Unten: die Vertraeglichkeitsquote.
        oben, unten = [], []
        for _, z in tab[tab["species"] == spezies].iterrows():
            oben.append(f"{z['werkzeug']}:  ×{z['faktor']:.2f} "
                        f"[{z['hdi89_unten']:.2f}–{z['hdi89_oben']:.2f}]   "
                        f"ρ = {z['spearman']:.2f}   n = {z['n']}")
        for spalte in spalten:
            if len(zens):
                q = (zens[spalte] > zens["mic_mess"]).mean()
                unten.append(f"{spalte.replace('mic_','')}:  {q:.0%} oberhalb")
        if len(zens):
            unten.append(f"n = {len(zens)}")
        for zeile, zeilen in ((0, oben), (1, unten)):
            if zeilen:
                achsen[zeile, k].text(
                    0.03, 0.97, "\n".join(zeilen),
                    transform=achsen[zeile, k].transAxes, va="top", ha="left",
                    fontsize=9, bbox={"facecolor": "white", "alpha": 0.85,
                                      "edgecolor": "0.8",
                                      "boxstyle": "round,pad=0.35"})

    achsen[0, 0].set_ylabel("vorhergesagte MIC [µM]\n\nunzensierte Messungen",
                            fontsize=10)
    achsen[1, 0].set_ylabel("vorhergesagte MIC [µM]\n\nrechtszensiert "
                            "(wahre MIC rechts vom Punkt)", fontsize=10)
    achsen[1, 2].legend(fontsize=10, loc="lower right", frameon=False,
                        markerscale=2)
    fig.suptitle("Vorhersage gegen Messung. Faktor = systematische Abweichung "
                 "(Posterior-Median, 89 %-HDI)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out:
        _speichere(fig, Path(out), dpi, cfg)
    return fig
