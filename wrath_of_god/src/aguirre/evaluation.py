"""evaluation.py -- Bewertung und Vergleich generierter Peptidsequenzen.

Architektur (nach Torres et al., Eval-Teil):

    SequenceSet   Datencontainer: Name + DataFrame (eine Zeile pro Sequenz)
    Scorer        berechnet Pro-Sequenz-Spalten (ProGen2, ESM-PPPL, APEX, Entropie)
    Evaluator     fuehrt mehrere Scorer aus, mit Cache auf Platte
    Filter        SequenceSet -> SequenceSet (z.B. MIC-Filter)
    Comparison    vergleicht mehrere SequenceSets, liefert Tabelle + Figure

Datenfluss:

    SequenceSet --Evaluator(Scorer...)--> SequenceSet (mit Score-Spalten)
                --Filter--> gefiltertes SequenceSet
                --Comparison({"train":..., "generiert":..., "gefiltert":...})--> ComparisonResult

Vertrag zwischen den Teilen (NICHT ohne Absprache aendern):
    * Pflichtspalte jedes SequenceSet: "sequence" (str, Grossbuchstaben, 20 Standard-AS)
    * Scorer haengen nur Spalten an, sie veraendern oder loeschen keine.
    * Spaltennamen der Scorer (siehe Scorer.columns):
        ppl_progen2   Perplexitaet unter ProGen2 (kleiner = natuerlicher)
        pppl_esm2     Pseudo-Perplexitaet unter ESM2
        mic_apex      vorhergesagte MIC (APEX)
        aa_entropy    Shannon-Entropie der AS-Verteilung der Sequenz (in bit)
"""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aguirre.config import Config
from aguirre.data import read_fasta
from aguirre.helpers import _shannons_H
cfg = Config()

AMINO_ACIDS = cfg.AS_alphabet


# ---------------------------------------------------------------------------
# 1. Datencontainer
# ---------------------------------------------------------------------------

@dataclass
class SequenceSet:
    """Eine benannte Menge von Sequenzen plus ihre berechneten Eigenschaften.

    Attributes:
        name: Anzeigename, wird in Plots und Tabellen als Gruppenlabel genutzt
            (z.B. "train", "generiert", "gefiltert").
        data: DataFrame mit mindestens der Spalte "sequence". Jede weitere
            Spalte ist eine Pro-Sequenz-Eigenschaft (Scorer-Ausgabe).
    """

    name: str
    data: pd.DataFrame

def __post_init__(self) -> None:
    """Prueft, dass die Spalte "sequence" existiert und nur gueltige AS enthaelt."""
    if "sequence" not in self.data.columns:
        raise ValueError(f"Spalte 'sequence' fehlt, vorhanden: {list(self.data.columns)}")
    seqs = self.data["sequence"]
    if seqs.isna().any() or (seqs == "").any():
        raise ValueError("Leere oder fehlende Sequenzen im Set.")
    invalid = set("".join(seqs)) - set(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Ungueltige Zeichen in Sequenzen: {sorted(invalid)}")

@classmethod
def from_fasta(cls, path: Path, name: str) -> "SequenceSet":
    ids, sequences = read_fasta(path)
    data = pd.DataFrame({"id": ids, "sequence": sequences})
    return cls(name=name, data=data)

@classmethod
def from_csv(cls, path: Path, name: str, sequence_col: str = "sequence") -> "SequenceSet":
    data = pd.read_csv(path)
    if sequence_col != "sequence":
        data = data.rename(columns={sequence_col: "sequence"})
    return cls(name=name, data=data)

def with_columns(self, new_cols: pd.DataFrame) -> "SequenceSet":
    if len(new_cols) != len(self.data):
        raise ValueError("new_cols hat eine andere Zeilenzahl als das Set.")
    # gleiche Spaltennamen werden ueberschrieben (z.B. bei erneutem Evaluator-Lauf)
    old = self.data.drop(columns=new_cols.columns.intersection(self.data.columns))
    data = pd.concat([old, new_cols.set_axis(old.index)], axis=1)
    return SequenceSet(name=self.name, data=data)

def subset(self, mask: pd.Series, name: str | None = None) -> "SequenceSet":
    if mask.dtype != bool or len(mask) != len(self.data):
        raise ValueError("mask muss ein bool-Series in der Laenge des Sets sein.")
    data = self.data.loc[mask.to_numpy()].reset_index(drop=True)
    return SequenceSet(name=name or self.name, data=data)

# ---------------------------------------------------------------------------
# 2. Scorer: berechnen Pro-Sequenz-Spalten
# ---------------------------------------------------------------------------

class Scorer(ABC):
    """Basisklasse: nimmt Sequenzen, gibt ein DataFrame mit Score-Spalten zurueck.

    Ein Scorer kennt weder Vergleiche noch Plots. Er muss nur wissen, wie man
    aus einer Sequenz eine oder mehrere Zahlen macht.

    Attributes:
        name: eindeutiger Name, dient als Cache-Schluessel (z.B. "progen2").
        columns: Namen der Spalten, die score() zurueckgibt.
    """

    name: str
    columns: list[str]

    @abstractmethod
    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Berechnet die Scores.

        Args:
            sequences: Sequenzen in beliebiger Reihenfolge.

        Returns:
            DataFrame mit len(sequences) Zeilen, gleicher Reihenfolge wie die
            Eingabe und genau den Spalten aus self.columns.
        """


class AminoAcidEntropyScorer(Scorer):
    """Shannon-Entropie der Aminosaeureverteilung pro Sequenz ("Residuenvielfalt").

    H = -sum_a p_a * log2(p_a), p_a = Anteil der AS a in der Sequenz.
    Niedrig = wenige AS dominieren, hoch = gleichmaessig gemischt (max. log2(20)).
    Braucht kein Modell, taugt daher als erster Test der Pipeline.
    """

    name = "aa_entropy"
    columns = ["aa_entropy"]

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        H = _shannons_H(sequences)
        return pd.DataFrame({self.columns[0]: H})


class ProGen2Scorer(Scorer):
    """Perplexitaet unter ProGen2 (autoregressiv, trainiert auf natuerlichen Proteinen).

    PPL = exp(-1/N * sum_i log p(x_i | x_<i)).  Kleiner = natuerlicher.
    Laeuft auf GPU -> in Batches arbeiten, Modell nur einmal laden.
    """

    name = "progen2"
    columns = ["ppl_progen2"]

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 64) -> None:
        """Laedt Modell und Tokenizer (siehe Cluster-Anleitung: Gewichte vorab laden)."""
        raise NotImplementedError

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError


class ESMPseudoPerplexityScorer(Scorer):
    """Pseudo-Perplexitaet unter ESM2 (maskiertes LM, Salazar et al. 2020).

    Jede Position wird einmal maskiert -> N Forward-Passes pro Sequenz.
    Alle maskierten Varianten einer Sequenz zu einem Batch buendeln.
    Zweitmetrik, weil der Generator selbst ESM2-Embeddings nutzt.
    """

    name = "esm2_pppl"
    columns = ["pppl_esm2"]

    def __init__(self, esm_variant: str = "8M", device: str = "cuda") -> None:
        raise NotImplementedError

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError


class APEXScorer(Scorer):
    """Vorhergesagte MIC (APEX, Wan et al.).

    Gibt zusaetzlich zu mic_apex ggf. weitere Spalten (z.B. pro Bakterienstamm) zurueck.
    Bei mehreren Staemmen: eine Spalte pro Stamm plus eine Aggregat-Spalte
    "mic_apex" (z.B. geometrisches Mittel), damit MICFilter eine feste Spalte hat.
    """

    name = "apex"
    columns = ["mic_apex"]

    def __init__(self, model_path: Path, device: str = "cuda") -> None:
        raise NotImplementedError

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Evaluator: Scorer ausfuehren, Ergebnisse cachen
# ---------------------------------------------------------------------------

class Evaluator:
    """Fuehrt eine Liste von Scorern ueber ein SequenceSet aus.

    Der Cache liegt pro Scorer auf Platte (z.B. Parquet), Schluessel ist ein
    Hash der Sequenz. So wird jede Sequenz pro Scorer nur einmal berechnet,
    auch wenn sie in mehreren Sets vorkommt oder ein Plot neu gebaut wird.
    """

    def __init__(self, scorers: Sequence[Scorer], cache_dir: Path | None = None) -> None:
        """
        Args:
            scorers: Scorer, die angewendet werden sollen.
            cache_dir: Ordner fuer den Cache; None = kein Caching.
        """
        raise NotImplementedError

    def run(self, seqset: SequenceSet) -> SequenceSet:
        """Wendet alle Scorer an und gibt ein neues SequenceSet mit den Score-Spalten zurueck.

        Ablauf pro Scorer: Sequenzen im Cache nachschlagen -> nur die fehlenden
        scoren -> Cache aktualisieren -> Spalten anhaengen.
        """
        raise NotImplementedError

    def _load_cache(self, scorer: Scorer) -> pd.DataFrame:
        """Liest den Cache eines Scorers (Index = Sequenz-Hash); leeres DataFrame, falls keiner existiert."""
        raise NotImplementedError

    def _save_cache(self, scorer: Scorer, cache: pd.DataFrame) -> None:
        """Schreibt den Cache eines Scorers."""
        raise NotImplementedError


def sequence_hash(sequence: str) -> str:
    """Stabiler Hash einer Sequenz (z.B. sha1 hexdigest), dient als Cache-Schluessel."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Filter: SequenceSet -> SequenceSet
# ---------------------------------------------------------------------------

class Filter(ABC):
    """Basisklasse fuer Filter. Ein Filter waehlt Sequenzen aus, er berechnet nichts."""

    @abstractmethod
    def apply(self, seqset: SequenceSet) -> SequenceSet:
        """Gibt das gefilterte SequenceSet zurueck (neuer Name, z.B. "<name>_gefiltert")."""


class MICFilter(Filter):
    """Behaelt Sequenzen mit vorhergesagter MIC <= max_mic (Spalte "mic_apex").

    Default 64.0 ist Kriterium (1) aus Torres et al., Methods "Computational
    filtering": mean MIC <= 64 umol/L. Die Kriterien (2) Aehnlichkeit zu
    bekannten AMPs und (3) paarweise Diversitaet sind NICHT Teil dieses
    Filters -- sie brauchen ein Alignment gegen DRAMP/DBAASP/APD3 und
    gehoeren in einen eigenen SimilarityFilter.
    """

    def __init__(self, max_mic: float = 64.0, column: str = "mic_apex") -> None:
        if not np.isfinite(max_mic) or max_mic <= 0:
            raise ValueError(f"max_mic muss eine positive Zahl sein, war {max_mic!r}.")
        self.max_mic = float(max_mic)
        self.column = column

    def apply(self, seqset: SequenceSet) -> SequenceSet:
        if self.column not in seqset.data.columns:
            raise KeyError(
                f"Spalte '{self.column}' fehlt in Set '{seqset.name}'. "
                f"Erst den APEXScorer ueber das Set laufen lassen. "
                f"Vorhanden: {list(seqset.data.columns)}"
            )
        werte = pd.to_numeric(seqset.data[self.column], errors="coerce")
        n_fehlend = int(werte.isna().sum())
        if n_fehlend:
            print(
                f"[MICFilter] {n_fehlend} von {len(werte)} Sequenzen ohne "
                f"'{self.column}'-Wert -> verworfen."
            )
        mask = werte.notna() & werte.le(self.max_mic)
        gefiltert = seqset.subset(mask, name=f"{seqset.name}_mic{self.max_mic:g}")
        print(
            f"[MICFilter] {len(gefiltert.data)} von {len(seqset.data)} Sequenzen "
            f"behalten ({len(gefiltert.data) / max(len(seqset.data), 1):.1%})."
        )
        return gefiltert


# ---------------------------------------------------------------------------
# 5. Statistik-Helfer
# ---------------------------------------------------------------------------

def _as_array(x) -> np.ndarray:
    """Series/Liste/Array -> float-Array ohne NaN."""
    return pd.Series(x).astype(float).dropna().to_numpy()


def mann_whitney(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Zweiseitiger Mann-Whitney-U-Test. Gibt (U, p) zurueck."""
    a, b = _as_array(a), _as_array(b)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Beide Gruppen brauchen mindestens einen Wert.")
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(U), float(p)


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    """Effektstaerke Cliff's delta in [-1, 1].

    delta = P(a > b) - P(a < b).  Ergaenzt den p-Wert, der bei grossem n
    fast immer signifikant wird.

    Rangbasiert statt paarweise gerechnet: delta = 2*U1/(n1*n2) - 1. Das ist
    exakt dasselbe Ergebnis inklusive Bindungen, aber O(n log n) statt
    O(n^2) -- bei 50.000 vs. 50.000 waeren es sonst 2,5 Mrd. Vergleiche.
    """
    a, b = _as_array(a), _as_array(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("Beide Gruppen brauchen mindestens einen Wert.")
    ranks = stats.rankdata(np.concatenate([a, b]))
    U1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(2.0 * U1 / (n1 * n2) - 1.0)


def cliffs_label(delta: float) -> str:
    """Grobe Einordnung von Cliff's delta nach Romano et al. 2006."""
    d = abs(delta)
    if np.isnan(d):
        return ""
    if d < 0.147:
        return "vernachlaessigbar"
    if d < 0.330:
        return "klein"
    if d < 0.474:
        return "mittel"
    return "gross"


# ---------------------------------------------------------------------------
# 6. Vergleiche: kennen die Vergleichsart, nicht die Daten
# ---------------------------------------------------------------------------

_UMLAUTE = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
            "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}


def _slug(text: str) -> str:
    """Macht aus einem Namen einen dateinamentauglichen String."""
    for k, v in _UMLAUTE.items():
        text = text.replace(k, v)
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-")
    return text or "unbenannt"


def build_stem(
    art: str,
    gruppen: Iterable[str],
    score: str | None = None,
    datum: date | None = None,
) -> str:
    """Baut den gemeinsamen Dateinamen-Stamm fuer Plot und CSV eines Vergleichs.

    Schema:  <art>_<score>_<gruppe1>-vs-<gruppe2>_<YYYY-MM-DD>
    Beispiel: "verteilung_aa-entropy_train-vs-generiert_2026-11-11"

    Ab vier Gruppen werden nur die ersten drei genannt, danach "plusN", damit
    der Dateiname nicht ausufert.
    """
    namen = [_slug(g) for g in gruppen]
    if len(namen) > 3:
        namen = namen[:3] + [f"plus{len(namen) - 3}"]

    teile = [_slug(art)]
    if score:
        teile.append(_slug(score))
    teile.append("-vs-".join(namen))
    teile.append((datum or date.today()).isoformat())
    return "_".join(teile)


def _gruppenfarben(labels: Iterable[str]) -> dict[str, str]:
    """Ordnet jeder Gruppe eine feste Farbe aus cfg.plot_farben zu.

    Die Zuordnung haengt nur an der Reihenfolge im groups-Dict, nicht daran,
    welche Gruppen in einer einzelnen Abbildung auftauchen. "train" hat damit
    im Violinplot dieselbe Farbe wie im Balkenplot -- auch im Enrichment-Plot,
    wo die Referenzgruppe selbst gar nicht vorkommt. Farbe gehoert zur Gruppe,
    nicht zur Position in der Abbildung.
    """
    labels = list(labels)
    farben = cfg.plot_farben
    if len(labels) > len(farben):
        print(
            f"[Plots] {len(labels)} Gruppen, aber nur {len(farben)} Farben in "
            f"cfg.plot_farben -- Farben wiederholen sich. Besser die Palette "
            f"erweitern oder weniger Gruppen je Abbildung."
        )
    return {label: farben[i % len(farben)] for i, label in enumerate(labels)}


@dataclass
class ComparisonResult:
    """Ergebnis eines Vergleichs.

    Attributes:
        table: Kennzahlen/Tests als DataFrame (zum Speichern als CSV oder fuer die Thesis).
        figures: Name -> matplotlib-Figure.
        stem: Dateiname ohne Endung, aus build_stem(). Tabelle und Plots eines
            Vergleichs tragen denselben Stamm und gehoeren damit sichtbar
            zusammen. Wird beim Vergleich gebaut, wo Gruppennamen und Score
            noch bekannt sind -- nicht erst beim Speichern.
    """

    table: pd.DataFrame
    figures: dict = field(default_factory=dict)
    stem: str = "vergleich"

    def save(
        self,
        out_dir: Path,
        prefix: str | None = None,
        csv: bool = True,
        dpi: int | None = None,
        close: bool = False,
    ) -> list[Path]:
        """Speichert Tabelle als CSV und alle Figures als PNG nach out_dir.

        Args:
            out_dir: Zielordner, wird angelegt falls noetig.
            prefix: ueberschreibt self.stem, falls gesetzt.
            csv: Tabelle mitspeichern. False, wenn nur die Plots neu gebaut
                werden -- die teuren APEX-/ProGen2-/ESM-Scores stehen dann
                schon in der CSV vom Cluster-Lauf.
            dpi: Aufloesung der PNGs. None -> cfg.plot_dpi.
            close: Figures nach dem Speichern schliessen (im Notebook: False,
                im Skript: True, sonst laeuft der Speicher voll).

        Returns:
            Liste der geschriebenen Pfade.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix or self.stem
        dpi = cfg.plot_dpi if dpi is None else dpi
        geschrieben: list[Path] = []

        if csv:
            pfad = out_dir / f"{stem}.csv"
            self.table.to_csv(pfad, index=False)
            geschrieben.append(pfad)

        for fig_name, fig in self.figures.items():
            suffix = f"__{_slug(fig_name)}" if len(self.figures) > 1 or fig_name else ""
            pfad = out_dir / f"{stem}{suffix}.png"
            fig.savefig(pfad, dpi=dpi, bbox_inches="tight")
            geschrieben.append(pfad)
            if close:
                plt.close(fig)

        return geschrieben


class Comparison(ABC):
    """Basisklasse: vergleicht beliebig viele benannte SequenceSets."""

    @abstractmethod
    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        """
        Args:
            groups: Gruppenlabel -> SequenceSet, z.B.
                {"train": ..., "generiert": ..., "gefiltert": ...}.
        """

    @staticmethod
    def _pruefe_gruppen(groups: Mapping[str, SequenceSet]) -> None:
        """Faellt frueh und mit klarer Meldung, statt spaeter im Plot."""
        if len(groups) < 2:
            raise ValueError(
                f"Ein Vergleich braucht mindestens zwei Gruppen, bekam {len(groups)}."
            )
        for label, s in groups.items():
            if not isinstance(s, SequenceSet):
                raise TypeError(
                    f"Gruppe '{label}' ist kein SequenceSet, sondern {type(s).__name__}."
                )
            if len(s.data) == 0:
                raise ValueError(f"Gruppe '{label}' ist leer.")


class DistributionComparison(Comparison):
    """Vergleicht die Verteilung EINER numerischen Spalte zwischen den Gruppen.

    Verwendbar fuer ppl_progen2, pppl_esm2, mic_apex, aa_entropy.
    Plots: Violin und ECDF. Tabelle in Langform: Zeilen mit art == "kennzahl"
    beschreiben je eine Gruppe (n, Median, Mittelwert, Quartile), Zeilen mit
    art == "test" je ein Gruppenpaar (Mann-Whitney-U, p, Cliff's delta).
    """

    def __init__(self, column: str, log_scale: bool = False) -> None:
        self.column = column
        self.log_scale = log_scale

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        self._pruefe_gruppen(groups)

        werte: dict[str, np.ndarray] = {}
        for label, s in groups.items():
            if self.column not in s.data.columns:
                raise KeyError(
                    f"Spalte '{self.column}' fehlt in Gruppe '{label}'. "
                    f"Vorhanden: {list(s.data.columns)}"
                )
            v = _as_array(s.data[self.column])
            if len(v) == 0:
                raise ValueError(
                    f"Gruppe '{label}' hat keinen gueltigen Wert in '{self.column}'."
                )
            werte[label] = v

        zeilen = []
        for label, v in werte.items():
            zeilen.append({
                "art": "kennzahl", "gruppe": label, "referenz": "",
                "spalte": self.column, "n": len(v),
                "median": float(np.median(v)), "mittelwert": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                "U": np.nan, "p": np.nan, "cliffs_delta": np.nan, "effekt": "",
            })

        for a_label, b_label in itertools.combinations(werte, 2):
            U, p = mann_whitney(werte[a_label], werte[b_label])
            d = cliffs_delta(werte[a_label], werte[b_label])
            zeilen.append({
                "art": "test", "gruppe": a_label, "referenz": b_label,
                "spalte": self.column, "n": len(werte[a_label]) + len(werte[b_label]),
                "median": np.nan, "mittelwert": np.nan, "std": np.nan,
                "q25": np.nan, "q75": np.nan,
                "U": U, "p": p, "cliffs_delta": d, "effekt": cliffs_label(d),
            })

        return ComparisonResult(
            table=pd.DataFrame(zeilen),
            figures={"violin": self._violin(werte), "ecdf": self._ecdf(werte)},
            stem=build_stem("verteilung", groups.keys(), score=self.column),
        )

    def _plotwerte(self, v: np.ndarray) -> np.ndarray:
        """Bei log_scale auf log10 umrechnen; nicht-positive Werte fliegen raus."""
        if not self.log_scale:
            return v
        positiv = v[v > 0]
        if len(positiv) < len(v):
            print(f"[DistributionComparison] {len(v) - len(positiv)} Werte <= 0 "
                  f"fuer die log-Darstellung verworfen.")
        return np.log10(positiv)

    def _achsenlabel(self) -> str:
        return f"log10({self.column})" if self.log_scale else self.column

    def _violin(self, werte: Mapping[str, np.ndarray]):
        labels = list(werte)
        daten = [self._plotwerte(werte[l]) for l in labels]
        fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.5, 4.2))
        teile = ax.violinplot(daten, showmedians=True, showextrema=False)
        farben = _gruppenfarben(labels)
        for label, koerper in zip(labels, teile["bodies"]):
            koerper.set_facecolor(farben[label])
            koerper.set_alpha(0.65)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(self._achsenlabel())
        ax.set_title(f"Verteilung: {self.column}")
        ax.grid(axis="y", alpha=cfg.plot_grid_alpha)
        fig.tight_layout()
        return fig

    def _ecdf(self, werte: Mapping[str, np.ndarray]):
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        farben = _gruppenfarben(werte)
        for label, v in werte.items():
            x = np.sort(v)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where="post", color=farben[label],
                    linewidth=2.0, label=f"{label} (n={len(x)})")
        if self.log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(self.column)
        ax.set_ylabel("Anteil <= x")
        ax.set_title(f"ECDF: {self.column}")
        ax.grid(alpha=cfg.plot_grid_alpha)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


class CompositionComparison(Comparison):
    """Vergleicht die Aminosaeurehaeufigkeit (gesamtes Set) zwischen Gruppen.

    Plot: gruppierte Balken pro AS. Tabelle: Anteil pro AS und Gruppe,
    zusaetzlich log2-Enrichment gegen eine Referenzgruppe (z.B. gefiltert vs.
    ungefiltert -> K, L, R wie bei Torres).
    """

    def __init__(self, reference: str, alphabet: str = AMINO_ACIDS) -> None:
        """reference: Gruppenlabel, gegen das das Enrichment berechnet wird."""
        self.reference = reference
        self.alphabet = "".join(sorted(alphabet))

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        self._pruefe_gruppen(groups)
        if self.reference not in groups:
            raise KeyError(
                f"Referenzgruppe '{self.reference}' nicht in den Gruppen {list(groups)}."
            )

        anteile: dict[str, pd.Series] = {}
        n_residuen: dict[str, int] = {}
        for label, s in groups.items():
            zaehler = Counter("".join(s.data["sequence"]))
            gesamt = sum(zaehler.values())
            n_residuen[label] = gesamt
            anteile[label] = pd.Series(
                {AS: zaehler.get(AS, 0) / gesamt for AS in self.alphabet}
            )

        ref = anteile[self.reference]
        zeilen = []
        for label, anteil in anteile.items():
            for AS in self.alphabet:
                p, p_ref = anteil[AS], ref[AS]
                enrich = np.log2(p / p_ref) if (p > 0 and p_ref > 0) else np.nan
                zeilen.append({
                    "gruppe": label, "referenz": self.reference, "aminosaeure": AS,
                    "anteil": p, "anteil_referenz": p_ref,
                    "log2_enrichment": 0.0 if label == self.reference else enrich,
                    "n_residuen": n_residuen[label],
                    "n_sequenzen": len(groups[label].data),
                })

        return ComparisonResult(
            table=pd.DataFrame(zeilen),
            figures={"anteile": self._balken(anteile),
                     "enrichment": self._enrichment(anteile)},
            stem=build_stem("komposition", groups.keys(), score="aa-anteil"),
        )

    def _balken(self, anteile: Mapping[str, pd.Series]):
        labels = list(anteile)
        farben = _gruppenfarben(labels)
        x = np.arange(len(self.alphabet))
        breite = 0.8 / len(labels)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for i, label in enumerate(labels):
            ax.bar(x + i * breite - 0.4 + breite / 2,
                   [anteile[label][AS] for AS in self.alphabet],
                   width=breite, label=label, color=farben[label], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(list(self.alphabet))
        ax.set_xlabel("Aminosaeure")
        ax.set_ylabel("Anteil im Set")
        ax.set_title("Aminosaeure-Komposition")
        ax.grid(axis="y", alpha=cfg.plot_grid_alpha)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def _enrichment(self, anteile: Mapping[str, pd.Series]):
        # Farben ueber ALLE Gruppen bestimmen, nicht nur ueber die geplotteten:
        # sonst bekaeme die erste Nicht-Referenz hier die Farbe der Referenz.
        farben = _gruppenfarben(anteile)
        labels = [l for l in anteile if l != self.reference]
        ref = anteile[self.reference]
        x = np.arange(len(self.alphabet))
        breite = 0.8 / max(len(labels), 1)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for i, label in enumerate(labels):
            werte = [
                np.log2(anteile[label][AS] / ref[AS])
                if (anteile[label][AS] > 0 and ref[AS] > 0) else np.nan
                for AS in self.alphabet
            ]
            ax.bar(x + i * breite - 0.4 + breite / 2, werte,
                   width=breite, label=label, color=farben[label], alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(list(self.alphabet))
        ax.set_xlabel("Aminosaeure")
        ax.set_ylabel(f"log2(Anteil / Anteil in '{self.reference}')")
        ax.set_title(f"Anreicherung gegen '{self.reference}'")
        ax.grid(axis="y", alpha=cfg.plot_grid_alpha)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig


class PositionalComparison(Comparison):
    """Haeufigste Aminosaeuren pro Position ("Residuen per Token").

    Fuer jede Position 1..L und jede Gruppe: die top_k haeufigsten AS mit
    ihrem Anteil. Plot: Heatmap Position x AS oder Tabelle der Top-k.
    Positionen jenseits der Sequenzlaenge einer Sequenz werden ignoriert.
    """

    def __init__(self, top_k: int = 3, max_length: int = 50,
                 alphabet: str = AMINO_ACIDS, vmax: float | None = None) -> None:
        """
        Args:
            top_k: wie viele AS je Position in die Tabelle kommen.
            max_length: laengste beruecksichtigte Position.
            alphabet: Zeichenvorrat der Heatmap-Achse.
            vmax: Obergrenze der Farbskala. None = echter Maximalwert.
                Bei Sets mit dominantem Startresiduum (generierte Sequenzen
                beginnen fast immer mit M, ~0.86) frisst dieser eine Punkt
                die ganze Skala -- dann z.B. vmax=0.3 setzen. Die Colorbar
                beschriftet die Kappung dann.
        """
        if top_k < 1:
            raise ValueError(f"top_k muss >= 1 sein, war {top_k}.")
        self.top_k = top_k
        self.max_length = max_length
        self.alphabet = "".join(sorted(alphabet))
        self.vmax = vmax

    def _matrix(self, sequences: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(max_length, n_AS) Zaehlmatrix und (max_length,) Sequenzen je Position."""
        lut = np.full(128, -1, dtype=np.int16)
        for i, AS in enumerate(self.alphabet):
            lut[ord(AS)] = i

        counts = np.zeros((self.max_length, len(self.alphabet)), dtype=np.int64)
        n_pro_position = np.zeros(self.max_length, dtype=np.int64)

        for seq in sequences:
            codes = lut[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)][:self.max_length]
            if (codes < 0).any():
                raise ValueError(f"Zeichen ausserhalb des Alphabets in: {seq[:30]}...")
            L = len(codes)
            counts[np.arange(L), codes] += 1
            n_pro_position[:L] += 1

        return counts, n_pro_position

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        self._pruefe_gruppen(groups)

        zeilen = []
        figures = {}
        for label, s in groups.items():
            counts, n_pos = self._matrix(list(s.data["sequence"]))
            with np.errstate(invalid="ignore", divide="ignore"):
                anteile = counts / n_pos[:, None]
            anteile[n_pos == 0] = np.nan

            for pos in range(self.max_length):
                if n_pos[pos] == 0:
                    continue
                reihe = anteile[pos]
                beste = np.argsort(-reihe)[:self.top_k]
                for rang, idx in enumerate(beste, start=1):
                    zeilen.append({
                        "gruppe": label, "position": pos + 1, "rang": rang,
                        "aminosaeure": self.alphabet[idx],
                        "anteil": float(reihe[idx]),
                        "n_sequenzen_an_position": int(n_pos[pos]),
                    })

            figures[label] = self._heatmap(anteile, n_pos, label)

        return ComparisonResult(
            table=pd.DataFrame(zeilen), figures=figures,
            stem=build_stem("positionen", groups.keys(), score=f"top{self.top_k}"),
        )

    def _heatmap(self, anteile: np.ndarray, n_pos: np.ndarray, label: str):
        genutzt = int((n_pos > 0).sum())
        daten = anteile[:genutzt].T  # (n_AS, Positionen)
        fig, ax = plt.subplots(figsize=(max(6.0, genutzt * 0.2), 5.0))
        bild = ax.imshow(daten, aspect="auto", origin="lower", cmap=cfg.plot_cmap,
                         interpolation="nearest", vmin=0.0, vmax=self.vmax)
        ax.set_yticks(range(len(self.alphabet)))
        ax.set_yticklabels(list(self.alphabet), fontsize=7)
        schritt = max(1, genutzt // 25)
        ax.set_xticks(range(0, genutzt, schritt))
        ax.set_xticklabels(range(1, genutzt + 1, schritt), fontsize=7)
        ax.set_xlabel("Position")
        ax.set_ylabel("Aminosaeure")
        ax.set_title(f"Positionsprofil: {label}")
        beschriftung = "Anteil an dieser Position"
        if self.vmax is not None and np.nanmax(daten) > self.vmax:
            beschriftung += f" (Skala bei {self.vmax:g} gekappt)"
        fig.colorbar(bild, ax=ax, label=beschriftung)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# 7. Orchestrierung
# ---------------------------------------------------------------------------

def run_evaluation(
    train: SequenceSet,
    generated: SequenceSet,
    evaluator: Evaluator,
    mic_filter: MICFilter,
    out_dir: Path,
) -> None:
    """Gesamte Auswertung:

    1. Evaluator auf train und generated anwenden.
    2. mic_filter auf generated anwenden -> "gefiltert".
    3. Comparisons ausfuehren (Verteilungen der Scorer-Spalten, Komposition,
       Positionen) und mit ComparisonResult.save() nach out_dir schreiben.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: argparse (Pfade, Scorer-Auswahl, max_mic) und run_evaluation aufrufen.
    raise NotImplementedError