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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from aguirre.config import Config
from aguirre.data import read_fasta
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
    # Verhält sich wie eine __init__
    name: str
    data: pd.DataFrame

def __post_init__(self) -> None: #Gut, wenn es fehlt 
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
        raise NotImplementedError


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
    """Behaelt Sequenzen mit vorhergesagter MIC <= max_mic (Spalte "mic_apex")."""

    def __init__(self, max_mic: float, column: str = "mic_apex") -> None:
        raise NotImplementedError

    def apply(self, seqset: SequenceSet) -> SequenceSet:
        raise NotImplementedError

class LengthFilter(Filter):
    """Behaelt Sequenzen mit min_len <= length <= max_len (Spalte "length")."""

    def __init__(self, min_len: int, max_len: int) -> None:
        if min_len > max_len:
            raise ValueError(f"min_len ({min_len}) groesser als max_len ({max_len}).")
        self.min_len = min_len
        self.max_len = max_len

    @classmethod
    def from_config(cls, cfg: Config) -> "LengthFilter":
        return cls(min_len=cfg.pep_min_len, max_len=cfg.pep_max_len)

    def apply(self, seqset: SequenceSet) -> SequenceSet:
        mask = seqset.data["length"].between(self.min_len, self.max_len)
        return seqset.subset(mask, name=f"{seqset.name}_len{self.min_len}-{self.max_len}")


# ---------------------------------------------------------------------------
# 5. Statistik-Helfer
# ---------------------------------------------------------------------------

def mann_whitney(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Zweiseitiger Mann-Whitney-U-Test. Gibt (U, p) zurueck."""
    raise NotImplementedError


def cliffs_delta(a: pd.Series, b: pd.Series) -> float:
    """Effektstaerke Cliff's delta in [-1, 1].

    delta = P(a > b) - P(a < b).  Ergaenzt den p-Wert, der bei grossem n
    fast immer signifikant wird.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 6. Vergleiche: kennen die Vergleichsart, nicht die Daten
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Ergebnis eines Vergleichs.

    Attributes:
        table: Kennzahlen/Tests als DataFrame (zum Speichern als CSV oder fuer die Thesis).
        figures: Name -> matplotlib-Figure.
    """

    table: pd.DataFrame
    figures: dict = field(default_factory=dict)

    def save(self, out_dir: Path, prefix: str) -> None:
        """Speichert Tabelle als CSV und alle Figures als PNG nach out_dir."""
        raise NotImplementedError


class Comparison(ABC):
    """Basisklasse: vergleicht beliebig viele benannte SequenceSets."""

    @abstractmethod
    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        """
        Args:
            groups: Gruppenlabel -> SequenceSet, z.B.
                {"train": ..., "generiert": ..., "gefiltert": ...}.
        """


class DistributionComparison(Comparison):
    """Vergleicht die Verteilung EINER numerischen Spalte zwischen den Gruppen.

    Verwendbar fuer ppl_progen2, pppl_esm2, mic_apex, aa_entropy.
    Plot: Violin oder ECDF. Tabelle: Median, Mittelwert, n und je Gruppenpaar
    Mann-Whitney-p plus Cliff's delta.
    """

    def __init__(self, column: str, log_scale: bool = False) -> None:
        raise NotImplementedError

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        raise NotImplementedError


class CompositionComparison(Comparison):
    """Vergleicht die Aminosaeurehaeufigkeit (gesamtes Set) zwischen Gruppen.

    Plot: gruppierte Balken pro AS. Tabelle: Anteil pro AS und Gruppe,
    zusaetzlich log2-Enrichment gegen eine Referenzgruppe (z.B. gefiltert vs.
    ungefiltert -> K, L, R wie bei Torres).
    """

    def __init__(self, reference: str) -> None:
        """reference: Gruppenlabel, gegen das das Enrichment berechnet wird."""
        raise NotImplementedError

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        raise NotImplementedError


class PositionalComparison(Comparison):
    """Haeufigste Aminosaeuren pro Position ("Residuen per Token").

    Fuer jede Position 1..L und jede Gruppe: die top_k haeufigsten AS mit
    ihrem Anteil. Plot: Heatmap Position x AS oder Tabelle der Top-k.
    Positionen jenseits der Sequenzlaenge einer Sequenz werden ignoriert.
    """

    def __init__(self, top_k: int = 3, max_length: int = 50) -> None:
        raise NotImplementedError

    def compare(self, groups: Mapping[str, SequenceSet]) -> ComparisonResult:
        raise NotImplementedError


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