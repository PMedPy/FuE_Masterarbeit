"""external.py -- Scorer, die ein fremdes CLI-Tool in einem eigenen Env aufrufen.

Macrel, AMPlify, AntiBP3 und HemoPI sind dasselbe Muster: FASTA schreiben,
CLI rufen, CSV parsen, Reihenfolge wiederherstellen. Das ist ein neuer
MECHANISMUS, kein neuer Vertrag -- score(sequences) -> DataFrame bleibt.

    SubprocessScorer    Template-Method: macht die Verrohrung einmal
    MacrelScorer        Vote + Haemolyse (zwei Familien, zwei Spalten)
    AMPlifyScorer       Vote
    AntiBP3Scorer       Vote + Gram-Aufloesung
    HemoPIScorer        Sicherheit
    DummyVoteScorer     deterministischer Platzhalter ohne installierte Tools

Conda-Hygiene: fue-amp bleibt sauber. Jedes Tool hat sein eigenes Env, weil
die Abhaengigkeiten konfligieren. Aufruf ueber `conda run -n <env> ...`.
Die Env-Namen stehen in Config, nicht hier.
"""

from __future__ import annotations

import hashlib
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd

from aguirre.config import Config, TMP_DIR
from aguirre.data import write_fasta
from aguirre.evaluation import Scorer

cfg = Config()

# Zeichen, an denen fast alle Tools scheitern (ANIA verwirft sie explizit).
NICHT_KANONISCH = set("BJOUXZ")


# ---------------------------------------------------------------------------
# 1. Basisklasse
# ---------------------------------------------------------------------------

class SubprocessScorer(Scorer, ABC):
    """Ruft ein Fremdtool als Subprocess und gibt dessen Ausgabe als Spalten.

    Unterklassen fuellen nur build_command() und parse_output() sowie die
    Klassenattribute. Alles andere -- Gate, Chunking, Tempdateien, NaN-Zeilen,
    Fehlerbehandlung -- macht score() einmal fuer alle.

    Attributes:
        env: Name der Conda-Umgebung des Tools (aus Config).
        min_len: kleinste bewertbare Laenge, None = keine Grenze.
            AMPScanner: 10, Macrel: None.
        max_len: groesste bewertbare Laenge. Macrel: 100.
        erlaubt_nicht_kanonisch: False -> Sequenzen mit BJOUXZ bekommen NaN.
        chunk_size: Sequenzen pro Tool-Aufruf. Zu gross sprengt den Speicher
            des Tools, zu klein kostet Prozess-Startzeit.
        timeout_s: Abbruch eines einzelnen Aufrufs.
    """

    env: ClassVar[str]
    min_len: ClassVar[int | None] = None
    max_len: ClassVar[int | None] = None
    erlaubt_nicht_kanonisch: ClassVar[bool] = False
    chunk_size: ClassVar[int] = 5000
    timeout_s: ClassVar[int] = 3600

    @property
    def fingerprint(self) -> str:
        """Cache-Identitaet: Name PLUS Toolversion.

        Ohne die Version liefert der Cache nach einem `pip install -U`
        stillschweigend alte Zahlen -- und das faellt erst beim Schreiben der
        Diskussion auf.
        """
        return f"{self.name}-v{self.tool_version()}"

    def tool_version(self) -> str:
        """Liest die Version des Tools, einmal pro Prozess gecacht.

        TODO(paul): einmal `<tool> --version` ueber _run() rufen, Ausgabe auf
        die Versionsnummer reduzieren. Bei Fehlschlag "unbekannt" liefern --
        aber dann WARNEN, weil der Cache damit unsicher wird.
        """
        raise NotImplementedError

    # -- Vertrag nach aussen -------------------------------------------------

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Nicht ueberschreiben. Ablauf in sechs Schritten:

        1. Ergebnis-DataFrame mit NaN in voller Eingabelaenge anlegen.
        2. Gate: _bewertbar() maskiert zu kurze, zu lange, nicht-kanonische.
        3. Deduplizieren (dict.fromkeys erhaelt die Reihenfolge) -- bei 50 000
           generierten Sequenzen sparen die Duplikate echte Rechenzeit.
        4. In chunk_size-Bloecken: FASTA schreiben, build_command(), _run(),
           parse_output().
        5. Ergebnisse per Sequenz-Dict auf die Eingabereihenfolge zurueckmappen.
        6. Pruefen, dass Zeilenzahl und Spalten exakt stimmen, dann zurueck.

        Returns:
            DataFrame, len(sequences) Zeilen, Spalten == self.columns.
            Nicht bewertbare Sequenzen: NaN-Zeile, NIEMALS fehlende Zeile.
        """
        raise NotImplementedError

    # -- Haken fuer Unterklassen --------------------------------------------

    @abstractmethod
    def build_command(self, fasta: Path, out: Path) -> list[str]:
        """Kommandozeile als Liste (kein shell=True, kein String-Join).

        Soll `conda run -n {self.env} ...` voranstellen -- dafuer _conda_run()
        benutzen, damit alle Tools gleich aufgerufen werden.
        """

    @abstractmethod
    def parse_output(self, out: Path) -> pd.DataFrame:
        """Tool-Ausgabe -> DataFrame mit Spalte "sequence" + self.columns.

        Muss ueber "sequence" joinbar sein, nicht ueber die Zeilenposition:
        manche Tools sortieren um oder lassen Zeilen weg.
        """

    # -- Helfer --------------------------------------------------------------

    def _bewertbar(self, sequences: Sequence[str]) -> np.ndarray:
        """Bool-Maske: welche Sequenzen darf das Tool ueberhaupt sehen.

        Das Gate gehoert in den Scorer, nicht in den Aufrufer -- sonst muss
        jede Auswertung die Laengengrenzen aller Tools kennen.
        """
        raise NotImplementedError

    def _conda_run(self, *args: str) -> list[str]:
        """["conda", "run", "-n", self.env, *args]."""
        raise NotImplementedError

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """Fuehrt cmd aus, mit timeout_s und eingefangenem stderr.

        Bei returncode != 0 eine Exception mit den letzten stderr-Zeilen
        werfen. Ein stilles Weiterlaufen mit halber Ausgabe ist der
        gefaehrlichste Fehlerfall: die Zahlen sehen dann plausibel aus.
        """
        raise NotImplementedError

    def _tmpdir(self) -> Path:
        """Eigenes Unterverzeichnis unter TMP_DIR, nach fingerprint benannt."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Vote-Familie: "Ist es ein AMP?"
# ---------------------------------------------------------------------------

class MacrelScorer(SubprocessScorer):
    """Macrel: Random Forest, konservativster der drei Vote-Tools.

    Liefert ZWEI Groessen aus zwei Familien -- den AMP-Klassifikator (Vote)
    und die Haemolyse-Propensitaet (Sicherheit). Zwei Spalten, getrennt
    aggregiert; p_hemo_macrel darf nie in den AMP-Vote.

    CLI: `macrel peptides --fasta <in> --output <dir>`, Ergebnis ist eine
    gezippte TSV im Ausgabeverzeichnis. Spalten dort: AMP_probability,
    hemolytic (ja/nein) -- Mapping in parse_output() festlegen.
    Referenz: https://github.com/BigDataBiology/macrel
    """

    name = "macrel"
    columns = ["p_macrel", "p_hemo_macrel"]
    env = "amp-macrel"  # TODO(paul): nach Config verschieben
    max_len = 100

    def build_command(self, fasta: Path, out: Path) -> list[str]:
        raise NotImplementedError

    def parse_output(self, out: Path) -> pd.DataFrame:
        raise NotImplementedError


class AMPlifyScorer(SubprocessScorer):
    """AMPlify: Bi-LSTM mit Attention, Sensitivitaet auf AMPScanner-Niveau.

    CLI: `AMPlify -s <fasta> -od <dir>`, Ausgabe eine TSV mit Score und
    Prediction. Der Score ist KEINE Wahrscheinlichkeit, sondern log-skaliert
    -- in parse_output() auf 0..1 bringen und die Transformation im Docstring
    festhalten, sonst ist die Schwelle 0,5 bedeutungslos.
    Referenz: https://github.com/bcgsc/AMPlify
    """

    name = "amplify"
    columns = ["p_amplify"]
    env = "amp-amplify"  # TODO(paul): nach Config verschieben

    def build_command(self, fasta: Path, out: Path) -> list[str]:
        raise NotImplementedError

    def parse_output(self, out: Path) -> pd.DataFrame:
        raise NotImplementedError


class AntiBP3Scorer(SubprocessScorer):
    """AntiBP3: hohe Sensitivitaet bei niedriger Falschpositivrate.

    Liefert zusaetzlich die Gram-Aufloesung (positiv/negativ/variabel) und
    koppelt die Klassifikation damit an dieselbe Spezies-Achse wie ANIA.

    CLI: `antibp3 -i <fasta> -o <csv> -s {1,2,3}` (1=gram+, 2=gram-,
    3=gram-variabel). Fuer p_antibp3 den gram-variablen Kopf nehmen, die
    beiden anderen fuer gram_antibp3 -- Entscheidung in parse_output()
    dokumentieren.
    Referenz: https://webs.iiitd.edu.in/raghava/antibp3/
    """

    name = "antibp3"
    columns = ["p_antibp3", "gram_antibp3"]
    env = "amp-antibp3"  # TODO(paul): nach Config verschieben

    def build_command(self, fasta: Path, out: Path) -> list[str]:
        raise NotImplementedError

    def parse_output(self, out: Path) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Sicherheits-Familie: "Ist es toxisch?"
# ---------------------------------------------------------------------------

class HemoPIScorer(SubprocessScorer):
    """HemoPI: Referenzstandard fuer Haemolyse, SVM auf HemoPI-1..3.

    Polaritaet ist umgekehrt: ein hoher Wert ist ein Ausschlussgrund, kein
    Beleg. Gehoert deshalb in einen eigenen Vote, nie in den AMP-Vote.
    Referenz: https://webs.iiitd.edu.in/raghava/hemopi/
    """

    name = "hemopi"
    columns = ["p_hemo_hemopi"]
    env = "amp-hemopi"  # TODO(paul): nach Config verschieben

    def build_command(self, fasta: Path, out: Path) -> list[str]:
        raise NotImplementedError

    def parse_output(self, out: Path) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Platzhalter
# ---------------------------------------------------------------------------

class DummyVoteScorer(Scorer):
    """Deterministischer Unsinn in Vote-Form -- wie DummyAPEXBackend.

    Zweck: VoteRule, AgreementMatrix und report.py bauen und testen, ohne vier
    Conda-Umgebungen zu installieren. Seed aus sha1(sequence), damit dieselbe
    Sequenz immer denselben Wert bekommt. Die Werte haben KEINE biologische
    Bedeutung -- ein Durchlauf hiermit prueft nur die Verrohrung.

    Args:
        spalten: welche Vote-Spalten simuliert werden sollen.
        p_positiv: Zielanteil Positiver, damit Konsensstufen ueberhaupt
            unterschiedlich ausfallen.
    """

    name = "dummy_vote"

    def __init__(
        self,
        spalten: Sequence[str] = ("p_macrel", "p_amplify", "p_antibp3"),
        p_positiv: float = 0.6,
    ) -> None:
        self.columns = list(spalten)
        self.p_positiv = p_positiv

    @property
    def fingerprint(self) -> str:
        return f"{self.name}-p{self.p_positiv:.2f}-{'_'.join(self.columns)}"

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        out = np.empty((len(sequences), len(self.columns)))
        for r, seq in enumerate(sequences):
            seed = int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            # Beta um p_positiv herum: korreliert zwischen den Spalten, weil
            # derselbe Seed -- das ahmt echte Tools besser nach als iid.
            basis = rng.beta(2.0, 2.0 * (1 - self.p_positiv) / self.p_positiv)
            out[r] = np.clip(basis + rng.normal(0, 0.12, len(self.columns)), 0, 1)
        return pd.DataFrame(out, columns=self.columns)
