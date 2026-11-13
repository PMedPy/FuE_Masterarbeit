"""apex_backends.py -- Backends fuer den APEXScorer.

Trennt den Scorer-Vertrag (DataFrame mit festen Spalten) von der Frage,
WOHER die MIC-Vorhersagen kommen. Alles APEX-Spezifische sitzt hier und in
apex_net.py; evaluation.py importiert nur APEXBackend fuer den Type-Hint.

    APEXBackend         abstrakte Schnittstelle
    DummyAPEXBackend    deterministischer Platzhalter, ohne Gewichte lauffaehig
    ApexNativeBackend   echte Vorhersagen ueber den portierten Inferenzpfad

Quelle der Stammliste und der Modelle:
    https://gitlab.com/machine-biology-group-public/apex-pathogen (MIT)
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from aguirre.apex_net import (
    APEX_MAX_PEPTIDE_LEN,
    load_apex_ensemble,
    predict_mic,
)

# Reihenfolge exakt wie pathogen_list in APEX_predict.py -- nicht umsortieren,
# sie bestimmt die Spaltenzuordnung der Modellausgabe.
PATHOGEN_LIST: list[str] = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]


class APEXBackend(ABC):
    """Die EINZIGE Stelle im Projekt, die APEX kennt.

    Attributes:
        strains: Namen der Pathogenstaemme in der Spaltenreihenfolge von
            predict(). Legt die Spaltennamen des APEXScorer fest.
    """

    strains: list[str]

    @abstractmethod
    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """MIC-Vorhersagen in µmol/L.

        Returns:
            Array (len(sequences), len(self.strains)), Zeilenreihenfolge wie
            die Eingabe. Nicht bewertbare Sequenzen ergeben eine NaN-Zeile --
            niemals eine fehlende Zeile, sonst verschiebt sich die Zuordnung
            zum SequenceSet.
        """


class DummyAPEXBackend(APEXBackend):
    """Platzhalter ohne Gewichte -- deterministischer Unsinn.

    Zweck: Evaluator, MICFilter und Comparisons bauen und mit pytest testen,
    ohne das Gewichts-Bundle zu brauchen. Die Werte sind reproduzierbar und
    liegen im richtigen Bereich, haben aber KEINE biologische Bedeutung.

    Achtung: Ein Durchlauf hiermit prueft nur die Verrohrung (Shapes, Spalten,
    Reihenfolge), nie die Semantik.
    """

    strains = PATHOGEN_LIST

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        out = np.empty((len(sequences), len(self.strains)), dtype=float)
        for r, seq in enumerate(sequences):
            seed = int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            # MIC ist log-verteilt -> im log2-Raum ziehen, dann zurueck.
            # loc=5 entspricht einem Median um 2**5 = 32 µmol/L.
            out[r] = 2.0 ** rng.normal(loc=5.0, scale=1.5, size=len(self.strains))
        return out


class ApexNativeBackend(APEXBackend):
    """APEX in-process, auf den portierten Gewichten (siehe apex_net.py).

    Kein Subprocess, kein zweites Env, keine hartcodierte CSV. Sequenzen ueber
    der Laengengrenze bekommen eine NaN-Zeile, statt -- wie im Original --
    still aus der Ausgabe zu verschwinden.
    """

    strains = PATHOGEN_LIST

    def __init__(
        self,
        bundle_path: Path,
        device: str = "cpu",
        batch_size: int = 512,
        max_len: int = APEX_MAX_PEPTIDE_LEN,
    ) -> None:
        bundle_path = Path(bundle_path)
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"{bundle_path} fehlt. Einmalig erzeugen mit "
                "apex_net.export_state_dicts(<apex-repo>, <bundle_path>)."
            )
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.models = load_apex_ensemble(bundle_path, device=device)

        n_task = self.models[0].fc4.out_features
        if n_task != len(self.strains):
            raise ValueError(
                f"Modell liefert {n_task} Tasks, strains hat "
                f"{len(self.strains)} Eintraege -- Spaltenzuordnung waere falsch."
            )

    @classmethod
    def from_config(cls, cfg) -> "ApexNativeBackend":
        return cls(
            bundle_path=cfg.apex_bundle_path,
            device=cfg.apex_device,
            batch_size=cfg.apex_batch_size,
            max_len=cfg.apex_max_len,
        )

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        out = np.full((len(sequences), len(self.strains)), np.nan)
        if len(sequences) == 0:
            return out

        # Deduplizieren spart Rechenzeit, der Laengenfilter haelt die harte
        # APEX-Grenze ein. dict.fromkeys erhaelt die Reihenfolge.
        anfragen = [s for s in dict.fromkeys(sequences) if len(s) <= self.max_len]
        if not anfragen:
            return out

        mic = predict_mic(
            self.models, anfragen, device=self.device, batch_size=self.batch_size
        )

        tabelle = dict(zip(anfragen, mic))
        for r, seq in enumerate(sequences):
            zeile = tabelle.get(seq)
            if zeile is not None:
                out[r] = zeile
        return out
