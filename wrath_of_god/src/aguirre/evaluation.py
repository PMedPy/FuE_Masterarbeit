"""evaluation.py -- Bewertung und Vergleich generierter Peptidsequenzen.

Architektur (nach Torres et al., Eval-Teil):

    SequenceSet   Datencontainer: Name + DataFrame (eine Zeile pro Sequenz)
    Scorer        berechnet Pro-Sequenz-Spalten (ProGen2, ESM-PPPL, APEX, Entropie)
    Evaluator     fuehrt mehrere Scorer aus, mit Cache auf Platte
    Filter        SequenceSet -> SequenceSet (MIC-, Laengenfilter)
    Comparison    vergleicht mehrere SequenceSets, liefert Tabelle + Figure

Datenfluss:

    SequenceSet --Evaluator(Scorer...)--> SequenceSet (mit Score-Spalten)
                --Filter--> gefiltertes SequenceSet
                --Comparison({"train":..., "generiert":..., "gefiltert":...})--> ComparisonResult

Vertrag zwischen den Teilen (NICHT ohne Absprache aendern):

    * Pflichtspalte jedes SequenceSet: "sequence" (str, Grossbuchstaben,
      20 Standard-AS aus cfg.AS_alphabet).
    * Automatische Spalte: "length" (int). Wird in __post_init__ aus
      "sequence" berechnet und bei jeder Ableitung neu gesetzt -- sie kann
      also nicht veralten. "sequence" ist die einzige Wahrheit.
    * "sequence" und "length" sind geschuetzt (SequenceSet.PROTECTED);
      with_columns() lehnt sie ab.
    * Scorer haengen nur Spalten an, sie veraendern oder loeschen keine.
    * Jeder Scorer hat einen fingerprint: die Cache-Identitaet. Dort muss
      ALLES hinein, was das Ergebnis veraendert (Modellname, Schwellen).
      Sonst liefert der Cache nach einem Modellwechsel stillschweigend
      alte Zahlen.
    * Spaltennamen der Scorer (siehe Scorer.columns):
        aa_entropy           Shannon-Entropie der AS-Verteilung (in bit)
        ppl_progen2          Perplexitaet unter ProGen2 (kleiner = natuerlicher)
        pppl_esm2            Pseudo-Perplexitaet unter ESM2 (Salazar et al.)
        mic_apex             Aggregat der APEX-MIC ueber alle Staemme (µmol/L,
                             ARITHMETISCHES Mittel -- so ist es mit Torres'
                             Schwelle von 64 vergleichbar; MICFilter nutzt diese)
        mic_apex_geom        dasselbe als geometrisches Mittel (µmol/L). MIC ist
                             log-verteilt, fuer Verteilungsvergleiche ist das
                             die richtige Groesse
        apex_n_inaktiv       Anzahl Staemme mit MIC > Schwelle (Diagnose)
        mic_apex__<stamm>    eine Spalte je Pathogenstamm (µmol/L, roh)

Schwere Abhaengigkeiten (torch, fair-esm, transformers) werden ERST beim
Instanziieren der jeweiligen Scorer importiert, nicht beim Import des
Moduls. Dadurch laesst sich evaluation.py auf einer Maschine ohne ML-Stack
oeffnen und fuer Filter, Statistik und Plots benutzen.
"""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aguirre.config import Config
from aguirre.data import read_fasta, write_fasta, run_mmseqs_search
from aguirre.helpers import shannons_H

if TYPE_CHECKING:  # nur fuer Typpruefer -- zur Laufzeit nicht importiert
    from aguirre.apex_backends import APEXBackend

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
            "length" wird automatisch ergaenzt.
    """

    name: str
    data: pd.DataFrame

    # Spalten, die der Container selbst verwaltet. Kein Scorer darf sie
    # anfassen. ClassVar, damit @dataclass daraus kein __init__-Feld macht.
    PROTECTED: ClassVar[tuple[str, ...]] = ("sequence", "length")

    def __post_init__(self) -> None:
        """Validiert "sequence" und berechnet "length" daraus."""
        if "sequence" not in self.data.columns:
            raise ValueError(
                f"Spalte 'sequence' fehlt, vorhanden: {list(self.data.columns)}"
            )
        seqs = self.data["sequence"]
        if seqs.isna().any() or (seqs == "").any():
            raise ValueError("Leere oder fehlende Sequenzen im Set.")
        invalid = set("".join(seqs)) - set(AMINO_ACIDS)
        if invalid:
            raise ValueError(f"Ungueltige Zeichen in Sequenzen: {sorted(invalid)}")

        # "sequence" ist die einzige Wahrheit: eine mitgelieferte length-Spalte
        # wird bewusst ueberschrieben statt uebernommen. Weil subset() und
        # with_columns() neue SequenceSets bauen, laeuft das hier jedes Mal
        # erneut -- "length" kann deshalb nicht veralten.
        self.data = self.data.assign(length=seqs.str.len().astype(int))

    @classmethod
    def from_fasta(cls, path: Path, name: str) -> "SequenceSet":
        ids, sequences = read_fasta(path)
        data = pd.DataFrame({"id": ids, "sequence": sequences})
        return cls(name=name, data=data)

    @classmethod
    def from_csv(
        cls, path: Path, name: str, sequence_col: str = "sequence"
    ) -> "SequenceSet":
        data = pd.read_csv(path)
        if sequence_col != "sequence":
            data = data.rename(columns={sequence_col: "sequence"})
        return cls(name=name, data=data)

    def with_columns(self, new_cols: pd.DataFrame) -> "SequenceSet":
        """Haengt Score-Spalten an. Gleiche Namen werden ueberschrieben."""
        if len(new_cols) != len(self.data):
            raise ValueError("new_cols hat eine andere Zeilenzahl als das Set.")

        verletzt = [c for c in new_cols.columns if c in self.PROTECTED]
        if verletzt:
            raise ValueError(
                f"Geschuetzte Spalten {verletzt} duerfen nicht ueberschrieben "
                f"werden (geschuetzt: {list(self.PROTECTED)})."
            )

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
        name: sprechender Kurzname (z.B. "progen2").
        columns: Namen der Spalten, die score() zurueckgibt. Darf auch
            Instanzattribut sein, wenn die Spalten erst zur Laufzeit
            feststehen (siehe APEXScorer).
    """

    name: str
    columns: list[str]

    @property
    def fingerprint(self) -> str:
        """Cache-Identitaet. ALLES, was das Ergebnis veraendert, gehoert hinein.

        Default ist der blosse Name -- korrekt nur fuer parameterfreie,
        deterministische Scorer. Jeder Scorer mit einem Modell oder einer
        Schwelle dahinter MUSS das ueberschreiben, sonst liefert der Cache
        nach einer Aenderung stillschweigend alte Zahlen.

        Der Rueckgabewert wird als Dateiname benutzt: keine Schraegstriche.
        """
        return self.name

    @abstractmethod
    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Berechnet die Scores.

        Args:
            sequences: Sequenzen in beliebiger Reihenfolge.

        Returns:
            DataFrame mit len(sequences) Zeilen, gleicher Reihenfolge wie die
            Eingabe und genau den Spalten aus self.columns. Nicht bewertbare
            Sequenzen ergeben eine NaN-Zeile -- niemals eine fehlende Zeile,
            sonst verschiebt sich die Zuordnung zum SequenceSet.
        """


class AminoAcidEntropyScorer(Scorer):
    """Shannon-Entropie der Aminosaeureverteilung pro Sequenz ("Residuenvielfalt").

    H = -sum_a p_a * log2(p_a), p_a = Anteil der AS a in der Sequenz.
    Niedrig = wenige AS dominieren, hoch = gleichmaessig gemischt (max. log2(20)).
    Braucht kein Modell, taugt daher als erster Test der Pipeline -- und faengt
    als Einziger den Mode Collapse ab, bei dem PPL faelschlich gut aussieht.
    """

    name = "aa_entropy"
    columns = ["aa_entropy"]

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame({self.columns[0]: shannons_H(sequences)})


class ProGen2Scorer(Scorer):
    """Perplexitaet unter ProGen2 (autoregressiv, trainiert auf natuerlichen Proteinen).

        PPL = exp(-1/N * sum_i log p(x_i | x_<i))

    Kleiner = natuerlicher. ProGen2 ist kausal, ein Forward-Pass liefert also
    alle N Konditionalen auf einmal.

    Besonderheiten des Hub-Ports (hugohrban/progen2-*):
      * Die Sequenzmarker sind die ZEICHEN "1" und "2", nicht <|bos|>/<|eos|>.
        Der Tokenizer fuegt nichts automatisch an -- "1" + seq + "2" ist
        unsere Aufgabe.
      * Der Tokenizer ist ein rohes tokenizers.Tokenizer, kein HF-AutoTokenizer.
      * Das Vokabular hat 30 Eintraege. Beim Scoren wird NICHT auf die 20
        kanonischen AS renormiert, sonst waere die Zahl nicht mehr mit der
        Literatur vergleichbar.

    Das abschliessende "2" geht bewusst NICHT in den Mittelwert ein: so ist N
    exakt die Sequenzlaenge.

    Batching: Sequenzen verschiedener Laenge werden RECHTS gepaddet. Eine
    Attention-Maske braucht es dafuer nicht -- kausal plus Right-Padding heisst,
    dass keine echte Position je ein Pad sieht. Die LOSS-Maske dagegen ist
    Pflicht, sonst haengt die PPL einer Sequenz davon ab, welche anderen
    zufaellig im selben Batch lagen.
    """

    name = "progen2"
    columns = ["ppl_progen2"]

    @property
    def fingerprint(self) -> str:
        # batch_size steht hier bewusst nicht drin: es aendert die Ausfuehrung,
        # nicht das Ergebnis.
        return f"progen2__{self.cfg.progen_model_name.replace('/', '-')}"

    def __init__(self, config: Config, batch_size: int | None = None) -> None:
        """
        Args:
            config: liefert progen_model_name, progen_batch_size, device.
            batch_size: ueberschreibt cfg.progen_batch_size, falls gesetzt.
        """
        # Lazy: transformers/tokenizers/torch erst hier, nicht beim Modulimport.
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM

        self.cfg = config
        self.device = config.device
        self.batch_size = int(batch_size or config.progen_batch_size)
        if self.batch_size < 1:
            raise ValueError(f"batch_size muss >= 1 sein, war {self.batch_size}.")

        # trust_remote_code: ProGen2 ist keine eingebaute transformers-
        # Architektur, die Modellklasse kommt als Datei aus dem Hub-Repo.
        # Braucht transformers < 5 (get_head_mask wurde in 5 entfernt).
        model = AutoModelForCausalLM.from_pretrained(
            config.progen_model_name, trust_remote_code=True
        )
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)

        self.tokenizer = Tokenizer.from_pretrained(config.progen_model_name)

        # IDs aus dem Tokenizer holen, nicht hartkodieren: bei einem Wechsel
        # auf eine andere ProGen2-Groesse koennte die Reihenfolge abweichen.
        self.bos_id = self.tokenizer.token_to_id("1")
        self.eos_id = self.tokenizer.token_to_id("2")
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        for bez, tid in (("BOS '1'", self.bos_id), ("EOS '2'", self.eos_id),
                         ("PAD", self.pad_id)):
            if tid is None:
                raise KeyError(f"Token fuer {bez} nicht im Tokenizer gefunden.")

        for AS in config.AS_alphabet:
            if self.tokenizer.token_to_id(AS) is None:
                raise KeyError(
                    f"'{AS}' aus cfg.AS_alphabet hat keine ID im ProGen2-Tokenizer."
                )

    # -- oeffentlich -------------------------------------------------------

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        return self._score_batch(sequences)

    def _ids(self, seq: str) -> list[int]:
        """Sequenz -> Token-IDs mit den Markern "1" und "2"."""
        return self.tokenizer.encode("1" + seq + "2").ids

    # -- gebatchte Fassung -------------------------------------------------

    def _score_batch(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Mehrere Sequenzen je Forward, rechts gepaddet, mit Loss-Maske."""
        import torch
        import torch.nn.functional as F
        from tqdm.auto import tqdm

        n_seq = len(sequences)
        if n_seq == 0:
            return pd.DataFrame({self.columns[0]: []})

        laengen = np.array([len(s) for s in sequences])
        if (laengen == 0).any():
            raise ValueError("Leere Sequenz -- Perplexitaet ist nicht definiert.")

        # Nach Laenge sortieren: so landen aehnlich lange Sequenzen im selben
        # Batch und das Padding bleibt klein. "stable" haelt die Reihenfolge
        # gleich langer Sequenzen fest -- reproduzierbar statt beliebig.
        ordnung = np.argsort(laengen, kind="stable")

        # NaN als Startwert, damit eine Luecke beim Zuruecksortieren auffaellt.
        ppl = np.full(n_seq, np.nan, dtype=float)

        with torch.no_grad():
            for start in tqdm(
                range(0, n_seq, self.batch_size), desc="ProGen2", leave=False
            ):
                idx = ordnung[start : start + self.batch_size]
                ids_liste = [self._ids(sequences[i]) for i in idx]
                L_max = max(len(x) for x in ids_liste)

                # Rechts mit pad_id auffuellen. Der Wert ist fuer das Ergebnis
                # egal (die Loss-Maske schneidet ihn weg), aber pad_id ist das
                # Token, das dafuer vorgesehen ist.
                ids = torch.full(
                    (len(idx), L_max), self.pad_id, dtype=torch.long, device=self.device
                )
                for r, x in enumerate(ids_liste):
                    ids[r, : len(x)] = torch.tensor(
                        x, dtype=torch.long, device=self.device
                    )

                logits = self.model(ids).logits          # (B, L_max, V)

                # Shift: Logits an Position j sagen Token j+1 vorher.
                pred = logits[:, :-1, :]                 # (B, L_max-1, V)
                ziel = ids[:, 1:]                        # (B, L_max-1)

                # reduction="none" ist der Kern: model(..., labels=...) oder
                # ein gemitteltes cross_entropy gaebe EINEN Skalar ueber den
                # ganzen Batch -- wir brauchen eine Zahl pro Sequenz.
                verlust = F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)),
                    ziel.reshape(-1),
                    reduction="none",
                ).view(ziel.shape)                       # (B, L_max-1)

                # Maske: ziel-Index j gehoert zu Token j+1, gueltig sind
                # j < n_residuen. Das schliesst in einem Ausdruck sowohl das
                # abschliessende "2" (j == n) als auch das Padding (j > n) aus.
                n_res = torch.as_tensor(
                    laengen[idx], dtype=torch.long, device=self.device
                )
                spalten = torch.arange(L_max - 1, device=self.device)
                maske = spalten.unsqueeze(0) < n_res.unsqueeze(1)      # (B, L_max-1)

                if not torch.equal(maske.sum(1), n_res):
                    raise AssertionError(
                        "Maske deckt nicht genau die Residuen ab -- "
                        f"{maske.sum(1).tolist()} statt {n_res.tolist()}."
                    )

                # Gewichteter Mittelwert je Zeile: Nenner ist die ECHTE Laenge,
                # nicht die Tensorbreite.
                mittel = (verlust * maske).sum(1) / maske.sum(1)

                # mps kennt kein float64 -- erst auf die CPU, dann exp.
                ppl[idx] = torch.exp(mittel.detach().cpu().double()).numpy()

        if np.isnan(ppl).any():
            raise AssertionError(
                f"{int(np.isnan(ppl).sum())} Sequenzen ohne Ergebnis -- "
                "beim Zuruecksortieren ist etwas verrutscht."
            )
        return pd.DataFrame({self.columns[0]: ppl})

    # -- Referenzfassung ---------------------------------------------------

    def _score_schleife(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Eine Sequenz nach der anderen, ohne Padding. Die Referenz.

        Bleibt im Code, damit _score_batch dagegen getestet werden kann:
            a = sc._score_schleife(seqs)["ppl_progen2"].to_numpy()
            b = sc._score_batch(seqs)["ppl_progen2"].to_numpy()
            assert np.allclose(a, b, rtol=1e-4)
        """
        import torch
        import torch.nn.functional as F
        from tqdm.auto import tqdm

        ppl_akk: list[float] = []
        with torch.no_grad():
            for seq in tqdm(sequences, desc="ProGen2 (Schleife)", leave=False):
                ids = torch.tensor(
                    self._ids(seq), dtype=torch.long, device=self.device
                ).unsqueeze(0)                       # (1, L+2)

                logits = self.model(ids).logits      # (1, L+2, V)

                # Zweimal abschneiden, aus zwei verschiedenen Gruenden:
                #   :-1  Shift -- Logits an Position i sagen Token i+1 vorher
                #   :-1  das EOS "2" soll nicht in den Mittelwert
                pred = logits[0, :-2, :]             # (L, V)
                ziel = ids[0, 1:-1]                  # (L,)
                if pred.shape[0] != len(seq) or ziel.shape[0] != len(seq):
                    raise AssertionError("Shift passt nicht zur Sequenzlaenge.")

                ce = F.cross_entropy(pred, ziel)     # rohe Logits, nats
                ppl_akk.append(float(torch.exp(ce.detach().cpu().double())))

        return pd.DataFrame({self.columns[0]: ppl_akk})


class ESMPseudoPerplexityScorer(Scorer):
    """Pseudo-Perplexitaet unter ESM-2 (maskiertes LM, Salazar et al. 2020).

        PPPL = exp(-1/N * sum_i log p(x_i | x_\\i))

    "Pseudo", weil das Produkt der Konditionalen keine gueltige gemeinsame
    Verteilung ergibt: jede bedingt auf ALLE anderen Positionen, die Terme
    ueberlappen sich. Bei ProGen2 teleskopiert die Kettenregel exakt, hier
    nicht. PPPL ist damit nicht mit ppl_progen2 vergleichbar -- zwei Spalten,
    zwei Skalen.

    Kosten: N Forward-Passes pro Sequenz (eine Maske je Position). Die N
    maskierten Varianten EINER Sequenz sind aber alle exakt gleich lang und
    lassen sich deshalb ohne Padding zu einem Batch (N, L+2) stapeln --
    "intra-Sequenz-Batching". Aus N Forwards wird einer.

    WICHTIG -- Zirkularitaet: Der Generator dekodiert ueber den lm_head von
    ESM-2. Bewertet man mit demselben Modell, misst man zum Teil den eigenen
    Decoder. Deshalb cfg.esm_scorer_model_name BEWUSST verschieden von
    cfg.esm_model_name halten (Generator 650M, Bewertung 8M). Der
    unabhaengige Richter ist ProGen2 -- andere Architektur, anderes Korpus.
    """

    name = "esm2_pppl"
    columns = ["pppl_esm2"]

    ERLAUBTE_MODELLE = ("esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D")

    @property
    def fingerprint(self) -> str:
        # max_masken steht hier BEWUSST nicht drin: es stueckelt nur den
        # Batch, jede Zeile bleibt unabhaengig, das Ergebnis aendert sich
        # nicht. In den Fingerabdruck gehoert nur, was die Zahl veraendert.
        return f"esm2_pppl__{self.cfg.esm_scorer_model_name}"

    def __init__(self, config: Config, max_masken: int = 256) -> None:
        """
        Args:
            config: liefert esm_scorer_model_name, esm_scorer_embed_dim, device.
            max_masken: so viele maskierte Kopien gehen hoechstens gemeinsam
                durchs Modell. Der Batch waechst quadratisch (N Kopien à N+2
                Token), bei 50 Residuen sind das 2.600 Token -- nichts. Bei
                einem 365er-Kontrollprotein waeren es 134.000, und mit dem
                650M-Modell wird das eng. Rein eine Speichergrenze.
        """
        # Lazy: fair-esm und torch erst hier, nicht beim Modulimport.
        import esm

        self.cfg = config
        self.device = config.device
        self.max_masken = int(max_masken)
        if self.max_masken < 1:
            raise ValueError(f"max_masken muss >= 1 sein, war {max_masken}.")

        mode = config.esm_scorer_model_name
        if mode not in self.ERLAUBTE_MODELLE:
            raise ValueError(
                f"Unbekanntes ESM-Modell '{mode}', erlaubt: {self.ERLAUBTE_MODELLE}"
            )
        if mode == config.esm_model_name:
            print(
                f"[ESMPseudoPerplexityScorer] Warnung: Bewertungsmodell '{mode}' ist "
                "identisch mit dem Generator-ESM (cfg.esm_model_name). Die "
                "Pseudo-Perplexitaet misst dann teilweise den eigenen Decoder."
            )

        model, alphabet = getattr(esm.pretrained, mode)()
        if model.embed_dim != config.esm_scorer_embed_dim:
            raise ValueError(
                f"ESM liefert embed_dim={model.embed_dim}, Config sagt "
                f"{config.esm_scorer_embed_dim}. Beides zusammen setzen."
            )

        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)

        self.batch_converter = alphabet.get_batch_converter()
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx

    # -- oeffentlich -------------------------------------------------------

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        return self._score_batch(sequences)

    # -- gemeinsame Vorarbeit ---------------------------------------------

    def _tokenisieren(self, seq: str):
        """(1, N+2)-Tokens auf dem Device, mit Shape-Pruefung."""
        if not seq:
            raise ValueError("Leere Sequenz -- PPPL ist nicht definiert.")
        _, _, tokens = self.batch_converter([("seq", seq)])
        # Tensor.to() ist NICHT in-place -- Rueckgabe zuweisen.
        tokens = tokens.to(self.device)
        if tuple(tokens.shape) != (1, len(seq) + 2):
            raise ValueError(
                f"Unerwartete Token-Shape {tuple(tokens.shape)}, "
                f"erwartet {(1, len(seq) + 2)}."
            )
        return tokens

    # -- gebatchte Fassung -------------------------------------------------

    def _score_batch(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Intra-Sequenz-Batching: die N Maskierungen einer Sequenz auf einmal.

        Alle N Kopien sind exakt gleich lang -- kein Padding, keine
        Attention-Maske, keine Loss-Maske. Die ganze Fehlerquelle des
        Batchings ueber verschiedene Sequenzen entfaellt hier.
        """
        import torch
        from tqdm.auto import tqdm

        pppl_akk: list[float] = []
        with torch.no_grad():
            for seq in tqdm(sequences, desc="ESM-PPPL", leave=False):
                tokens_orig = self._tokenisieren(seq)
                N = len(seq)

                # Zielwerte lesen, BEVOR maskiert wird. clone(), sonst zeigt
                # der Slice in denselben Speicher, den wir gleich ueberschreiben.
                echte_ids = tokens_orig[0, 1 : N + 1].clone()          # (N,)

                teile = []
                for start in range(0, N, self.max_masken):
                    # Positionen dieses Stuecks; 0-basiert in der Sequenz.
                    pos = torch.arange(
                        start, min(start + self.max_masken, N), device=self.device
                    )                                                   # (M,)
                    m = int(pos.numel())
                    zeilen = torch.arange(m, device=self.device)        # (M,)

                    # repeat, NICHT expand: expand teilt Speicher, das
                    # Maskieren wuerde alle Zeilen gleichzeitig treffen und
                    # nach einem Schritt saesse die Maske ueberall.
                    tokens = tokens_orig.repeat(m, 1)                   # (M, N+2)

                    # Residuum i liegt hinter dem BOS auf Token-Index i+1.
                    # Zwei Indexvektoren gleicher Laenge werden PAARWEISE
                    # abgebildet (advanced indexing), nicht als Kreuzprodukt:
                    # Zeile j bekommt genau Position pos[j] maskiert.
                    tokens[zeilen, pos + 1] = self.mask_idx

                    logits = self.model(tokens)["logits"]               # (M, N+2, 33)

                    # Dieselbe Diagonale wieder herauslesen.
                    diag = logits[zeilen, pos + 1]                      # (M, 33)
                    # Softmax ueber ALLE 33 Token, nicht auf die 20 kanonischen
                    # einschraenken -- sonst ist es eine andere Groesse.
                    lp = torch.log_softmax(diag, dim=-1)[zeilen, echte_ids[pos]]
                    if tuple(lp.shape) != (m,):
                        raise AssertionError(
                            f"Diagonale ergab {tuple(lp.shape)}, erwartet {(m,)} -- "
                            "vermutlich Kreuzprodukt statt paarweiser Indizierung."
                        )
                    # Auf CPU in float64 sammeln: mps kennt kein float64, und
                    # der Mittelwert soll nicht in float32 entstehen.
                    teile.append(lp.detach().cpu().double())

                log_ps = torch.cat(teile)                               # (N,)
                if log_ps.numel() != N:
                    raise AssertionError(
                        f"{log_ps.numel()} Positionen fuer Laenge {N}."
                    )
                # Erst mitteln, dann exponenzieren -- genau einmal.
                pppl_akk.append(float(torch.exp(-log_ps.mean())))

        return pd.DataFrame({self.columns[0]: pppl_akk})

    # -- Referenzfassung ---------------------------------------------------

    def _score_schleife(self, sequences: Sequence[str]) -> pd.DataFrame:
        """Eine Maske nach der anderen. Langsam, aber die Referenz.

        Bleibt im Code, damit _score_batch dagegen getestet werden kann:
            a = sc._score_schleife(seqs)["pppl_esm2"].to_numpy()
            b = sc._score_batch(seqs)["pppl_esm2"].to_numpy()
            assert np.allclose(a, b, rtol=1e-5)
        """
        import torch
        from tqdm.auto import tqdm

        pppl_akk: list[float] = []
        with torch.no_grad():
            for seq in tqdm(sequences, desc="ESM-PPPL (Schleife)", leave=False):
                tokens_orig = self._tokenisieren(seq)
                log_ps: list[float] = []
                for i in range(len(seq)):
                    tokens = tokens_orig.clone()
                    echte_id = tokens[0, i + 1].item()
                    tokens[0, i + 1] = self.mask_idx
                    logits = self.model(tokens)["logits"]
                    lp = torch.log_softmax(logits[0, i + 1], dim=-1)[echte_id]
                    log_ps.append(float(lp))
                if len(log_ps) != len(seq):
                    raise AssertionError("Positionszahl weicht von der Laenge ab.")
                pppl_akk.append(math.exp(-sum(log_ps) / len(log_ps)))

        return pd.DataFrame({self.columns[0]: pppl_akk})


class APEXScorer(Scorer):
    """Vorhergesagte MIC (APEX, Wan et al.) gegen mehrere Pathogenstaemme.

    Spalten:
        mic_apex           arithmetisches Mittel ueber alle Staemme, µmol/L
        mic_apex_geom      geometrisches Mittel ueber alle Staemme, µmol/L
        apex_n_inaktiv     Anzahl Staemme mit MIC > inaktiv_schwelle
        mic_apex__<stamm>  je Stamm, roh, µmol/L

    Kleiner = potenter. BEIDE Aggregate werden gefuehrt, weil sie zwei
    verschiedene Fragen beantworten:

      mic_apex (arithmetisch)  Torres schreibt "we used the mean prediction
          MIC to sort the peptides" und filtert bei <= 64 µmol/L. Woertlich
          gelesen ist das das arithmetische Mittel. MICFilter greift auf diese
          Spalte zu, damit unsere Selektion mit der Publikation vergleichbar
          bleibt.
      mic_apex_geom            MIC wird in Zweierverduennungen gemessen, lebt
          also auf einer Log-Skala. Fuer Verteilungsvergleiche, Mediane und
          Effektstaerken ist das geometrische Mittel die richtige Groesse --
          das arithmetische laesst sich vom schlechtesten Stamm dominieren.

    Die Wahl ist nicht kosmetisch: bei einer Schwelle von 64 µmol/L entscheidet
    sie, welche Peptide durchkommen. Deshalb stehen beide Spalten in den Daten
    und die Entscheidung im Methodenteil.

    Es wird NICHT geklemmt. APEX saettigt erst bei 1e6 µM (ReLU am Ausgang),
    und das geometrische Mittel vertraegt einzelne Saettiger. Ein frueher
    Deckel wuerde den Arbeitsbereich abschneiden statt der Saettigung: auch
    gute AMPs liegen im Mittel ueber 90 µM. inaktiv_schwelle wird nur
    gezaehlt.

    Die Spalten stehen erst zur Laufzeit fest (sie haengen an den Staemmen des
    Backends), deshalb ist columns hier Instanz- und nicht Klassenattribut.
    """

    name = "apex"

    def __init__(
        self,
        backend: "APEXBackend",
        batch_size: int = 256,
        inaktiv_schwelle: float = 64.0,
    ) -> None:
        """
        Args:
            backend: liefert die MIC-Matrix (siehe aguirre.apex_backends).
            batch_size: Sequenzen je Backend-Aufruf.
            inaktiv_schwelle: Grenze in µmol/L, ab der ein Stamm als "nicht
                getroffen" GEZAEHLT wird. Nie geklemmt. 64 ist Torres'
                Selektionsschwelle.
        """
        self.backend = backend
        self.batch_size = batch_size
        self.inaktiv_schwelle = float(inaktiv_schwelle)
        self.columns = (
            ["mic_apex", "mic_apex_geom", "apex_n_inaktiv"]
            + [f"mic_apex__{s}" for s in backend.strains]
        )

    @classmethod
    def from_config(cls, backend: "APEXBackend", cfg_in: Config) -> "APEXScorer":
        return cls(
            backend=backend,
            batch_size=cfg_in.apex_batch_size,
            inaktiv_schwelle=cfg_in.apex_inaktiv_schwelle,
        )

    @property
    def fingerprint(self) -> str:
        # Das Backend allein genuegt nicht: die Schwelle veraendert
        # apex_n_inaktiv, gehoert also in die Cache-Identitaet.
        return f"apex__{type(self.backend).__name__}__inakt{self.inaktiv_schwelle:g}"

    def score(self, sequences: Sequence[str]) -> pd.DataFrame:
        if len(sequences) == 0:
            return pd.DataFrame(columns=self.columns)

        bloecke = []
        for start in range(0, len(sequences), self.batch_size):
            batch = list(sequences[start : start + self.batch_size])
            mic = np.asarray(self.backend.predict(batch), dtype=float)
            erwartet = (len(batch), len(self.backend.strains))
            if mic.shape != erwartet:
                raise ValueError(f"Backend lieferte {mic.shape}, erwartet {erwartet}.")
            bloecke.append(mic)

        mic = np.vstack(bloecke)

        n_inaktiv = (mic > self.inaktiv_schwelle).sum(axis=1).astype(float)
        n_inaktiv[np.isnan(mic).any(axis=1)] = np.nan   # nicht bewertbar

        data = {
            # np.mean und geometrisches_mittel reichen NaN beide durch: eine
            # Sequenz, die das Backend nicht bewerten konnte, bekommt auch
            # kein Aggregat.
            "mic_apex": mic.mean(axis=1),
            "mic_apex_geom": geometrisches_mittel(mic),
            "apex_n_inaktiv": n_inaktiv,
        }
        for j, stamm in enumerate(self.backend.strains):
            data[f"mic_apex__{stamm}"] = mic[:, j]
        return pd.DataFrame(data, columns=self.columns)

class KnownAMPSimilarityScorer(Scorer):

    name = "ident_bekannt"
    columns = ["max_ident_bekannt"]

    def __init__(self):
        super().__init__()


    @property
    def fingerprint(self):
        return f"ident_bekannt__{pfad.stem}__{pfad.stat().st_size}"

    def score(self, sequences):
        write_fasta(list(sequences), path= TMP_DIR / "query.fasta")
        


# ---------------------------------------------------------------------------
# 3. Evaluator: Scorer ausfuehren, Ergebnisse cachen
# ---------------------------------------------------------------------------

class Evaluator:
    """Fuehrt eine Liste von Scorern ueber ein SequenceSet aus, mit Cache.

    Cache-Schluessel ist die SEQUENZ SELBST, nicht ihr Hash: Peptide sind
    <= 50 Zeichen, Kollisionen gibt es per Definition nicht, und eine
    Parquet-Datei, die man oeffnen und lesen kann, laesst sich debuggen.

    Eine Datei je Scorer-Fingerabdruck. NaN-Ergebnisse werden NICHT gecacht:
    sie koennen eine deterministische Ablehnung sein oder die Folge eines
    Absturzes -- und einen Absturz will man nicht einbetonieren.

    Nicht parallel ausfuehren: zwei Prozesse auf derselben Cache-Datei
    verlieren Schreibvorgaenge.
    """

    def __init__(
        self,
        scorers: Sequence[Scorer],
        cache_dir: Path | None = None,
        flush_every: int = 2000,
    ) -> None:
        """
        Args:
            scorers: anzuwendende Scorer.
            cache_dir: Ordner fuer den Cache; None = kein Caching.
            flush_every: nach so vielen neu berechneten Sequenzen wird der
                Cache geschrieben. Begrenzt den Verlust bei einem Abbruch.
        """
        self.scorers = list(scorers)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.flush_every = flush_every

        belegt: dict[str, str] = {}
        for sc in self.scorers:
            for spalte in sc.columns:
                if spalte in belegt:
                    raise ValueError(
                        f"Spalte '{spalte}' wird von '{belegt[spalte]}' und "
                        f"'{sc.fingerprint}' belegt."
                    )
                belegt[spalte] = sc.fingerprint

    def run(self, seqset: SequenceSet) -> SequenceSet:
        """Wendet alle Scorer an und gibt ein neues SequenceSet zurueck."""
        sequences = seqset.data["sequence"].tolist()
        bloecke = [self._run_scorer(sc, sequences) for sc in self.scorers]
        return seqset.with_columns(pd.concat(bloecke, axis=1))

    # -- pro Scorer --------------------------------------------------------

    def _run_scorer(self, scorer: Scorer, sequences: list[str]) -> pd.DataFrame:
        cache = self._load_cache(scorer)

        # dedupliziert und in stabiler Reihenfolge -- dieselbe Sequenz wird
        # einmal gerechnet, auch wenn sie mehrfach im Set steht.
        fehlend = [s for s in dict.fromkeys(sequences) if s not in cache.index]
        if fehlend:
            print(
                f"[Evaluator] {scorer.fingerprint}: {len(fehlend)} von "
                f"{len(set(sequences))} Sequenzen neu berechnen."
            )
            cache = self._score_missing(scorer, fehlend, cache)

        # reindex stellt die EINGABEreihenfolge her, dupliziert Mehrfach-
        # vorkommen und setzt NaN fuer alles, was nicht im Cache steht.
        # Positionelles Zusammensetzen waere hier der klassische stille
        # Fehler: die Zahlen bleiben plausibel, nur die Zuordnung rutscht.
        werte = cache.reindex(sequences)[list(scorer.columns)]
        return werte.reset_index(drop=True)

    def _score_missing(
        self, scorer: Scorer, fehlend: list[str], cache: pd.DataFrame
    ) -> pd.DataFrame:
        for start in range(0, len(fehlend), self.flush_every):
            block = fehlend[start : start + self.flush_every]
            neu = scorer.score(block)

            if len(neu) != len(block):
                raise ValueError(
                    f"{scorer.fingerprint}.score() gab {len(neu)} Zeilen fuer "
                    f"{len(block)} Sequenzen zurueck."
                )
            if list(neu.columns) != list(scorer.columns):
                raise ValueError(
                    f"{scorer.fingerprint}.score() lieferte {list(neu.columns)}, "
                    f"erwartet {list(scorer.columns)}."
                )

            neu = neu.set_axis(pd.Index(block, name="sequence"))
            # Reihenfolge wichtig: erst NaN-Zeilen weg, dann Zeitstempel --
            # sonst ist computed_at nie NaN und der Filter laeuft ins Leere.
            neu = neu[~neu.isna().any(axis=1)]
            neu = neu.assign(
                computed_at=pd.Timestamp.now().isoformat(timespec="seconds")
            )

            # Nicht concat mit dem leeren Rahmen: der wuerde seine
            # object-Dtypes mitschleppen und die Zahlenspalten verderben.
            cache = neu if cache.empty else pd.concat([cache, neu])
            self._save_cache(scorer, cache)
        return cache

    # -- Cache -------------------------------------------------------------

    def _cache_path(self, scorer: Scorer) -> Path:
        return self.cache_dir / f"{scorer.fingerprint}.parquet"

    def _load_cache(self, scorer: Scorer) -> pd.DataFrame:
        # Null-Objekt: vollstaendig geformt, nur zeilenlos. Dadurch
        # funktionieren "in cache.index", reindex und .empty unveraendert,
        # und cache_dir=None braucht nirgends ein Sonder-if.
        leerer_cache = pd.DataFrame(
            columns=list(scorer.columns) + ["computed_at"],
            index=pd.Index([], name="sequence", dtype="object"),
        )
        if self.cache_dir is None:
            return leerer_cache
        pfad = self._cache_path(scorer)
        if not pfad.is_file():
            return leerer_cache

        df = pd.read_parquet(pfad)
        fehlt = [c for c in scorer.columns if c not in df.columns]
        if fehlt:
            raise ValueError(
                f"Cache {pfad} passt nicht zu '{scorer.fingerprint}': "
                f"Spalten {fehlt} fehlen. Datei loeschen oder Fingerabdruck pruefen."
            )
        return df

    def _save_cache(self, scorer: Scorer, cache: pd.DataFrame) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ziel = self._cache_path(scorer)
        # Erst daneben schreiben, dann atomar umbenennen: ein Abbruch mitten
        # im Schreiben hinterlaesst sonst eine unlesbare Cache-Datei.
        tmp = ziel.with_suffix(".parquet.tmp")
        cache.to_parquet(tmp)
        tmp.replace(ziel)


# ---------------------------------------------------------------------------
# 4. Filter: SequenceSet -> SequenceSet
# ---------------------------------------------------------------------------

class Filter(ABC):
    """Basisklasse fuer Filter. Ein Filter waehlt Sequenzen aus, er berechnet nichts."""

    @abstractmethod
    def apply(self, seqset: SequenceSet) -> SequenceSet:
        """Gibt das gefilterte SequenceSet zurueck (neuer Name)."""


class LengthFilter(Filter):
    """Behaelt Sequenzen mit min_len <= length <= max_len (Spalte "length").

    Vor dem APEXScorer sinnvoll: APEX ueberspringt Sequenzen ueber 50
    Residuen, die bekaemen sonst eine NaN-Zeile.
    """

    def __init__(self, min_len: int, max_len: int) -> None:
        if min_len > max_len:
            raise ValueError(f"min_len ({min_len}) groesser als max_len ({max_len}).")
        self.min_len = int(min_len)
        self.max_len = int(max_len)

    @classmethod
    def from_config(cls, cfg_in: Config) -> "LengthFilter":
        return cls(min_len=cfg_in.pep_min_len, max_len=cfg_in.pep_max_len)

    def apply(self, seqset: SequenceSet) -> SequenceSet:
        mask = seqset.data["length"].between(self.min_len, self.max_len)
        gefiltert = seqset.subset(
            mask, name=f"{seqset.name}_len{self.min_len}-{self.max_len}"
        )
        print(
            f"[LengthFilter] {len(gefiltert.data)} von {len(seqset.data)} Sequenzen "
            f"behalten ({len(gefiltert.data) / max(len(seqset.data), 1):.1%})."
        )
        return gefiltert


class MICFilter(Filter):
    """Behaelt Sequenzen mit vorhergesagter MIC <= max_mic (Spalte "mic_apex").

    Default 64.0 ist Kriterium (1) aus Torres et al., Methods "Computational
    filtering": mean MIC <= 64 umol/L. Die Kriterien (2) Aehnlichkeit zu
    bekannten AMPs (<= 0.60) und (3) paarweise Diversitaet (<= 0.40) sind
    NICHT Teil dieses Filters -- sie brauchen ein Alignment gegen
    DRAMP/DBAASP/APD3 und gehoeren in einen eigenen SimilarityFilter.

    NaN bedeutet "nicht bewertbar" und wird verworfen: ungeprueft heisst
    nicht bestanden.
    """

    def __init__(self, max_mic: float = 64.0, column: str = "mic_apex") -> None:
        if not np.isfinite(max_mic) or max_mic <= 0:
            raise ValueError(f"max_mic muss eine positive Zahl sein, war {max_mic!r}.")
        self.max_mic = float(max_mic)
        self.column = column

    @classmethod
    def from_config(cls, cfg_in: Config, column: str = "mic_apex") -> "MICFilter":
        return cls(max_mic=cfg_in.apex_inaktiv_schwelle, column=column)

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


def geometrisches_mittel(werte: np.ndarray, axis: int = 1) -> np.ndarray:
    """Mittel im Log-Raum. Fuer log-verteilte Groessen wie MIC.

    Ein geometrisches Mittel IST ein arithmetisches Mittel auf der
    Log-Skala -- genau die Skala, auf der MIC gemessen wird (Zweier-
    verduennungen). Dadurch vertraegt es einzelne Saettigungswerte, die ein
    arithmetisches Mittel dominieren wuerden.

    NaN wird durchgereicht: eine Sequenz, die das Backend nicht bewerten
    konnte, bekommt auch kein Aggregat.
    """
    werte = np.asarray(werte, dtype=float)
    if (werte <= 0).any():          # NaN <= 0 ist False, stoert also nicht
        raise ValueError("Nur positive Werte (Log-Mittel).")
    return np.exp(np.log(werte).mean(axis=axis))


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

    ISO-Datum, damit der Dateimanager chronologisch sortiert. Ab vier Gruppen
    werden nur die ersten drei genannt, danach "plusN", damit der Dateiname
    nicht ausufert.
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

    Verwendbar fuer ppl_progen2, pppl_esm2, mic_apex, aa_entropy, length.
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

# Spalten, die standardmaessig als Verteilung verglichen werden. Die
# stammspezifischen mic_apex__* bleiben absichtlich draussen: elf weitere
# Vergleiche je Lauf sind keine Auswertung mehr, sondern ein Papierstau.
STANDARD_SCORE_SPALTEN: tuple[str, ...] = (
    "length",
    "aa_entropy",
    "ppl_progen2",
    "pppl_esm2",
    "mic_apex",
    "mic_apex_geom",
    "apex_n_inaktiv",
)

# Spalten, die log-skaliert dargestellt werden (MIC ist log-verteilt).
LOG_SPALTEN: tuple[str, ...] = ("mic_apex", "mic_apex_geom")


def run_evaluation(
    train: SequenceSet,
    generated: SequenceSet,
    evaluator: Evaluator,
    mic_filter: MICFilter,
    out_dir: Path,
    score_spalten: Sequence[str] | None = None,
    referenz: str = "train",
    speichern: bool = True,
) -> dict[str, ComparisonResult]:
    """Gesamte Auswertung: scoren, filtern, vergleichen, speichern.

    Ablauf:
        1. Evaluator auf train und generated anwenden.
        2. mic_filter auf generated anwenden -> "gefiltert".
        3. Je Score-Spalte eine DistributionComparison, dazu
           CompositionComparison und PositionalComparison.
        4. Alles mit ComparisonResult.save() nach out_dir schreiben.

    Args:
        train: Referenzset (natuerliche AMPs).
        generated: Ausgabe des Diffusionsmodells.
        evaluator: bereits mit Scorern bestueckt.
        mic_filter: Torres-Kriterium (1).
        out_dir: Zielordner fuer CSVs und PNGs.
        score_spalten: zu vergleichende Spalten. None -> STANDARD_SCORE_SPALTEN,
            auf die tatsaechlich vorhandenen eingeschraenkt.
        referenz: Gruppenlabel fuer das Enrichment in CompositionComparison.
        speichern: False, um die Ergebnisse nur zurueckzubekommen (Notebook).

    Returns:
        Name des Vergleichs -> ComparisonResult. Die Figures sind offen,
        im Skript also ggf. selbst schliessen.
    """
    out_dir = Path(out_dir)

    print(f"[run_evaluation] scoren: '{train.name}' ({len(train.data)} Sequenzen)")
    train_bewertet = evaluator.run(train)
    print(f"[run_evaluation] scoren: '{generated.name}' ({len(generated.data)} Sequenzen)")
    generated_bewertet = evaluator.run(generated)

    gefiltert = mic_filter.apply(generated_bewertet)

    groups: dict[str, SequenceSet] = {
        "train": train_bewertet,
        "generiert": generated_bewertet,
    }
    if len(gefiltert.data) == 0:
        print(
            "[run_evaluation] Warnung: nach dem MIC-Filter ist nichts uebrig -- "
            "die Gruppe 'gefiltert' wird uebersprungen."
        )
    else:
        groups["gefiltert"] = gefiltert

    if referenz not in groups:
        raise KeyError(f"Referenz '{referenz}' nicht in den Gruppen {list(groups)}.")

    # Nur Spalten vergleichen, die in ALLEN Gruppen stehen und numerisch sind.
    kandidaten = list(score_spalten) if score_spalten else list(STANDARD_SCORE_SPALTEN)
    spalten = [
        c for c in kandidaten
        if all(c in s.data.columns for s in groups.values())
        and all(_as_array(s.data[c]).size > 0 for s in groups.values())
    ]
    uebersprungen = [c for c in kandidaten if c not in spalten]
    if uebersprungen:
        print(f"[run_evaluation] uebersprungen (fehlt oder leer): {uebersprungen}")

    ergebnisse: dict[str, ComparisonResult] = {}

    for spalte in spalten:
        ergebnisse[f"verteilung_{spalte}"] = DistributionComparison(
            column=spalte, log_scale=spalte in LOG_SPALTEN
        ).compare(groups)

    ergebnisse["komposition"] = CompositionComparison(reference=referenz).compare(groups)
    ergebnisse["positionen"] = PositionalComparison(
        max_length=cfg.pep_max_len
    ).compare(groups)

    if speichern:
        geschrieben: list[Path] = []
        for res in ergebnisse.values():
            geschrieben += res.save(out_dir)
        print(f"[run_evaluation] {len(geschrieben)} Dateien nach {out_dir} geschrieben.")

    return ergebnisse


if __name__ == "__main__":
    # TODO: argparse (Pfade, Scorer-Auswahl, max_mic) und run_evaluation aufrufen.
    raise NotImplementedError