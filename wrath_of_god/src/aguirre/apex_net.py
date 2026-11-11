"""apex_net.py -- portierter APEX-Inferenzpfad.

Quelle: APEX (Wan, Torres, de la Fuente-Nunez et al.)
        https://gitlab.com/machine-biology-group-public/apex-pathogen
        Commit: <HASH EINTRAGEN>
Lizenz: MIT, Copyright (c) 2025 F W
        Volltext: external/apex-pathogen/LICENSE

Portiert aus APEX_models.py und utils.py, damit APEX unter torch >= 2.6 in
DIESER Umgebung laeuft statt in einem eigenen Env mit torch 1.11.

UNVERAENDERT -- jede Aenderung hier bricht die numerische Aequivalenz zur
publizierten Version und damit die Vergleichbarkeit mit Torres et al.:
    * die forward-Mathematik
    * Vokabular, Start/End-Marker und Padding-Schema aus onehot_encoding
    * die Ruecktransformation MIC = 10**(6 - y)
    * das ARITHMETISCHE Mittel ueber die 8 Ensemble-Modelle im MIC-Raum

BEWUSST GEAENDERT:
    * unbekannte Zeichen werfen einen Fehler statt still zu Padding zu werden
    * Hyperparameter werden aus dem state_dict abgeleitet, nicht aus Dateinamen
    * Typannotationen; der ungenutzte Variable-Import ist entfallen
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vokabular und Kodierung  (aus utils.make_vocab / utils.onehot_encoding)
# ---------------------------------------------------------------------------

APEX_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"   # Reihenfolge wie im Original!
APEX_VOCAB: dict[str, int] = {"0": 0, "1": 1, "2": 2}   # pad, start, end
APEX_VOCAB.update({aa: 3 + i for i, aa in enumerate(APEX_AMINO_ACIDS)})

APEX_MAX_LEN = 52                       # start + 50 Residuen + end
APEX_MAX_PEPTIDE_LEN = APEX_MAX_LEN - 2
MIC_INACTIVE = 1.0e6                    # Saettigungswert: ReLU klemmt auf 0


def encode_sequences(
    sequences: Sequence[str], max_len: int = APEX_MAX_LEN
) -> np.ndarray:
    """Peptide -> Indexmatrix (len(sequences), max_len), rechts mit 0 gepaddet.

    Achtung: Das APEX-Vokabular hat eine EIGENE Reihenfolge und ist nicht
    identisch mit cfg.AS_alphabet. Nicht vermischen.
    """
    X = np.zeros((len(sequences), max_len), dtype=np.int64)
    erlaubt = set(APEX_AMINO_ACIDS)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        unbekannt = set(seq) - erlaubt
        if unbekannt:
            raise ValueError(
                f"Sequenz {i}: Zeichen {sorted(unbekannt)} nicht im APEX-Vokabular. "
                "(Das Original haette sie still zu Padding gemacht.)"
            )
        a_seq = "1" + seq[: max_len - 2] + "2"
        for j, zeichen in enumerate(a_seq[:max_len]):
            X[i, j] = APEX_VOCAB[zeichen]
    return X


# ---------------------------------------------------------------------------
# Architektur  (aus APEX_models.py -- Attributnamen muessen exakt bleiben,
# sonst passt das state_dict nicht)
# ---------------------------------------------------------------------------

class PeptideEmbeddings(nn.Module):
    """Eingefrorene AAindex-Embeddings. Gewichte kommen aus dem state_dict."""

    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.aa_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.aa_embedding.weight.requires_grad_(False)   # Original: freeze=True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aa_embedding(x)


class ApexNet(nn.Module):
    """APEX-Basislerner: Embedding -> biGRU -> Attention -> FC-Stapel.

    Klassenname frei waehlbar (steht nicht im state_dict), Attributnamen nicht.
    """

    def __init__(
        self,
        emb_size: int,
        num_rnn_layers: int,
        dim_h: int,
        dim_latent: int,
        num_task: int,
        vocab_size: int = len(APEX_VOCAB),
        max_len: int = APEX_MAX_LEN,
    ) -> None:
        super().__init__()

        self.peptideEmb = PeptideEmbeddings(vocab_size, emb_size)
        self.dim_emb = emb_size
        self.dim_h = dim_h
        self.dim_latent = dim_latent

        self.rnn = nn.GRU(
            emb_size, dim_h, num_layers=num_rnn_layers,
            batch_first=True, dropout=0.1, bidirectional=True,
        )
        self.layernorm = nn.LayerNorm(dim_h * 2)
        self.attn1 = nn.Linear(dim_h * 2 + emb_size, max_len)
        self.attn2 = nn.Linear(dim_h * 2, 1)

        self.fc0 = nn.Linear(dim_h * 2, dim_h)

        # Regressionskopf (MIC je Pathogen) -- der, den APEX_predict.py nutzt
        self.fc1 = nn.Linear(dim_h, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim_latent // 2)
        self.fc3 = nn.Linear(dim_latent // 2, dim_latent // 4)
        self.fc4 = nn.Linear(dim_latent // 4, num_task)
        self.ln1 = nn.LayerNorm(dim_latent)
        self.ln2 = nn.LayerNorm(dim_latent // 2)
        self.ln3 = nn.LayerNorm(dim_latent // 4)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)

        # Klassifikationskopf -- ungenutzt, aber im state_dict enthalten.
        # Weglassen wuerde load_state_dict(strict=True) brechen.
        self.fc1_ = nn.Linear(dim_h, dim_latent)
        self.fc2_ = nn.Linear(dim_latent, dim_latent // 2)
        self.fc3_ = nn.Linear(dim_latent // 2, dim_latent // 4)
        self.fc4_ = nn.Linear(dim_latent // 4, 1)
        self.ln1_ = nn.LayerNorm(dim_latent)
        self.ln2_ = nn.LayerNorm(dim_latent // 2)
        self.ln3_ = nn.LayerNorm(dim_latent // 4)
        self.dp1_ = nn.Dropout(0.1)
        self.dp2_ = nn.Dropout(0.1)
        self.dp3_ = nn.Dropout(0.1)

    # --- ab hier: unveraendert aus dem Original ---------------------------
    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = self.peptideEmb(x)
        out, _h = self.rnn(x)
        out = self.layernorm(out)
        attn_weights1 = F.softmax(self.attn1(torch.cat((out, x), 2)), dim=2)
        out = torch.bmm(attn_weights1, out)
        attn_weights2 = F.softmax(self.attn2(out), dim=1)
        out = torch.sum(attn_weights2 * out, dim=1)
        return self.fc0(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._trunk(x)
        out = self.dp1(F.relu(self.ln1(self.fc1(out))))
        out = self.dp2(F.relu(self.ln2(self.fc2(out))))
        out = self.dp3(F.relu(self.ln3(self.fc3(out))))
        out = self.fc4(out)
        return F.relu(out)          # -> y >= 0  -> MIC <= 1e6 µM

    def clf_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._trunk(x)
        out = self.dp1_(F.relu(self.ln1_(self.fc1_(out))))
        out = self.dp2_(F.relu(self.ln2_(self.fc2_(out))))
        out = self.dp3_(F.relu(self.ln3_(self.fc3_(out))))
        return self.fc4_(out)
    # ----------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor]) -> "ApexNet":
        """Baut das Netz aus den Shapes des state_dict -- keine Dateinamen noetig."""
        vocab_size, emb_size = sd["peptideEmb.aa_embedding.weight"].shape
        modell = cls(
            emb_size=emb_size,
            num_rnn_layers=sum(
                1 for k in sd if re.fullmatch(r"rnn\.weight_ih_l\d+", k)
            ),
            dim_h=sd["rnn.weight_hh_l0"].shape[1],
            dim_latent=sd["fc1.weight"].shape[0],
            num_task=sd["fc4.weight"].shape[0],
            vocab_size=vocab_size,
        )
        modell.load_state_dict(sd)      # strict=True: faengt jede Abweichung
        modell.eval()
        return modell


# ---------------------------------------------------------------------------
# Gewichte: einmalig exportieren, danach portabel laden
# ---------------------------------------------------------------------------

def export_state_dicts(repo_path: Path, out_path: Path) -> Path:
    """EINMALIG ausfuehren: gepickelte Originalmodelle -> portables Bundle.

    Danach wird das APEX-Repo zum Rechnen nicht mehr gebraucht, nur noch
    fuer den Aequivalenztest.

    Achtung: weights_only=False entpickelt und fuehrt dabei Code aus. Das ist
    hier vertretbar (MIT, benannte Quelle, Code gelesen) -- aber bewusst.
    """
    import sys

    repo_path = Path(repo_path).resolve()

    pfade = sorted((repo_path / "APEX_pathogen_models").glob("APEX_*"))
    if not pfade:
        raise FileNotFoundError(f"Keine Modelle unter {repo_path}")

    # Das APEX-Repo hat ein utils.py im Wurzelverzeichnis. Dauerhaft in
    # sys.path waere es ein Schattenimport-Risiko -- daher nur so lange
    # drin, wie das Entpickeln es braucht.
    sys.path.insert(0, str(repo_path))          # AMP_model muss importierbar sein
    try:
        bundle = []
        for p in pfade:
            original = torch.load(p, map_location="cpu", weights_only=False)
            bundle.append(
                {
                    "name": p.name,
                    "state_dict": {
                        k: v.cpu() for k, v in original.state_dict().items()
                    },
                }
            )
    finally:
        if sys.path and sys.path[0] == str(repo_path):
            sys.path.pop(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    return out_path


def load_apex_ensemble(bundle_path: Path, device: str = "cpu") -> list[ApexNet]:
    """Laedt das Bundle -- ohne Entpickeln, also mit weights_only=True."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    modelle = []
    for eintrag in bundle:
        m = ApexNet.from_state_dict(eintrag["state_dict"]).to(device)
        m.eval()
        m.requires_grad_(False)
        modelle.append(m)
    return modelle


# ---------------------------------------------------------------------------
# Inferenz  (Nachbearbeitung unveraendert aus APEX_predict.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_mic(
    models: Sequence[ApexNet],
    sequences: Sequence[str],
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """MIC-Vorhersagen in µmol/L, Shape (len(sequences), num_task)."""
    if not models:
        raise ValueError("Leeres Ensemble -- es wurden keine Modelle geladen.")
    if len(sequences) == 0:
        return np.zeros((0, models[0].fc4.out_features), dtype=float)

    X = encode_sequences(sequences)
    summe: np.ndarray | None = None

    for modell in models:
        teile = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            y = modell(xb).detach().cpu().numpy()
            teile.append(10.0 ** (6.0 - y))        # UNVERAENDERT
        mic = np.vstack(teile)
        summe = mic if summe is None else summe + mic

    return summe / float(len(models))              # arithmetisch, UNVERAENDERT