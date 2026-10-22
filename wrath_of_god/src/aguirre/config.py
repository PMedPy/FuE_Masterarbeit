"""Abschnitt 0 -- Zentrale Konfiguration.

Eine einzige Dataclass, die ALLE Hyperparameter und Pfade haelt. Jede andere
Klasse bekommt sie im Konstruktor uebergeben, statt sich Werte selbst zu
suchen. Damit ist ein Trainingslauf vollstaendig durch ein Config-Objekt
beschrieben -- und reproduzierbar, sobald du es als YAML mitschreibst.

Zu befuellen (Vorschlag, Gruppen):
    Daten      : data_path, min_len, max_len, val_split, seed
    Modell     : esm_model_name, embed_dim, pep_max_len, n_esm_layers
    Diffusion  : timesteps, beta_schedule, objective, loss_type
    Training   : batch_size, lr, n_epochs, ema_decay, device
    Pfade      : run_dir, cache_dir

Methoden, die du brauchen wirst:
    to_yaml(path)    -- Config neben den Checkpoint schreiben
    from_yaml(path)  -- Lauf spaeter exakt rekonstruieren
"""

from dataclasses import dataclass
from pathlib import Path
import torch

#==Modulkonstanten==
PROJECT_ROOT = Path(__file__).resolve().parents[3] # resolve macht aus einem relativen (zu dieser Config!), einen Absoluten pfad. parents[3] geht vier Verzeichnisse zurück -> FuE_Masterarbeit

# Rohdaten aus Torres-Repo
DATA_RAW = PROJECT_ROOT / "Torres_Model" / "note" / "eval" / "train_eval.csv"

# Arbeitsverzeichnisse
DATA_DIR = PROJECT_ROOT / "wrath_of_god" / "data"
CACHE_DIR = DATA_DIR/ "cache"
RUNS_DIR = PROJECT_ROOT / "wrath_of_god" / "runs"
TMP_DIR = PROJECT_ROOT / "wrath_of_god" / "tmp"
SPLIT_DIR = PROJECT_ROOT / "wrath_of_god" / "splits"
TRACKED_DIR = PROJECT_ROOT / "wrath_of_god" / "data_tracked"

#==Configurations==
@dataclass
class Config:
    """Alle Hyperparameter eines Laufs an einer Stelle."""
    #==Grunlegendes==
    seed: int = 666
    AS_alphabet: str = "LAGVSERTIDPKQNFYMHWC"
    esm_model_name: str = "esm2_t33_650M_UR50D" # "esm2_t33_650M_UR50D" oder "esm2_t6_8M_UR50D"
    embed_dim: int = 1280 # 1280 oder 320
    pep_max_len: int = 50
    pep_min_len: int = 6 # Darunter biologische Unmöglichkeit. Filter
    lower_seq2_threshold: float = 0.3 # Spart Rechenzeit
    upper_seq2_threshold: float = 0.75 #
    device: str = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
        )
    #==Architektur==
    n_esm_layers: int | None = None
    
    #==Schedule==
    timesteps: int = 1000
    time_cond: str = "per_position"   # Kann "per_position" oder "global" sein
    beta_schedule: str = "cosine"     # "cosine" | "linear"
    objective: str = "pred_x0"        # "pred_x0" | "pred_noise" | "pred_v"
    loss_type: str = "l2"             # "l2" | "l1"
    loss_reduction: str = "sequence"  # "sequence", "none" oder "token"
    loss_weight_scheme: str = "none"         # "none" | "p2" | "min_snr"
    loss_weight_normalize: bool = True # Für Stabilisierung von Loss-Skala, führt jedoch zu Batchweiser Renomierung. 
                                       # Bei großen Bachtes kein Unterschied zur Globel Renomierung.
    loss_weight_gamma: float = 5.0     # P2/min_snr-Korrektur nach Choi et al. 2022, Torresdefault: 0 für P2 = 1 für min_snr = 5
    loss_weight_k: float = 1.0         # Numerische Stabilität bei P2-Korrektur
    #==Trainer==
    batch_size: int = 16
    learning_rate: float = 1e-4
    n_epochs: int = 100
    ema_decay: float = 0.995
    #train_num_steps: float = 1e5
    adam_betas: tuple[float] = (0.9, 0.99)
    max_grad_norm: float = 1.0
    ema_update_every: int = 1

