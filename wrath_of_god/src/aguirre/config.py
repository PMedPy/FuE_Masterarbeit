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
PROJECT_ROOT = Path(__file__).resolve().parents[3] # resolve macht aus einem relativen, einen Absoluten pfad. parents[3] geht vier Verzeichnisse zurück -> FuE_Masterarbeit

# Rohdaten aus Torres-Repo
DATA_RAW = PROJECT_ROOT / "Torres_Model" / "note" / "eval" / "train_eval.csv"

# Arbeitsverzeichnisse
DATA_DIR = PROJECT_ROOT / "wrath_of_god" / "data"
CACHE_DIR = DATA_DIR/ "cache"
RUNS_DIR = PROJECT_ROOT / "wrath_of_god" / "runs"


#==Configurations==
@dataclass
class Config:
    """Alle Hyperparameter eines Laufs an einer Stelle."""
    AS_alphabet: str = "LAGVSERTIDPKQNFYMHWC"
    esm_model_name: str = "esm2_t6_8M_UR50D" # "esm2_t33_650M_UR50D" oder "esm2_t6_8M_UR50D"
    embed_dim: int = 320 # 1280 oder 320
    pep_max_len: int = 42
    device: str = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
        )

