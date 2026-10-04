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
import torch


@dataclass
class Config:
    """Alle Hyperparameter eines Laufs an einer Stelle."""
    esm_model_name: str = "esm2_t6_8M_UR50D" # "esm2_t33_650M_UR50D" oder "esm2_t6_8M_UR50D"
    embed_dim: int = 320
    pep_max_len: int = 42
    device: str = (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
        )

