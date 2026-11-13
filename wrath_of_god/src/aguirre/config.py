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
    loss_weight_scheme: str = "min_snr"         # "none" | "p2" | "min_snr"
    loss_weight_normalize: bool = True # Für Stabilisierung von Loss-Skala, führt jedoch zu Batchweiser Renomierung. 
                                       # Bei großen Bachtes kein Unterschied zur Globel Renomierung.
    loss_weight_gamma: float = 5.0     # P2/min_snr-Korrektur nach Choi et al. 2022, Torresdefault: 0 für P2 = 1 für min_snr = 5
    loss_weight_k: float = 1.0         # Numerische Stabilität bei P2-Korrektur
    #==Trainer==
    batch_size: int = 16
    learning_rate: float = 1e-4
    n_epochs: int = 100                # Torres Default ist 102
    ema_decay: float = 0.995
    shuffle: bool = True
    eval_subset_fract: float = 0.2
    eval_subset_max: int = 350
    eval_subset_min: int = 100
    eval_t_grid: int = 20              # Stuetzstellen des festen t-Rasters in der Eval
    #train_num_steps: float = 1e5
    adam_betas: tuple[float, float] = (0.9, 0.99)
    max_grad_norm: float = 1.0
    ema_update_every: int = 1
    #==Evaluation==
    progen_model_name: str = "hugohrban/progen2-medium"  # "hugohrban/progen2-medium" oder "hugohrban/progen2-small"
    progen_batch_size: int = 32
    # BEWUSST ein anderes Modell als esm_model_name oben: der Generator
    # dekodiert ueber den lm_head des 650M-ESM. Mit demselben Modell zu
    # bewerten hiesse, den eigenen Decoder zu messen. Beide Felder immer
    # zusammen aendern -- der Scorer prueft embed_dim.
    esm_scorer_model_name: str = "esm2_t6_8M_UR50D" # "esm2_t6_8M_UR50D" oder "esm2_t33_650M_UR50D"
    esm_scorer_embed_dim: int = 320 # 320 (8M) oder 1280 (650M)

    #==APEX==
    apex_bundle_path: Path = DATA_DIR / "models" / "apex_state_dicts.pt"
    apex_device: str = "cpu"        # GRU auf mps erst mit Aequivalenztest freigeben
    apex_batch_size: int = 512
    apex_max_len: int = 50
    apex_inaktiv_schwelle: float = 64.0  

    #==Plots (Evaluation)==
    # Kategoriale Palette: eine Farbe je Gruppe (train, generiert, gefiltert...).
    # Zuweisung in fester Reihenfolge -- eine Gruppe behaelt ihre Farbe ueber
    # alle Abbildungen hinweg, auch wenn in einer Abbildung eine Gruppe fehlt
    # (z.B. die Referenz im Enrichment-Plot).
    #
    # Okabe-Ito, die uebliche barrierefreie Palette fuer Fachpublikationen.
    # Geprueft: alle Nachbarpaare bleiben bei Deuteranopie, Protanopie und
    # Tritanopie unterscheidbar, und die Helligkeiten liegen weit genug
    # auseinander fuer den Schwarzweissdruck. Beim Aendern beides mitpruefen.
    plot_farben: tuple[str, ...] = (
        "#0072B2",   # Blau
        "#E69F00",   # Orange
        "#009E73",   # Gruen
        "#D55E00",   # Zinnoberrot
        "#56B4E9",   # Hellblau
        "#CC79A7",   # Violett
    )
    # Sequenzielle Colormap fuer Heatmaps (Positionsprofile). Ein Farbton von
    # hell nach dunkel -- kein Regenbogen, sonst liest man Spruenge in die
    # Daten hinein, die nicht da sind. Alternativen: "cividis", "magma".
    plot_cmap: str = "viridis"
    plot_dpi: int = 200                # Aufloesung gespeicherter PNGs
    plot_grid_alpha: float = 0.3       # Deckkraft der Hilfslinien