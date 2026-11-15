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

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import torch
import yaml

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

def _yaml_safe(obj):
    """Rekursiv yaml-taugliche Typen erzwingen: Path -> str, Tupel -> Liste.

    Liegt hier und nicht in helpers.py, weil Config selbst es braucht und
    helpers.py bereits aus config.py importiert -- andersherum waere es ein
    Zirkelimport. helpers.py reicht den Namen weiter.
    """
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


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

    amp_referenz_fasta: Path = TRACKED_DIR / "torres_train.fasta"
    similarity_max_ident: float = 0.60   # Torres' Kriterium (2)
    similarity_cov_mode: int = 2
    similarity_coverage: float = 0.8
    similarity_evalue: float = 1e-3

    diversity_max_ident: float = 0.40

    scorer_auswahl: tuple[str, ...] = (
    "aa_entropy", "ident_bekannt", "apex", "progen2", "esm2_pppl",
    )

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
        "#E71D36",   # 
        "#2EC4B6",   # 
        "#CBF3F0",   # 
        "#FFBF69",   # 
        "#FF9F1C",   # 
        "#CC79A7",   # 
    )
    # Sequenzielle Colormap fuer Heatmaps (Positionsprofile). Ein Farbton von
    # hell nach dunkel -- kein Regenbogen, sonst liest man Spruenge in die
    # Daten hinein, die nicht da sind. Alternativen: "cividis", "magma".
    plot_cmap: str = "viridis"
    plot_dpi: int = 200                # Aufloesung gespeicherter PNGs
    plot_grid_alpha: float = 0.3       # Deckkraft der Hilfslinien

    # ==== Serialisierung ====

    def to_yaml(self, path: str | Path) -> Path:
        """Config nach YAML schreiben -- die Provenienz eines Laufs.

        Jeder Lauf schreibt seine Config neben die Ergebnisse. Ohne sie sieht
        man einer Abbildung nach drei Monaten nicht mehr an, welches Modell,
        welche Schwelle, welcher Datensatz dahinterstand.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                _yaml_safe(asdict(self)),
                f,
                sort_keys=False,      # Feldreihenfolge der Dataclass behalten
                allow_unicode=True,   # Umlaute lesbar statt \uXXXX
            )
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Config aus YAML rekonstruieren.

        YAML kennt weder Tupel noch Path -- beide kommen als Liste bzw. str
        zurueck. Hier werden sie anhand der Feld-Annotationen zurueckgewandelt;
        ohne das waere z.B. apex_bundle_path ein str und .is_file() wuerde
        mit AttributeError sterben.
        """
        with open(Path(path), encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}

        bekannt = {fd.name: fd for fd in fields(cls)}
        unbekannt = sorted(set(d) - set(bekannt))
        if unbekannt:
            raise ValueError(
                f"Unbekannte Felder in {path}: {unbekannt}. Stammt die Datei "
                "aus einer aelteren Config-Version?"
            )

        for name, wert in list(d.items()):
            typ = str(bekannt[name].type)
            if "Path" in typ and isinstance(wert, str):
                d[name] = Path(wert)
            elif typ.startswith("tuple") and isinstance(wert, list):
                d[name] = tuple(wert)
        return cls(**d)