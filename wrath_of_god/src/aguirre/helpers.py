"""Häufigste Helperfunctions"""

#==Importe==
import torch
import pandas as pd
from pathlib import Path
from aguirre.config import Config, TRACKED_DIR
from dataclasses import asdict
cfg = Config()

# Baut Padding oder Validmask (tensor B,L) aus Längen (siehe Data_Coordinator)
def masker(lengths: torch.Tensor, valid_mask: bool = False, pep_max_len: int = cfg.pep_max_len):
    """(B,) Längen -> (B, L) bool, True = Padding, bei valid_mask = True ist False = Padding."""

    #Guardian
    too_long = lengths > pep_max_len # Bool (B,)
    if too_long.any(): # .any() schaut, ob ein Element der Bedingung widerspricht
        raise ValueError(
            f"{int(too_long.sum())} von {lengths.numel()} Sequenzen sind laenger \n"
            f"als pep_max_len = {pep_max_len}. Groesste gefundene Laenge: \n"
            f"{int(lengths.max())}."
        )

    #Maske
    padding_mask = torch.arange(pep_max_len, device= lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)
    if valid_mask:
        return ~padding_mask
    else: 
        return padding_mask

# yaml-Saver, um config yaml-tauglich zu machen.

def _yaml_safe(obj):
    """Rekursiv yaml-taugliche Typen erzwingen: Path -> str, Tupel -> Liste."""
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
    
# Akkumulatoren
class Akkumulator:
    """Mittelwert ueber viele Batches, ohne Zwischenmittelung.

    Akkumuliert Summe und Anzahl getrennt und teilt genau einmal -- dadurch
    ist das Ergebnis das echte Gesamtmittel, auch wenn der letzte Batch
    kleiner ist. Ein Mittel aus Batch-Mittelwerten waere es nicht.

    Shape-agnostisch: `value` und `n` duerfen Skalare sein (ein Loss) oder
    Tensoren gleicher Form (z.B. (L,) fuer eine Kurve pro Positionsindex).
    Skalar und Vektor brauchen dieselbe Klasse, weil Addition und Division
    elementweise arbeiten.


    Beispiel (t-Raster):
        acc = Akkumulator()
        for t_val in t_grid:
            acc.reset()
            for batch in loader:
                acc.update(loss, B)
            acc.compute()            # ein Wert pro t
        zeile = acc.output()         # (n_grid,)
    """

    def __init__(self):
        self.sum = 0.0        # Skalar oder Tensor, waechst per +=
        self.count = 0.0      # dito
        self.akku = []     # list[Tensor], je ein compute()-Ergebnis

    def reset(self):
        """Laufenden Mittelwert leeren. Akku bleibt."""
        self.sum = 0.0
        self.count = 0.0

    def hard_reset(self):
        """Alles leeren, auch der Akku."""
        self.reset()
        self.akku = []

    def update(self, value, n):
        """Einen Batch dazunehmen.

        Args:
            value: Mittelwert dieses Batches (Skalar oder Tensor).
                Kommt typischerweise aus reduce_loss().
            n: Anzahl der Elemente, ueber die `value` gemittelt wurde --
                bei einem Sequenz-Loss also B, nicht 1. Gleiche Form wie
                `value` oder ein Skalar.
        """
        if torch.is_tensor(value):
            value = value.detach()           # sonst haengt der Autograd-Graph
        if torch.is_tensor(n):
            n = n.detach()

        self.sum = self.sum + value * n
        self.count = self.count + n

    def compute(self):
        """Abschnitt abschliessen: Mittel bilden, in die Akku legen.

        Returns:
            Tensor (0-dim bei skalarem Input), auf CPU und losgeloest.
            Leere Zaehler ergeben 0.0, nicht NaN.
        """
        if isinstance(self.count, float) and self.count == 0.0:
            raise RuntimeError(
                "compute() ohne vorheriges update() -- es gibt nichts zu mitteln."
            )

        total = torch.as_tensor(self.sum, dtype=torch.float32).detach()
        n = torch.as_tensor(self.count, dtype=torch.float32).detach()
        mean = (total / n.clamp(min=1.0)).cpu()

        self.akku.append(mean)
        return mean

    def output(self):
        """Akku als ein Tensor.

        Returns:
            (n_compute,) bei skalaren Werten, (n_compute, ...) sonst.
        """
        if not self.akku:
            raise RuntimeError("Akku ist leer -- erst compute() aufrufen.")
        return torch.stack(self.akku).clone()

    #==Architekturrelevante Felder==
    _ARCH_KEYS = (
        "esm_model_name", "embed_dim", "pep_max_len", "n_esm_layers",
        "time_cond", "timesteps", "beta_schedule", "objective",
    )

    def _check_cfg_kompatibel(self, cfg_ckpt: dict) -> None:
        """Bricht ab, wenn der Checkpoint zu einer anderen Architektur gehoert."""
        jetzt = asdict(self.cfg)
        abweichend = {
            k: (cfg_ckpt.get(k), jetzt.get(k))
            for k in self._ARCH_KEYS
            if cfg_ckpt.get(k) != jetzt.get(k)
        }
        if abweichend:
            raise ValueError(
                "Checkpoint passt nicht zur aktuellen Config.\n"
                + "\n".join(
                    f"  {k}: Checkpoint={alt!r}, jetzt={neu!r}"
                    for k, (alt, neu) in abweichend.items()
                )
            )


# Pfad der Standards-CSV -> eine Stelle, an der der Name gebaut wird
def standards_path(cfg_in: Config | None = None):
    """Pfad der Standards-CSV.

    Konvention aus 04_standardize.py: Modell UND Laengenfilter stehen im
    Namen, weil 8M-Werte fuer 650M-Embeddings wertlos sind. Der Name wird
    hier gebaut und nirgends sonst -- load_standards und der Sampler rufen
    diese Funktion.

    Args:
        cfg_in: Config. None -> Modul-Default.
    """
    c = cfg if cfg_in is None else cfg_in
    return TRACKED_DIR / f"standards_{c.esm_model_name}_L{c.pep_max_len}.csv"     

# Laedt mean/std der Embedding-Dimensionen aus der Standards-CSV
def load_standards(cfg_in: Config | None = None):
    """mean/std pro Embedding-Dimension laden -> Einzige Quelle.

    Der Sampler liest hier. Data_coordinator liest die CSV derzeit noch
    selbst (data.py, STANDARDS_PATH) -> zwei Lesepfade auf dieselbe Datei,
    das gehoert zusammengefuehrt. Bis dahin: wer 04_standardize.py neu
    laufen laesst, aendert beide Pfade gleichzeitig.

    Args:
        cfg_in: Config. None -> Modul-Default

    Returns:
        (mean, std), beide (D,) float32.
    """
    c = cfg if cfg_in is None else cfg_in
    path = standards_path(c)

    #Guardian
    if not path.exists():
        raise FileNotFoundError(
            f"Standards nicht gefunden: {path}\n"
            "Erst scripts/04_standardize.py laufen lassen."
        )
    df = pd.read_csv(path, header=0)
    for col in ("mean", "std"):
        if col not in df.columns:
            raise ValueError(
                f"Spalte '{col}' fehlt in {path}. Gefunden: {list(df.columns)}"
            )

    mean = torch.tensor(df["mean"].to_numpy(), dtype=torch.float32)   # (D,)
    std = torch.tensor(df["std"].to_numpy(), dtype=torch.float32)     # (D,)

    if mean.shape[0] != c.embed_dim:
        raise ValueError(
            f"Standards haben D={mean.shape[0]}, Config sagt {c.embed_dim}. "
            "Falsches ESM-Modell oder falsche CSV."
        )
    if (std <= 0).any():
        raise ValueError("std enthaelt Werte <= 0 -- Division wuerde explodieren.")

    return mean, std
