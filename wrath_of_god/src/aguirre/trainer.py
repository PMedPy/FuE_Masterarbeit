"""Abschnitt 6 -- Training, Validierung, Checkpoints.

Komposition, keine Vererbung: der Trainer BEKOMMT Modell, Dataloader,
Optimizer, Config -- er erbt von niemandem.

Verantwortlichkeiten:
    fit()              Epochenschleife
    train_one_epoch()  -> mittlerer Loss
    validate()         -> Val-Loss (torch.no_grad)
    save_checkpoint()  Gewichte + EMA + Optimizer-State + Epoche
    load_checkpoint()  Lauf fortsetzen

EMA (ema_pytorch): ein zweiter, langsam nachziehender Gewichtssatz. Generiert
wird AUS DEM EMA-MODELL, nicht aus den rohen Gewichten -- siehe generate.py
bei Torres. Daran denken, sonst sind deine Samples schlechter als sie sein
muessten.

RunManager-Rolle: pro Lauf ein Ordner runs/<datum>_<name>/ mit
    config.yaml       die exakte Config dieses Laufs
    metrics.csv       Loss pro Epoche
    checkpoint_*.pt   Gewichte
    plots/
Dadurch ist "Modelle vergleichen" spaeter nur ein Skript, das ueber runs/*
iteriert. Ohne config.yaml weisst du nach drei Laeufen nicht mehr, welche
Learning Rate im besten drin war.
"""


class Trainer:
    """Trainingsschleife, Validierung, EMA, Checkpointing."""

    # TODO


class RunManager:
    """Legt den Lauf-Ordner an und schreibt Config, Metriken, Plots."""

    # TODO
