"""06 -- Trainingslauf.

Sechster Schritt. Laedt die vorberechneten Embeddings, baut Denoiser und
Trainer und schreibt alles in einen Lauf-Ordner.

IN  : data/cache/<split>-Set_<modell>_L<max_len>.pt
      data_tracked/standards_<modell>_L<max_len>.csv   (bei standardize)
OUT : runs/train_<name>_<datum>/
          config.yaml
          metrics/metrics.csv
          metrics/loss_per_t.csv
          checkpoints/checkpoint_last.pt
          checkpoints/checkpoint_best.pt
          final.pt
          plots/lernkurven.png, plots/t_heatmap.png
          report.pdf

Aufruf:
    python scripts/06_train.py --name cluster01 --esm 8M --epochs 50
    python scripts/06_train.py --name smoke --esm 8M --epochs 2 --debug 200
"""

import argparse
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")      # kein Display auf dem Cluster

import torch
from torch.utils.data import DataLoader, Subset

from aguirre.config import Config
from aguirre.data import Data_coordinator
from aguirre.denoiser import build_denoiser
from aguirre.esm_wrapper import ESMWrapper
from aguirre.trainer import RunManager, Trainer

ESM_MODELLE = {
    "8M": ("esm2_t6_8M_UR50D", 320),
    "650M": ("esm2_t33_650M_UR50D", 1280),
}


def parse_args():
    p = argparse.ArgumentParser(description="Training des Diffusionsmodells.")
    p.add_argument("--name", required=True, help="Kurzname fuer den Lauf-Ordner")
    p.add_argument("--esm", default="8M", choices=sorted(ESM_MODELLE))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    p.add_argument("--loss-reduction", default=None,
                   choices=["none", "token", "sequence"])
    p.add_argument("--loss-weight", default=None,
                   choices=["none", "p2", "min_snr"])
    p.add_argument("--debug", type=int, default=0, metavar="N",
                   help="nur N Train- und N/4 Val-Sequenzen")
    p.add_argument("--no-standardize", action="store_true")
    return p.parse_args()


def build_cfg(args) -> Config:
    esm_name, embed_dim = ESM_MODELLE[args.esm]
    aenderungen = {"esm_model_name": esm_name, "embed_dim": embed_dim}

    if args.epochs is not None:
        aenderungen["n_epochs"] = args.epochs
    if args.batch_size is not None:
        aenderungen["batch_size"] = args.batch_size
    if args.lr is not None:
        aenderungen["learning_rate"] = args.lr
    if args.device is not None:
        aenderungen["device"] = args.device
    if args.loss_reduction is not None:
        aenderungen["loss_reduction"] = args.loss_reduction
    if args.loss_weight is not None:
        aenderungen["loss_weight_scheme"] = args.loss_weight

    return replace(Config(), **aenderungen)


def build_data(cfg: Config, args):
    standardize = not args.no_standardize
    train_ds = Data_coordinator(split="train", cfg=cfg, standardize=standardize)
    val_ds = Data_coordinator(split="val", cfg=cfg, standardize=standardize)

    if args.debug:
        g = torch.Generator().manual_seed(cfg.seed)

        def small(ds, n):
            idx = torch.randperm(len(ds), generator=g)[:n].tolist()
            return Subset(ds, idx)

        train_ds = small(train_ds, args.debug)
        val_ds = small(val_ds, max(args.debug // 4, cfg.batch_size))

    train_d = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_d = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_ds, val_ds, train_d, val_d


def main():
    args = parse_args()
    cfg = build_cfg(args)
    torch.manual_seed(cfg.seed)

    train_ds, val_ds, train_d, val_d = build_data(cfg, args)

    wrapper = ESMWrapper(cfg)
    model = build_denoiser(cfg, wrapper.model)
    del wrapper                      # ESM wird nach dem Kopieren der Layer nicht mehr gebraucht

    run_manager = RunManager(cfg=cfg, run_name=args.name)
    trainer = Trainer(
        Model=model,
        Train_Data=train_d,
        Val_Data=val_d,
        run_manager=run_manager,
        cfg=cfg,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Lauf       : {run_manager.run_dir}")
    print(f"Modell     : {cfg.esm_model_name}  D={cfg.embed_dim}  L={cfg.pep_max_len}")
    print(f"Parameter  : {params / 1e6:.1f} Mio trainierbar")
    print(f"Device     : {cfg.device}")
    print(f"Daten      : {len(train_ds)} train / {len(val_ds)} val, batch {cfg.batch_size}")
    print(f"Loss       : {cfg.loss_type} / {cfg.loss_reduction} / w(t)={cfg.loss_weight_scheme}")
    print(f"Epochen    : {cfg.n_epochs}", flush=True)

    trainer.fit()
    trainer.print_report()

    print(f"\nFertig. Artefakte in {run_manager.run_dir}", flush=True)


if __name__ == "__main__":
    main()
