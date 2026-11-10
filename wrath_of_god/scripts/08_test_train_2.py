"""Weiterer Test mit 8M Modell. Untersuchung mit 100 Epochen und loss_weight scheme 'min.snr',
und pred_noise"""



import matplotlib
matplotlib.use("Agg")      # kein Display auf dem Cluster

import torch
from torch.utils.data import DataLoader, Subset

from aguirre.config import Config
from aguirre.data import Data_coordinator
from aguirre.denoiser import build_denoiser
from aguirre.esm_wrapper import ESMWrapper
from aguirre.trainer import RunManager, Trainer



# Config
cfg = Config()
cfg.objective = "pred_v"     # direkt nach Config(), vor allem anderen


# DataSet erstellen
train_ds = Data_coordinator(split= "train", cfg = cfg, standardize= True)
val_ds = Data_coordinator(split = "val", cfg = cfg, standardize= True)
train_d = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_d = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

# Model 
wrapper = ESMWrapper(cfg)
model = build_denoiser(cfg, wrapper.model)
del wrapper

torch.manual_seed(cfg.seed)

# RunManager und Trainer
run_manager = RunManager(cfg= cfg, run_name= "pred_noise_v_run")
trainer = Trainer(
    Model= model,
    Train_Data= train_d,
    Val_Data= val_d,
    run_manager= run_manager,
    cfg= cfg
)

# Schleife:
trainer.fit()
trainer.print_report()