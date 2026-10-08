"""Abschnitt 4 -- Noise-Schedule und Diffusionsprozess.

Diese Datei kennt die MATHEMATIK, aber kein Netz-Innenleben. Sie bekommt ein
Modell uebergeben und ruft es auf. Diese Grenze ist die wichtigste im ganzen
Projekt: dadurch wird dein Architektur-Experiment ein Einzeiler-Tausch.

Rahmen: DDPM (Ho et al. 2020, arXiv:2006.11239), nicht score-based.
Sampling ist der ancestral sampler, keine Langevin-Dynamik.
Objective im Torres-Checkpoint: 'pred_x0' (das Netz schaetzt x_0, nicht eps).

NoiseSchedule
    Berechnet betas, alphas, alpha_bar und alle abgeleiteten Wurzeln EINMAL
    im Konstruktor und legt sie als Buffer ab (register_buffer -> wandern mit
    .to(device) und landen im state_dict).

    extract(a, t, shape): holt pro Batch-Element den Skalar a[t] und formt ihn
    auf (B, 1, 1) um, damit er ueber L und D broadcastet.

DiffusionProcess
    q_sample(x_0, t, noise)  -> x_t          (B, L, D)
        Ueberblendung Signal <-> Rauschen, varianzerhaltend.
    p_losses(x_0, t, mask)   -> Skalar
        Loss nur auf nicht-gepaddeten Positionen!
    p_sample(x_t, t)         -> x_{t-1}      (B, L, D)
    sample(n, design_len)    -> x_0          (N, L, D)
"""

import math

import torch
import torch.nn as nn
import torch.nn.fuctional as F

""" Die Beta-Schedules

Erklaerung noch einfuegen

"""

def linear_beta_schedule(timesteps)                                   #Eigentlich Nutzlos, lediglich vllt nuetzlich mal einen run im vergleich zu "cosine" zu machen als starkes Argument warum "cosine" genutzt wurde
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008)
    steps = timesteps +1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) **2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


beta_schedules = {                                         #Ersetzt Zeilen "206 - 2011" in ampdiffusion.py
    "linear": linear_beta_schedule
    "cosine": cosine_beta_schedule
}



class NoiseSchedule(nn.Module):
    """betas, alphas, alpha_bar und abgeleitete Groessen als Buffer.
    
    Erklaerung hinzufuegen
    
    """
    def __init__(self, timesteps: 1000, beta_schedule: "cosine")
        super().__init__()

        if beta_schedule not in beta_schedules:
            raise ValueError(
                f"unbekannter beta_schedule {beta_schedule},"
                f"erlaubt: {sorted(beta_schedules)}"
            )

        betas = beta_schedules[beta_schedule](timesteps)                      #Durch Dictionary muss die Funktion vorher aus beta_schedule ruasgenommen und gestartet werden
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def reg(name: str, val: torch.Tensor)                                #lambda durch Funktion ersetzt -> Uebersichtlicher und besser Wiederverwendbar
            self.register_buffer(name, val.to(torch.float32))

        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)

        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod                                                                  #Da kein "self" Vorhanden waere das ganze ohne "staticmethod" irrefuehrend
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor)           #Tensoren angegeben -> Bessere Uebersichtlichkeit

        b = t.shape
        out = a.gather(-1, t.to(a.device))                                         #"t.to(a.device)" zieht spaeter einfacher alles auf GPU bzw dahin wo a liegt, damit es verrechnet werden kann -> Reine Sicherheit
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


    
      


class DiffusionProcess(nn.Module):
    """Forward-Rauschen, Loss und Reverse-Sampling. Wrappt den Denoiser.
    
    Erklaerung hinzufuegen
    
    """

    
