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

def linear_beta_schedule(timesteps):                                   #Eigentlich Nutzlos, lediglich vllt nuetzlich mal einen run im vergleich zu "cosine" zu machen als starkes Argument warum "cosine" genutzt wurde
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps +1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) **2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


beta_schedules = {                                         #Ersetzt Zeilen "206 - 2011" in ampdiffusion.py
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule
}



class NoiseSchedule(nn.Module):
    """betas, alphas, alpha_bar und abgeleitete Groessen als Buffer.
    
    Erklaerung hinzufuegen
    
    """
    def __init__(self, timesteps: int=1000, beta_schedule: str="cosine"):
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

        def reg(name: str, val: torch.Tensor):                                #lambda durch Funktion ersetzt -> Uebersichtlicher und besser Wiederverwendbar
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
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor):           #Tensoren angegeben -> Bessere Uebersichtlichkeit

        b = t.shape
        out = a.gather(-1, t.to(a.device))                                         #"t.to(a.device)" zieht spaeter einfacher alles auf GPU bzw dahin wo a liegt, damit es verrechnet werden kann -> Reine Sicherheit
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


    
      
class DiffusionProcess(nn.Module):
    """Forward-Rauschen, Loss und Reverse-Sampling. Wrappt den Denoiser.
    
    Erklaerung hinzufuegen
    
    """
    def __init__(
        self,
        model,
        *,
        seq_length: int = 40,
        embed_dim: int = 320,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        objective: str = "pred_x0",
        loss_type: str = "l2",
        auto_normalize: bool = False,
        norm_scale: float = 4.0,
        norm_shift: float = 0.5,
    ):

        super().__init__()

        if objective not in {"pred_x0", "pred_noise", "pred_v"}:
            raise ValueError(f"unbekanntes objective {objective}")                 #Aufruf der Fehler, wenn etwas nicht stimmt schon hier statt erst spaeter
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"unbekannter loss_type {loss_type}")

        self.model = model
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.objective = objective
        self.loss_type = loss_type

        self.schedule = NoiseSchedule(timesteps=timesteps, beta_schedule=beta_schedule)
        self.num_timesteps = self.schedule.timesteps

        """
        Erklaerung einfuegen

        """
        self.auto_normalize = auto_normalize
        self.norm_scale = float(norm_scale)
        self.norm_shift = float(norm_shift)

    def normalize (self, x: torch.Tensor):
        if not self.auto_normalize:
            return x
        return x / self.norm_scale + self.norm_shift

    def denormalize(self, z: torch.Tensor):
        if not self.auto_normalize:
            return z
        return (z - self.normshift) * self.norm_scale


    def q_sample(
            self,
             x_start: torch.Tensor,
             t: torch.Tensor,
             noise,
             padding_mask
    ):
        
        if noise is None:
            noise = torch.rand_like(x_start)            #Ersetzt Lambda und exist Term aus ampdiffusion (Z. 45 - 52)

        extract = self.schedule.extract
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)                 #Reine shape Zusatzzeile -> (B, 1, 1)
        sqrt_1map = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)     #Reine shape Zusatzzeile -> (B, 1, 1)
        x_t = sqrt_ab * x_start + sqrt_1map * noise                                            #(B, L, D)
        
        if padding_mask is not None:
            x_t = x_t.masked_fill(padding_mask.unsqueeze(-1), 0.0)        #(B, L, D) -> Padding Maske schon hier einbauen?
        
        return x_t

    def add_noise(                                 #Ersetzt def forward ampdiffusion (Z. 500 ff)
            self,
            x_start: torch.Tensor,
            t,
            noise,
            padding_mask,
            normalize: bool = True
    ):

        b = x_start.shape[0]
        device = x_start.device                  # Wieder Absicherung fuer richtiges Device

        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        x0 = self.normalize(x_start) if normalize else x_start
        x_t = self.q_sample(x0, t, noise=noise, padding_mask=padding_mask)
        return x_t, noise, t

    def predict_v(
            self,
            x_start: torch.Tensor,
            t: torch.tensor,
            noise: torch.tensor
    ):

        extract = self.schedule.extract
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)                  #Reine shape Zusatzzeile -> (B, 1, 1)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)      #Reine shape Zusatzzeile -> (B, 1, 1)
        return sqrt_ab * noise - sqrt_1mab * x_start                                            # (B, L, D)

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == "l1" else F.mse_loss

    def check_shape(self, x: torch.Tensor):         #Verbesserung der Fehlermeldung aus def p_losses (Z. 464), welche zwar bei shape Fehlern abbricht aber nichts genaues aussagt
       
        if x.ndim != 3 or x.shape[1] != self.seq_length or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"Embedding hat Shape {tuple(x.shape)}, erwartet "
                f"(B, {self.seq_length}, {self.embed_dim}). "
                "Haeufigste Ursache: CLS/EOS wurden nicht abgeschnitten. "
                "42 statt 40 Token verwendet?!"
            )

    def p_losses(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            padding_mask,
            noise
    ):

        if self.model is None:
            raise RuntimeError("DiffusionProcess model nicht uebergeben")     #Absicherung, dass alles auch mit mehreren py Branches korrekt aufgerufen wurde.
        self.check_shape(x_start)

        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        model_out = self.model(x_t, t, padding_mask=padding_mask)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            target = self.predict_v(x_start, t, noise)

        loss = self.loss_fn(model_out, target, reduction="none")

        if padding_mask is not None:
            keep = (~padding_mask).unsqueeze(-1).to(loss.dtype)
            denom = (keep.sum(dim=(1,2)) * loss.shape[-1].clamp(min=1.0))
            loss = (loss * keep).sum(dim=(1, 2)) / denom
        else:
            loss = loss.flatten(start_dim=1).mean(dim=1)

        return loss.mean()

    def forward(
        self,
        x_start: torch.Tensor,
        padding_mask,
        noise 
    ):

        self.check_shape(x_start)
        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        x_start = self.normalize(x_start)

        return self.p_losses(x_start, t, padding_mask=padding_mask, noise=noise)