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
  

Diese Datei enthaelt lediglich den Forward-Pfad und einige buffer (posterior) des Reverse-Pfades. Sampling und der Reverse-Pfad
ist in "sampler.py" und "denoiser.py"

Rausgelassen wurden:
'log_one_minus_alphas_cumprod'                                      -> Altlast aus anderem Paper (wird nicht verwendet)
'self_condition' und 'x_self_cond'                                  -> Werden in generate.py auf 'False' gesetzt 
'sampling_timesteps', 'is_ddim_sampling' und 'ddim_sampling_eta'    -> Wird in generate.py auf 'False' gesetzt weshalb es immer timesteps = Config nutzt
'p2_loss_weight'                                                    -> gamma = 0, dadurch Zahl**0 = 1 fuer alle Werte und Regionen 
'normalize', 'denormalize', 'auto-normalize'                        -> Ist auf "False" gestellt und damit  hinfaellig

Stattdessen wurde:
'loss_elements'        -> elementweiser Fehler (B, L, D), unreduziert. Naht zum Trainer fuer Diagnosen
'reduce_loss'          -> Reduktion von der Mathematik getrennt; Modus ueber cfg.loss_reduction
'loss_reduction'       -> "none" (Torres) | "token" | "sequence"; ersetzt das Bool mask_loss
'snr'                  -> Buffer alpha_bar / (1 - alpha_bar), Signal-Rausch-Verhaeltnis pro Schritt
'loss_weight(t)'       -> w(t) pro Batch-Element; verallgemeinert das entfernte p2_loss_weight
'loss_weight_scheme'   -> "none" (= Torres) | "p2" (Choi et al. 2022) | "min_snr" (Hang et al. 2023)
'weight'-Argument      -> p_losses/reduce_loss nehmen w(t) an; None reproduziert Torres exakt

"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from aguirre.config import Config

""" Die Beta-Schedules
 beta_t ist die Varianz des Rauschens, das in EINEM Schritt dazukommt:
     q(x_t | x_{t-1}) = N(x_t ; sqrt(1 - beta_t) * x_{t-1},  beta_t * I)

 Der Schedule legt also fest, WIE SCHNELL Information zerstoert wird, alles
 andere folgt daraus zwangslaeufig.

 Cave!
 linear_beta_schedule wird im Code nie aufgerufen und stammt noch von Ho et al. (2020),
 Cosine muesste das Signal stabiler ueber die Schritte verteilt halten. Kann man aber nochmal testen

Beide Funktionen rechnen in float64 und werden erst beim Registrieren als
Buffer auf float32 gecastet. Grund: alphas_cumprod ist ein Produkt ueber
1000 Faktoren < 1 und laeuft gegen ~1e-9; in float32 akkumuliert das
sichtbaren Rundungsfehler.
"""

def linear_beta_schedule(timesteps):        #Eigentlich Nutzlos, lediglich vllt nuetzlich mal einen run im vergleich zu "cosine" zu machen als starkes Argument warum "cosine" genutzt wurde
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """Kosinus-Schedule aus Nichol & Dhariwal (2021)
    nicht beta wird definiert, sondern direkt alpha_bar:

        alpha_bar(t) = cos( ((t/T + s) / (1 + s)) * pi/2 )^2

    normiert auf alpha_bar(0) = 1. Die betas werden rueckgerechnet:

        beta_t = 1 - alpha_bar(t) / alpha_bar(t-1)

        s = 0.008 verhindert, dass beta_1 exakt 0 wird (sonst Division durch 0
    in den abgeleiteten Groessen).
    """
    steps = timesteps +1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) **2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


beta_schedules = {                        #Ersetzt Zeilen "206 - 211" in ampdiffusion.py
    "linear": linear_beta_schedule,
    "cosine": cosine_beta_schedule
}


class NoiseSchedule(nn.Module):
    """betas, alphas, alpha_bar und abgeleitete Groessen als Buffer.
    Haelt nur Konstanten -- keine Parameter, kein forward().

     Args:
        timesteps: Anzahl Diffusionsschritte T wird von DiffusionProcess aus Config übergeben.
        beta_schedule: wird von DiffusionProcess aus Config übergeben.
    
    """
    def __init__(self, timesteps: int, beta_schedule: str):
        super().__init__()

        if beta_schedule not in beta_schedules:
            raise ValueError(
                f"unbekannter beta_schedule {beta_schedule},"
                f"erlaubt: {sorted(beta_schedules)}"
            )

        betas = beta_schedules[beta_schedule](timesteps)            #Durch Dictionary muss die Funktion vorher aus beta_schedule rausgenommen und gestartet werden
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # Grundgroessen
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min= 1e-8)

        def reg(name: str, val: torch.Tensor):                                #lambda durch Funktion ersetzt -> Uebersichtlicher und besser Wiederverwendbar
            self.register_buffer(name, val.to(torch.float32))

        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("snr", snr.clamp(max=1e4)) # Clamp für numerische Instabilität

        # fuer q(x_t | x_0) also q_sample
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Umrechnung zwischen x_0-, eps- und v-Parametrisierung fuer im Training objective='pred_v' -> genutzt wird aber pred_x_0
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # fuer die Posterior q(x_{t-1} | x_t, x_0), hier ungenutzt, aber schonmal da als Vorbereitung fuer Reverse-Pfad 
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        reg("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod                                                                   #Da kein "self" Vorhanden waere das ganze ohne "staticmethod" irrefuehrend
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):           #Tensoren angegeben -> Bessere Uebersichtlichkeit
        """a[t] herausgreifen und auf (B, 1, 1) umformen.

        Args:
            a: Buffer der Laenge T, z.B. sqrt_alphas_cumprod.        
            t: Zeitschritt pro Batch-Element.                  
            x_shape: Shape des Tensors, auf den gebroadcastet wird.    

        Returns:
            (B, 1, 1), broadcastet damit ueber L und D,
            jedes Batch-Element bekommt seinen eigenen Skalar.
         """
        
        b = t.shape[0]                                                
        out = a.gather(-1, t.to(a.device))                         #(B,); "t.to(a.device)" zieht spaeter einfacher alles auf GPU bzw dahin wo a liegt, damit es verrechnet werden kann -> Reine Sicherheit
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))        #(B, 1, 1)


    
      
class DiffusionProcess(nn.Module):
    """Forward-Rauschen und Trainings-Loss. Wrappt den Denoiser.
    
  Args:
        model: Denoiser mit Signatur model(x, t, padding_mask=...)
            -> (B, L, D). Darf None sein, wenn nur gerauscht wird.
        cfg: Config-Objekt. Gelesen werden:
            pep_max_len   -> L, Sequenzlaenge in Residuen
            embed_dim     -> D
            timesteps     -> T
            beta_schedule -> Default = "cosine" | "linear"
            objective     -> "pred_x0" | "pred_noise" | "pred_v"
            loss_type     -> "l2" | "l1"
            lost_reduction     -> Default = "Sequence"
    """
    def __init__(self, model, cfg: Config):
        
        super().__init__()

        if cfg.objective not in {"pred_x0", "pred_noise", "pred_v"}:
            raise ValueError(f"unbekanntes objective {cfg.objective}")                 #Aufruf der Fehler, wenn etwas nicht stimmt schon hier statt erst spaeter
        if cfg.loss_type not in {"l1", "l2"}:
            raise ValueError(f"unbekannter loss_type {cfg.loss_type}")
        if cfg.loss_reduction not in {"none", "token", "sequence"}:
            raise ValueError(f"unbekannte loss_reduction {cfg.loss_reduction}")
        if cfg.loss_weight_scheme not in {"none", "p2", "min_snr"}:
            raise ValueError(f"unbekanntes loss_weight_scheme {cfg.loss_weight_scheme}")

        self.model = model
        self.cfg = cfg
        self.pep_max_len = cfg.pep_max_len
        self.embed_dim = cfg.embed_dim
        self.objective = cfg.objective
        self.loss_type = cfg.loss_type
        self.loss_reduction = cfg.loss_reduction
        self.loss_weight_scheme = cfg.loss_weight_scheme
        self.loss_weight_gamma = cfg.loss_weight_gamma
        self.loss_weight_k = cfg.loss_weight_k
        self.loss_weight_normalize = cfg.loss_weight_normalize

        #Scheduler
        self.schedule = NoiseSchedule(timesteps=cfg.timesteps, beta_schedule=cfg.beta_schedule)
        self.num_timesteps = self.schedule.num_timesteps


        # p2_loss_weight entfernt, da gesetztes gamma = 0 
        # -> ueberall exakt 1, also identisch


    def q_sample(
            self,
             x_start: torch.Tensor,
             t: torch.Tensor,
             noise = None,
    ):
        """Forward-Prozess: x_0 in einem Sprung auf Schritt t rauschen.

            q(x_t | x_0) = N( x_t ; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I )

        umgeschrieben mit dem Reparametrisierungstrick:

            x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps

             Args:
            x_start: x_0, ESM-Embedding.                
            t: Zeitschritt pro Batch-Element.               
            noise: eps. None -> frisch gezogen.                     

        Returns:
            x_t                                           
         """
        
        if noise is None:
            noise = torch.randn_like(x_start)            #(B, L, D) Ersetzt Lambda und exist Term aus ampdiffusion (Z. 45 - 52)

        extract = self.schedule.extract
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)                 #Reine shape Zusatzzeile -> (B, 1, 1)
        sqrt_1map = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)     #Reine shape Zusatzzeile -> (B, 1, 1)
        x_t = sqrt_ab * x_start + sqrt_1map * noise                                            #(B, L, D)
        
        return x_t                                                        #(B, L, D)

    def predict_v(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor
    ):
        """v-Parametrisierung (Salimans & Ho 2022, Appendix D).

            v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0

        Nur fuer objective='pred_v' gebraucht: dann ist v das Ziel, auf das
        der Denoiser regressiert.

        Args:
            x_start: x_0.                                          
            t: Zeitschritt.                                          
            noise: eps.                                              

        Returns:
            v                                                        
        """

        extract = self.schedule.extract
        sqrt_ab = extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)                  #Reine shape Zusatzzeile -> (B, 1, 1)
        sqrt_1mab = extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)      #Reine shape Zusatzzeile -> (B, 1, 1)
        return sqrt_ab * noise - sqrt_1mab * x_start                                            # (B, L, D)

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == "l1" else F.mse_loss

    def check_shape(self, x: torch.Tensor):         #Verbesserung der Fehlermeldung aus ampdiffusion: def p_losses (Z. 464), welche zwar bei shape Fehlern abbricht aber nichts genaues aussagt
        """Frueher Check der Shapes mit Meldung
         Args:
              x : zu pruefendes Embedding
        """
        
        if x.ndim != 3 or x.shape[1] != self.pep_max_len or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"Embedding hat Shape {tuple(x.shape)}, erwartet "
                f"(B, {self.pep_max_len}, {self.embed_dim}). "
                "Haeufigste Ursache: CLS/EOS wurden nicht abgeschnitten."
            )

    def loss_elements(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            padding_mask: torch.Tensor,
            noise = None,
    )-> torch.Tensor:
        """Berechnet, loss für alle Elemente in (B,L,D)"""
        #Guardian
        if self.model is None:
                    raise RuntimeError("DiffusionProcess model nicht uebergeben")     #Absicherung, dass alles auch mit mehreren py Branches korrekt aufgerufen wurde.
        self.check_shape(x_start)
        if noise is None:
            noise = torch.randn_like(x_start)                        #(B, L, D)

        x_t = self.q_sample(x_start, t, noise=noise) # Ziehen aus Zeit-Abhängiger Verteilung

        model_out = self.model(x_t, t, padding_mask=padding_mask)    #(B, L, D)

        if self.objective == "pred_noise":
            target = noise                                           #(B, L, D)
        elif self.objective == "pred_x0":
            target = x_start                                         #(B, L, D)
        elif self.objective == "pred_v":
            target = self.predict_v(x_start, t, noise)               #(B, L, D)
        else: 
            raise ValueError(
                f"Es gibt kein Objective '{self.objective}', ein zulässiges Objektive wählen:\n"
                "pred_noise\n"
                "pred_x0\n"
                "pred_v"
            )
        
        #Ausgabe mit Guardian:
        loss = self.loss_fn(model_out, target, reduction="none")
        if loss.shape != x_start.shape:
            raise ValueError(
                            f"Loss hat falsche Shapes. Erwartet: {x_start.shape}, Tatsächlich: {loss.shape}"
                            "Der Denoiser gibt einen anderen Shape zurueck als er bekommt. loss_fn =/= x_start"
                        )
        return loss #(B, L, D)

    def loss_weight(self, t:torch.Tensor) -> torch.Tensor | None:
        """Gewichte w(t) pro Batch-Element fuer die Loss-Gewichtung über t
        Args: 
            t:(B,), long, Zeitschritt pro Batch-Element
        
        Returns:
            float , oder None bei Schema 'none' (dann greift in reduce_loss der if weight
            is not none -Zweig nicht)
            """
        
        
        if self.loss_weight_scheme == "none":
            return None
        
        snr = self.schedule.extract(self.schedule.snr, t, t.shape)   # (B,)

        if self.loss_weight_scheme == "p2":
            w = 1 / (self.loss_weight_k + snr )**self.loss_weight_gamma
            return w
        if self.loss_weight_scheme == "min_snr":
            capped = snr.clamp(max = self.loss_weight_gamma)
            if self.objective == "pred_x0":
                w = capped

            elif self.objective == "pred_noise":
                num = capped
                denom = snr
                w = num / denom

            elif self.objective == "pred_v":
                num = capped
                denom = snr + 1.0
                w = num / denom
            
            else: 
                raise ValueError(f"unbekanntes objektive {self.objective}")
            
        else: 
            raise ValueError(
                f"unbekanntes loss_weight_scheme {self.loss_weight_scheme}"
            )
        return w
          
                      
    def reduce_loss(
            self,
            loss_el: torch.Tensor,
            padding_mask: torch.Tensor,
            *,
            weight: torch.Tensor | None = None,
            mode: str | None = None,
    ) -> torch.Tensor:
        """Elementweisen Fehler zu einem Skalar mitteln.

        Alle Modi sind gewichtete Mittel UEBER SEQUENZEN und unterscheiden
        sich nur im Gewicht omega_b:
            "none"      omega_b = 1/B, ohne Maske (reproduziert Torres)
            "sequence"  omega_b = 1/B, maskiert   -- jede Sequenz zaehlt gleich
            "token"     omega_b = L_b / sum(L)    -- jedes Residuum zaehlt gleich

        Torres mittelt ueber ALLE Positionen, also auch ueber das Padding.
        Dort ist das Ziel konstant 0 -- diese Positionen sind trivial und
        verduennen das Gradientensignal proportional zum Padding-Anteil:
        bei durchschnittlich halb gefuellten Sequenzen faellt der Loss rund
        um Faktor 2 zu niedrig aus (gemessen: 1.33 vs 0.72).

        Args:
            loss_el: (B, L, D) elementweiser Fehler aus loss_elements().
            padding_mask: (B, L) bool, True = Padding.
            weight: (B,) optionale Zusatzgewichtung pro Batch-Element,
                z.B. w(t) einer P2-/Min-SNR-Gewichtung. None = aus.
            mode: ueberschreibt cfg.loss_reduction. None = Config benutzen.

        Returns:
            () Skalar.
        """
        #Guardian
        mode = self.loss_reduction if mode is None else mode
        if mode not in {"none", "token", "sequence"}:
            raise ValueError(
                f"unbekannter mode {mode!r}, erlaubt: 'none', 'token', 'sequence'"
            )
        if loss_el.ndim != 3 or loss_el.shape[:2] != padding_mask.shape:
            raise ValueError(
                f"loss_el hat Shape {tuple(loss_el.shape)}, padding_mask "
                f"{tuple(padding_mask.shape)}. Erwartet (B, L, D) und (B, L)."
            )

        # Torres mittelt ueber ALLE Positionen, also auch ueber das Padding.
        # Dort ist das Ziel konstant 0 -- diese Positionen sind trivial und
        # verduennen das Gradientensignal proportional zum Padding-Anteil:
        # bei durchschnittlich halb gefuellten Sequenzen faellt der Loss
        # rund um Faktor 2 zu niedrig aus (gemessen: 1.33 vs 0.72).    

        B, L, D = loss_el.shape
        keep = (~padding_mask).unsqueeze(-1).to(loss_el.dtype)        # (B, L, 1)
        valid_b = keep.sum(dim=(1, 2))                                # (B,)  = L_b

        if mode == "none":
            per_seq = loss_el.flatten(start_dim=1).mean(dim=1)        # (B,)
            omega = loss_el.new_full((B,), 1.0 / B)                   # (B,)
        else:
            denom = (valid_b * D).clamp(min=1.0)                      # (B,)
            per_seq = (loss_el * keep).sum(dim=(1, 2)) / denom        # (B,)
            if mode == "sequence":
                omega = loss_el.new_full((B,), 1.0 / B)               # (B,)
            else:                                                     # "token"
                omega = valid_b / valid_b.sum().clamp(min=1.0)        # (B,)

        if weight is not None:
            if weight.shape != (B,):
                raise ValueError(
                    f"weight hat Shape {tuple(weight.shape)}, erwartet ({B},)."
                )
            omega = omega * weight.to(dtype=omega.dtype, device=omega.device)
            if self.loss_weight_normalize:
                omega = omega / omega.sum().clamp(min=1e-12)              # neu normieren

        return (per_seq * omega).sum()                                # ()
            
    def p_losses(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            padding_mask: torch.Tensor,
            noise=None,
            *,
            weight: torch.Tensor | None = None,
            mode: str | None = None,
    ) -> torch.Tensor:
        """Ein Trainingsschritt: rauschen, vorhersagen lassen, vergleichen.

        Returns:
            () Skalar-Loss.
        """
        loss_el = self.loss_elements(x_start, t, padding_mask, noise=noise)   # (B, L, D)
        return self.reduce_loss(loss_el, padding_mask, weight=weight, mode=mode)
    

    def forward(
        self,
        x_start: torch.Tensor,
        padding_mask: torch.Tensor,
        noise = None,
        mode: str | None = None,
    ):
        """Trainings-Entry-Point: t ziehen, Loss.

        Args:
            x_start: ROHE ESM-Embeddings.                            
            padding_mask: True = Padding.                            
            noise: eps. None -> frisch gezogen.                    

        Returns:
            Skalar-Loss.                                           
        """

        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()    #(B,)

        return self.p_losses(x_start, t, padding_mask=padding_mask, noise=noise, weight= self.loss_weight(t), mode= mode)        #()
