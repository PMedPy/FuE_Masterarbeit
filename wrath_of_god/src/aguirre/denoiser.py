"""Abschnitt 5 -- Das Netz.

Weiss NICHTS ueber Diffusion. Bekommt ein verrauschtes Embedding und einen
Zeitschritt, gibt eine Vorhersage zurueck. Fertig.

    forward(x, t, padding_mask) -> (B, L, D)
        x             (B, L, D)  verrauschtes Embedding
        t             (B,)       long, Zeitschritt pro Batch-Element
        padding_mask  (B, L)     bool, True = Padding

Aufbau nach Torres:
    1. Zeit-Embedding: SinusoidalPosEmb -> MLP -> (B, L, 2D) oder bei "global" (B, 1, 2D),
       in scale und shift gesplittet  -->  x = x * (scale + 1) + shift
       (FiLM-Konditionierung: das Netz erfaehrt ueber eine affine
       Transformation, wie verrauscht sein Input ist)
    2. Padding-Position auf 0 setzen
    3. Eingangs-MLP
    4. Vortrainierte ESM-2-Attention-Layer, Gewichte als Initialisierung
       uebernommen und mittrainiert
    5. Residual-Verbindung + LayerNorm
    6. Ausgangs-MLP

Shape-Falle: die ESM-Layer erwarten (L, B, D), nicht (B, L, D).
Torres transponiert vor der Schleife und danach zurueck.
Die padding_mask bleibt dabei (B, L), da ESM dies so erwartet.

Eigene Keyword-Only-Argumente (`*,` in der Signatur) verwenden -- im Original
rutscht in ddim_sample ein Argument positionell in den falschen Slot.

Rausgelassen wurden:
'classes'                          -> Wird nie aufgerufen
'self_condition', 'x_self_cond'    -> In generate.py auf "False" gesetzt
'if exists(t)'                     -> Unnoetiger Check, da es hinter t = self.time_mlp(t) steht und somit immer "True" ist
'layer_idx', 'attn'                -> Schleifenvariable wird nie gelesen
'r = x.clone()'                    -> Clonen unnoetig, da x neu zugewiesen wird (keine In-place Operation dazwischen)
'device = x.device'                -> Wird nie aufgerufen
'time_dim = embed_dim'             -> Von Torres veraendert (urspruenglich = embed_dim * 4), dadurch nun unnoetig
"""

import math
import copy

import torch
import torch.nn as nn

from esm.modules import ESM1bLayerNorm
from aguirre.config import Config


class SinusoidalPosEmb(nn.Module):
    """Zeitschritt t -> (B, dim) Fourier-Features.
    
    Der Skalar t wird auf dim/2 Frequenzen abgebildet und als sin/cos-Paar ausgegeben.
    Ohne das bekaeme das MLP eine einzelne Zahl zwischen 0 und T und
    muesste die Aufloesung selbst lernen.

    Args:
        dim: Ausgabedimension, muss gerade sein (haelftig sin, haelftig cos).
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim muss gerade sein, ist {dim}")    #Pruefung, ob volle dim übergeben wird -> halbe dim faellt sonst weg
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        Args:
            t: (B,) long, Zeitschritt pro Batch-Element.

        Returns:
            (B, dim) float.
        """
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=t.device) * -freq)
        emb = t[:, None].float() * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DenoiseTransformer(nn.Module):
    """Schaetzt x_0 (bzw. eps) aus x_t und t.
    
    Bekommt einfache Werte, kein Config-Objekt.
    Erst bei def build_denoiser(cfg, esm_modell) werden die Werte aus der Config uebergeben.

    Args:
        embed_dim: D. ESM2-8M: 320, ESM2-650M: 1280.
        pep_max_len: L, Sequenzlaenge in Residuen (CLS/EOS sind ab).
        esm_layers: nn.ModuleList vortrainierter ESM-2-TransformerLayer.
            PFLICHT und trainierbar -- siehe trainable_esm_layers().
        time_cond: "per_position" (Torres) oder "global". Siehe time_mlp.
    """

    def __init__(
            self,
            *,
            embed_dim: int,
            pep_max_len: int,
            esm_layers: nn.ModuleList,
            time_cond: str
    ):
        super().__init__()

        if time_cond not in {"per_position", "global"}:
            raise ValueError(f"unbekanntes time_cond {time_cond}, erlaubt 'per_position' und 'global'")
        if esm_layers is None or len(esm_layers) == 0:
            raise ValueError("esm_layers fehlen! Denoiser wird zum reinen MLP")

        self.embed_dim = embed_dim
        self.pep_max_len = pep_max_len
        self.time_cond = time_cond

        # Torres gibt JEDER Position ihr eigenes scale/shift-Paar. Die letzte
        # Linear-Schicht hat dadurch embed_dim * (embed_dim * L * 2) Gewichte:
        #   ESM2-8M,   D= 320, L=42  ->   8.6 Mio  (94 % des eigenen Netzes)
        #   ESM2-650M, D=1280, L=50  -> 164.0 Mio  (mit Adam ca. 2.6 GB VRAM)
        # "global" gibt stattdessen EIN scale/shift-Paar aus und broadcastet
        # es ueber L -- die uebliche FiLM-Form, bei D=1280 nur 3.3 Mio.
        # Default erstmal Torres, aber vllt global interessant?
        if time_cond == "per_position":
            time_out = embed_dim * pep_max_len * 2
        else:
            time_out = embed_dim * 2

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, time_out)
        )

        #Eingangs_MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim) 
        )

        #Vortrainiertes ESM-2-Attention
        self.esm_layers = esm_layers
        self.emb_layer_norm_after = ESM1bLayerNorm(embed_dim)

        #Ausgangs-MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            *,
            padding_mask: torch.Tensor
    ):
         """Ein Entrausch-Schritt.

         Args:
            x: (B, L, D) verrauschtes Embedding.
            t: (B,) Zeitschritt pro Batch-Element.
            padding_mask: (B, L) bool, True = Padding. Keyword-only und Pflicht
            im Original rutscht in ddim_sample ein Argument positionell in den falschen Slot
            (model_predictions wird dort mit self_cond an der Stelle von design_len gerufen).
            Wenn padding_mask = None wird nichts maskiert.

        Returns:
            (B, L, D) Vorhersage, je nach objective aus noiser: x_0, eps oder v.
        """

         B, L, D = x.shape
         if L != self.pep_max_len or D != self.embed_dim:
             raise ValueError(
                 f"x hat shape {x.shape}, erwartet (B, {self.pep_max_len}, {self.embed_dim}). "
                 f"Check, ob die cfg.pep_max_len und cfg.embed_dim zu den ESMWrapper Embeddings passen?"
             )
         if padding_mask is not None and (padding_mask.shape) !=(B, L):
             raise ValueError(
                 f"padding_mask hat Shape {padding_mask.shape}, erwartet {(B, L)}."
             )

        #FiLM: scale und shift aus dem Zeitschritt 
         emb = self.time_mlp(t)                                            #(B, 2DL) oder (B, 2D)
         if self.time_cond == "per_position":
             emb = emb.reshape(B, self.pep_max_len, 2 * self.embed_dim)    #(B, L, 2D)
         elif self.time_cond == "global":
             emb = emb.reshape(B, 1, 2 * self.embed_dim)                   #(B, 1, 2D) 
         else:
             raise ValueError(
                 f"Eingegebene time_cond unbekannt {self.time_cond}, zulaessig 'per_position' und 'global'"
             )     
         scale, shift = emb.chunk(2, dim=-1)                               #(B, L|1, D)  
         x = x * (scale + 1.0) + shift                                     #(B, L, D)

        #Padding nullen -> Noetig weil shift auf Padding-Position addiert wird und q_sample dort ohnehin schon Rauschen hinterlässt
         if padding_mask is not None:
             x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)            #(B, L, D)

        #In ESM-Konvention (L, B, D) umwandeln
         x = x.transpose(0, 1)                                             #(L, B, D)
         r = x
         x = self.mlp(x)                                                   #(L, B, D)

        #Attention. padding_mask bleibt (B, L) so wie ESM diese braucht
         for layer in self.esm_layers:
             x, _ = layer(x, self_attn_padding_mask=padding_mask)          #(L, B, D)

        #Residual + LayerNorm
         x = x + r                                                         #(L, B, D)
         x = self.emb_layer_norm_after(x)                                  #(L, B, D)

        #Zurueck in urspruengliches shape (B, L, D)
         x = x.transpose(0, 1)                                             #(B, L, D)
         return self.out_mlp(x)                                            #(B, L, D)


def trainable_esm_layers(esm_model, n_layers: int = None):
    """Eigene, trainierbare Kopie der ESM-2-Attention-Layer.
    Der ESMWrapper setzt requires_grad_(False) auf sein ESM-Modell -> Modell ist eingefroren!
    Wuerde man diese layers direkt hereinreichen, waere der halbe Denoiser eingefroren (ohne Fehlermeldung!) 
    der Loss wuerde kaum sinken, deshalb hier eine deepcopy mit requires_grad_(True)

    Die Kopie ist Wichtig, da Encoder und Denoiser sich nicht dieselben Gewichte teilen sollen.
    Der Encoder muss eingefroren bleiben, sonst verschiebt sich waehrend des Trainings der Raum, in den diffundiert wird.

    Args:
        esm_model: geladenes ESM-2-Modell
        n_layers: nur die ersten n Layer uebernehmen. None = alle.

    Returns:
        nn.ModuleList trainierbarer TransformerLayer.
    """

    layers = list(esm_model.layers)

    if n_layers is not None:
        if not 0 < n_layers <= len(layers):
            raise ValueError(f"n_esm_layers={n_layers}, Modell hat {len(layers)} Layer.")
        layers = layers[:n_layers]

    layers = nn.ModuleList(copy.deepcopy(layers))
    layers.requires_grad_(True)
    return layers


def build_denoiser(cfg: Config, esm_model):
    """Denoiser aus der Config bauen.

    Nutzt das bereits geladene ESM des Wrappers, statt es ein zweites Mal zu laden
    -> Spart Rechenzeit.

    Args:
        cfg: Config. Gelesen werden embed_dim, pep_max_len, n_esm_layers, time_cond.
        esm_model: geladenes ESM2-Modell aus config.

    Returns:
        DenoiseTransformer, bereit fuer DiffusionProcess(denoiser, cfg).
    """

    layers = trainable_esm_layers(esm_model, cfg.n_esm_layers)
     
    return DenoiseTransformer(
        embed_dim=cfg.embed_dim,
        pep_max_len=cfg.pep_max_len,
        esm_layers=layers,
        time_cond=cfg.time_cond
    )
 