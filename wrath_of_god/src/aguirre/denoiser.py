"""Abschnitt 5 -- Das Netz.

Weiss NICHTS ueber Diffusion. Bekommt ein verrauschtes Embedding und einen
Zeitschritt, gibt eine Vorhersage zurueck. Fertig.

    forward(x, t, padding_mask=None) -> (B, L, D)
        x             (B, L, D)  verrauschtes Embedding
        t             (B,)       long, Zeitschritt pro Batch-Element
        padding_mask  (B, L)     bool, True = Padding

Aufbau nach Torres:
    1. Zeit-Embedding: SinusoidalPosEmb -> MLP -> (B, L, 2D),
       in scale und shift gesplittet  -->  x = x * (scale + 1) + shift
       (FiLM-Konditionierung: das Netz erfaehrt ueber eine affine
       Transformation, wie verrauscht sein Input ist)
    2. Eingangs-MLP
    3. Vortrainierte ESM-2-Attention-Layer, Gewichte als Initialisierung
       uebernommen und mittrainiert
    4. Residual-Verbindung + LayerNorm
    5. Ausgangs-MLP

Shape-Falle: die ESM-Layer erwarten (L, B, D), nicht (B, L, D).
Torres transponiert vor der Schleife und danach zurueck.

Eigene Keyword-Only-Argumente (`*,` in der Signatur) verwenden -- im Original
rutscht in ddim_sample ein Argument positionell in den falschen Slot.
"""

import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Zeitschritt t -> (B, dim) Fourier-Features."""

    # TODO


class DenoiseTransformer(nn.Module):
    """Schaetzt x_0 (bzw. eps) aus x_t und t."""

    # TODO
