"""Abschnitt 2 -- ESM-2 als Encoder UND Decoder.

Beide Richtungen gehoeren in EINE Klasse: sie brauchen dasselbe Modell,
dasselbe `alphabet` und dieselbe Indexliste der 20 Standard-Aminosaeuren.
Getrennt waeren es zwei Stellen, an denen die Token-Indizes auseinanderlaufen
koennen -- und ESM wuerde zweimal geladen.

Es gibt hier KEINEN gelernten Latent Space: der Raum ist der
Repraesentationsraum von ESM-2, der Decoder ist ESM-2s eigener `lm_head`.
Beide bleiben eingefroren (eval-Modus, kein Gradient).

    encode(seqs: list[str])  -> (B, L, D) float32
    decode(emb: (B, L, D))   -> list[str]

Fuer decode(): Logits ueber das volle ESM-Vokabular (33 Token), dann nur die
20 Spalten der Standard-Aminosaeuren auswaehlen und darueber argmax --
sonst generiert das Modell Masken- und Sondertoken.

Modell: esm2_t6_8M_UR50D  ->  D = 320, 6 Layer
"""

import torch.nn as nn


class ESMWrapper(nn.Module):
    """Eingefrorener ESM-2: Sequenz <-> Embedding."""

    # TODO
