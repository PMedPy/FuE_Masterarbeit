"""Abschnitt 7 -- Generative Inferenz.

Kette:
    Checkpoint laden (EMA-Gewichte!)
      -> DiffusionProcess.sample(n, design_len)   -> (N, L, D)
      -> ESMWrapper.decode(...)                   -> list[str]
      -> auf design_len kuerzen
      -> CSV

design_len ist die gewuenschte Peptidlaenge in Aminosaeuren; in Tokens sind
es design_len + 2 (CLS + EOS). Die Padding-Maske steuert, welche Positionen
beim Denoising ueberhaupt beachtet werden -- so kommen variable Laengen aus
einem Modell mit fester Tensorbreite L = 42.
"""


class Sampler:
    """Laedt einen Checkpoint und generiert Sequenzen."""

    # TODO
