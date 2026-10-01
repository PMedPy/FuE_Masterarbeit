"""Torrinator -- Latent-Diffusion-Modell zur De-novo-Generierung von AMPs.

Nachbau und Erweiterung von Torres et al. (2025), AMPDiffusion.

Globale Konventionen fuer dieses Paket
--------------------------------------
Shapes werden als Kommentar hinter jeder Tensor-Zeile notiert:

    x = self.mlp(x)              # (B, L, D)

Symbole:
    B  Batchgroesse
    L  Sequenzlaenge in Tokens = 42  (40 Residuen + CLS + EOS)
    D  Embedding-Dimension   = 320  (ESM2-8M, esm2_t6_8M_UR50D)
    T  Anzahl Diffusionsschritte = 1000
"""

__version__ = "0.1.0"
