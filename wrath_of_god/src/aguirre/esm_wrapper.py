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
import torch 
import esm

class ESMWrapper(nn.Module):
    """Wrapper um Denoiser: Aufbau eines Embeddingraumes und decoding in Aminosäurelänge
    """
    def __init__(self, device: str = "cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        AS_list = alphabet.all_toks[4:-7]
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        self.device = device
        

    @torch.no_grad()
    def ESM_encode(self, seqs: list[str]) -> torch.Tensor: 
        """"""
        data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
        model = self.model
        model.eval().to(self.device)
        _, _,seq_tokens = self.batch_converter(data) # dims (B,L_batch)
        out = model(seq_tokens.to(self.device), repr_layers = [model.num_layers]) #dims (B)
        reps = out["representations"][model.num_layers]
        reps_sliced = reps[:,1:-1,:] #Abschneiden von BOS und EOS
        return reps_sliced.cpu()


    @torch.no_grad()
    def ESM_decode():
        pass

