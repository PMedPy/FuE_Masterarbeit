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

import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import esm
from aguirre.config import Config

class ESMWrapper(nn.Module):
    """Wrapper um Denoiser: Aufbau eines Embeddingraumes und decoding in Aminosäurelänge
    ### Args:
        - device (string, wird im Geräte unabhängigen Code definiert.), "cpu" als default.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        
        #==Fehler bei falschen Modus auffangen==
        mode = cfg.esm_model_name
        assert mode in ("esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D"), f"Unbekannter Modus '{mode}', erlaubt sind 'esm2_t33_650M_UR50D' und 'esm2_t6_8M_UR50D'"

        #==Auswahl Moden==
        if mode == "esm2_t33_650M_UR50D":
            model,alphabet = esm.pretrained.esm2_t33_650M_UR50D() # Tupel ESM2-Model und Tokentabelle
        elif mode == "esm2_t6_8M_UR50D":
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        assert model.embed_dim == cfg.embed_dim, f"ESM liefert {model.embed_dim}, Config sagt {cfg.embed_dim}"

        model.eval()
        model.requires_grad_(False)
        batch_converter = alphabet.get_batch_converter() # Baut Sequenz in ESM-2 Tokens (Vokabular: 33 Tokens) um. dims: (B,L)
        AS = cfg.AS_alphabet #Alphabet der 20 kanonischen Aminosäuren
        AS_list = torch.tensor([alphabet.tok_to_idx[a] for a in AS], dtype= torch.long)

        #==Attribute Instanzen==
        self.register_buffer("AS_list", AS_list) # Register Buffer dient als nicht lernbarer Parameter und wird später in der state_dict verfügbar sein
        self.AS = AS
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = batch_converter
        self.device = cfg.device
        self.L_max = cfg.pep_max_len
        self.to(self.device)
        

    @torch.no_grad() #Protokoll für Backprop ausschalten
    def encode(self, seqs: List[str]) -> tuple[torch.Tensor, list[int]]: 
        """
        ## Encoder.
        ### Introduction: 
        Transformiert Peptidsequenzen in ESM-2 Embeddings (representations) in gewünschter Modelgröße (8M, oder 650M-Model). 
        Peptidsequenzen (Liste an Strings), werden nach Embedding auf maximale batch-länge mittels ESM-2 Internen Padding  (siehe batch_converter) verlängert und durch eigene Längen per Pad-Funktion auf maximale Länge (default 42) padden. 

        - Notiz: BOS und EOS werden erst nach Attentionblöcke abgeschnitten, sodass Positionsinformationen sich schon in der Embeddingdimension D befinden. 
        -  Decoder: Keine Attention-Layer, nur Linear -> GELU -> LayerNorm -> Linear. 
        ### Args:
        - Batch an Sequenzen (Liste and Strings) z.B ["ACDEFGHIKLMN", "KLVFFAED"]

        ### Output: 
        - Tupel aus dem Embedded Tensor dims (B,L_Max, D), mit L_max der maximalen Peptidlänge und D, der Embeddingdimension und der Liste an Sequenzlängen (B).
        ransformiert Peptidsequenzen in ESM-2 Embeddings in gewünschter Modelgröße (8M, oder 650M-Model). 
        Peptidsequenzen (Liste an Strings), werden nach Embedding auf maximale batch-länge mittels ESM-2 Internen Padding  (siehe batch_converter) verlängert und durch eigene Längen per Pad-Funktion auf maximale Länge (default 42) padden. 
        """

        #==Fehler bei L_batch > L_max auffangen==
        max_len = max(len(s)for s in seqs) 
        assert max_len  <= self.L_max, f"Sequenz mit {max_len} Residuen, L_max ist {self.L_max}"
        data = [(f"seq{i}", s) for i, s in enumerate(seqs)] # Paare aus Sequenzname + Sequenz in Liste gepackt
        seqs_lengths = torch.tensor([len(s) for s in seqs]).unsqueeze(1).to(self.device) #dims (B,1) jeweilige AS-Sequenzlängen
        model = self.model

        #==ESM-2 Inferenz==
        _, _,seq_tokens = self.batch_converter(data) # dims: (B,L_batch+2), mit PADs zum Auffüllen
        out = model(seq_tokens.to(self.device), repr_layers = [model.num_layers]) #dims: (B) #Inferenz mit pLM
        reps = out["representations"][model.num_layers] # dims: (B,L_batch+2, D)
        reps_sliced = reps[:,1:-1,:] #Abschneiden von BOS und EOS ,dims: (B,L_batch, D)

        #==Zero-Padding mit F.pad() für maximale Länge==
        zero_pad_amount = self.L_max - reps_sliced.size(1) # dims:(B)
        reps_padded = F.pad(reps_sliced,(0,0,0,zero_pad_amount), mode= "constant", value = 0).to(self.device) # dims: (B,L_max,D)

        #==Paddingmaske mit booleans==
        positions = torch.arange(self.L_max).unsqueeze(0).to(self.device) #dims: (1, L_max)
        padding_mask = positions < seqs_lengths # dims: (1, L_max) 

        #==Ausgabe==
        reps_return = reps_padded * padding_mask.unsqueeze(-1) # Aufgefüllt mit batch_converter Padding und 0 (hier Broadcasting über B und D?)
        return reps_return.cpu(), seqs_lengths.squeeze(1).cpu().tolist() # dims: representations (B,L_max, dims), Sequenzlängenliste

    @torch.no_grad()
    def decode(self, reps: torch.Tensor, lengths: List[int], names: List[str] = None) -> dict:
        """## Decoder.
        Benutzt Language Modeling Head (LM-Head) von ESM-2 in gewünschter Modellgröße (8M oder 650M), um Sequenz aus Embeddingraum (320, oder 1280) in Sequenzstring zu übersetzen. 

        ### Introduction:
        - Lm-Head gibt wahrscheinlichste Aminosäure an den jeweiligen Positionen aus.
        - Kleiner Head, nur mit:  Linear -> GELU -> LayerNorm -> Linear, keine Attention-Layer.

        ### Args:
        - Representations: torch.Tensor, shape (B, L_max, D) (Embeddings aus Encode oder generierte Sequenzen in Embeddingraum aus dem Diffusionsmodell)
        - lengths: Liste aus Integer, Längen der AMPs aus Encode, oder dem Diffusionsmodell für Abschneiden der Paddingsequenzen.
        - names: Liste an Strings: Bezeichnung für decodierte Sequenzen

        ### Output:
        - Dictionairy aus  Name: Sequenz als String
        """
        
        names = names or [f"Sequenz {i+1}" for i in range(reps.size(0))]
        assert len(names) == reps.size(0)
        logits = self.model.lm_head(reps.to(self.device)) # lm_head: Linear -> GELU -> LayerNorm -> Linear. dims: (B, L_max, 33)
        seq_tokens = torch.argmax(logits[:,:, self.AS_list], dim=-1) # Nur die Tokens für die 20 kanonischen AS. dims: (B,L_batch)
        
        results = {} #Leeres Dicitonairy
        for name, row, l in zip(names, seq_tokens.cpu().tolist(), lengths): #
            seq = "" #Leerer string
            for i in row[:l]:  #Für jede Sequenz wird über deren Position bis l iteriert
                seq += self.AS[i] 
            results[name] = seq 

        return results
        
