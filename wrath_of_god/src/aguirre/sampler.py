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

import pandas as pd
import torch

from pathlib import Path
from aguirre.config import Config, CACHE_DIR
from aguirre.denoiser import build_denoiser
from aguirre.diffusion import DiffusionProcess
from aguirre.esm_wrapper import ESMWrapper
from aguirre.helpers import masker, load_standards


def _clean_state_dict(state: dict, ema: bool):
    """State-Dict auf die Keys von DiffusionProcess zurechtschneiden.
    
        Haendelt zwei Praefixe:
            'ema_model.'  ema_pytorch legt das Schattenmodell darunter ab.
                          -> 'online_model.'-Keys werden verworfen.
            'module.'     torch.nn.DataParallel (Torres nutzt das).
    
        Args:
            state: geladenes State-Dict.
            ema: True -> nur 'ema_model.'-Keys behalten.
    
        Returns:
            State-Dict passend zu DiffusionProcess.load_state_dict().
        """
    if ema:
        state = {
            k[len("ema_model."):]: v
            for k, v in state.items()
            if k.startwith("ema_model.")
        }
        if not state:
            raise ValueError(
                "Im EMA-State liegen keine 'ema_model.'-Keys!"
                "Anderes EMA-Paket oder falscher Key im Checkpoint?"
            )

        return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
    }



class Sampler:
    """Laedt einen Checkpoint und generiert Sequenzen.
            Args:
                diffusion: DiffusionProcess mit geladenen Gewichten, auf cfg.device.
                esm: ESMWrapper -- wird fuer decode() gebraucht, sein ESM-Modell
                    hat ausserdem die Layer fuer den Denoiser geliefert.
                cfg: Config des Laufs.
                mean: (D,) Mittelwert der Standardisierung, oder None wenn der
                    Lauf nicht standardisiert war.
                std: (D,) Standardabweichung, oder None.
            
    """

    def __init__(
            self,
            diffusion: DiffusionProcess,
            esm: ESMWrapper,
            cfg: Config,
            *,
            mean: torch.Tensor | None,
            std: torch.Tensor | None
    ):
        if (mean is None) != (std is None):
            raise ValueError(
                "mean und std nur gemeinsam setzten oder gemeinsam auf 'None' stellen"
                )

        self.diffusion = diffusion.eval()
        self.esm = esm
        self.cfg = cfg
        self.device = cfg.device
        self.mean = mean
        self.std = std

    def _to_esm_space(self, x: torch.Tensor):
        """Standardisierung zurueckrechnen: Modellraum -> ESM-Raum.
        
        Ohne mean/std die Identitaet. MUSS vor ESMWrapper.decode() laufen --
        der lm_head erwartet rohe ESM-Repraesentationen.
        
        Args:
            x: (B, L, D) wie aus DiffusionProcess.sample().
        
        Returns:
            (B, L, D) im ESM-Raum.
        """

        if self.mean is None:
            return x
        return x * self.std.to(x) + self.mean.to(x)    #(B, L, D)

    #Konstruktoren

    @classmethod
    def _build(cls, cfg: Config, standardize: bool):
        """Module bauen, ohne Gewichte zu laden. Gemeinsame Basis.
        
        ESM wird GENAU EINMAL geladen: der Wrapper braucht es fuer decode(),
        der Denoiser bekommt Kopien seiner Attention-Layer (siehe
        trainable_esm_layers -- der Encoder bleibt eingefroren).
        """
            
        esm = ESMWrapper(cfg)
        denoiser = build_denoiser(cfg, esm.model)
        diffusion = DiffusionProcess(denoiser, cfg).to(cfg.device)
        
        mean, std = load_standards(cfg) if standardize else (None, None)
        return cls(diffusion, esm, cfg, mean=mean, std=std)

    @classmethod
    def from_untrained(cls, cfg: Config | None, standardize: bool = True):
        """Vollstaendige Kette mit ZUFALLSGEWICHTEN -- zum Testen ohne Checkpoint.
        
        Shapes, Laufzeit und Speicherbedarf sind identisch zum echten Lauf.
        Die Sequenzen sind Unsinn, das ist der Punkt.
        
        Args:
            cfg: Config. None -> Default.
            standardize: mean/std aus der Standards-CSV laden.
        """

        cfg = Config() if cfg is None else cfg
        return cls._build(cfg, standardize)

    @classmethod
    def from_checkpoint(
            cls,
            path: str | Path,
            cfg: Config | None,
            *,
            standardize: bool | None,
            use_ema: bool = True,
            strict: bool = True,
    ):
        """Sampler aus einem Checkpoint bauen.
        
        UNGETESTET, solange Trainer.save_checkpoint() nicht existiert -- der
        erwartete Aufbau steht im Modul-Docstring (Checkpoint-Vertrag).
        
        Die Config kommt aus dem Checkpoint, nicht von aussen: ein Lauf mit
        esm2_t6_8M (D=320) laesst sich sonst stumm mit der 650M-Default-Config
        laden und scheitert erst beim ersten Matmul.
        
        Args:
            path: .pt-Datei.
            cfg: ueberschreibt die Config im Checkpoint. Normalerweise None.
            standardize: ob der Lauf auf standardisierten Embeddings
                trainiert wurde. None -> aus dem Checkpoint lesen; fehlt
                der Schluessel dort, bricht die Methode ab statt zu raten.
            use_ema: EMA-Gewichte nehmen. Fast immer richtig.
            strict: an load_state_dict weitergegeben.
        
        Returns:
            Sampler, bereit fuer .sample().
         """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {path}")

        #weight_only=False, weil der Checkpoint die Config mitbringt. Nur eigene Datein laden.
        data = torch.load(path, map_location="cpu", weights_only=False)

        if cfg is None:
            if "cfg" not in data:
                raise KeyError(
                    f"Checkpoint {path.name} enthaelt keinen 'cfg'-Eintrag. "
                    "Entweder Config explizit uebergeben oder den Trainer an den Checkpoint anpassen "
                )
            raw = data["cfg"]
            cfg = raw if isinstance(raw, Config) else Config(**raw)

        #Gewichtssatz
        key = "ema" if use_ema else "model"
        if key not in data:
            other = "model" if use_ema else "ema"
            raise KeyError(
                f"Checkpoint hat keinen Eintrag '{key}' (vorhanden: "
                f"{sorted(data)}). Mit use_ema={not use_ema} waere es '{other}'."
            )
        state = _clean_state_dict(data[key], ema=use_ema)

        #Standardisierung -> Der Checkpoint traegt mean/std nicht mit, also muss er das Flag mitbringen.
        if standardize is None:
            if "standardize" not in data:
                raise KeyError(
                    f"Checkpoint {path.name} sagt nicht, ob der Lauf "
                    "standardisiert war, und standardize= wurde nicht "
                    "uebergeben. Bitte explizit angeben -- raten waere hier "
                    "stumm falsch (siehe Checkpoint-Vertrag im Modul-Docstring)."
                )
            standardize = bool(data["standardize"])


        #Bauen und befuellen
        self = cls._build(cfg, standardize=standardize)
        missing, unexpected = self.diffusion.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"[Sampler] fehlende Keys: {list(missing)}")
            print(f"[Sampler] unerwartete Keys: {list(unexpected)}")
        self.diffusion.eval()
        return self

    #Laengen

    def _resolve_lengths(
            self,
            lenghts,
            n: int | None
    ):
        """Laengenangabe auf (N,) normieren.
        
        Args:
            lengths: ein int (gleiche Laenge fuer alle, dann ist n Pflicht),
                oder eine Liste/ein Tensor mit einer Laenge pro Sequenz.
            n: Anzahl Sequenzen, nur bei int-Angabe gebraucht.
        
        Returns:
            (N,).
        """
        if isinstance(lenghts, int):
            if n is None:
                raise ValueError("Bei einer einzelnen Laenge muss n angegeben werden.")
            out = torch.full((n,), lenghts, dtype=torch.long)
        else:
            out = torch.as_tensor(lenghts, dtype=torch.long).flatten()
            if n is not None and out.numel() != n:
                raise ValueError(
                    f"{out.numel()} Laengen, aber n={n}. Entweder n weglassen "
                    "oder genau n Laengen uebergeben."
                )

        #Guardian
        lo, hi = self.cfg.pep_min_len, self.cfg.pep_max_len
        if (out < 1).any():
            raise ValueError("Laengen < 1 sind nicht generierbar.")
        if (out > hi).any():
            raise ValueError(
                f"Groesste Laenge {int(out.max())} > pep_max_len {hi}."
            )
        if (out < lo).any():
            print(
                f"[Sampler] Hinweis: {int((out < lo).sum())} Laengen unter "
                f"pep_min_len {lo} -- biologisch fragwuerdig, wird trotzdem generiert."
            )
        return out

    def draw_lengths(
            self,
            n: int,
            *,
            source: str = "train",
            seed: int | None,
    ):
        """Laengen ziehen.

        Die Laengenverteilung bestimmt mit, wie die generierten Sequenzen
        aussehen -- wer uniform zieht, vergleicht hinterher nicht mehr
        sauber gegen die Trainingsverteilung oder gegen Torres.

        Args:
            n: Anzahl Laengen.
            source: "train" -> empirisch aus den gecachten Train-Embeddings
                (richtig fuer Vergleiche). "uniform" -> gleichverteilt in
                [pep_min_len, pep_max_len].
            seed: fuer reproduzierbare Ziehung. None -> cfg.seed.

        Returns:
            (n,) long.
        """
        g = torch.Generator().manual_seed(self.cfg.seed if seed is None else seed)

        if source == "uniform":
            lo, hi = self.cfg.pep_min_len, self.cfg.pep_max_len
            return torch.randint(lo, hi + 1, (n,), generator=g, dtype=torch.long)

        if source != "train":
            raise ValueError(f"unbekannte source {source!r}, erlaubt 'train' | 'uniform'")

        cache = (
            CACHE_DIR
            / f"train-Set_{self.cfg.esm_model_name}_L{self.cfg.pep_max_len}.pt"
        )
        if not cache.exists():
            raise FileNotFoundError(
                f"Train-Cache nicht gefunden: {cache}\n"
                "Entweder scripts/05_build_embeddings.py laufen lassen oder "
                "source='uniform' nutzen."
            )
        pool = torch.as_tensor(torch.load(cache)["lengths"], dtype=torch.long).flatten()
        idx = torch.randint(0, pool.numel(), (n,), generator=g)
        return pool[idx]

    #Generieren

    @torch.no_grad()
    def sample_embeddings(
            self,
            lengths,
            n: int | None = None,
            *,
            batch_size: int | None = None,
            progress: bool = True,
    ):
        """Embeddings generieren -- im ESM-Raum, fertig fuer decode().

        Laeuft in Batches: 1000 Reverse-Schritte auf dem 650M-Denoiser
        passen fuer N=100 nicht in einem Rutsch in den Speicher.

        Args:
            lengths: Laengen (siehe _resolve_lengths).
            n: Anzahl, nur bei einer einzelnen int-Laenge noetig.
            batch_size: None -> cfg.batch_size.
            progress: Fortschrittsbalken pro Batch.

        Returns:
            (emb, lengths): (N, L, D) float32 auf CPU und (N,) long.
        """
        lens = self._resolve_lengths(lengths, n)                      # (N,)
        bs = self.cfg.batch_size if batch_size is None else batch_size
        if bs < 1:
            raise ValueError(f"batch_size muss >= 1 sein, ist {bs}")

        out = []
        for start in range(0, lens.numel(), bs):
            chunk = lens[start:start + bs].to(self.device)            # (b,)
            padding_mask = masker(chunk, pep_max_len=self.cfg.pep_max_len)   # (b, L)

            x = self.diffusion.sample(padding_mask, progress=progress)       # (b, L, D)
            x = self._to_esm_space(x)                                        # (b, L, D)
            out.append(x.float().cpu())

        return torch.cat(out, dim=0), lens                            # (N, L, D), (N,)

    @torch.no_grad()
    def sample(
            self,
            lengths,
            n: int | None = None,
            *,
            batch_size: int | None = None,
            names: list[str] | None = None,
            progress: bool = True,
    ):
        """Sequenzen generieren.

        Args:
            lengths: Laengen (siehe _resolve_lengths).
            n: Anzahl, nur bei einer einzelnen int-Laenge noetig.
            batch_size: None -> cfg.batch_size.
            names: Bezeichner. None -> 'gen_0000' ...
            progress: Fortschrittsbalken.

        Returns:
            DataFrame mit Spalten name, length, sequence.
        """
        emb, lens = self.sample_embeddings(
            lengths, n, batch_size=batch_size, progress=progress
        )
        lens_list = lens.tolist()

        if names is None:
            names = [f"gen_{i:04d}" for i in range(len(lens_list))]
        elif len(names) != len(lens_list):
            raise ValueError(f"{len(names)} Namen, aber {len(lens_list)} Sequenzen.")

        results = self.esm.decode(emb, lens_list, names=names)        # dict[name] = seq

        df = pd.DataFrame(
            {
                "name": names,
                "length": lens_list,
                "sequence": [results[k] for k in names],
            }
        )

        #Guardian: decode() schneidet auf length -- wenn das nicht passt,
        #sind lengths und Maske auseinandergelaufen.
        bad = df["sequence"].str.len() != df["length"]
        if bad.any():
            raise RuntimeError(
                f"{int(bad.sum())} Sequenzen haben nicht die angeforderte Laenge. "
                "lengths und padding_mask passen nicht zusammen."
            )
        return df

    def to_csv(self, df: pd.DataFrame, path) -> Path:
        """DataFrame wegschreiben.

        Args:
            df: Ergebnis von sample().
            path: Zieldatei. Elternordner wird angelegt.

        Returns:
            Der geschriebene Pfad.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path