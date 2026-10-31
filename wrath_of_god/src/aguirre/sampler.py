"""Abschnitt 7 -- Generative Inferenz

Kette:
    Lauf laden (EMA-Gewichte!)
      -> DiffusionProcess.sample(padding_mask)   -> (N, L, D) im Modellraum
      -> _to_esm_space(...)                      -> (N, L, D) im ESM-Raum
      -> ESMWrapper.decode(...)                  -> dict[name] = Sequenz
      -> DataFrame -> CSV

Modellunabhaengig: welches ESM-2 (und damit welches D und L) gilt, steht in
der Config des Laufs und wird aus dem Checkpoint gelesen.

STANDARDISIERUNG
Der Normalfall: Data_coordinator rechnet (x - mean) / std pro
Embedding-Dimension, mit den Statistiken aus
data_tracked/standards_<modell>_L<max_len>.csv (04_standardize.py). Der
Sampler rechnet das vor dem Decoder zurueck.

Der Checkpoint sagt nicht, ob ein Lauf standardisiert war, deshalb ist
standardize=True der Default -- das ist 06_train.py ohne --no-standardize.
Wer MIT --no-standardize trainiert hat, uebergibt standardize=False, sonst
rechnet der Sampler zurueck, was nie umgerechnet wurde.

Damit das nicht still schiefgeht, misst sample_embeddings das Ergebnis
nach: bei einem standardisierten Lauf muss im Modellraum ungefaehr
Streuung 1 herauskommen. Liegt sie weit daneben, war der Lauf entweder
nicht standardisiert, das Netz ist kollabiert, oder die Standards-CSV
gehoert nicht zu diesem Lauf.

Checkpoint-Vertrag -- wortgleich zu trainer.RunManager:
    "model"     DenoiseTransformer.state_dict()
    "ema"       dasselbe fuer das EMA-Schattenmodell. Hieraus wird gesampelt.
    "epoch"     int, zuletzt abgeschlossene Epoche
    "best_val"  float
    "cfg"       dataclasses.asdict(cfg), ein einfaches Dict

"""

from dataclasses import fields, replace
from pathlib import Path

import pandas as pd
import torch

from aguirre.config import Config, CACHE_DIR
from aguirre.denoiser import build_denoiser
from aguirre.diffusion import DiffusionProcess
from aguirre.esm_wrapper import ESMWrapper
from aguirre.helpers import masker, load_standards, standards_path


class Sampler:
    """Laedt einen Trainingslauf und generiert Sequenzen.

    Nicht direkt bauen -> Sampler.load() nehmen.

    Args:
        diffusion: DiffusionProcess mit geladenen Gewichten, auf cfg.device.
        esm: ESMWrapper fuer decode(). Sein ESM-Modell hat ausserdem die Layer fuer den Denoiser geliefert.
        cfg: Config des Laufs, aus dem Checkpoint.
        mean: (D,) Mittelwert der Standardisierung, oder None bei einem Lauf ohne Standardisierung.
        std: (D,) Standardabweichung, oder None.
        herkunft: Protokoll des Laufs (Quelle, Epoche, best_val, EMA).
    """

    def __init__(
            self,
            diffusion: DiffusionProcess,
            esm: ESMWrapper,
            cfg: Config,
            mean: torch.Tensor | None,
            std: torch.Tensor | None,
            herkunft: dict
    ):
        self.diffusion = diffusion.eval()
        self.esm = esm
        self.cfg = cfg
        self.device = cfg.device
        self.mean = mean
        self.std = std
        self.herkunft = herkunft

    @property
    def denoiser(self):
        """Das Netz selbst -> Ziel des load_state_dict beim Laden."""
        return self.diffusion.model

    def __repr__(self):
        return (
            f"Sampler({self.cfg.esm_model_name}, D={self.cfg.embed_dim}, "
            f"L={self.cfg.pep_max_len}, T={self.diffusion.num_timesteps}, "
            f"device={self.device}, quelle={self.herkunft['quelle']})"
        )

    @classmethod
    def load(
            cls,
            quelle: str | Path,
            tag: str = "final",
            *,
            standardize: bool = True,
            use_ema: bool = True,
            device: str | None = None
    ):
        """Sampler aus einem Lauf-Ordner oder einer einzelnen .pt-Datei.

        Lauf-Ordner von RunManager:
            runs/train_<name>_<datum>/
                final.pt                        tag="final"
                checkpoints/checkpoint_best.pt  tag="best"
                checkpoints/checkpoint_last.pt  tag="last"

        Args:
            quelle: Lauf-Ordner oder direkt eine .pt-Datei.
            tag: nur bei einem Ordner. "final" | "best" | "last" | z.B. "epoch_042".
            standardize: ob der Lauf auf standardisierten Embeddings trainiert wurde.
                True = Default von Skript 06_train.py;
                False nur fuer Laeufe mit 'no-standardize'.
            use_ema: EMA-Gewichte nehmen. Fast immer richtig -> gesampelt wird aus der EMA, wie bei generate.py Torres gelassen.
            device: None -> das Geraet, das Config auf dieser Maschine ermittelt.

        Returns:
            Sampler, bereit fuer .sample().
        """
        quelle = Path(quelle)

        if quelle.is_dir():
            pfad = (quelle / "final.pt" if tag == "final"
                    else quelle / "checkpoints" / f"checkpoint_{tag}.pt")
            if not pfad.is_file():
                vorhanden = ["final"] if (quelle / "final.pt").is_file() else []
                vorhanden += sorted(
                    p.stem.removeprefix("checkpoint_")
                    for p in (quelle / "checkpoints").glob("checkpoint_*.pt")
                )
                raise FileNotFoundError(
                    f"{pfad} existiert nicht. Vorhandene Tags in "
                    f"{quelle.name}: {vorhanden if vorhanden else '(keine)'}"
                )
        elif quelle.is_file():
            pfad = quelle
        else:
            raise FileNotFoundError(f"Weder Ordner noch Datei: {quelle}")

        #weights_only=False, weil der Checkpoint die Config mitbringt.
        #Explizit gesetzt, damit torch nicht wegen des Defaults warnt.
        state = torch.load(pfad, map_location="cpu", weights_only=False)

        schluessel = "ema" if use_ema else "model"
        if schluessel not in state:
            raise KeyError(
                f"Checkpoint hat keinen Eintrag '{schluessel}' (vorhanden: "
                f"{sorted(state)}). Mit use_ema={not use_ema} waere es "
                f"'{'model' if use_ema else 'ema'}'."
            )

        #Nur Felder uebernehmen, die Config heute kennt, sonst bricht ei aelterer Checkpoint ab, sobald die Dataclass ein Feld gewinnt oder verliert
        #Fehlende Felder laufen auf ihren Default.
        erlaubt = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in state["cfg"].items() if k in erlaubt})
        cfg = replace(cfg, device=device or Config().device)

        #ESM wird genau einmal geladen: der Wrapper braucht es fuer decode(), der Denoiser bekommt Kopien seiner Attention-Layer
        #(trainable_esm_layers -- der Encoder bleibt eingefroren, sonst verschiebt sich der Raum, in den diffundiert wurde).
        esm = ESMWrapper(cfg)
        denoiser = build_denoiser(cfg, esm.model)
        diffusion = DiffusionProcess(denoiser, cfg).to(cfg.device)
        mean, std = load_standards(cfg) if standardize else (None, None)

        try:
            denoiser.load_state_dict(state[schluessel], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Gewichte aus '{schluessel}' passen nicht zu dem Denoiser, "
                f"den die Config beschreibt. Die Config kommt aus dem "
                f"Checkpoint, es liegt also nicht an einem Default -- "
                f"haeufigste Ursache ist ein Lauf, der mit einem ANDEREN "
                f"Netz trainiert wurde als build_denoiser baut.\n{e}"
            ) from e

        return cls(
            diffusion, esm, cfg, mean, std,
            herkunft={
                "quelle": f"{pfad.parent.name}/{pfad.name}",
                "use_ema": use_ema,
                "epoch": state.get("epoch"),
                "best_val": state.get("best_val"),
            },
        )

    def _to_esm_space(self, x: torch.Tensor):
        """Modellraum -> ESM-Raum. Ohne Standardisierung die Identitaet.

        Umkehrung der Standardisierung aus data.py (emb.sub_(mean).div_(std)).
        Muss vor ESMWrapper.decode() laufen: der lm_head erwartet rohe ESM-Repraesentationen, und deren Streuung liegt pro Dimension
        deutlich unter 1 -- nicht bei 1 wie im Modellraum.

        Padding-Positionen wandern dabei von 0 auf mean. Das ist egal,
        decode() schneidet auf lengths ab.

        Args:
            x: (B, L, D) wie aus DiffusionProcess.sample().

        Returns:
            (B, L, D) im ESM-Raum.
        """
        if self.mean is None:
            return x
        return x * self.std.to(x) + self.mean.to(x)              # (B, L, D)

    #Laengen

    def _resolve_lengths(self, lengths, n: int | None):
        """Laengenangabe auf (N,) normieren.

        Args:
            lengths: ein int (gleiche Laenge fuer alle, dann ist n Pflicht) oder eine Liste/ein Tensor mit einer Laenge pro Sequenz.
            n: Anzahl Sequenzen, nur bei int-Angabe gebraucht.

        Returns:
            (N,) long.
        """
        if isinstance(lengths, int):
            if n is None:
                raise ValueError("Bei einer einzelnen Laenge muss n angegeben werden.")
            out = torch.full((n,), lengths, dtype=torch.long)
        else:
            out = torch.as_tensor(lengths, dtype=torch.long).flatten()
            if n is not None and out.numel() != n:
                raise ValueError(
                    f"{out.numel()} Laengen, aber n={n}. Entweder n weglassen "
                    "oder genau n Laengen uebergeben."
                )

        lo, hi = self.cfg.pep_min_len, self.cfg.pep_max_len
        if out.numel() == 0:
            raise ValueError("Keine Laengen uebergeben.")
        if (out < 1).any():
            raise ValueError("Laengen < 1 sind nicht generierbar.")
        if (out > hi).any():
            raise ValueError(f"Groesste Laenge {int(out.max())} > pep_max_len {hi}.")
        if (out < lo).any():
            print(f"[Sampler] Hinweis: {int((out < lo).sum())} Laengen unter "
                  f"pep_min_len {lo} -- biologisch fragwuerdig, wird trotzdem generiert.")
        return out

    def draw_lengths(self, n: int, *, source: str = "train", seed: int | None = None):
        """Laengen ziehen.

        Die Laengenverteilung bestimmt mit, wie die generierten Sequenzen aussehen
        -> wer uniform zieht, vergleicht hinterher nicht mehr sauber gegen die Trainingsverteilung oder gegen Torres.

        Args:
            n: Anzahl Laengen.
            source: "train" | "val" | "test" -> empirisch aus den gecachten
                Embeddings des Splits (richtig fuer Vergleiche).
                "uniform" -> gleichverteilt in [pep_min_len, pep_max_len].
            seed: None -> cfg.seed.

        Returns:
            (n,) long.
        """
        g = torch.Generator().manual_seed(self.cfg.seed if seed is None else seed)

        if source == "uniform":
            lo, hi = self.cfg.pep_min_len, self.cfg.pep_max_len
            return torch.randint(lo, hi + 1, (n,), generator=g, dtype=torch.long)

        if source not in {"train", "val", "test"}:
            raise ValueError(f"unbekannte source {source!r}, erlaubt 'train' | "
                             "'val' | 'test' | 'uniform'")

        #Gleicher Pfad wie Data_coordinator und 05_build_embeddings.py
        cache = CACHE_DIR / f"{source}-Set_{self.cfg.esm_model_name}_L{self.cfg.pep_max_len}.pt"
        if not cache.exists():
            raise FileNotFoundError(
                f"Cache nicht gefunden: {cache}\n"
                "Entweder scripts/05_build_embeddings.py laufen lassen oder "
                "source='uniform' nutzen."
            )
        pool = torch.as_tensor(
            torch.load(cache, map_location="cpu", weights_only=True)["lengths"],
            dtype=torch.long,
        ).flatten()
        return pool[torch.randint(0, pool.numel(), (n,), generator=g)]

    #Generieren

    @torch.no_grad()
    def sample_embeddings(
            self,
            lengths,
            n: int | None = None,
            *,
            batch_size: int | None = None,
            progress: bool = True,
            seed: int | None = None,
            to_esm_space: bool = True,
    ):
        """Embeddings generieren.

        Laeuft in Batches: die volle Reverse-Kette fuer viele Sequenzen passt nicht in einem Rutsch in den Speicher.

        Args:
            lengths: Laengen (siehe _resolve_lengths) oder draw_lengths().
            n: Anzahl, nur bei einer einzelnen int-Laenge noetig.
            batch_size: None -> cfg.batch_size.
            progress: Fortschrittsbalken pro Batch.
            seed: setzt den GLOBALEN torch-RNG -- DiffusionProcess.sample zieht mit torch.randn
                ohne eigenen Generator. None -> RNG nicht anfassen.
            to_esm_space: Standardisierung zurueckrechnen. True ist fuer decode() Pflicht.
                False gibt den Modellraum zurueck, in dem das Netz gelernt hat -> nur dort sind generierte und echte Embeddings
                direkt vergleichbar. Ohne Standardisierung sind beide Raeume dasselbe.

        Returns:
            (emb, lengths): (N, L, D) float32 auf CPU und (N,) long.
        """
        lens = self._resolve_lengths(lengths, n)                 # (N,)
        bs = self.cfg.batch_size if batch_size is None else batch_size
        if bs < 1:
            raise ValueError(f"batch_size muss >= 1 sein, ist {bs}")

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.diffusion.eval()

        out = []
        for start in range(0, lens.numel(), bs):
            chunk = lens[start:start + bs].to(self.device)        # (b,)
            padding_mask = masker(chunk, pep_max_len=self.cfg.pep_max_len)   # (b, L)

            x = self.diffusion.sample(padding_mask, progress=progress)       # (b, L, D)
            if to_esm_space:
                x = self._to_esm_space(x)                         # (b, L, D)
            out.append(x.float().cpu())

        emb = torch.cat(out, dim=0)                               # (N, L, D)

        #Plausibilitaet der Standardisierung
        #Der Checkpoint sagt nicht, ob der Lauf standardisiert war -> das Ergebnis sagt es: im Modellraum ist das Trainingsziel Streuung 1.
        #Gemessen wird nur ueber echte Residuen; ueber den ganzen Tensor wuerde das Padding die Streuung Richtung 0 ziehen.
        if self.mean is not None:
            modellraum = (emb - self.mean) / self.std if to_esm_space else emb
            valid = masker(lens, valid_mask=True, pep_max_len=self.cfg.pep_max_len)
            streuung = float(modellraum[valid].std())
            if not 0.5 <= streuung <= 2.0:
                print(
                    f"[Sampler] Warnung: Streuung im Modellraum ist "
                    f"{streuung:.3f}, erwartet ungefaehr 1. Moegliche Ursachen: "
                    f"der Lauf wurde mit --no-standardize trainiert (dann "
                    f"standardize=False uebergeben), das Netz ist kollabiert "
                    f"oder untrainiert, oder "
                    f"{standards_path(self.cfg).name} gehoert nicht zu "
                    f"diesem Lauf."
                )

        return emb, lens                                          # (N, L, D), (N,)

    @torch.no_grad()
    def decode_embeddings(
            self,
            emb: torch.Tensor,
            lengths,
            *,
            names: list[str] | None = None,
            batch_size: int | None = None
    ):
        """Embeddings im ESM-Raum -> DataFrame mit name, length, sequence.

        Auch der lm_head laeuft in Batches: er bildet (B, L, D) auf
        (B, L, 33) ab, das wird fuer viele Sequenzen sonst unnoetig gross.

        Args:
            emb: (N, L, D) im ESM-Raum, nicht im Modellraum.
            lengths: (N,) oder Liste.
            names: Bezeichner, muessen eindeutig sein. None -> 'gen_0000' ...
            batch_size: None -> cfg.batch_size.
        """
        lens_list = torch.as_tensor(lengths, dtype=torch.long).flatten().tolist()
        if emb.shape[0] != len(lens_list):
            raise ValueError(f"{emb.shape[0]} Embeddings, aber {len(lens_list)} Laengen.")

        if names is None:
            names = [f"gen_{i:04d}" for i in range(len(lens_list))]
        elif len(names) != len(lens_list):
            raise ValueError(f"{len(names)} Namen, aber {len(lens_list)} Sequenzen.")
        elif len(set(names)) != len(names):
            #decode() gibt ein Dict zurueck -- doppelte Namen fielen still zusammen
            raise ValueError("Namen muessen eindeutig sein, decode() liefert ein Dict.")

        bs = self.cfg.batch_size if batch_size is None else batch_size
        seqs: list[str] = []
        for start in range(0, len(lens_list), bs):
            stop = start + bs
            teil = self.esm.decode(
                emb[start:stop], lens_list[start:stop], names=names[start:stop]
            )
            seqs.extend(teil[k] for k in names[start:stop])

        df = pd.DataFrame({"name": names, "length": lens_list, "sequence": seqs})

        #Guardian: decode() schneidet auf length -- passt das nicht, sind
        #lengths und Padding-Maske auseinandergelaufen.
        bad = df["sequence"].str.len() != df["length"]
        if bad.any():
            raise RuntimeError(
                f"{int(bad.sum())} Sequenzen haben nicht die angeforderte Laenge. "
                "lengths und padding_mask passen nicht zusammen."
            )
        return df

    @torch.no_grad()
    def sample(
            self,
            lengths,
            n: int | None = None,
            *,
            batch_size: int | None = None,
            names: list[str] | None = None,
            progress: bool = True,
            seed: int | None = None
    ) -> pd.DataFrame:
        """Sequenzen generieren -- die ganze Kette.

        Args:
            lengths: Laengen (siehe _resolve_lengths) oder draw_lengths().
            n: Anzahl, nur bei einer einzelnen int-Laenge noetig.
            batch_size: None -> cfg.batch_size.
            names: Bezeichner. None -> 'gen_0000' ...
            progress: Fortschrittsbalken.
            seed: reproduzierbare Ziehung (globaler RNG).

        Returns:
            DataFrame mit Spalten name, length, sequence.
        """
        emb, lens = self.sample_embeddings(
            lengths, n, batch_size=batch_size, progress=progress, seed=seed
        )
        return self.decode_embeddings(emb, lens, names=names, batch_size=batch_size)

    def to_csv(self, df: pd.DataFrame, path) -> Path:
        """DataFrame wegschreiben. Elternordner wird angelegt."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
