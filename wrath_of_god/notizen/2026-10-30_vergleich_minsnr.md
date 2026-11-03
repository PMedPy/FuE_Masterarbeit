# Vergleich: Min-SNR-Gewichtung gegen ungewichtet

Zwei Laeufe, ESM2-8M, D=320, L=50, 18,2 Mio trainierbare Parameter, Batch 16,
15 507 Train- / 1 939 Val-Sequenzen, Tesla V100 auf bioinf-euryale.

| | `train_cluster01_2026-10-28_1310` | `train_min_snr_test_run_2026-10-30_1010` |
|---|---|---|
| `loss_weight_scheme` | `none` | `min_snr`, gamma = 5.0 |
| `n_epochs` | 50 | 100 |
| `loss_reduction` | `sequence` | `sequence` |
| Endstand (gemeldeter Loss) | train 0.1288 / val 0.1410 | train 0.0191 / val 0.0264 |

---

## 0. Datengrundlage dieser Auswertung

Alle Zahlen und die Abbildung stammen aus den CSV-Dateien der beiden
Lauf-Ordner, **nicht** aus den Logs:

```
runs/<lauf>/metrics/loss_per_t.csv    Zeile epoch=49 bzw. epoch=99,
                                      20 Spalten t0 ... t999
runs/<lauf>/metrics/metrics.csv       Spalten val_token_loss, train_token_loss,
                                      train_loss, val_sequence_loss
```

Wie `loss_per_t.csv` zustande kommt (`Trainer.evaluate`):

- **EMA-Modell**, `model.eval()`, unter `torch.no_grad()`
- festes t-Raster: `torch.linspace(0, 999, 20).long()` -> 0, 52, 105, ... 999
- festes Rauschen: `torch.Generator(device="cpu")` mit `cfg.seed`, **pro
  t-Wert zurueckgesetzt** -- jede Zelle sieht ueber alle Epochen und ueber
  beide Laeufe hinweg dasselbe epsilon
- laengenstratifizierte Teilmenge des Val-Satzes: nach Laenge sortiert, jede
  k-te gezogen; n = min(max(0.2 * 1939, 100), 350) = **350 Sequenzen**
- Reduktion hart verdrahtet auf `mode="sequence"`, `weight=None`

Der letzte Punkt ist der entscheidende: **die Messung ist von der
Trainingsgewichtung unabhaengig** und deshalb ueber Laeufe hinweg vergleichbar.

---

## 1. Der faire Vergleich: min_snr gewinnt um 11 %

Der gemeldete Loss (0.0264 gegen 0.1410) ist **nicht** vergleichbar -- die
Zielfunktionen unterscheiden sich. Vergleichbar ist der **Token-Loss**, der in
`train()` und `evaluate()` fest mit `weight=None` berechnet wird:

| Epoche | cluster01 | min_snr | Differenz |
|---|---|---|---|
| 9 | 0.1868 | 0.1835 | -1.8 % |
| 24 | 0.1578 | 0.1439 | -8.8 % |
| **49** | **0.1411** | **0.1256** | **-11.0 %** |
| 99 | -- | 0.1164 | |

Bei gleicher Epochenzahl entrauscht das min_snr-Modell real um 11 % besser.
Der Rest des scheinbaren Vorsprungs war Umgewichtung.

## 2. Der Handel: besser unterhalb von t = 820, schlechter darueber

Siehe `2026-10-30_t_loss_vergleich.png`. Epoche 49, beide ungewichtet gemessen:

| t | cluster01 | min_snr | |
|---|---|---|---|
| 0 | 0.0371 | 0.0079 | **-78.7 %** |
| 105 | 0.0414 | 0.0198 | -52.3 % |
| 210 | 0.0507 | 0.0298 | -41.2 % |
| 473 | 0.0684 | 0.0423 | -38.2 % |
| 630 | 0.0781 | 0.0518 | -33.6 % |
| 736 | 0.1024 | 0.0819 | -20.0 % |
| 788 | 0.1435 | 0.1323 | -7.8 % |
| 841 | 0.2273 | 0.2345 | **+3.2 %** |
| 893 | 0.3535 | 0.3824 | +8.2 % |
| 946 | 0.5179 | 0.5578 | +7.7 % |
| 999 | 0.6983 | 0.7915 | **+13.3 %** |

Vorzeichenwechsel bei t ~ 820. Genau das sagt die Theorie voraus: fuer
`pred_x0` ist w(t) = min(SNR(t), gamma) bei grossem t klein, die Gewichtung
verlagert Gradientenbudget von hohem zu niedrigem t. Hier ist der Effekt
quantifiziert.

**Offen bleibt die Sampling-Frage.** Der Reverse-Pfad startet bei t = T, also
dort, wo das Modell schlechter geworden ist. Ob der Gewinn in den vielen
niedrigen t das aufwiegt, entscheidet keine Loss-Zahl, sondern erst der
Sampler mit den Metriken aus `evaluation.py`.

## 3. min_snr lernt bei hohem t kaum weiter (Epoche 49 -> 99)

| t | E49 | E99 | |
|---|---|---|---|
| 0 | 0.0079 | 0.0038 | -51.5 % |
| 315 | 0.0358 | 0.0292 | -18.4 % |
| 630 | 0.0518 | 0.0411 | -20.8 % |
| 788 | 0.1323 | 0.1185 | -10.4 % |
| 946 | 0.5578 | 0.5385 | -3.5 % |
| 999 | 0.7915 | 0.7511 | -5.1 % |

Der hohe t-Bereich bleibt auch mit doppelter Epochenzahl nahezu unbeweglich --
konsistent mit der Annahme, dass der Fehler dort weitgehend irreduzibel ist.

## 4. Erledigt: die t0-Pruefung war kein Bug

cluster01 lag bei t = 0 auf 0.0371, was nach einem Fehler im Forward-Pfad
aussah. min_snr erreicht dort **0.0038**, eine Zehnerpotenz tiefer. Die
Aufgabe bei t = 0 ist also sehr wohl loesbar; cluster01 hatte nur nie Gewicht
darauf verwendet. Symptom der ungewichteten Zielfunktion, kein Bug.

## 5. Erledigt: die time_mlp-Hypothese bekommt keine Unterstuetzung

Token/Sequence-Verhaeltnis bei cluster01 (beide ungewichtet, also gueltig):

    Epoche 0: 1.091      letzte Epoche: 1.101

Konstant. Der hoehere Fehler langer Peptide ist eine Eigenschaft der Daten,
keine Trainingspathologie. Die Vermutung, dass positionsspezifische
FiLM-Slices bei hohen Indizes verhungern, wuerde einen **wachsenden** Abstand
vorhersagen.

## 6. KORREKTUR: die Kovarianz-Diagnose gilt nur bei gleichem w(t)

Bei min_snr betraegt dasselbe Verhaeltnis 1.301 -> **6.340**. Das ist keine
Kovarianz.

Die Herleitung

    L_tok - L_seq = (B / sum(L)) * Cov(L_b, e_b)

setzt voraus, dass sich beide Reduktionen **nur im Gewicht omega_b**
unterscheiden. In `Trainer.train()` ist der Sequence-Loss aber gewichtet und
der Token-Loss nicht:

```python
loss     = reduce_loss(loss_el, p_mask, weight=self.Scheduler_train.loss_weight(t=t), mode=cfg.loss_reduction)
tok_loss = reduce_loss(loss_el, p_mask, weight=None, mode="token")
```

Die Differenz vermischt damit zwei Effekte, und 6.34 ist schlicht der
Umgewichtungsfaktor.

**Fix: drei Groessen loggen statt zwei.**

| Groesse | wofuer |
|---|---|
| sequence, **gewichtet** | das, was optimiert wird |
| sequence, **ungewichtet** | Vergleich zwischen Laeufen |
| token, **ungewichtet** | Kovarianz-Diagnose gegen die Zeile darueber |

Die mittlere fehlt bisher. Zwei Zeilen in `train()` und `evaluate()`, plus
zwei Spalten in `_append_metrics`.

---

## Naechste Schritte

1. Drei-Groessen-Logging einbauen (Punkt 6)
2. Heatmap-Varianten: Log-Farbskala und Differenz-Heatmap gegen Epoche 0
3. **Referenzlauf `loss_weight_scheme="none"` ueber 100 Epochen** -- das Paar
   ist bisher nur bei Epoche 49 sauber gematcht
4. `sampler.py` fertigstellen; erst damit laesst sich die Sampling-Frage aus
   Punkt 2 beantworten
5. Drittes Experiment: `time_cond="global"` gegen `per_position`
