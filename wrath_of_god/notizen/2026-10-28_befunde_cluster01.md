# Befunde aus Lauf `train_cluster01_2026-10-28_1310`

Erster Trainingslauf auf dem THM-Cluster (bioinf-euryale, Tesla V100-PCIE-32GB),
ESM2-8M, D=320, L=50, 18,2 Mio trainierbare Parameter, 15 507 Train- / 1 939
Val-Sequenzen, Batch 16, 50 Epochen in ca. 23 Minuten.

Konfiguration: `objective=pred_x0`, `loss_type=l2`, `loss_reduction=sequence`,
`loss_weight_scheme=none`, `beta_schedule=cosine`, `time_cond=per_position`,
`timesteps=1000`, `lr=1e-4`, `ema_decay=0.995`.

Endstand: train 0.1288, val 0.1410, best_val 0.1402.

---

## 1. Der Lauf ist nicht auskonvergiert

Beide Kurven fallen bis zur letzten Epoche. Die Generalisierungslücke wächst
langsam und ist unkritisch:

| Epoche | train | val | Abstand |
|---|---|---|---|
| 20 | 0.1609 | 0.1620 | -0.001 |
| 35 | 0.1374 | 0.1474 | 0.010 |
| 49 | 0.1288 | 0.1410 | 0.012 |

Dass der Val-Loss in den ersten ~15 Epochen unter dem Train-Loss liegt, ist
kein Fehler: der Train-Loss mittelt ueber die **ganze** Epoche inklusive der
schlechten Gewichte am Anfang, der Val-Loss wird am **Ende** gemessen.

**Konsequenz:** laenger trainieren ist begruendbar. Reproduktionsanker fuer
Torres sind 100 000 Schritte bei Batch 16, bei 15 507 Sequenzen also

    100000 * 16 / 15507 ~= 103 Epochen

## 2. Token- vs. Sequence-Loss misst eine Kovarianz

Beide Modi sind gewichtete Mittel ueber dieselben Pro-Sequenz-Fehler e_b und
unterscheiden sich nur im Gewicht. Die Differenz ist bis auf einen positiven
Faktor die Kovarianz zwischen Sequenzlaenge und mittlerem Fehler:

    L_tok - L_seq = sum_b e_b * (L_b/sum(L) - 1/B)
                  = (B / sum(L)) * Cov(L_b, e_b)

Der Term (L_b/sum(L) - 1/B) ist positiv fuer ueberdurchschnittlich lange und
negativ fuer kurze Sequenzen. Das Vorzeichen der Differenz sagt also direkt,
ob lange Peptide systematisch hoehere Fehler haben.

**Beobachtung:** Token liegt ueber Sequence (Epoche 0: 0.620 vs 0.570;
Epoche 49: ~0.140 vs 0.129). Lange Peptide haben demnach den hoeheren Fehler
pro Zelle.

**Einschraenkung, die eine Fehlinterpretation verhindert:** der Abstand besteht
schon bei Epoche 0, also beim untrainierten Netz. Das Verhaeltnis ist bei
Epoche 0 etwa 1.088 und am Ende etwa 1.087 -- praktisch konstant. Das spricht
fuer eine Eigenschaft der **Daten** (laengere Peptide haben groessere
Embedding-Betraege pro Position), nicht fuer ein Trainingsproblem. Die
time_mlp-Hypothese (positionsspezifische FiLM-Slices bei hohen Indizes
verhungern) wuerde einen **wachsenden** Abstand vorhersagen.

TODO: Verhaeltnis aus `metrics/metrics.csv` exakt nachrechnen, bevor es in die
Ausarbeitung geht -- die Zahlen oben sind aus der Grafik abgelesen.

## 3. Hauptbefund: das Gradientenbudget verpufft bei hohem t

Aus `plots/t_heatmap.png`:

- Bis t ~ 600 liegt der Loss flach bei etwa 0.1.
- Ab t ~ 700 steigt er steil, jenseits von 900 auf 0.6-0.7.
- **Ueber die Epochen veraendert sich der Bereich t > 800 praktisch nicht**
  (von Epoche 2 bis 49 unveraendert dunkel).

Bei `pred_x0` ist das im hohen t-Bereich erwartbar -- dort ist kaum
Information ueber x_0 uebrig, der Fehler ist weitgehend irreduzibel.

Ueberschlag des Beitrags zum Gesamtloss (t wird uniform gezogen):

    0.2 * 0.6 + 0.8 * 0.1 = 0.12 + 0.08 = 0.20

**Rund 60 % des Gesamtlosses und damit des Gradientenbudgets fliessen in einen
Bereich, der sich ueber 50 Epochen nicht verbessert hat.** Gemessen, nicht
vermutet.

## 4. Daraus folgt das erste Experiment: Min-SNR

Fuer `pred_x0` lautet die Min-SNR-Gewichtung (Hang et al. 2023)

    w(t) = min(SNR(t), gamma),    gamma ~ 5

Bei grossem t ist die SNR klein, also ist w dort klein: die Gewichtung daempft
genau den Bereich, der in der Heatmap unbeweglich ist, und deckelt zugleich die
trivialen kleinen t. Kein Umbau noetig, nur ein Flag:

    python scripts/06_train.py --name minsnr01 --esm 8M --epochs 103 --loss-weight min_snr

**Offene Frage, die das Experiment beantwortet:** beim Sampling startet der
Reverse-Pfad bei t = T. Wird das Modell dort noch weniger trainiert, koennten
die ersten Entrauschungsschritte schlechter werden. Hang et al. finden, dass es
trotzdem hilft.

**Wichtig fuer den Vergleich:** zwei Laeufe mit verschiedenem
`loss_weight_scheme` haben verschiedene Zielfunktionen. Der Loss ist dann
**nicht** vergleichbar. Verglichen wird ueber Sampling-Metriken, oder beide
Modelle werden mit demselben ungewichteten Loss auf dem Val-Set evaluiert.

## 5. Darstellungsproblem der Heatmap

Die Farbskala laeuft von 0.05 bis 0.72, der Bereich t < 600 bewegt sich
zwischen 0.09 und 0.13 -- rund 3 % der Skala und damit unsichtbar. Der
Gesamtloss ist von 0.16 auf 0.13 gefallen, in der Grafik sieht man davon
nichts.

Zwei Abhilfen, beide einzeilig in `summarize_eval`:

- **Log-Farbskala:** `norm=matplotlib.colors.LogNorm()` im `imshow`-Aufruf.
  Zeigt relative statt absoluter Aenderungen.
- **Differenz-Heatmap:** `matrix - matrix[0]` statt `matrix` plotten, also die
  Veraenderung gegenueber Epoche 0. Dafuer ist Braun-zu-Blau genau richtig --
  echtes Diverging mit sinnvoller Null in der Mitte (Braun = schlechter,
  Blau = besser).

Beide einbauen: absolute Heatmap mit Log-Skala fuer die Struktur ueber t, die
Differenz-Heatmap fuer den Lernfortschritt.

## 6. Offene Plausibilitaetspruefung

Spalte `t0` in `metrics/loss_per_t.csv` pruefen. Bei `pred_x0` ist x_t ~ x_0 bei
t = 0, das Netz muesste dort fast perfekt sein, also ein Loss nahe 0. Liegt der
Wert bei 0.05 statt 0.005, lohnt ein zweiter Blick: entweder ist alpha_bar_0
nicht exakt 1, oder die triviale Aufgabe bekommt zu wenig Gewicht.

---

## Naechste Schritte

1. `t0`-Spalte pruefen, Token/Sequence-Verhaeltnis aus der CSV nachrechnen
2. Heatmap-Varianten (Log + Differenz) in `trainer.py` einbauen
3. Referenzlauf mit 103 Epochen, `loss_weight_scheme=none`
4. Experiment `minsnr01` mit 103 Epochen, Vergleich ueber Sampling-Metriken
5. Weiteres Experiment: `time_cond=global` gegen `per_position`
   (94 % der eigenen Parameter sitzen im positionsspezifischen FiLM)
