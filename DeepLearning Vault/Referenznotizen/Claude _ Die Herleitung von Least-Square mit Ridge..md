19-08-2026
Tags: #FuE #MachineLearning 
Status: #unextended



## Schritt 1 — MLE unter Gauß-Rauschen ergibt Least Squares

Annahme: Jede Beobachtung ist Modell plus Gauß-Rauschen, $y_i = x_i^T\beta + \varepsilon_i$ mit $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Die Likelihood einer Beobachtung ist dann die Gauß-Dichte:

$$p(y_i \mid \beta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp!\left(-\frac{(y_i - x_i^T\beta)^2}{2\sigma^2}\right)$$

```
p(y_i \mid \beta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^T\beta)^2}{2\sigma^2}\right)
```

Alle Beobachtungen zusammen (unabhängig → Produkt), dann log (Produkt → Summe):

$$\log p(y \mid \beta) = -\frac{1}{2\sigma^2} \sum_i (y_i - x_i^T\beta)^2 + \text{const}$$

```
\log p(y \mid \beta) = -\frac{1}{2\sigma^2} \sum_i (y_i - x_i^T\beta)^2 + \text{const}
```

Das $\exp$ verschwindet durch den log, das Quadrat aus dem Exponenten bleibt. Jetzt **maximieren** nach $\beta$: Die Konstante und der Vorfaktor $-\frac{1}{2\sigma^2}$ ändern das Maximum nicht. Maximieren eines negativen Terms = Minimieren des positiven:

$$\hat\beta_{MLE} = \arg\min_\beta \sum_i (y_i - x_i^T\beta)^2 = \arg\min_\beta |y - X\beta|^2$$

```
\hat\beta_{MLE} = \arg\min_\beta \|y - X\beta\|^2
```

Das ist **Least Squares** — die Normalengleichung. Gauß-Rauschen im MLE _ist_ Kleinste-Quadrate. Der Grund, warum ausgerechnet die _quadrierte_ Abweichung minimiert wird, steht im Exponenten der Gauß-Dichte: Das Quadrat kommt von dort.

## Schritt 2 — MAP fügt den Prior hinzu, der Strafterm fällt heraus

MAP addiert $\log p(\beta)$. Setz einen **Gauß-Prior** an: $\beta \sim \mathcal{N}(0, \tau^2)$ — „ich glaube a priori, die Koeffizienten sind klein, normalverteilt um null". Sein Logarithmus:

$$\log p(\beta) = -\frac{1}{2\tau^2}|\beta|^2 + \text{const}$$

```
\log p(\beta) = -\frac{1}{2\tau^2}\|\beta\|^2 + \text{const}
```

Dieselbe Gauß-Form, jetzt zentriert um 0, mit $|\beta|^2$ im Exponenten. MAP maximiert log-Likelihood **plus** log-Prior:

$$\hat\beta_{MAP} = \arg\min_\beta \Big[ \underbrace{|y - X\beta|^2}_{\text{aus der Likelihood}} + \underbrace{\frac{\sigma^2}{\tau^2}|\beta|^2}_{\text{aus dem Prior}} \Big]$$

```
\hat\beta_{MAP} = \arg\min_\beta \left[ \|y - X\beta\|^2 + \frac{\sigma^2}{\tau^2}\|\beta\|^2 \right]
```

Das ist **exakt Ridge**, mit $\lambda = \sigma^2/\tau^2$. Der Strafterm $\lambda|\beta|^2$, den du in Lektion 33 als „bestrafe große Koeffizienten" kennengelernt hast, ist wörtlich der **log eines Gauß-Priors**. Er ist in die quadrierte Klammer gefallen, genau wie du sagtest.

Und die Interpretation von $\lambda = \sigma^2/\tau^2$ ist wunderschön:

- **kleiner Prior-Varianz $\tau^2$** (starke Überzeugung „β ist klein") → großes $\lambda$ → starke Regularisierung
- **großer Prior-Varianz $\tau^2$** (vage Überzeugung) → kleines $\lambda$ → schwache Regularisierung
- $\tau^2 \to \infty$ (kein Vorwissen, „flacher Prior") → $\lambda \to 0$ → zurück zu MLE/OLS

Der letzte Punkt schließt den Kreis: **MLE ist MAP mit unendlich vagem Prior.** Kein Vorwissen = flacher Prior = kein Strafterm = reines Least Squares.

## Die eine Zeile zum Mitnehmen

$$\underbrace{\text{Ridge}}_{\text{ML-Sprache}} = \underbrace{\text{MAP mit Gauß-Prior}}_{\text{Bayes-Sprache}}, \qquad \lambda = \frac{\sigma^2}{\tau^2}$$

Der Regularisierungsparameter $\lambda$ ist das Verhältnis von Rausch-Varianz zu Prior-Varianz. „Regularisierung stärke" und „wie stark glaube ich an kleine Koeffizienten" sind dieselbe Zahl.

---

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]