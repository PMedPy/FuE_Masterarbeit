23-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# SSE, Normalverteilung & Entropie — der gemeinsame Kern

> Warum sitzt dieselbe Gauß-Glocke hinter der Methode der kleinsten Quadrate, der Normal-Likelihood und dem Maximum-Entropie-Prinzip? Drei Wege, ein Objekt.

---

## Die drei Aussagen in einem Satz

Die **kleinsten Quadrate** sind das, was man **rechnet**, wenn man eine **Normal-Likelihood maximiert**, und die Normalverteilung ist die Verteilung, die man **wählen sollte**, wenn man sich ehrlich nur auf **Mittelwert und Varianz** festlegt — denn sie ist deren **Maximum-Entropie-Verteilung**.

---

## 1. Normal-Likelihood $\Rightarrow$ Sum of Squared Errors

Die Log-Likelihood eines normalverteilten Fehlers ist eine **negative Quadratsumme**:

$$ \log p(y \mid \mu, \sigma) = -\frac{1}{2\sigma^2}\sum_i (y_i - \mu_i)^2 ;-; n\log\sigma ;+; \text{const}. $$

Bei festem $\sigma$ hängt alles nur am $\sum (y_i-\mu_i)^2$. **Die Likelihood maximieren = die Quadratsumme minimieren.** Das ist exakt OLS / Least Squares (Gauß, 1809).

**Als Sprache gelesen:** Das $(\cdot)^2$ ist ein _Energie-/Fehlerterm_ (wie $\lVert\cdot\rVert^2$). Quadratisch heißt: ein Punkt, der doppelt so weit weg liegt, kostet _viermal_ so viel — große Abweichungen werden überproportional bestraft. Genau diese harte Bestrafung ist auch der Grund, warum Least Squares **nicht robust** gegen Ausreißer ist (schwere Ränder bräuchten eine Student-t-Likelihood).

---

## 2. Warum _die Normal_? Maximum-Entropie

Die Wahl der Quadratsumme ist keine Willkür, sondern folgt aus einem Prinzip. Unter **allen** Verteilungen mit gegebenem Mittel $\mu$ und gegebener Varianz $\sigma^2$ ist die Normalverteilung die mit der **größten Entropie** — der am wenigsten festlegenden, "flachsten" Verteilung, die noch zu diesen zwei Momenten passt.

**Herleitung (Lagrange).** Maximiere $H[p] = -\int p\ln p,dx$ unter $\int p =1$, $\int x,p=\mu$, $\int (x-\mu)^2 p=\sigma^2$:

$$ -\ln p(x) - 1 - \lambda_0 - \lambda_1 x - \lambda_2 (x-\mu)^2 = 0 ;;\Rightarrow;; p(x) \propto \exp!\big(-\lambda_2 (x-\mu)^2\big) = \mathcal{N}(\mu,\sigma^2). $$

Die Glocke **fällt direkt heraus**. Die _eine_ quadratische Nebenbedingung (Varianz) ist genau das, was die flache Linie zu einer einzelnen, symmetrischen Glocke biegt.

**Als Sprache gelesen:** Entropie $H = -\mathbb{E}[\ln p]$ ist die _erwartete Überraschung_ / das Maß der Unwissenheit. Der $\ln$ macht aus dem Produkt unabhängiger Überraschungen eine Summe (Additivität von Information). **Maximieren** heißt: "nimm so wenig an wie erlaubt, streue die Wahrscheinlichkeit so gleichmäßig wie möglich". Eine Normal-Likelihood behauptet daher **nicht**, dass die Daten normal _sind_ — sie sagt: "Ich lege mich nur auf Mittel und Varianz fest und schmuggle nichts darüber hinaus ein."

---

## 3. Die Synthese: drei Straßen, eine Glocke

|Weg|Aussage|Stichwort|
|---|---|---|
|**Generativ**|Summe vieler kleiner unabhängiger Fluktuationen (endliche Varianz) $\to$ Normal|CLT|
|**Rechnerisch**|Normal-Log-Likelihood $= -,$Quadratsumme $\Rightarrow$ Maximieren $=$ Least Squares|OLS|
|**Inferentiell**|Normal $=$ Maximum-Entropie bei fixem $(\mu,\sigma^2)$ $\Rightarrow$ annahmeärmste Wahl|MaxEnt|

> **Additive Fehler erzeugen sie, kleinste Quadrate rechnen mit ihr, maximale Entropie rechtfertigt sie.**

---

## 4. Bezug zu Deep Learning

- **MSE-Loss = Gauß-Likelihood.** Ein Netz mit Mean-Squared-Error-Verlust ist _implizit_ ein Maximum-Likelihood-Schätzer unter der Annahme normalverteilter, homoskedastischer Residuen. Der MSE ist kein neutraler "Abstand", sondern eine **probabilistische Festlegung**.
- **Cross-Entropy-Loss = kategoriale Likelihood.** Dieselbe Logik für Klassifikation: Der Cross-Entropy-Verlust ist die negative Log-Likelihood einer Kategorial-/Bernoulli-Verteilung. _Jeder_ gängige Loss ist eine versteckte Verteilungswahl. Das ist der eigentliche Grund, warum Loss-Funktionen "vom Himmel zu fallen scheinen" — sie sind negative Log-Likelihoods.
- **Softmax = Maximum-Entropie.** Die Softmax-Verteilung ist die MaxEnt-Verteilung über Klassen unter linearen Logit-Constraints — dasselbe Lagrange-Argument wie oben, nur diskret.
- **Robuste Losses.** Will man Ausreißer-Toleranz (Huber-Loss, MAE), wechselt man implizit die Likelihood (Student-t, Laplace) — exakt der Tail-Tausch aus der Robustheits-Diskussion.

## 5. Bezug zur Bayes-Statistik

- **Regularisierung = Prior, im Log-Raum addiert.** Ein **L2/Weight-Decay**-Strafterm $-\lambda\lVert\theta\rVert^2$ ist exakt ein **Gauß-Prior** auf die Gewichte; **L1** ist ein **Laplace-Prior**. MAP-Schätzung $=$ "Loss $+$ Strafterm" $=$ "$-\log$ Likelihood $-\log$ Prior".
- **MaxEnt-Prior.** Ein Gauß-Prior ist selbst MaxEnt für einen Parameter mit bekannter Skala — "minimale Annahme, richtig gemacht" führt zu _mildem Schrumpfen_, nicht zum flachen Prior.
- **Exponentialfamilie = MaxEnt-Familie.** Jede MaxEnt-Verteilung unter Momentbedingungen hat die Form $\exp(\sum_k \lambda_k T_k(x))$. Die Normal ist das Mitglied für die sufficient statistics $(x, x^2)$.
- **Loss-Minimierung vs. Posterior.** Reine Loss-Minimierung liefert nur den **Modus** (Punktschätzer); die Bayes-Sicht liefert die **ganze Verteilung** über $\theta$ und damit kalibrierte Unsicherheit. Derselbe Loss, zwei Antwort-Ebenen.

---

## Kernsatz für später

> Ein Loss ist nie nur "Abstand". Jeder Loss ist eine negative Log-Likelihood, also eine **Annahme über die Datenentstehung**; jeder Regularisierer ist ein **Prior**. SSE/MSE ist der Spezialfall "ich lege mich auf Mittel und Varianz fest und sonst nichts" — und genau das macht die Maximum-Entropie-Eigenschaft der Normalverteilung explizit.

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]