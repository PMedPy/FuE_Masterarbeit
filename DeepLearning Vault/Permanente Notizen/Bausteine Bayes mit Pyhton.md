01-07-2026
Tags: #FuE #MachineLearning 
Status: #unextended

Tags: #Python #FuE #Bayes Status: #unextended

# Das PyMC-Ökosystem – Landkarte

> Die drei Ebenen unter einem Bayesianischen Modell in Python: **welches Werkzeug welche Frage beantwortet.** Bezug: PyMC-Kurs Sitzung 1. Umgebung: `lern`.

---

## Die drei Ebenen auf einen Blick

Ein Bayesianisches Modell in Python durchläuft drei Werkzeugebenen. Jede beantwortet eine andere Art von Frage:

|Ebene|Bibliothek|Beantwortet die Frage|Objekt / Ergebnis|
|---|---|---|---|
|**Fundament**|`scipy.stats`|„Wie verhält sich _eine_ Verteilung?"|Verteilungsobjekt|
|**Modellbau**|`pymc`|„Wie hängen Parameter und Daten zusammen, und was ist der Posterior?"|`InferenceData`|
|**Auswertung**|`arviz`|„Hat das Sampling funktioniert, und was sagt der Posterior?"|Diagnostik + Plots|

Der rote Faden: `scipy.stats` liefert die **Bausteine** (einzelne Verteilungen), `pymc` **verknüpft** sie zu einem Modell und zieht den Posterior, `arviz` **prüft und interpretiert** das Ergebnis. Man steigt beim Arbeiten von oben nach unten _durch_ – aber verstehen muss man von unten nach oben.

---

## Ebene 1 – scipy.stats: das Vokabular

`scipy.stats` kennt kein Bayes, keine Priors, keinen Posterior. Es ist reine **Wahrscheinlichkeitsrechnung**: Es beschreibt einzelne Verteilungen und beantwortet Fragen über sie.

Eine Verteilung ist ein **Objekt** (frozen distribution), das man mit festen Parametern erzeugt und dann befragt:

```python
from scipy import stats
d = stats.norm(100, 15)   # frozen: Parameter fixiert
```

Die vier Kernfragen an jedes Verteilungsobjekt:

|Methode|Frage|Richtung|
|---|---|---|
|`.pdf(x)` / `.pmf(k)`|„Wie dicht / wahrscheinlich ist die Stelle x?"|Parameter → Dichte|
|`.cdf(x)`|„Wieviel Masse liegt unter x?"|Wert → kumulierte W.|
|`.ppf(q)`|„Unter welcher Stelle liegen q% der Masse?"|W. → Wert|
|`.rvs(size)`|„Zieh Stichproben."|Verteilung → Daten|

Zwei Merkpunkte:

- **Dichte ist keine Wahrscheinlichkeit.** Bei stetigen Verteilungen kann `.pdf(x)` größer als 1 sein. Erst das Integral wird Wahrscheinlichkeit: $\text{cdf}(x) = \int_{-\infty}^{x} \text{pdf}(t), dt$.
- **Diskret vs. stetig:** Bei diskreten Verteilungen heißt die Dichte-Methode `.pmf` (mass statt density), und dort _ist_ `.pmf(k)` eine echte Wahrscheinlichkeit.

`.ppf` ist die Umkehrung der `.cdf`: $\text{ppf}(q) = \text{cdf}^{-1}(q)$. Sie ist der Motor hinter Quantilen und Intervallgrenzen.

> [!info] Warum das das Fundament ist Wenn PyMC eine Likelihood auswertet, fragt es im Kern nach einer `.pdf` / `.pmf` an einer Stelle. `scipy.stats` ist die Sprache, in der „Dichte an einer Stelle" überhaupt formulierbar ist. Ohne diese Ebene ist PyMC eine Blackbox.

---

## Ebene 2 – PyMC: der Modellbau

PyMC nimmt die Verteilungen aus Ebene 1 und **verknüpft** sie zu einem generativen Modell. Es fügt drei Dinge hinzu, die scipy nicht hat: den Begriff **Prior vs. Likelihood**, das **Zusammensetzen** mehrerer Verteilungen, und einen **Sampler**, der den Posterior zieht.

Die immer gleiche Anatomie:

```python
import pymc as pm

with pm.Model() as modell:
    theta = pm.Beta("theta", alpha=1, beta=1)              # Prior
    k     = pm.Binomial("k", n=10, p=theta, observed=7)    # Likelihood (observed!)
    idata = pm.sample(2000, tune=1000, chains=4)           # Sampling → Posterior
```

Der entscheidende Unterschied zu scipy:

| scypi.stats | PyMC |
| ----------- | ---- |
|             |      |

||scipy.stats|PyMC|
|---|---|---|
|Verteilung|einzeln, isoliertverknüpft im `with`-Block|
|Parameter|feste Zahlen|können selbst Verteilungen sein|
|`observed`|gibt es nicht|macht aus Prior eine Likelihood|
|Ergebnis|Zahlen auf Abruf|Posterior-Samples (`InferenceData`)|

Das Schlüsselwort **`observed=`** ist die Nahtstelle: Ohne es ist eine Verteilung ein unbeobachteter Parameter (Prior). Mit `observed` wird sie an Daten gebunden (Likelihood). Der Sampler (NUTS) wertet die kombinierte Log-Dichte $\log p(\theta) + \log p(\text{Daten} \mid \theta)$ ab und zieht daraus den Posterior; die Evidence kürzt sich dabei heraus.

$$\log \text{score}(\theta) = \log p(\theta) + \log p(\text{Daten} \mid \theta)$$

---

## Ebene 3 – ArviZ: die Auswertung

PyMC gibt ein **`InferenceData`**-Objekt zurück – tausende Posterior-Samples, organisiert in Gruppen (`posterior`, `prior`, `observed_data`, ...). ArviZ ist das Werkzeug, um dieses Objekt zu **prüfen** und zu **interpretieren**.

Zwei Aufgaben:

**1. Diagnostik – hat das Sampling funktioniert?**

|Größe|Bedeutung|Zielwert|
|---|---|---|
|`r_hat`|Konvergenz der Ketten|≈ 1,00 (> 1,01 = Warnung)|
|`ess_bulk` / `ess_tail`|effektive Stichprobengröße|hoch (Hunderte+)|
|Trace-Plot|Kettenverläufe|„fuzzy caterpillars", überlappend|

**2. Interpretation – was sagt der Posterior?**

```python
import arviz as az
az.summary(idata)          # Mittelwert, sd, HDI, r_hat, ess
az.plot_posterior(idata)   # Posterior mit HDI
az.plot_trace(idata)       # Konvergenz visuell
```

> [!warning] API-Drift beachten ArviZ-Argumentnamen wandern zwischen Versionen (z.B. `hdi_prob` → `prob` bzw. `ci_prob` in neueren Versionen). Immer in der **eigenen** Umgebung prüfen. Argumentnamen sind nicht einmal zwischen Funktionen derselben Library garantiert gleich.

ArviZ-Plots geben `matplotlib.axes`-Objekte zurück – man kann sie also mit vorhandenem Matplotlib-Können weiterbearbeiten und als PDF speichern. Hier schließt sich der Kreis zu bestehenden Plot- und Export-Workflows.

---

## Wie die Ebenen zusammenspielen

Ein vollständiger Durchlauf, mit der jeweils zuständigen Ebene markiert:

```python
from scipy import stats     # Ebene 1
import pymc as pm           # Ebene 2
import arviz as az          # Ebene 3

# (Ebene 1: Verteilungen verstehen – hier implizit im Prior-Wissen)

with pm.Model() as modell:                     # Ebene 2: Modellbau
    theta = pm.Beta("theta", alpha=1, beta=1)
    k     = pm.Binomial("k", n=10, p=theta, observed=7)
    idata = pm.sample(2000, tune=1000, chains=4)

az.summary(idata)                              # Ebene 3: Auswertung
az.plot_trace(idata)
```

Kurzform des Datenflusses:

**Verteilungswissen (scipy)** → **Modell + Posterior (PyMC)** → **Diagnostik + Interpretation (ArviZ)**

---

## Einordnung: was NICHT dazugehört

Zur Abgrenzung, damit die Landkarte scharfe Ränder hat:

- **`scipy.optimize`, Gradientenverfahren, Backprop** – gehören zur _Optimierungs_-Welt (z.B. Deep-Learning-Training), nicht zum Sampling-basierten Bayesianischen Workflow. Andere Achse.
- **`numpy`** – liegt _unter_ allem als Array-Maschinerie, ist aber kein Statistik-Werkzeug im engeren Sinn.
- **`pandas` / `matplotlib`** – rahmen den Workflow ein (Daten rein, Bilder raus), sind aber nicht Teil der Inferenz-Kette selbst.

> [!note] Merksatz `scipy.stats` = _eine_ Verteilung befragen. `pymc` = Verteilungen zu einem Modell verknüpfen und Posterior ziehen. `arviz` = Posterior prüfen und deuten. Drei Fragen, drei Werkzeuge.

# Referenzen
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]