10-06-2026
Tags: #Python  #FuE 
Status: #unextended

# PyMC –  (Einstieg)

> Referenz für die ersten PyMC-Modelle. Bezug: Bayes Thema 2. Umgebung: `lern` (Python 3.13, neuere ArviZ-API).

---

## Die Anatomie jedes Modells

Immer dieselben drei Schritte – egal wie viele Parameter:

```python
import pymc as pm
import arviz as az
import numpy as np

with pm.Model() as modell:
    # 1. PRIOR(S)   – Überzeugung über Parameter VOR den Daten
    theta = pm.Beta("theta", alpha=1, beta=1)

    # 2. LIKELIHOOD – wie die Daten vom Parameter abhängen (observed!)
    k = pm.Binomial("k", n=10, p=theta, observed=7)

    # 3. SAMPLING   – NUTS zieht den Posterior
    idata = pm.sample(2000, tune=1000, chains=4)
```

**Merke:** Der `with`-Block ist der Klebstoff. Jede Zeile registriert sich beim Modell; das Modell summiert die Log-Dichten zu `log score(θ) = log prior + log likelihood`. Der Sampler wertet diese Funktion nur ab – die Evidence kürzt sich raus.

---

## Wichtige Verteilungen (Priors)

|Verteilung|Wofür|Wertebereich|PyMC|
|---|---|---|---|
|**Beta**|Wahrscheinlichkeiten, Anteile|[0, 1]|`pm.Beta("p", alpha=1, beta=1)`|
|**Normal**|kontinuierliche Messwerte, Effekte|(−∞, ∞)|`pm.Normal("mu", mu=0, sigma=1)`|
|**HalfNormal**|Standardabweichungen, nur positiv|[0, ∞)|`pm.HalfNormal("sigma", sigma=1)`|
|**LogNormal**|positive Größen über Größenordnungen (Konz., EC50, kcat)|(0, ∞)|`pm.LogNormal("ec50", mu=0, sigma=1)`|
|**Uniform**|flacher Prior mit Grenzen|[a, b]|`pm.Uniform("x", lower=0, upper=1)`|
|**Exponential**|positive Skalen, Wartezeiten|(0, ∞)|`pm.Exponential("lam", lam=1)`|

### Beta-Prior – Formgefühl

- `Beta(1, 1)` → flach (uniform), „keine Ahnung"
- `Beta(2, 2)` → sanft um 0,5 zentriert
- `Beta(0.5, 0.5)` → U-förmig, Masse an den Rändern
- `Beta(a, b)` → Mittelwert = a / (a + b); größere Werte = schmaler/sicherer

---

## Wichtige Likelihoods

|Verteilung|Datentyp|PyMC|
|---|---|---|
|**Binomial**|k Erfolge aus n Versuchen|`pm.Binomial("y", n=n, p=theta, observed=k)`|
|**Bernoulli**|einzelne 0/1-Ausgänge (Array)|`pm.Bernoulli("y", p=theta, observed=daten)`|
|**Normal**|kontinuierliche Messwerte mit Rauschen|`pm.Normal("y", mu=mu, sigma=sigma, observed=daten)`|
|**Poisson**|Zähldaten (Counts, Raten)|`pm.Poisson("y", mu=rate, observed=daten)`|

> **`observed=`** ist das, was eine Verteilung von Prior zu Likelihood macht. Ohne `observed` = unbeobachteter Parameter (Prior). Mit `observed` = an Daten gebunden.

---

## Sampling-Aufruf

```python
idata = pm.sample(
    draws=2000,     # echte Samples PRO Kette (nach dem Einschwingen)
    tune=1000,      # Einschwing-Schritte (werden verworfen)
    chains=4,       # unabhängige Ketten parallel → für R-hat-Diagnostik
    random_seed=42  # Reproduzierbarkeit
)
```

- **draws**: mehr = glatterer Posterior, aber langsamer. 2000 ist ein guter Start.
- **tune**: Sampler justiert hier seine Schrittweite. Verworfen, nicht im Ergebnis.
- **chains**: mehrere Ketten von verschiedenen Startpunkten → erlaubt Konvergenzprüfung.
- Ergebnis `idata` = ArviZ **InferenceData**-Objekt.

---

## Posterior auswerten (ArviZ)

```python
# Numerische Zusammenfassung: Mittelwert, sd, HDI, R-hat, ESS
az.summary(idata)

# Posterior-Verteilung plotten (mit HDI)
az.plot_posterior(idata)

# Trace-Plot: Ketten-Verläufe + Verteilung (Konvergenz prüfen)
az.plot_trace(idata)

# HDI direkt abgreifen
az.hdi(idata, hdi_prob=0.94)   # ACHTUNG API: siehe Hinweis unten
```

### Diagnostik – worauf achten

- **r_hat** ≈ 1,00 → Ketten konvergiert. > 1,01 = Warnsignal.
- **ess_bulk / ess_tail** → effektive Stichprobengröße; hoch ist gut (Hunderte+).
- **Trace-Plot**: die Ketten sollen wie „fuzzy caterpillars" aussehen, überlappend, ohne Trends oder Sprünge.

---

## ⚠️ API-Drift ArviZ (eigene Umgebung beachten!)

In neueren ArviZ-Versionen heißen die Argumente teils anders:

- `az.hdi(...)`: `hdi_prob` → **`prob`**
- `az.summary(...)`: `hdi_prob` → **`ci_prob`**

> Immer in der eigenen Umgebung prüfen – Argumentnamen wandern zwischen Versionen und sind nicht zwischen Funktionen derselben Library übertragbar.

---

## Posterior Predictive (Modellvalidierung, später)

```python
with modell:
    ppc = pm.sample_posterior_predictive(idata)

az.plot_ppc(ppc)   # simulierte vs. echte Daten vergleichen
```

---

## Mini-Mapping zu Thema 1

|Handrechnung (diskret)|PyMC (kontinuierlich)|
|---|---|
|Prior = 2 Zahlen|`pm.Beta(...)` = Dichtekurve|
|Likelihood θ^k (1−θ)^(n−k)|`pm.Binomial(..., observed=k)`|
|Evidence von Hand summiert|NUTS umgeht sie (kürzt sich raus)|
|Posterior = 2 Zahlen|`idata` = tausende Samples → Kurve|

# Referenzen