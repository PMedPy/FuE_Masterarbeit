16-07-2026
Tags: #Python  #FuE 
Status: #unextended


# ArviZ – Cheat Sheet

> Nachschlagewerk für die Bayesianische Auswertung. Bezug: PyMC-Kurs Sitzung 3b. Umgebung: `lern`. **ArviZ gehört nicht zu PyMC.** Es ist backend-agnostisch (PyMC, Stan, NumPyro, emcee) – deshalb die Trennung: PyMC baut und sampelt, ArviZ wertet aus.


---

## Bereich 1 – Zugriff

**Frage:** Wie komme ich an die Samples?

|Funktion|Zweck|Ergebnis|
|---|---|---|
|`idata.groups`|welche Gruppen sind da?|Liste|
|`idata.posterior`|die Samples|`xarray.Dataset`|
|`idata.posterior["theta"]`|ein Parameter|`DataArray`, shape `(chain, draw)`|
|`az.extract(idata)`|chain+draw zu `sample` verschmelzen|`DataArray`, shape `(4000,)`|
|`.values.flatten()`|direkt nach NumPy|`ndarray`|

```python
idata.posterior["theta"].shape        # (4, 1000) = chain × draw
az.extract(idata)                     # sample-Dimension: 4000
idata.posterior["theta"].values.flatten()   # NumPy-Array (4000,)
```

**Die zwei Achsen:**

- `chain` – unabhängige Sampler-Ketten (nur für Diagnostik)
- `draw` – Samples pro Kette (nach dem Einschwingen)

> [!important] Merksatz Ketten **getrennt** halten zum Prüfen (R-hat braucht sie), **vereinen** zum Auswerten (Histogramm ist die Herkunft egal).

**Gruppen im `InferenceData`:**

| Gruppe                           | Inhalt                                    |
| -------------------------------- | ----------------------------------------- |
| `posterior`                      | gezogene Parameter-Samples                |
| `sample_stats`                   | Sampler-Diagnostik (Energie, Divergenzen) |
| `observed_data`                  | die `observed=`-Daten                     |
| `prior` / `posterior_predictive` | nur wenn explizit erzeugt                 |

---

## Bereich 2 – Diagnostik

**Frage:** Hat das Sampling funktioniert?

|Funktion|Kennzahl|Zielwert|
|---|---|---|
|`az.rhat(idata)`|Konvergenz der Ketten|≈ 1.00 (> 1.01 = Warnung)|
|`az.ess(idata)`|effektive Stichprobengröße|hoch (Hunderte+)|
|`az.mcse(idata)`|Monte-Carlo-Fehler|klein rel. zu `sd`|

```python
az.rhat(idata)    # 1.0013
az.ess(idata)     # 1752
az.mcse(idata)    # 0.0031
```

**Warum einzeln, wenn `summary` alles hat?** Für programmatische Prüfung in Pipelines:

```python
if float(az.rhat(idata)["theta"]) > 1.01:
    raise ValueError("Sampler nicht konvergiert")
```

**Visuelle Diagnostik:**

|Plot|Zeigt|
|---|---|
|`az.plot_trace(idata)`|Kettenverläufe + Dichte („fuzzy caterpillars")|
|`az.plot_rank(idata)`|Rangverteilung der Ketten – schärfer als Trace|
|`az.plot_energy(idata)`|NUTS-Effizienz der Sprünge|
|`az.plot_ess(idata)`|ESS über Quantile hinweg|
|`az.plot_autocorr(idata)`|Autokorrelation der Samples|

> [!note] ESS vs. Anzahl Samples `ess_bulk = 1752` bei 4000 Samples ist normal: MCMC-Samples sind **autokorreliert**, aufeinanderfolgende Ziehungen ähneln sich. ESS sagt, wie vielen _unabhängigen_ Samples die Kette entspricht.

**Beziehung mcse ↔ ess:**

$\text{mcse} \approx \dfrac{\text{sd}}{\sqrt{\text{ESS}}}$

Prüfbeispiel: $0.129/\sqrt{1752} = 0.0031$.

**Wichtige Unterscheidung:**

- `sd` = echte Unsicherheit über den Parameter (bleibt, egal wie lang gesampelt wird)
- `mcse` = Rechenungenauigkeit (schrumpft mit mehr draws)

---

## Bereich 3 – Zusammenfassung

**Frage:** Was sagt der Posterior?

```python
az.summary(idata)   # -> pandas DataFrame!
```

Spalten (neuere Versionen):

|Spalte|Bedeutung|
|---|---|
|`mean`, `sd`|Posterior-Mittelwert und -Streuung|
|`eti89_lb` / `eti89_ub`|Intervallgrenzen (89 %, gleichschwänzig)|
|`ess_bulk`|effektive Samples im Zentrum → für mean/sd|
|`ess_tail`|effektive Samples in den Rändern → für Intervallgrenzen|
|`r_hat`|Konvergenz|
|`mcse_mean`, `mcse_sd`|Monte-Carlo-Fehler|

**Die zwei Intervalltypen:**

|Funktion|Typ|Definition|
|---|---|---|
|`az.hdi(idata)`|Highest Density Interval|**kürzestes** Intervall mit der Masse|
|`az.eti(idata)`|Equal-Tailed Interval|gleich viel Masse links/rechts (= Quantile)|

```python
az.hdi(idata)   # theta: [0.481, 0.880]
az.eti(idata)   # theta: [0.452, 0.860]
```

Bei **symmetrischen** Posteriors fast identisch. Bei **schiefen** driften sie auseinander – ETI kann Bereiche enthalten, die weniger plausibel sind als ausgeschlossene.

> [!info] Warum 89 % / 94 %? Bewusst „krumme" Werte, um die 95 %-Konvention zu brechen (McElreath macht denselben Witz). Botschaft: keine magische Schwelle – das Intervall ist Beschreibung, kein Test.

**Plots:**

|Plot|Zweck|
|---|---|
|`az.plot_posterior(idata)`|Dichte mit Intervall|
|`az.plot_forest(idata)`|viele Parameter kompakt untereinander (hierarchische Modelle)|
|`az.plot_pair(idata)`|Parameterkorrelationen aufdecken|
|`az.plot_dist(idata)`|einzelne Verteilung|

---

## Bereich 4 – Modellkritik

**Frage:** Ist das Modell gut? (Später relevant, hier zur Einordnung.)

|Funktion|Zweck|
|---|---|
|`az.loo(idata)`|Leave-One-Out Cross-Validation (Vorhersagegüte ohne Refit)|
|`az.compare({"m1": i1, "m2": i2})`|Modellvergleich|
|`az.plot_compare(...)`|Vergleich visuell|
|`az.plot_ppc_dist(ppc)`|Posterior Predictive Check|
|`az.psense(idata)`|Prior-Sensitivität|
|`az.bayes_factor(idata)`|Bayes-Faktor|

Posterior Predictive erzeugen (PyMC-Seite):

```python
with modell:
    ppc = pm.sample_posterior_predictive(idata)
az.plot_ppc_dist(ppc)
```

---

## Bereich 5 – Überlapp mit anderen Bibliotheken

**Frage:** Was ist direkt integriert, was übersetzt man?

### Das Schichtbild

|Bibliothek|Rolle gegenüber ArviZ|
|---|---|
|**xarray**|_ist_ ArviZ' Datenstruktur – `InferenceData` besteht aus `xarray.Dataset`|
|**NumPy**|liegt darunter – jedes `DataArray` hat `.values` → `ndarray`|
|**Pandas**|Ausgangstür – `az.summary` gibt direkt ein DataFrame|
|**Matplotlib**|Plot-Backend – ArviZ-Plots geben Axes zurück|
|**SciPy**|**kein** direkter Überlapp – ArviZ hat eigene Statistik-Implementierungen|

### Warum xarray und nicht Pandas?

Ein DataFrame ist flach (Zeilen × Spalten). Ein Posterior hat **benannte Dimensionen** (`chain`, `draw`, bei Vektorparametern mehr). xarray ist Pandas für höherdimensionale, benannte Achsen. Man muss xarray nicht lernen – nur wissen, wie man wieder herauskommt.

### Übersetzungstabelle: ArviZ ↔ NumPy

Verifizierte Äquivalenzen (identische Zahlen):

|ArviZ|NumPy-Äquivalent|
|---|---|
|`az.eti(idata)` (89 %)|`np.percentile(samples, [5.5, 94.5])`|
|`az.summary(...)["mean"]`|`samples.mean()`|
|`az.summary(...)["sd"]`|`samples.std()`|
|`az.hdi(idata)`|**kein** einfaches Äquivalent – braucht Suche nach kürzestem Intervall|

```python
samples = idata.posterior["theta"].values.flatten()

np.percentile(samples, [5.5, 94.5])   # [0.4524, 0.8603]
az.eti(idata)["theta"].values         # [0.4524, 0.8603]  ← identisch
```

> [!important] Der Punkt ETI ist „nur" Quantile – das kann NumPy auch. HDI nicht: Es erfordert die Suche nach dem _kürzesten_ Intervall. Das ist ArviZ' eigentlicher Mehrwert bei den Intervallen.

### Übersetzung: ArviZ ↔ Pandas

```python
az.summary(idata)                  # ist BEREITS ein DataFrame
az.summary(idata).to_excel("ergebnisse.xlsx")   # direkt in den xlsx-Workflow
```

Für die Rohsamples nach Pandas:

```python
df = idata.to_dataframe()                    # klassische API
# oder manuell:
df = pd.DataFrame({"theta": samples})
```

### Übersetzung: ArviZ ↔ Matplotlib

ArviZ-Plots sind Matplotlib-Plots – man kann sie weiterbearbeiten:

```python
ax = az.plot_posterior(idata)
ax.set_title("Enzymaktivität – Posterior")
plt.savefig("abbildungen/posterior.pdf")     # Vektorgrafik-Workflow
```

Umgekehrt: Alles, was ArviZ plottet, geht auch von Hand:

```python
samples = idata.posterior["theta"].values.flatten()
fig, ax = plt.subplots()
ax.hist(samples, bins=40, density=True)      # das IST der Posterior
```

> [!note] Wann ArviZ, wann selbst? ArviZ für Standardgrößen (Diagnostik, HDI, Forest-Plots bei vielen Parametern) – da steckt echte Logik drin. Selbst plotten, wenn die Abbildung in ein Dokument mit eigenem Stil soll. Beides ist derselbe Posterior.

### SciPy – die Nicht-Beziehung

`scipy.stats` und ArviZ berühren sich **nicht** direkt. Aber sie treffen sich in der Validierung:

```python
# Analytischer Posterior (scipy) vs. NUTS-Samples (PyMC/ArviZ)
x = np.linspace(0, 1, 200)
ax.hist(idata.posterior["theta"].values.flatten(), bins=40, density=True)
ax.plot(x, stats.beta(8, 4).pdf(x))    # Beta(1+7, 1+3), konjugiert
```

Deckt sich die Kurve mit dem Histogramm, hat der Sampler die analytische Wahrheit gefunden – ohne die Formel zu kennen.

---

## Arbeitsmuster

Immer dieselbe Reihenfolge – **erst prüfen, dann interpretieren**:

1. `az.summary(idata)` → `r_hat` und `ess` anschauen
2. Bei Auffälligkeiten: `az.plot_trace(idata)` oder `az.plot_rank(idata)`
3. Erst wenn sauber: `az.plot_posterior(idata)`, `az.hdi(idata)`, Interpretation
4. Modellkritik: `az.plot_ppc_dist(ppc)`, `az.loo(idata)`

> [!warning] Grundregel Ein Posterior aus einem nicht-konvergierten Sampler ist wertlos – egal wie schön die Zahlen aussehen.

---

## API-Drift-Sammelstelle

ArviZ-Argumentnamen und Rückgabetypen wandern zwischen Versionen. **Immer in der eigenen Umgebung prüfen.**

|Alt|Neu (beobachtet)|
|---|---|
|`idata.groups()` (Methode)|`idata.groups` (Property)|
|`az.hdi(..., hdi_prob=)`|`az.hdi(..., prob=)`|
|`az.summary(..., hdi_prob=)`|`az.summary(..., ci_prob=)`|
|Spalten `hdi_3%` / `hdi_97%`|`eti89_lb` / `eti89_ub`|
|`InferenceData`|`DataTree` (in sehr neuen Versionen)|
|Plots geben `Axes`|teils `PlotCollection`|

Prüfbefehle für die eigene Umgebung:

```python
import arviz as az
print(az.__version__)
print(type(idata).__name__)              # InferenceData oder DataTree?
print(type(az.plot_posterior(idata)))    # Axes oder PlotCollection?
[f for f in dir(az) if f.startswith("plot_")]   # verfügbare Plots
```

> [!warning] Merksatz Argumentnamen sind **nicht** zwischen Funktionen derselben Library übertragbar – `hdi_prob` in der einen Funktion heißt `ci_prob` in der anderen.

# Referenzen

# Referenzen