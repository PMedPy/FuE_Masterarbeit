10-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Die Verwendung von Wahrscheinlichkeiten als Quantification von Unsicherheit ist eine Bayesisch Perspektive. 

Man unterscheidet die Bayesische Sicht und die [[Frequentist Perspective]]:
- Frequentisch: Jede Wahrscheinlichkeit stellt die relative Häufigkeit  bei unendlich wiederholten Zufallsexperimenten da. Wahrscheinlichkeit ist also eine Eigenschaft der Welt.
- Bayesisch: Die Wahrscheinlichkeit ist ein Maß für die Unsicherheit oder den Glauben. Bayesische Sicht dann, wenn kein "Zufallsexperiment" vorliegt. 

# Was Bishop meint
Die Bayesische Sicht ist nach Bishop allgemeiner: Sie funktioniert auch bei Ereignissen, die Wiederholbar sind UND, deren Wiederholung keinen Sinn macht:
- "Wie wahrscheinlich ist es, dass dieses spezifische Enzym einen $k_{cat}$ über 100 s⁻¹ hat?"
- "Wie wahrscheinlich ist es, dass die Riemann-Hypothese stimmt?"
- "Wie wahrscheinlich ist es, dass der wahre Parameter $\mu$  zwischen 4 und 5 liegt?"
Frequentisch betrachtet keine Aussage über die Riemann-Hypothese gemacht werden, da die Fragen **nicht sinnvoll** sind.

# Wahrscheinlichkeit nach Bayes:

Die Wahrscheinlichkeit nach Bayes hat die Form: 
$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}$$
Wobei:
$p(\theta)$ Beobachtung über den Parameter vor Datenerhebung (z.B Literaturwerte) (Oder Hypotethisches Modell)
$p(\mathcal{D} \mid \theta)$ Likelihood: Die Wahrscheinlichkeit der Daten unter einen bestimmten $\theta$ (Unter welchen $\theta$ wären die Werte möglich). (Zeigt die Menge an Pfaden, mit der die Daten erklärbar sind und optimiert bei der [[Likelihood Maximization]] mit den meisten Pfaden, siehe Baum)
$p(\theta \mid \mathcal{D})$ die Posteriori Wahrscheinlichkeit nach dem Daten vorhanden sind (Was gewollt ist)
$p(\mathcal{D})$ Die Evidenz (Normalisierungskonstante.)
# Beispiel von Claude: Enzymkinetik

## Die Formel in ihrer abstrakten Form

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)  p(\theta)}{p(\mathcal{D})}$$



Mit den vier Bauteilen:

|Symbol|Name|Was es bedeutet|
|---|---|---|
|$\theta$|Parameter|Die unbekannte Größe, über die du etwas lernen willst|
|$\mathcal{D}$|Daten|Was du tatsächlich gemessen hast|
|$p(\theta)$|Prior|Was du _vor_ den Messungen über $\theta$ glaubst|
|$p(\mathcal{D} \mid \theta)$|Likelihood|Wie wahrscheinlich die Daten unter einem hypothetischen $\theta$ wären|
|$p(\theta \mid \mathcal{D})$|Posterior|Dein aktualisierter Glaube über $\theta$, nach den Daten|
|$p(\mathcal{D})$|Evidenz|Normalisierungskonstante|

## Das Enzymkinetik-Beispiel — Schritt für Schritt zugeordnet

### Was ist $\theta$?

$\theta$ ist hier **der wahre, unbekannte Wert von $k_{cat}$** für dein spezifisches Enzym. Eine einzelne Zahl in der Einheit s⁻¹.

Du kennst diesen Wert nicht. Genau deshalb betreibst du überhaupt Statistik. Wenn du ihn wüsstest, gäbe es nichts zu schätzen.

$$\theta = k_{cat} \quad \text{(unbekannt, in s}^{-1}\text{)}$$


### Was ist $\mathcal{D}$?

$\mathcal{D}$ ist deine **konkrete Messreihe**:

$$\mathcal{D} = {x_1, x_2, x_3, x_4, x_5} = {47, 53, 42, 58, 51}$$

$$\mathcal{D} = \{x_1, x_2, x_3, x_4, x_5\} = \{47, 53, 42, 58, 51\}$$

Das sind fünf reale Zahlen, die du im Labor ermittelt hast. Sie sind _gegeben_ — kein Zufall mehr, sondern Fakten.

### Was ist der Prior $p(\theta)$?

Das ist dein **Vorwissen**, bevor du die Messungen gemacht hast. Du hast in die Literatur geschaut und gesehen: Ähnliche Enzyme haben $k_{cat}$ um die 50 s⁻¹, mit gewisser Streuung. Das formalisierst du als:

$$p(\theta) = \mathcal{N}(\theta \mid 50,, 10^2)$$

$$p(\theta) = \mathcal{N}(\theta \mid 50,\, 10^2)$$

**Wie liest man das?** Diese Verteilung sagt: "Bevor ich messe, halte ich $\theta = 50$ für am plausibelsten. Werte zwischen 30 und 70 sind alle ernsthaft im Spiel. Werte unter 20 oder über 80 halte ich für sehr unwahrscheinlich."

Wichtig: **Der Prior ist eine Verteilung über $\theta$, nicht über Messwerte.** Das ist ein häufiger Stolperstein. $p(\theta)$ beschreibt nicht, wie deine Messungen verteilt sind, sondern wie du über den unbekannten Parameter denkst.

### Was ist die [[Likelihood]] $p(\mathcal{D} \mid \theta)$?

Die Likelihood beantwortet folgende **hypothetische** Frage: _"Wenn der wahre $k_{cat}$ den Wert $\theta$ hätte — wie wahrscheinlich wären dann genau diese fünf Messwerte?"_

Du nimmst an, einzelne Messungen sind verrauscht um den wahren Wert (z. B. wegen Pipettierfehler, Detektorrauschen):

$$x_i \sim \mathcal{N}(\theta, \sigma^2) \quad \text{mit} \quad \sigma = 5$$


$$x_i \sim \mathcal{N}(\theta, \sigma^2) \quad \text{mit} \quad \sigma = 5$$

Für die ganze Messreihe (Annahme: Messungen unabhängig):

$$p(\mathcal{D} \mid \theta) = \prod_{i=1}^{5} \mathcal{N}(x_i \mid \theta, 5^2)$$

$$p(\mathcal{D} \mid \theta) = \prod_{i=1}^{5} \mathcal{N}(x_i \mid \theta, 5^2$$

**Anschauliches Beispiel:**

- Wäre $\theta = 50$, wären Werte wie 47, 53, 42, 58, 51 sehr plausibel → Likelihood **hoch**.
- Wäre $\theta = 100$, wären diese niedrigen Messwerte sehr seltsam → Likelihood **sehr niedrig**.
- Wäre $\theta = 30$, wäre der Wert 58 ungewöhnlich weit oben → Likelihood **mittel-niedrig**.

Die Likelihood "bewertet" also für jeden hypothetischen $\theta$-Wert, wie gut er zu den tatsächlich beobachteten Daten passt. Sie ist als Funktion von $\theta$ zu verstehen, mit fixen Daten.

### Was ist der Posterior $p(\theta \mid \mathcal{D})$?

Das ist das, was du eigentlich willst: deine **aktualisierte Überzeugung** über $\theta$, nachdem du Prior und Daten kombiniert hast. Im konjugierten Gauß-Fall fällt das analytisch sauber heraus:

$$p(\theta \mid \mathcal{D}) = \mathcal{N}(\theta \mid 50{,}19,, 4{,}76)$$

**Wie liest man das?** "Nach Berücksichtigung meines Vorwissens _und_ meiner fünf Messungen halte ich $\theta \approx 50{,}2$ für am plausibelsten. Mit 95 % Wahrscheinlichkeit liegt der wahre Wert zwischen etwa 45,9 und 54,5."

Beachte den Vergleich: Der Prior hatte Varianz $10^2 = 100$. Der Posterior hat Varianz $4{,}76$ — **viel schärfer**. Genau das ist Lernen: Unsicherheit wird durch Daten reduziert.

### Was ist die Evidenz $p(\mathcal{D})$?

Im Nenner steht die Wahrscheinlichkeit, dieses spezielle $\mathcal{D}$ überhaupt zu beobachten — gemittelt über _alle möglichen_ $\theta$-Werte:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta), p(\theta), d\theta$$

```latex
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta
```

In der Praxis ist das eine Zahl (kein Funktional von $\theta$), die nur dafür sorgt, dass der Posterior als Verteilung zu 1 integriert. Bei konjugierten Modellen (wie hier) muss man die Evidenz oft gar nicht explizit ausrechnen — sie fällt automatisch heraus. Bei komplizierten Modellen ist sie der schwerste Teil und der Grund, warum man Approximationen wie MCMC oder Variational Inference braucht.

## Die Formel mit deinen Größen ausgeschrieben

$$\underbrace{p(k_{cat} \mid \mathcal{D})}_{\text{was du wissen willst}} = \frac{\overbrace{\prod_{i=1}^{5} \mathcal{N}(x_i \mid k_{cat}, 5^2)}^{\text{Likelihood: passen die Daten zu } k_{cat}?} \cdot \overbrace{\mathcal{N}(k_{cat} \mid 50, 10^2)}^{\text{Prior: was die Literatur sagt}}}{\underbrace{p(\mathcal{D})}_{\text{Normalisierung}}}$$


## Die intuitive Geschichte hinter der Formel

Es lohnt sich, das einmal in Worten zu erzählen:

> _"Ich starte mit einer Vermutung über $k_{cat}$, gestützt auf Literatur (Prior). Dann mache ich Messungen. Für jeden möglichen Wert von $k_{cat}$ frage ich: Wären meine Messdaten mit diesem Wert vereinbar? (Likelihood). Werte, die sowohl zur Literatur passen als auch die Daten gut erklären, bekommen hohe Posterior-Wahrscheinlichkeit. Werte, die entweder dem Literaturwissen widersprechen oder die Daten schlecht erklären, werden bestraft."_

Das ist Bayes in einem Absatz. Die Formel ist nur die mathematische Verdichtung dieser Geschichte.

## Eine letzte Verständnishilfe: Wer "spricht" hier eigentlich?

Manchmal hilft es, sich klar zu machen, **welcher Bauteil welche Information bringt**:

- $p(\theta)$ **kommt aus dir** (bzw. aus der Literatur, deinem Vorwissen). Es ist deine Eingabe.
- $p(\mathcal{D} \mid \theta)$ **kommt aus deinem Messmodell**. Du modellierst, wie die Welt Daten erzeugt.
- $\mathcal{D}$ **kommt aus dem Labor**. Es sind die rohen Fakten.
- $p(\theta \mid \mathcal{D})$ **ist das Ergebnis**. Das, was du nach der Rechnung in der Hand hast.

Die Bayes-Formel ist im Grunde ein Rezept: _Werfe deine Eingaben (Prior + Messmodell + Daten) in den Mixer, raus kommt das Update deines Wissens._

---

Ein häufiger Aha-Moment: Wenn du das einmal an einem konkreten Beispiel wie diesem nachvollzogen hast, siehst du die Bayes-Formel in jedem späteren ML-Kontext wieder — Bayesian Regression, Gaussian Processes, Variational Autoencoders, Bayesian Neural Networks. Es ist immer dasselbe Schema, nur die Verteilungen werden komplizierter.

Sag Bescheid, wenn du irgendeinen der vier Bauteile noch genauer aufdröseln willst — die Likelihood ist erfahrungsgemäß der trickreichste, weil sie "rückwärts gelesen" werden muss (als Funktion von $\theta$ bei festen Daten).

# Claude Beispiel: Statistische Auswertung Zellkultur

> Vergleich zwischen klassisch-frequentistischer und bayesscher Auswertung eines typischen Zellkultur-Experiments mit $n=3$ biologischen Replikaten.

---

## Szenario: Wirkung eines Inhibitors auf Zellwachstum

Du hast eine Zellkultur und willst wissen, ob ein neu synthetisierter Inhibitor das Wachstum bremst. Du machst — realistisch — $n=3$ biologische Replikate pro Bedingung und misst die **Verdopplungszeit** in Stunden:

|Bedingung|Replikat 1|Replikat 2|Replikat 3|Mittelwert|SD|
|---|--:|--:|--:|--:|--:|
|Kontrolle|24|22|25|23,67|1,53|
|Inhibitor|28|31|27|28,67|2,08|

---

## Klassisch-frequentistische Standardauswertung

Was du wahrscheinlich aus der Bachelorarbeit kennst:

- **t-Test** (zweiseitig, ungleiche Varianzen): $t \approx 3{,}35$, $p \approx 0{,}03$
- **Schlussfolgerung im klassischen Stil:** _"Der Inhibitor verlängert die Verdopplungszeit signifikant ($p < 0{,}05$)."_
- Sternchen ans Balkendiagramm, fertig.

### Was du _nicht_ sagen kannst, aber wahrscheinlich gemeint hast

- Wie **groß** ist der Effekt?
- Wie **sicher** bin ich, dass er real ist?
- Wie **verlässlich** ist mein Befund?
- Ist der Effekt biologisch **relevant**?

Auf diese eigentlich interessanten Fragen liefert der p-Wert prinzipiell keine Antwort.

---

## Bayessche Auswertung

### Modell

Verdopplungszeiten sind normalverteilt um einen Gruppen-Mittelwert:

$$x_{C,i} \sim \mathcal{N}(\mu_C, \sigma^2), \qquad x_{I,j} \sim \mathcal{N}(\mu_I, \sigma^2)$$

### Parameter, die uns interessieren

$$\theta = (\mu_C,, \mu_I,, \sigma) \quad \text{und vor allem: } \Delta = \mu_I - \mu_C$$

$\Delta$ ist die eigentlich biologisch interessante Größe — der Effekt des Inhibitors auf die Verdopplungszeit.

### Priors (aus Vorwissen über Zellkultur)

|Parameter|Prior|Begründung|
|---|---|---|
|$\mu_C$|$\mathcal{N}(24, 5^2)$|Säugerzellen typisch 20–30 h|
|$\mu_I$|$\mathcal{N}(24, 10^2)$|Wirkung unbekannt, breiterer Prior|
|$\sigma$|$\text{HalfCauchy}(5)$|Schwacher, aber realistischer Prior auf die Streuung|

### Posterior

Bei diesem Modell nicht mehr analytisch lösbar, aber mit MCMC (z. B. PyMC oder Stan) eine Sache von wenigen Sekunden. Beispielhaftes Ergebnis:

|Größe|Posterior-Mittelwert|95 % Credible Interval|
|---|--:|:-:|
|$\mu_C$|23,7 h|[21,9; 25,4]|
|$\mu_I$|28,6 h|[26,4; 30,9]|
|$\Delta$|4,9 h|[2,1; 7,7]|
|$P(\Delta > 0)$|0,98|—|
|$P(\Delta > 2)$|0,93|—|

---

## Was ist hier so viel besser?

### 1. Verteilung über $\Delta$, nicht nur Punktschätzung

Du kannst direkt fragen:

- **"Wirkt der Inhibitor überhaupt?"** → $P(\Delta > 0) = 0{,}98$
- **"Ist der Effekt biologisch relevant (> 2 h)?"** → $P(\Delta > 2) = 0{,}93$

Das sind die Fragen, die Biologen eigentlich stellen wollen.

### 2. Effektgröße mit Unsicherheit

> _"Der Inhibitor verlängert die Verdopplungszeit um wahrscheinlich 2–8 Stunden, mit dem wahrscheinlichsten Wert bei etwa 5 Stunden."_

Eine Aussage, die direkt biologisch interpretierbar ist.

### 3. Vorwissen sinnvoll einbringen

Wenn aus früheren Experimenten bekannt ist, dass dieser Inhibitor-Typ Effekte von 3–10 h hat, kannst du diesen Prior wählen. Das macht die Schätzung bei $n=3$ stabiler.

### 4. Argumentation _für_ die Nullhypothese möglich

Wenn $P(|\Delta| < 1) = 0{,}8$, hast du echte Evidenz dafür, dass der Effekt **klein** ist — nicht nur das frequentistische _"wir wissen es nicht"_.

---

## Praktische Umsetzung mit PyMC

```python
import pymc as pm
import numpy as np

control = np.array([24, 22, 25])
inhibitor = np.array([28, 31, 27])

with pm.Model() as model:
    # Priors
    mu_C = pm.Normal('mu_C', mu=24, sigma=5)
    mu_I = pm.Normal('mu_I', mu=24, sigma=10)
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    # Likelihood
    pm.Normal('obs_C', mu=mu_C, sigma=sigma, observed=control)
    pm.Normal('obs_I', mu=mu_I, sigma=sigma, observed=inhibitor)
    
    # Größe, die uns wirklich interessiert
    delta = pm.Deterministic('delta', mu_I - mu_C)
    
    # Sampling
    trace = pm.sample(2000)
```

Das war's. Du bekommst die Posterior-Samples und kannst dann **alle** Fragen daran stellen.

---

## Zuordnung zur Bayes-Grundformel

Zur Erinnerung — die zentrale Formel:

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) , p(\theta)}{p(\mathcal{D})}$$

|Bauteil|In unserem Zellkultur-Beispiel|
|---|---|
|$\theta$|$(\mu_C, \mu_I, \sigma)$ — die Modellparameter|
|$\mathcal{D}$|Die 6 gemessenen Verdopplungszeiten|
|$p(\theta)$ (Prior)|Was wir vor dem Experiment über die Parameter glauben|
|$p(\mathcal{D} \mid \theta)$ (Likelihood)|Wie wahrscheinlich die Messdaten unter einem hypothetischen $\theta$ wären|
|$p(\theta \mid \mathcal{D})$ (Posterior)|Aktualisierte Verteilung der Parameter nach den Daten|
|$p(\mathcal{D})$ (Evidenz)|Normalisierungskonstante|

---

## Fazit

|Aspekt|Frequentistisch ($t$-Test)|Bayessch|
|---|---|---|
|Aussage über Effektgröße|nein (nur Mittelwerte)|ja, mit Verteilung|
|Aussage über Effekt-Wahrscheinlichkeit|nein|ja, direkt interpretierbar|
|Vorwissen nutzbar|nein (formal)|ja, über Prior|
|Argumentation für H₀|nicht möglich|möglich|
|Verständlichkeit für Biologen|hoch (gewohnt)|hoch (intuitiver, aber ungewohnt)|
|Aufwand|minimal|überschaubar mit PyMC/Stan|
|Konventionsakzeptanz|sehr hoch|wächst, aber noch in der Minderheit|

**Pragmatische Empfehlung:** Beides nebeneinander präsentieren — das wirkt nicht aufgesetzt, sondern _zusätzlich informativ_. Niemand wird etwas dagegen haben, und du übst eine Methode, die in der nächsten Generation an Bedeutung gewinnen wird.

---

## Weiterführende Literatur

- **Kruschke, J. (2014):** _Doing Bayesian Data Analysis._ Sehr freundlicher Einstieg, mit R/JAGS.
- **McElreath, R. (2020):** _Statistical Rethinking._ Vielleicht das beste moderne Lehrbuch — kostenloser Vorlesungs-Kurs auf YouTube.
- **PyMC-Dokumentation:** https://www.pymc.io
- **ASA Statement on p-Values (Wasserstein & Lazar, 2016):** Die institutionelle Referenz zur Kritik an der frequentistischen Standardpraxis.

# Referenzen
[@bishopDeepLearningFoundations2024]