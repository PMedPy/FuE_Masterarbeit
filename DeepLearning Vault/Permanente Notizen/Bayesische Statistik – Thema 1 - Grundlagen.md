10-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended


> Sonderlektion zum FuE-Projekt (Bishop, Deep Learning Foundations). Einstieg in bayesisches Denken vor PyMC.

---

## 1. Die zwei Weltbilder

Der Kernunterschied: **was gilt als zufällig?**

**Frequentistisch**

- Parameter θ ist ein _fester, unbekannter_ Wert.
- Die _Daten_ sind zufällig.
- Wahrscheinlichkeit = relative Häufigkeit bei unendlicher Wiederholung.
- Konfidenzintervall: Aussage über die _Methode_, nicht über das eine Intervall.

**Bayesisch**

- Parameter θ ist _selbst eine Zufallsgröße_ – nicht weil er physikalisch schwankt, sondern weil unser **Wissen** unsicher ist.
- Wahrscheinlichkeit = Grad der Überzeugung (_degree of belief_).
- Erlaubt direkt: „Mit 95 % Wahrscheinlichkeit liegt θ in diesem Bereich."

> Bishop (Kap. 2): Wahrscheinlichkeit wird von Häufigkeiten auf eine **Quantifizierung von Unsicherheit** verallgemeinert. Genau das braucht ein Netz, das Vorhersageunsicherheit über kcat/Km ausgeben soll.

---

## 2. Der Satz von Bayes

$$p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta), p(\theta)}{p(\mathcal{D})}$$

|Term|Name|Bedeutung|
|---|---|---|
|$p(\theta)$|**Prior**|Überzeugung über θ _vor_ den Daten|
|$p(\mathcal{D} \mid \theta)$|**Likelihood**|Wie gut erklärt θ die beobachteten Daten?|
|$p(\theta \mid \mathcal{D})$|**Posterior**|Aktualisierte Überzeugung _nach_ den Daten|
|$p(\mathcal{D})$|**Evidence**|Normierungskonstante (θ-unabhängig)|

Bishops Leitsatz:

$$\text{posterior} \propto \text{likelihood} \times \text{prior}$$

Das Proportionalzeichen ∝ bedeutet: alle Faktoren, die **nicht von θ abhängen**, dürfen weggelassen werden (sie kürzen sich heraus).

### Die Evidence im Detail

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta), p(\theta), d\theta \quad\text{(diskret: Summe statt Integral)}$$

- Sie ist die **Summe der Zähler** (likelihood × prior) über alle Hypothesen/θ-Werte.
- In Worten: „Wahrscheinlichkeit der Daten, gemittelt über _alle_ möglichen θ, gewichtet mit dem Prior."
- **Warum θ darin verschwindet:** durch das Aufsummieren/Integrieren über _alle_ θ – nicht durch den Prior. Der Prior ist nur das _Gewicht_ in diesem Mittel.
- Dieses Integral ist für realistische Modelle analytisch fast nie lösbar → genau deshalb existiert MCMC / PyMC (Thema 2).

---

## ⭐ Baustelle 1: Normierung – worüber summiert sich was zu 1?

„Normiert über X" heißt: _summiert/integriert man über alle X, kommt 1 heraus._

||normiert über|summiert sich zu 1 über …|
|---|---|---|
|**Likelihood** $p(\mathcal{D}\mid\theta)$|die **Daten**|alle möglichen Datenausgänge bei _festem_ θ|
|**Posterior** $p(\theta\mid\mathcal{D})$|die **Hypothesen / θ**|alle θ-Werte bei _festen_ Daten|

- Die Likelihood summiert sich über θ **nicht** zu 1 → deshalb braucht es die Division durch die Evidence.
- Die Evidence ist genau die Brücke: sie zwingt den Posterior, sich über die Hypothesen zu 1 zu summieren.

**Merksatz:** _Likelihood normiert über Daten, Posterior normiert über Hypothesen._

---

## ⭐ Baustelle 2: Inferenz funktioniert nur RELATIV

Eine einzelne Likelihood-Zahl bedeutet **für sich genommen nichts**.

- „Likelihood = 0,512" ist weder gut noch schlecht.
- Erst der **Vergleich** mit der Alternative (z. B. 0,125) sagt etwas aus.
- Beispiel: Likelihood A = 0,002 vs. B = 0,001 → A ist _besser als B_, aber **nicht** automatisch eine _gute_ Erklärung. Vielleicht erklären beide die Daten mies, A nur weniger mies.

**Konsequenz:** Bayes braucht immer den vollständigen Hypothesenraum. Frage nie „ist θ = X wahr?", sondern „ist θ = X wahr **im Vergleich zu welchen Alternativen?**"

> Sonderfall: Betrachtet man nur _eine_ Hypothese, gibt es nichts zu lernen – der Posterior ist trivial 1, weil keine Alternative existiert, zu der Masse wandern könnte.

---

## 3. Sequenzielles Updaten

Der Posterior von heute ist der Prior von morgen:

$$p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \theta), \underbrace{p(\mathcal{D}_1 \mid \theta), p(\theta)}_{\text{Posterior nach } \mathcal{D}_1}$$

**Äquivalenz sequenziell ↔ gebündelt:** Updatet man Datenpunkt für Datenpunkt oder alle auf einmal, kommt **dasselbe** heraus – aber nur unter zwei Bedingungen:

1. **Exakte Präzision** (gerundete Zwischenstände brechen die Äquivalenz – im Skript-Test selbst gesehen!).
2. Datenpunkte sind **unabhängig** (die Likelihood faktorisiert).

> FuE-Bezug: Bei iterativem/batchweisem Updaten volle Präzision mittragen, nicht gerundete Zwischenstände.

---

## ⭐ Baustelle 3: Binomialkoeffizient – wann weglassen?

$$p(\mathcal{D}\mid\theta) = \binom{n}{k},\theta^{k}(1-\theta)^{n-k}$$

- **Weglassen erlaubt** bei der Posterior-Berechnung: $\binom{n}{k}$ ist **θ-unabhängig**, steht in jedem Zähler _und_ in der Evidence → kürzt sich vollständig heraus. (Solange man es bei _allen_ Hypothesen gleich handhabt.)
- **Mitzählen nötig**, wenn man die **absolute** Wahrscheinlichkeit der Daten will (z. B. „wie wahrscheinlich war dieses Ergebnis konkret?").

Grund: Der Koeffizient zählt nur, _in wie vielen Reihenfolgen_ das Ergebnis auftreten kann – er behandelt beide Hypothesen gleich und trägt deshalb **nichts zur Unterscheidung** bei. Für Inferenz über θ irrelevant.

---

## ⭐ Baustelle 4: Parameter vs. Hypothese

Zwei verschiedene Ebenen!

- **θ ist der Parameter** – die unbekannte Größe selbst (Grün-Wahrscheinlichkeit, EC₅₀, kcat). Die _Achse_.
- Eine **Hypothese** ist eine _Aussage über_ θ („θ = 0,5"). Ein _Punkt_ (oder Bereich) auf der Achse.

**Warum sie im diskreten Fall verschmolzen:** Bei nur zwei erlaubten Werten {0,5; 0,7} _war_ jeder Parameterwert eine Hypothese → die Wörter wurden synonym. Im Kontinuierlichen bricht das auf:

|Ebene|Diskret (Ballwerfer)|Kontinuierlich (PyMC)|
|---|---|---|
|**Parameter θ**|nur Werte {0,5; 0,7}|ganzes Intervall [0,1]|
|**Hypothese**|„θ = 0,5" (1 von 2)|„θ in [0,6; 0,75]" (Bereich)|
|**Prior**|2 Zahlen|Dichtekurve (z. B. Beta)|
|**Posterior**|2 Zahlen|Dichtekurve|

Bishops „Parameter" ist der präzise Begriff. Bayes liefert eine Wahrscheinlichkeitsverteilung **über** θ:

- Diskret → degeneriert zu „Wahrscheinlichkeit pro Hypothese".
- Kontinuierlich → volle Dichtekurve $p(\theta\mid\mathcal{D})$.

Die Wahrscheinlichkeit bezieht sich (bayesisch) direkt auf den **Parameter** – das ist frequentistisch verboten.

---

## 4. Der konzeptionelle Gewinn

- **Frequentistisch:** „Ist der Unterschied signifikant?" → p-Wert, Ja/Nein.
- **Bayesisch:** „Wie wahrscheinlich ist es, dass das neue Mittel besser wirkt als Cetirizin?" → direkte Zahl, z. B. $P(\theta_{\text{neu}} > \theta_{\text{cet}} \mid \mathcal{D}) = 0{,}87$.

Das ist die Aussage, die ein:e Entscheider:in tatsächlich will.

---

## 5. Werkzeugkasten für PyMC (Ausblick Thema 2)

|Werkzeug|Wofür|PyMC|
|---|---|---|
|**Beta-Verteilung**|Prior für Wahrscheinlichkeiten (0–1), z. B. θ|`pm.Beta("theta", alpha=1, beta=1)`|
|**Binomial/Bernoulli**|Likelihood für Zähldaten (k aus n)|`pm.Binomial("y", n=10, p=theta, observed=7)`|
|**Normal / Log-Normal**|kontinuierliche Messwerte / positive Größen über Größenordnungen (Konz., EC₅₀, kcat)|`pm.Normal`, `pm.LogNormal`|
|**Konjugierte Priors**|Prior+Likelihood → Posterior derselben Familie (Beta+Binomial→Beta); analytisch ohne MCMC|–|
|**MCMC / NUTS**|Posterior _samplen_ statt Evidence-Integral lösen|`pm.sample()`|
|**Posterior Predictive Checks**|Modellvalidierung: simuliere neue Daten, vergleiche mit echten|`pm.sample_posterior_predictive()`|
|**Diagnostik (ArviZ)**|Konvergenz prüfen: R̂ ≈ 1,0, ESS, Trace-Plots|`az.summary()`, `az.plot_trace()`|
|**Posterior-Zusammenfassung**|Mittelwert/Median, **HDI** (→ Thema 3)|`az.summary()`, `az.hdi()`|

**Ablauf:** Verteilungen wählen (Prior + Likelihood) → MCMC samplet Posterior → Diagnostik + predictive checks → Zusammenfassung mit HDI.

### Tool-Landschaft (PPLs)

Niemand baut Bayes selbst – man nutzt **probabilistische Programmiersprachen**:

- **PyMC** (Python) – zugänglich, große Community → unser Tool.
- **Stan** (eigene Sprache, via CmdStanPy) – akademischer Goldstandard, robuster, steiler.
- **NumPyro** (Python/JAX) – sehr schnell, für große Modelle / Skalierung.
- **brms** (R) – Formelsprache über Stan, beliebt in Sozialwissenschaften.

---

## Kernmerksätze

1. **posterior ∝ likelihood × prior** – θ-unabhängige Faktoren kürzen sich raus.
2. **Likelihood normiert über Daten, Posterior über Hypothesen.**
3. **Inferenz ist relativ** – eine einzelne Likelihood ist bedeutungslos, nur Vergleiche zählen.
4. **Evidence = Summe/Integral der Zähler**, macht den Posterior zu echten Wahrscheinlichkeiten.
5. **θ = Parameter** (Achse), Hypothese = Aussage über θ (Punkt/Bereich).
6. **Sequenziell = gebündelt** nur bei exakter Präzision + unabhängigen Daten.

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]