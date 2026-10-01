12-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# MCMC-Vertiefung: Evidence, Metropolis & die Energie-Analogie

> Sonderlektion Bayes, Thema 2 (Vertiefung). Warum MCMC ohne die Evidence auskommt, ein durchgerechnetes Metropolis-Beispiel, und die Brücke Posterior ↔ Energie ↔ Loss.

---

## 1. Warum sich die Evidence in MCMC herauskürzt

### Was der Sampler kennt – und was nicht

Der Sampler sieht nur die **unnormierte** Funktion (den Zähler des Bayes-Satzes):

$$\tilde{p}(\theta) = p(\mathcal{D}\mid\theta),p(\theta) \quad (\text{Likelihood} \times \text{Prior})$$

Der echte Posterior ist das geteilt durch die Evidence $Z$:

$$p(\theta\mid\mathcal{D}) = \frac{\tilde{p}(\theta)}{Z}, \qquad Z = \int \tilde{p}(\theta),d\theta$$

$Z$ ist eine **Konstante** (hängt nicht von θ ab) – nur die Zahl, die die Fläche auf 1 bringt. Der Sampler berechnet $Z$ **nie**.

### Lokale Kürzung (ein Schritt)

Im Akzeptanzverhältnis steht der Posterior zweimal – $Z$ kürzt sich:

$$\alpha = \frac{p(\theta'\mid\mathcal{D})}{p(\theta\mid\mathcal{D})} = \frac{\tilde{p}(\theta')/Z}{\tilde{p}(\theta)/Z} = \frac{\tilde{p}(\theta')}{\tilde{p}(\theta)}$$

### Globale Kürzung (ganze Verteilung) – der eigentliche Grund

Die Kette erfüllt **detailed balance** (detaillierte Bilanz):

$$\pi(\theta),T(\theta \to \theta') = \pi(\theta'),T(\theta' \to \theta)$$

Setzt man $\pi = \tilde{p}/Z$ und die Metropolis-Akzeptanz ein, steht $Z$ auf **beiden Seiten** als derselbe Faktor und kürzt sich – **für jedes Paar (θ, θ') im gesamten Raum**.

Damit ist der Posterior die **stationäre Verteilung** der Kette. Folge (Ergodizität): die Kette verbringt langfristig in jedem Bereich genau den Zeitanteil, der dem Posterior entspricht. Die **relative** Aufenthaltshäufigkeit ist:

$$\frac{\text{Zeit bei }\theta'}{\text{Zeit bei }\theta} = \frac{\tilde{p}(\theta')}{\tilde{p}(\theta)}$$

Da das überall gilt, hat die Aufenthaltsverteilung **die Form** des Posteriors. Die **Normierung** entsteht gratis durch „N Samples zählen, durch N teilen".

> **Kernsatz:** $Z$ ist nur ein globaler konstanter Vorfaktor, der die _Form_ nicht ändert. MCMC rekonstruiert die Form über relative Häufigkeiten; die Normierung erledigt das Zählen selbst. Deshalb wird $Z$ nie gebraucht.

### Physik-Brücke

$\tilde p(\theta) = e^{-E/k_BT}$, und $Z = \int e^{-E/k_BT}$ ist die **Zustandssumme** (partition function) – ebenfalls ein unlösbares Integral, das man nie berechnet. Bayes-Evidence ↔ Zustandssumme: mathematisch dasselbe.

---

## 2. Metropolis-Algorithmus – zwei Schritte durchgerechnet

**Modell:** θ ∈ [0,1], flacher Prior Beta(1,1) → $p(\theta)=1$, Likelihood Binomial n=40, k=28.

$$\tilde{p}(\theta) = \theta^{28}(1-\theta)^{12} \quad (\text{Binomialkoeffizient kürzt sich})$$

**Vorschlag:** symmetrischer Gauß-Schritt $\theta' = \theta + \mathcal{N}(0, 0{,}05)$ → Hastings-Korrektur entfällt (reines Metropolis). **Start:** $\theta_0 = 0{,}5$.

Rechnung im **Log-Raum** (numerisch stabil, wie PyMC):

### Schritt 1 – Vorschlag θ' = 0,62

$$\log\tilde{p}(0{,}62) = 28\ln(0{,}62) + 12\ln(0{,}38) = -13{,}38 - 11{,}62 = -25{,}00$$ $$\log\tilde{p}(0{,}5) = 40\ln(0{,}5) = -27{,}73$$ $$\ln\alpha = -25{,}00 + 27{,}73 = +2{,}73 ;\Rightarrow; \alpha = e^{2{,}73} \approx 15{,}3$$

$\alpha > 1$ → **immer akzeptieren** (0,62 liegt näher an den Daten 28/40 = 0,70 als 0,5). → **Neuer Zustand $\theta_1 = 0{,}62$.** Notiere 0,62.

### Schritt 2 – Vorschlag θ' = 0,55

$$\log\tilde{p}(0{,}55) = 28\ln(0{,}55) + 12\ln(0{,}45) = -16{,}74 - 9{,}58 = -26{,}33$$ $$\log\tilde{p}(0{,}62) = -25{,}00 \quad (\text{von eben})$$ $$\ln\alpha = -26{,}33 + 25{,}00 = -1{,}33 ;\Rightarrow; \alpha = e^{-1{,}33} \approx 0{,}265$$

$\alpha < 1$ (0,55 liegt weiter von 0,70 weg) → **würfeln**: ziehe $u \sim \text{Uniform}(0,1)$.

- $u = 0{,}18 < 0{,}265$ → akzeptieren, Zustand 0,55
- $u = 0{,}72 > 0{,}265$ → **ablehnen**, bleibe bei 0,62 und **notiere 0,62 erneut**

Sagen wir $u = 0{,}72$ → abgelehnt. → **Zustand $\theta_2 = 0{,}62$** (zum zweiten Mal notiert).

### Der Algorithmus in Kürze

1. Start bei θ.
2. Vorschlag θ' aus symmetrischer Verteilung um θ.
3. $\alpha = \min!\big(1, \tilde{p}(\theta')/\tilde{p}(\theta)\big)$.
4. $u \sim U(0,1)$: ist $u < \alpha$ → akzeptiere θ', sonst bleibe bei θ (und notiere θ erneut).
5. Wiederhole.

> **Der entscheidende Trick:** In Schritt 2 wird mit 26,5 % ein _schlechterer_ Wert akzeptiert. Das ist **kein Bug** – ohne dieses „Schlechter-Akzeptieren" hätte man einen _Optimierer_, der nur zum Gipfel klettert und dort klebt (= Punktschätzer). Das gelegentliche Akzeptieren lässt die Kette die **Flanken** erkunden → die Aufenthaltshäufigkeit zeichnet die ganze Glockenform nach, nicht nur die Spitze.

---

## 3. Posterior ↔ Energie ↔ Loss

### Das „Umdrehen" des Posteriors

Negativer Logarithmus des (unnormierten) Posteriors = **Energie**:

$$E(\theta) = -\log\tilde{p}(\theta) = -\log\big[p(\mathcal{D}\mid\theta),p(\theta)\big]$$

Aus der Boltzmann-Beziehung:

$$\tilde{p}(\theta) = e^{-E(\theta)} \quad\Longleftrightarrow\quad E(\theta) = -\log\tilde{p}(\theta)$$

|Posterior|Energie $E = -\log\tilde p$|
|---|---|
|**hoch** (wahrscheinliches θ)|**niedrig** (Talboden)|
|**niedrig** (unwahrscheinlich)|**hoch** (Berg)|
|Modus (wahrscheinlichstes θ)|globales Energieminimum|

### Die Loss-Funktion ist dieselbe Größe

In ML ist $-\log\tilde{p}(\theta)$ **exakt die Loss-Funktion**: $$\underbrace{-\log p(\mathcal{D}\mid\theta)}_{\text{neg. Log-Likelihood = Daten-Loss}} ;+; \underbrace{(-\log p(\theta))}_{\text{neg. Log-Prior = Regularisierung}}$$

→ **Loss minimieren = Energie minimieren = Posterior maximieren = Talboden finden.** (Das ist MAP / der Punktschätzer.)

### Warum „nur bergab" scheitert (= warum man Sampler statt Optimierer braucht)

- **Optimierer** (nur bessere Werte): Gradientenabstieg, nur bergab → landet im nächsten Talboden und bleibt kleben. Ergebnis: ein Punkt (der Modus).
- **Sampler** (auch schlechtere mit $\alpha = e^{-\Delta E}$): darf bergauf → besetzt die Flanken proportional → erzeugt die ganze Verteilung (inkl. `sd`, Unsicherheit). Kann zudem über Energie-_Pässe_ aus lokalen Tälern in tiefere entkommen (multimodal!).

Die Akzeptanz ist **buchstäblich der Boltzmann-Faktor**:

$$\alpha = \frac{\tilde p(\theta')}{\tilde p(\theta)} = \frac{e^{-E(\theta')}}{e^{-E(\theta)}} = e^{-\Delta E}$$

Bergauf ($\Delta E > 0$) → akzeptiert mit $e^{-\Delta E} < 1$: je steiler, desto seltener, aber nie null.

### Temperatur (Ausblick: schwer samplebare Posteriors)

Volle Form $\alpha = e^{-\Delta E/k_BT}$. **Simulated Annealing / Parallel Tempering**: aufheizen (großes $T$ → wilde Erkundung, raus aus lokalen Tälern), dann abkühlen ($T \to 1$ → im tiefsten Tal niederlassen). Dieselbe Temperatur wie in der MD.

---

## CLT in MCMC – kurze Notiz

Das CLT gilt für die **Schätzung von Posterior-Größen** (z. B. Mittelwert), aber mit $N_{\text{eff}}$ (ESS) statt N, weil MCMC-Samples **korreliert** sind (Markov-Eigenschaft):

$$\text{Fehler} \approx \frac{\sigma_{\text{post}}}{\sqrt{N_{\text{eff}}}} \quad (\text{= MCSE})$$

- **10 Samples**: ruckelige Form, Mittelwert schwankt stark, HDI unzuverlässig.
- **3000 Samples**: glatte Glocke, stabiler Mittelwert, belastbares HDI.
- Die _wahre_ Posterior-Form ändert sich nicht – nur die **Auflösung**.
- `ess_bulk` sagt, wie viele _unabhängige_ Samples die korrelierte Kette wert ist.

---

## FuE-Bezug

Bei Enzymkinetik-Fits (Michaelis-Menten, Hill) wandert die Kette durch den Parameterraum ($K_m$, $v_{max}$, …) und besucht jede Kombination proportional dazu, wie gut sie die Messpunkte erklärt. Schlechtere Fits werden nicht verboten, nur seltener besucht. → Ergebnis: pro Parameter eine **Verteilung** (Mittelwert + Unsicherheit), statt nur ein Best-Fit wie bei klassischer nichtlinearer Regression.

---

## Merksätze

1. **Evidence $Z$ ist ein globaler Vorfaktor** – kürzt sich global (detailed balance), nie gebraucht.
2. **MCMC rekonstruiert die Form, Normierung kommt durchs Zählen.**
3. **Schlechtere Werte akzeptieren ist der ganze Trick** – sonst Optimierer statt Sampler.
4. **Posterior umgedreht = Energie = Loss**; Modus = Talboden = MAP.
5. **Akzeptanz $= e^{-\Delta E}$** ist der Boltzmann-Faktor aus der MD.
6. **CLT gilt mit $N_{\text{eff}}$**, nicht N – wegen Sample-Korrelation.


# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]