22-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# A05 – Estimands & Estiplans

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 5_ Quelle: [YouTube – Lecture A05](https://www.youtube.com/watch?v=sYE8a95x-0E) _Ab dieser Folge: Modelle & Statistik in **PyMC** statt R._

---

## 1. Estimand, Estimator, Estimate — sauber getrennt

Die drei Wörter klingen ähnlich und werden ständig verwechselt. McElreath bringt sie mit einer Kuchen-Metapher auseinander (Igel-Kuchen): das Zielbild, das Rezept, der echte Kuchen.

|Begriff|Lat. Wurzel|Was es ist|Igel-Kuchen|
|---|---|---|---|
|**Estimand**|_„das, was geschätzt werden soll“_ (Gerundiv)|Die **präzise wissenschaftliche Zielgröße** — die Frage selbst, formal definiert (z. B. _der totale kausale Effekt von Geschlecht auf Gewicht_). Existiert **unabhängig** von Daten und Methode.|Das **Foto** des perfekten Igel-Kuchens — das Ziel.|
|**Estimator**|_„der Schätzende“_|Das **Rezept**: das statistische Modell **plus** Schätzverfahren, das aus Daten eine Antwort produziert. Eine **Funktion** Daten $\to$ Zahl/Verteilung.|Die **Zutatenliste + Anleitung**.|
|**Estimate**|_„das Geschätzte“_|Das **Ergebnis**, wenn der Estimator auf **echte Daten** trifft — bei Bayes eine ganze **Posterior-Verteilung**.|Der **tatsächliche** (oft unförmige) Kuchen, der herauskommt.|

McElreaths Kritik: In vielen Publikationen ist **kein klarer Estimand** erkennbar — man wirft Software auf Daten, ohne je eine formale Frage gestellt zu haben. Das Rezept wird gekocht, ohne dass jemand weiß, welchen Kuchen man eigentlich wollte. Der ganze Workflow steht und fällt damit, den Estimand **zuerst** zu fixieren.

> [!info] Vertiefung — warum die Trennung praktisch zählt Der Estimator wird **validiert**, indem man ihn auf **synthetische** Daten loslässt, deren wahren Estimand man kennt (Simulation-Based Validation aus A04). Findet das Rezept den bekannten Zielkuchen zurück, vertraust du ihm — _dann erst_ fütterst du echte Daten. Das ist die rote Schleife im Workflow-Diagramm: Estimand → Estimator → _synthetischer_ Estimate → Abgleich mit der bekannten Wahrheit.

---

## 2. Estimand 1 — der **totale** kausale Effekt von $S$ auf $W$

**Frage:** Welchen gesamten kausalen Einfluss hat das biologische Geschlecht $S$ auf das Gewicht $W$? DAG: $S \to W$ (direkt) **und** $S \to H \to W$ (indirekt über die Größe).

**Warum die Größe $H$ hier NICHT ins Modell darf.** Das ist der entscheidende Punkt und der Rückbezug auf A04: $H$ ist ein **Mediator** auf dem indirekten Pfad. Würde man $H$ konditionieren (stratifizieren), blockierte man $S \to H \to W$ und bekäme nur noch den _direkten_ Rest. Für den **totalen** Effekt muss man den Mediator **weglassen** — „nicht nach Post-Treatment-Variablen stratifizieren“.

**Das Modell** (Index-Variablen, ein Mittelwert pro Geschlecht):

$$ \begin{aligned} W_i &\sim \text{Normal}(\mu_i, \sigma) \ \mu_i &= \alpha_{S[i]} \ \alpha_j &\sim \text{Normal}(60, 10) \qquad (j \in {1,2}) \ \sigma &\sim \text{Uniform}(0, 10) \end{aligned} $$

$\alpha_1$ ist direkt das mittlere Frauengewicht, $\alpha_2$ das mittlere Männergewicht — gleiche Priors für beide, keine Referenzkategorie (siehe A04-Deep-Dive).

### In PyMC

```python
import pymc as pm
import numpy as np
import arviz as az

# d: DataFrame der !Kung-San-Erwachsenen (age >= 18) mit Spalten height, weight, sex
H = d.height.values
W = d.weight.values
S = d.sex.values                 # 1 = weiblich, 2 = männlich (McElreath-Kodierung)
S_idx = (S - 1).astype(int)      # 0/1 -> 0-basiert für Python-Indexing
Hbar = H.mean()

coords = {"sex": ["female", "male"]}

with pm.Model(coords=coords) as m_SW:          # ESTIMAND 1: totaler Effekt
    a     = pm.Normal("a", mu=60, sigma=10, dims="sex")   # ein Intercept je Geschlecht
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    mu    = a[S_idx]                                       # Index-Lookup = Embedding
    pm.Normal("W", mu=mu, sigma=sigma, observed=W)
    idata_SW = pm.sample(1000, tune=1000, chains=4, random_seed=1)
```

**Synthetische Validierung zuerst** (bevor echte Daten): künstliche Personen mit einem _bekannten_ Gewichtsunterschied erzeugen und prüfen, ob `m_SW` ihn zurückfindet.

```python
# synthetische Daten mit bekanntem weiblich/männlich-Mittel
S_sim   = np.repeat([1, 2], 500)
mu_true = np.where(S_sim == 1, 45.0, 55.0)     # bekannter Unterschied (Testwert)
W_sim   = np.random.normal(mu_true, 5.0)
# ... m_SW mit (S_sim, W_sim) fitten und prüfen, ob a ≈ [45, 55] zurückkommt
```

> [!note] Vertiefung — `a[S_idx]` ist derselbe Embedding-Lookup wie in A04 $\mu_i = \alpha_{S[i]}$ heißt in PyMC wörtlich `a[S_idx]`: integer-Code rein, passendes Element raus. Mit `dims="sex"` ist `a` ein Vektor der Länge 2; `a[S_idx]` „gathered“ pro Person den richtigen Eintrag. Skaliert unverändert auf 200 Kategorien — `shape`/`dims` ändern, Modellzeile bleibt gleich.

---

## 3. Posterior: Mittelwerte ≠ Individuen

Der Posterior von `m_SW` ist gemeinsam über $(\alpha_1, \alpha_2, \sigma)$. Im Pairs-Plot der Folie sind die drei **praktisch unkorreliert** (Korrelationen $\approx 0.0$) — logisch, denn es ist ein reines Intercept-Modell ohne geteilte Steigung. Frauenmittel $\alpha_1 \approx 41{-}42$ kg, Männermittel $\alpha_2 \approx 48{-}49$ kg.

Jetzt die zentrale Unterscheidung (Folie „Posterior means & predictions“):

```python
post   = az.extract(idata_SW)
aF, aM = post["a"].sel(sex="female"), post["a"].sel(sex="male")
sig    = post["sigma"]

# (1) Posterior der MITTELWERTE  -> KEINE Überlappung
#     aF und aM sind scharf getrennt (~42 vs ~48.5)

# (2) Posterior-PREDIKTION für INDIVIDUEN -> massive Überlappung
W_female = np.random.normal(aF.values, sig.values)
W_male   = np.random.normal(aM.values, sig.values)
```

Die **Mittelwerte** überlappen nicht — das Modell ist sicher, dass sich die _Durchschnitte_ unterscheiden. Die **vorhergesagten Individuen** überlappen massiv, weil $\sigma$ (die Streuung _innerhalb_ jedes Geschlechts) viel größer ist als die Differenz der Mittel. Zwei völlig verschiedene Fragen: „unterscheiden sich die Durchschnitte?“ vs. „kann ich das Geschlecht aus dem Gewicht einer Einzelperson erraten?“

---

## 4. „Always Be Contrasting“ — was ein Kontrast ist

Das ist dein expliziter Fokuspunkt. Ein **Kontrast** ist die **Posterior-Verteilung der Differenz** zweier Größen, berechnet **innerhalb jeder einzelnen Posterior-Ziehung**:

```python
W_contrast = (aM - aF).values        # pro Sample: Männermittel - Frauenmittel
```

Lies das als Operation: Für jede der z. B. 4000 Ziehungen $(\alpha_1^{(s)}, \alpha_2^{(s)})$ bildet man $\delta^{(s)} = \alpha_2^{(s)} - \alpha_1^{(s)}$, und die Verteilung dieser $\delta^{(s)}$ **ist** der Kontrast. Das ist „summarize last“ (A02) angewandt auf Gruppenunterschiede: erst die Differenz auf der vollen gemeinsamen Verteilung bilden, dann zusammenfassen.

McElreaths eiserne Regel: **Niemals** Unterschiede aus der **Überlappung** der Randverteilungen (oder aus dem Vergleich zweier Konfidenzintervalle / zweier p-Werte) ablesen. Man **muss** die Kontrastverteilung rechnen.

> [!important] Vertiefung — warum Überlappung lügt: die Kovarianz (Folie „never compare overlap“) Der mathematische Kern. Für die Differenz zweier Parameter gilt $$\mathrm{Var}(\alpha_2 - \alpha_1) = \mathrm{Var}(\alpha_2) + \mathrm{Var}(\alpha_1) - 2,\mathrm{Cov}(\alpha_1,\alpha_2).$$ Lies den $-2,\mathrm{Cov}$-Term: Er ist genau die Information, die in den **Randverteilungen unsichtbar** ist. Die Folie zeigt zwei **positiv korrelierte** Parameter — ihre Randdichten überlappen breit, aber weil sie _gemeinsam_ hoch/niedrig wandern, ist ihre **Differenz scharf** und klar von null weg (die schwarze „difference“-Kurve). Hätte man nur die zwei Marginals (oder zwei CIs) verglichen, hätte man „kein Unterschied“ geschlossen — falsch. Umgekehrt kann scheinbare Trennung einen unsicheren Kontrast verbergen. Nur die per-Sample-Differenz trägt die Kovarianz korrekt mit. _Deshalb_ ist „compare confidence intervals“ statistisch unzulässig.

---

## 5. Estimand 2 — der **direkte** Effekt (Stratifizierung & Zentrierung)

**Frage:** Wie viel des Gewichtsunterschieds bleibt, wenn man Männer und Frauen **gleicher Größe** vergleicht? Das isoliert den **direkten** Pfad $S \to W$ (z. B. Körperzusammensetzung), indem der indirekte Pfad über $H$ blockiert wird. Jetzt **muss** $H$ ins Modell — als Stratifikationsvariable.

**Zentrierung.** Man rechnet mit $H_i - \bar H$ statt $H_i$. Damit wird der Intercept $\alpha$ interpretierbar als Gewicht bei **durchschnittlicher** Größe (statt bei der unsinnigen Größe $0$) — exakt der A03-Vorgriff, jetzt eingelöst. Nebeneffekt: $\alpha$ und $\beta$ **entkorrelieren** im Posterior.

**Das Modell** (zwei Intercepts _und_ zwei Steigungen, je Geschlecht):

$$ \begin{aligned} W_i &\sim \text{Normal}(\mu_i, \sigma) \ \mu_i &= \alpha_{S[i]} + \beta_{S[i]},(H_i - \bar H) \ \alpha_j &\sim \text{Normal}(60,10), \quad \beta_j \sim \text{Uniform}(0,1), \quad \sigma \sim \text{Uniform}(0,10) \end{aligned} $$

### In PyMC

```python
with pm.Model(coords=coords) as m_SHW:         # ESTIMAND 2: direkter Effekt
    a     = pm.Normal("a", mu=60, sigma=10, dims="sex")
    b     = pm.Uniform("b", lower=0, upper=1, dims="sex")   # Steigung je Geschlecht
    sigma = pm.Uniform("sigma", lower=0, upper=10)
    mu    = a[S_idx] + b[S_idx] * (H - Hbar)                # zentrierte Größe
    pm.Normal("W", mu=mu, sigma=sigma, observed=W)
    idata_SHW = pm.sample(1000, tune=1000, chains=4, random_seed=1)
```

**Kontrast bei jeder Größe** (Folie „Contrasts at each height“) — der Kern-Output:

```python
xseq  = np.linspace(130, 190, 50)
post2 = az.extract(idata_SHW)
aF, aM = post2["a"].sel(sex="female"), post2["a"].sel(sex="male")
bF, bM = post2["b"].sel(sex="female"), post2["b"].sel(sex="male")

# mu je Geschlecht: Form (samples, heights)
muF = aF.values[:, None] + bF.values[:, None] * (xseq[None, :] - Hbar)
muM = aM.values[:, None] + bM.values[:, None] * (xseq[None, :] - Hbar)

mu_contrast = muF - muM                              # Kontrast (F - M) pro Höhe
contrast_mean = mu_contrast.mean(axis=0)
contrast_PI   = np.percentile(mu_contrast, [5.5, 94.5], axis=0)   # 89%-Intervall
```

**Ergebnis:** Der Kontrast $F - M$ liegt über _alle_ Größen hinweg nahe **null**. Bei gleicher Größe unterscheiden sich die Gewichte kaum. Fazit der Folie: **„Nearly all of the causal effect of S acts through H.“** In dieser historischen Stichprobe wirkt das Geschlecht fast vollständig **indirekt** über die Größe — der direkte Effekt ist vernachlässigbar. (Genau deshalb brauchte Estimand 1 das $H$ _nicht_: der gesamte Effekt läuft ohnehin durch den Größenkanal.)

---

## 6. Philosophischer Abschluss: „Modelle haben keine Annahmen“

McElreaths Schlussreflexion: Eine lineare Regression ist eine **hirnlose Maschine**. Die Normalverteilung der Residuen ist keine Naturbehauptung, sondern eine **Lizenz für bestimmte Berechnungen** (Mittelwert + Streuung schätzen, p-Werte rechnen). Die Gültigkeit einer **Mittelwertschätzung** hängt **nicht** davon ab, ob die Residuen real perfekt normal sind — die Normalverteilung dient nur als Arbeits-Prior.

> „Modelle haben keine Annahmen, **Menschen** haben Annahmen.“ Und: **starke Annahmen für starke Schlüsse** — Schlussfolgerungen ohne explizite Annahmen sind wie Meinungen ohne Begründung: trivial oder bedeutungslos.

Das führt direkt zu deinem letzten Fokuspunkt — _warum_ gerade die Normalverteilung diese „Lizenz“ ist.

---

## 7. Capstone: Lineare Regression ↔ Normalverteilung ↔ Entropie

Warum sitzt ausgerechnet die Gauß-Glocke im Herzen der linearen Regression? Drei **unabhängige** Wege führen zu ihr — und der dritte ist der, den du mit deinem Info-Theorie-Hintergrund am meisten genießen wirst.

**Weg 1 — generativ (CLT).** Aus A03: Summen vieler kleiner, unabhängiger Fluktuationen (endliche Varianz) konvergieren gegen die Normalverteilung. Additive Fehler $\Rightarrow$ Glocke. Das Fußballfeld.

**Weg 2 — rechnerisch (Least Squares).** Die Log-Likelihood der Normal ist $$ \log p(W \mid \mu,\sigma) = -\frac{1}{2\sigma^2}\sum_i (W_i - \mu_i)^2 + \text{const}. $$ Lies das $(\cdot)^2$ als **Energie/Fehler**: Die Likelihood zu **maximieren** heißt, die **Summe der quadrierten Residuen zu minimieren** — das ist OLS / Methode der kleinsten Quadrate (Gauß 1809, A03). Normal-Likelihood **ist** Least Squares.

**Weg 3 — inferentiell (Maximum-Entropie).** Das ist die eigentliche Antwort auf „warum diese Lizenz“. Unter **allen** Verteilungen mit gegebenem Mittel $\mu$ und gegebener Varianz $\sigma^2$ ist die Normalverteilung die mit der **maximalen Entropie** — die **am wenigsten festlegende**, „flachste“ Verteilung, die noch zu diesen zwei Momenten passt.

> [!important] Vertiefung — die Normal fällt aus Maximum-Entropie heraus (Lagrange) Maximiere die Shannon-Entropie $H[p] = -\int p(x)\ln p(x),dx$ unter drei Nebenbedingungen: $\int p = 1$, $\int x,p = \mu$, $\int (x-\mu)^2 p = \sigma^2$. Lagrange-Funktional, Variationsableitung nach $p$: $$\frac{\delta}{\delta p}\Big[-!\int p\ln p - \lambda_0!\int p - \lambda_1!\int x p - \lambda_2!\int (x-\mu)^2 p\Big] = 0$$ $$\Rightarrow\ -\ln p(x) - 1 - \lambda_0 - \lambda_1 x - \lambda_2 (x-\mu)^2 = 0 \ \Rightarrow\ p(x) \propto \exp!\big(-\lambda_2 (x-\mu)^2\big).$$ Das ist **exakt** eine Gauß-Glocke; die Konstanten $\lambda$ werden durch die Nebenbedingungen festgelegt, $\Rightarrow p = \mathcal{N}(\mu,\sigma^2)$.
> 
> Als Sprache gelesen: **Entropie** $H = -\mathbb{E}[\ln p]$ ist die _erwartete Überraschung_ / das Maß deiner Unwissenheit; der $\ln$ macht aus dem Produkt unabhängiger Überraschungen eine **Summe** (Additivität von Information). **Maximieren** heißt „streue die Wahrscheinlichkeit so gleichmäßig wie möglich — nimm so wenig an wie erlaubt“. Die **eine** quadratische Nebenbedingung (Varianz) ist genau das, was die flache Linie zu **einer** symmetrischen Glocke biegt. Mehr Struktur steckt nicht drin.

**Die Synthese.** Eine Normal-Likelihood in der Regression behauptet **nicht**, dass die Daten normal _sind_. Sie sagt: „Ich bin nur bereit, mich auf einen **Mittelwert** und eine **Varianz** festzulegen — und unter dieser Selbstbeschränkung nehme ich die **ehrlichste**, annahmeärmste Verteilung.“ Genau das ist McElreaths „Lizenz, Mittelwert und Streuung zu schätzen“. Lineare Regression = bedingten Mittelwert $\mu = \alpha + \beta x$ modellieren, mit dem **maximal-entropischen** Rauschmodell, das sich auf nichts außer den ersten zwei Momenten festlegt.

> [!note] Vertiefung — die Brücke zu deinem Info-Theorie-Werkzeug MaxEnt ist äquivalent zu **minimaler KL-Divergenz** zur (uninformativsten) Basisverteilung unter den Momentbedingungen: Du fügst der maximalen Unwissenheit _nur_ die Information „Mittel und Varianz“ hinzu, nichts sonst. Drei weitere Fäden, die sich hier treffen:
> 
> - **Regularisierende Priors** (A04): Ein Normal-Prior ist MaxEnt für einen Koeffizienten mit bekannter Skala — derselbe Gedanke, nur auf Parameter statt auf Residuen angewandt (= L2/Ridge).
> - **Exponentialfamilie**: _Jede_ MaxEnt-Verteilung unter Momentbedingungen ist von der Form $\exp(\sum \lambda_k T_k(x))$ — die Exponentialfamilie ist die „MaxEnt-Familie“. Die Normal ist ihr Mitglied für $(x, x^2)$.
> - **Jensen** ($\mathbb{E}[g(\theta)] \neq g(\mathbb{E}\theta)$, A02): derselbe Konvexitätskern, der „summarize last“ erzwingt, steckt auch hinter der Konkavität der Entropie, die das Maximum überhaupt eindeutig macht.

So konvergieren drei Straßen auf dieselbe Glocke: **additive Fehler** (CLT) erzeugen sie, **kleinste Quadrate** rechnen mit ihr, **maximale Entropie** rechtfertigt sie als die annahmeärmste Wahl. Lineare Regression ist der Ort, an dem sich diese drei treffen.

---

## 8. Ausblick

Mit klar definierten Estimands (total vs. direkt), Index-Variablen, Kontrasten und zentrierten Prädiktoren ist das Handwerkszeug komplett. Die nächste Vorlesung führt das **formale DAG-Framework für kausale Inferenz** ein: Regeln, die _automatisch_ sagen, welche Variablen man kontrollieren muss (und welche nicht), um einen gewünschten Effekt unverzerrt zu schätzen — die Verallgemeinerung der „Pre-/Post-Treatment“-Heuristik aus A04.

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A05, abgeglichen mit deinen Folien-Screenshots. Ergänzt/präzisiert: Prior $\alpha_j\sim\text{Normal}(60,10)$ und $\sigma\sim\text{Uniform}(0,10)$ aus der Posterior-Folie; der synthetische Testunterschied ist ein illustrativer Validierungswert (in der Vorlesung eine runde Zahl), nicht der reale Effekt (~6–7 kg). Sämtlicher **PyMC-Code** sowie die Sektionen zu **Kontrast-Kovarianz** und **Linear/Normal/Entropie** sind meine Ergänzungen zum Vortragsinhalt; die als „Vertiefung“ markierten Callouts sind nicht Teil des Originalvortrags._

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]