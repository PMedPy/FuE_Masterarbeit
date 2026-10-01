23-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# A06 – Elemental Confounds I (DAGs & der Fork)

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 6_ Quelle: [YouTube – Lecture A06](https://www.youtube.com/watch?v=lGR7D45Ww38)

---

## 1. Von Korrelation zu Kausalität

Korrelation ist **allgegenwärtig** und beweist nichts. McElreaths Beispiele: Heavy-Metal-Bands pro Kopf vs. Lebenszufriedenheit (Finnland vorn), Waffle-House-Dichte vs. Scheidungsrate, und der Klassiker — **Margarine-Pro-Kopf-Verbrauch vs. Scheidungsrate in Maine, $r = 0{,}993$**. Reine Artefakte.

Die zentrale Umdeutung der Vorlesung: Ein **Confound** ist _alles, was uns über die Kausalität in die Irre führt_. Und — das ist neu —

> Ein Confound entsteht **nicht nur** durch weggelassene Variablen (_omitted_), sondern oft durch **falsch einbezogene** Variablen (_included_).

Eine Variable ins Modell zu werfen kann genauso verzerren wie sie wegzulassen. _Welche_ Variablen man kontrolliert, ist keine Geschmacksfrage, sondern folgt aus der Kausalstruktur.

---

## 2. Heuristiken vs. Gründe — „a better way“

McElreath sortiert die übliche Praxis:

- **Schlechte Heuristiken:** Modelle nach $R^2$ wählen, Variablen mit nicht-signifikantem p-Wert rauswerfen, kausalfreie Entscheidungsbäume („welcher Test passt zum Spaltentyp?“), Konformität.
- **Okaye Heuristiken:** Prä-Treatment-Variablen einschließen, Post-Treatment ausschließen, nicht gegen Baseline, sondern gegen Alternativ-Treatment vergleichen.

Der Ausweg (Folie „a better way“):

> **Mache die kausalen Annahmen transparent, und benutze Logik, um daraus den Estimator abzuleiten.**

Kausale Modelle leben auf einem **Abstraktionsspektrum**:


| DAGs | SEMs | DiD/IV | ODEs |
| ---- | ---- | ------ | ---- |
|      |      |        |      |

||Abstrakt $\longrightarrow$ Bespoke|||
|---|---|---|---|
|**DAGs**|**SEMs**|**DiD / IV**|**ODEs**|
|nur Variablen + Pfeile, _keine_ Verteilungen|minimale Verteilungs-/Richtungshypothesen|ökonometrische Designs|volle Dynamik (z. B. SIR: $\frac{dS}{dt}=-\beta SI$)|

> [!info] Vertiefung — dein Kinetik-Modell sitzt rechts außen Das ODE-Ende des Spektrums ($\frac{dS}{dt}=-\beta SI,\ \frac{dI}{dt}=\beta SI-\gamma I,\dots$) ist genau die „bespoke“ Ebene, auf der dein CHO-Wachstumsmodell lebt. Ein DAG ($c \to {\mu,\delta} \to N$) ist die _abstrakte Skizze_ derselben Kausalität; die ODE ist ihre _vollständig ausspezifizierte_ Form. Beides ist dasselbe Kontinuum — der DAG sagt „was wirkt auf was“, die ODE sagt zusätzlich „mit welcher Funktion und Rate“.

_Historie:_ Urvater der DAGs ist **Sewall Wright**, der in den 1920ern _Pfaddiagramme_ erfand — an Fellmustern von Meerschweinchen.

---

## 3. DAG → Generatives Modell → Statistisches Modell (Schwerpunkt)

Das ist der Übergang, der oft unklar bleibt. Drei Ebenen derselben Sache, mit _einer_ Kernidee, die alles verbindet:

> **Ein Pfeil ist ein Term.** „$Z \to Y$“ heißt: $Z$ steht auf der rechten Seite der Gleichung für $Y$. Die Eltern eines Knotens (die Pfeile, die _hinein_zeigen) sind genau seine Prädiktoren.

||**DAG**|**Generatives Modell**|**Statistisches Modell**|
|---|---|---|---|
|**Was es ist**|qualitatives Skelett: Variablen + Kausalpfeile|DAG + Funktionen + Rauschen + Parameter_werte_|DAG + Likelihood + Priors über _unbekannte_ Parameter|
|**Richtung**|keine (nur Struktur)|**vorwärts** — Daten _erzeugen_|**rückwärts** — Parameter _schätzen_|
|**Verteilungen?**|**nein**|ja (jeder Knoten bekommt Rauschen)|ja (Likelihood + Priors)|
|**Gleichungen**|keine|**eine pro Knoten** (RHS = seine Eltern)|**eine für das Outcome** (Prädiktoren = vom DAG gewählt)|
|**Wozu**|überlegen, _was im Prinzip_ schätzbar ist|simulieren, validieren, $\mathrm{do}()$ verstehen|aus echten Daten den Estimand berechnen|

### Worked Example (Heirat/Scheidung)

**Ebene 1 — DAG.** Alter bei Heirat $A$ ist gemeinsame Ursache von Heiratsrate $M$ und Scheidungsrate $D$; fraglich ist, ob $M$ selbst auf $D$ wirkt: $$ A \to M, \qquad A \to D, \qquad M \to D ,? $$

**Ebene 2 — generatives Modell** (jeder Knoten eine Gleichung, RHS = Eltern, vorwärts laufbar): $$ \begin{aligned} A &\sim \text{Normal}(0,1) &&\text{(exogen, keine Eltern)} \ M &= \beta_{AM},A + \varepsilon_M, &&\varepsilon_M \sim \text{Normal}(0,\sigma_M) \quad (\text{Eltern von } M:\ A)\ D &= \beta_{MD},M + \beta_{AD},A + \varepsilon_D, &&\varepsilon_D \sim \text{Normal}(0,\sigma_D) \quad (\text{Eltern von } D:\ M, A) \end{aligned} $$ Setzt man zum Testen $\beta_{MD}=0$, erzeugt man Daten, in denen $M$ _nachweislich keinen_ Effekt hat — und prüft, ob der Estimator das zurückfindet.

**Ebene 3 — statistisches Modell** für den Estimand _„Effekt von $M$ auf $D$“_ (nur **eine** Gleichung, für das Outcome $D$): $$ \begin{aligned} D_i &\sim \text{Normal}(\mu_i, \sigma) \ \mu_i &= \alpha + \beta_M M_i + \beta_A A_i \end{aligned} $$ Warum steht $A$ hier? **Weil der DAG es vorschreibt:** $A$ ist eine gemeinsame Ursache (ein Fork, §5), öffnet also einen Hinterpfad $M \leftarrow A \to D$. Nur wenn man $A$ einschließt, misst $\beta_M$ den _kausalen_ Effekt statt der vom Alter induzierten Scheinassoziation. Geschätzt werden $(\alpha, \beta_M, \beta_A, \sigma)$.

> [!important] Vertiefung — _warum_ die beiden Modelle so verschieden aussehen Das ist der Knoten, an dem es klemmt. Generatives und statistisches Modell sind **dieselbe Kausalstruktur, von zwei Seiten gelesen**:
> 
> - Das **generative** Modell beschreibt das _ganze System_ — eine Gleichung pro Knoten —, weil man _alles_ erzeugen will, um vorwärts zu simulieren. Hier wird _jeder_ Pfeil zu einem Term in der Gleichung seines Zielknotens.
> - Das **statistische** Modell beschreibt nur die _eine_ inverse Frage: „gegeben die Daten, welcher Wert von $\beta_M$?“ Deshalb braucht es nur **eine** Gleichung — die für das Outcome $D$. Der DAG spielt hier eine _andere_ Rolle: nicht „schreibe jeden Pfeil als Term“, sondern „**entscheide, welche Prädiktoren** in die Outcome-Gleichung gehören“, damit der geschätzte Koeffizient kausal interpretierbar ist (Confounder rein, Mediatoren/Collider raus).
> 
> Kurz: **Generativ** = der DAG als Daten_fabrik_ (alle Knoten). **Statistisch** = der DAG als _Auswahlregel_ für die Prädiktoren der einen Outcome-Gleichung. Derselbe Graph, zwei Jobs.

---

## 4. Die vier elementaren Confounds

McElreath: Es gibt nur **vier** Elementarstrukturen („Ye Olde Causal Alchemy“, Avatar-Motiv). Jeder DAG ist aus ihnen zusammengesetzt.

|Confound|Struktur|$X,Y$ marginal?|Aktion für $X \to Y$|
|---|---|---|---|
|**Fork** (Gabel)|$X \leftarrow Z \to Y$|assoziiert (Scheinkorrelation)|**stratifizieren** nach $Z$|
|**Pipe** (Rohr)|$X \to Z \to Y$|assoziiert (über Mediator)|$Z$ **nicht** kontrollieren (sonst Pfad blockiert)|
|**Collider**|$X \to Z \leftarrow Y$|**un**assoziiert|$Z$ **nicht** kontrollieren (sonst Schein-Assoziation)|
|**Descendant**|$X \to Z \to Y$, $\ Z \to A$|erbt $Z$s Verhalten|$A$ wirkt wie ein abgeschwächtes $Z$|

Diese Vorlesung vertieft den **Fork**; Pipe, Collider, Descendant folgen.

---

## 5. Der Fork im Detail

$$ X \leftarrow Z \to Y, \qquad Z \text{ ist gemeinsame Ursache.} $$

Drei Aussagen (Folie):

1. $X$ und $Y$ **sind** assoziiert: $\ Y \not\perp!!!\perp X$.
2. Sie teilen die gemeinsame Ursache $Z$.
3. **Stratifiziert nach $Z$**, verschwindet die Assoziation: $\ Y \perp!!!\perp X \mid Z$.

Lies die letzte Zeile als Sprache: „**Sobald ich $Z$ kenne, sagt mir $X$ nichts Neues mehr über $Y$.**“ Beide hörten nur auf $Z$ — kennt man $Z$, ist der gemeinsame Draht durchtrennt. Das $\perp!!!\perp$ ist die **testbare Implikation** des DAGs: eine bedingte Unabhängigkeit, die man in den Daten prüfen kann.

> [!note] Vertiefung — der Fork _ist_ „ignorable unless shared“ (A04) Erinnerst du dich an die unbeobachteten Ursachen $U, V$ aus A04, die nur störten, wenn sie **geteilt** waren? Der Fork ist genau dieser Fall, jetzt benannt: $Z$ ist eine **geteilte** Ursache von $X$ _und_ $Y$. Eine Ursache, die nur auf _einen_ Knoten zeigt, geht ins Rauschen; eine, die auf _zwei_ zeigt, ist ein Fork und damit ein Confounder. „Welche Ursachen sind geteilt?“ — der Fork ist die Antwort in Reinform.

---

## 6. Statistischer Fork: Standardisierung & Priors

**Standardisierung (z-Scores):** Variablen auf Mittel 0, SD 1 bringen. Das eliminiert willkürliche Einheiten (McElreath: _„selbst Gott würde sagen, die Lichtgeschwindigkeit ist 1“_) und macht Priors übertragbar.

**Priors auf standardisierten Daten** (Prior-Predictive zeigt: flache Priors wie $\text{Normal}(0,10)$ erzeugen absurde, fast vertikale Geraden): $$ \alpha \sim \text{Normal}(0,,0.2), \quad \beta_M,\beta_A \sim \text{Normal}(0,,0.5), \quad \sigma \sim \text{Exponential}(1). $$

### In PyMC

```python
import pymc as pm, numpy as np, arviz as az

# d: WaffleDivorce; standardisieren
z = lambda x: (x - x.mean()) / x.std()
A = z(d.MedianAgeMarriage.values)   # Alter bei Heirat
M = z(d.Marriage.values)            # Heiratsrate
D = z(d.Divorce.values)             # Scheidungsrate

with pm.Model() as m_fork:
    a   = pm.Normal("a", 0, 0.2)
    bM  = pm.Normal("bM", 0, 0.5)
    bA  = pm.Normal("bA", 0, 0.5)
    sig = pm.Exponential("sigma", 1)
    mu  = a + bM*M + bA*A            # A drin, weil Fork -> Hinterpfad blocken
    pm.Normal("D", mu, sig, observed=D)
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=1)

az.summary(idata, var_names=["bM", "bA"])
```

Ergebnis: **$\beta_M \approx 0$**, $\beta_A$ deutlich negativ. Die Heiratsraten–Scheidungs-Korrelation war ein **Fork-Artefakt** über das Heiratsalter.

---

## 7. Interventionen: $\mathrm{do}(M)$ als Graph-Mutilation

Ein kausaler Effekt ist eine **Manipulation des generativen Modells**. $p(D \mid \mathrm{do}(M))$ = Verteilung von $D$, wenn wir $M$ aktiv setzen — was heißt: **alle Pfeile, die in $M$ zeigen, löschen** ($A \to M$ verschwindet) und $D$ simulieren.

```python
# Totaler kausaler Effekt von M auf D: Kontrast do(M=1) vs do(M=0)
post = az.extract(idata)

def sim_D_do(m, n=1000):
    # do(M=m): A->M gekappt, A behält Populationsverteilung (~Normal(0,1) standardisiert)
    A_pop = np.random.normal(0, 1, (post.sizes["sample"], n))
    mu = (post["a"].values[:, None]
          + post["bM"].values[:, None] * m
          + post["bA"].values[:, None] * A_pop)
    return np.random.normal(mu, post["sigma"].values[:, None])

causal_contrast = (sim_D_do(1.0) - sim_D_do(0.0)).mean(axis=1)   # pro Posterior-Ziehung
# -> Verteilung nahe 0
```

> [!important] Vertiefung — Koeffizient = kausaler Effekt ist eine _Ausnahme_ Hier liefert der Kontrast praktisch denselben Wert wie $\beta_M$. Das ist **kein** allgemeines Gesetz, sondern ein Spezialfall des **additiven, linearen Gauß-Modells**. McElreath warnt eindringlich: In **nicht-linearen** Modellen — logistische Regression, Poisson, und **deine ODE-Kinetik** — ist der Koeffizient _nicht_ der kausale Effekt. Dort **musst** du die Intervention explizit simulieren ($\mathrm{do}$, mutilieren, vorwärts rechnen, kontrastieren), weil sich Effekte nichtlinear mit dem Niveau anderer Variablen verschränken. Für dein IC50/Hill-Modell heißt das: den Inhibitor-Effekt _immer_ als simulierten Kontrast $\mu(c)-\mu(0)$ berechnen, nie als „den einen Parameter ablesen“.
> 
> (McElreaths Running Gag dazu: sein geplanter „Teufels-Statistikkurs“ beginnt mit der **Cauchy-Verteilung** — kein Mittel, unendliche Varianz — um die Illusionen der bequemen Gauß-Welt zu sprengen. Genau die schweren Ränder aus unserer Robustheits-Diskussion.)

**Für den Effekt von $A$** gilt das Spiegelbild: $M$ liegt auf dem Pfad $A \to M \to D$, ist also **Mediator** (Post-Treatment bzgl. $A$) — man darf $M$ **nicht** kontrollieren, sonst blockiert man einen Teil von $A$s Wirkung.

---

## 8. Ausblick

Der Fork ist erledigt; **Pipe, Collider und Descendant** folgen in „Elemental Confounds II“. Damit hat man die vier Bausteine, aus denen sich jeder beliebige DAG zusammensetzt — und kann _mechanisch_ (per do-Kalkül / Backdoor-Kriterium) ableiten, welches Adjustment-Set einen gegebenen Effekt unverzerrt schätzbar macht. Genau die Verallgemeinerung der „Pre-/Post-Treatment“-Heuristik aus A04/A05.

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A06, abgeglichen mit deinen Folien-Screenshots. Ergänzt: die konkreten Priors ($\alpha\sim\text{Normal}(0,0.2)$, $\beta\sim\text{Normal}(0,0.5)$, $\sigma\sim\text{Exponential}(1)$, in den Folien als Platzhalter), der vollständige **PyMC-Code** sowie die Schwerpunkt-Sektion **DAG → generativ → statistisch** und der do()-Interventions-Sim. Die als „Vertiefung“ markierten Callouts sind meine Ergänzungen, nicht Teil des Vortrags._
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]