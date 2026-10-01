19-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# A04 – Categories and Causes

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 4_ Quelle: [YouTube – Lecture A04](https://www.youtube.com/watch?v=GIdwLrW2nNo)

---

## 1. Recap: Prior-Predictive & die „Predigt über Priors“

Vor der Datenanalyse simuliert man die **Prior-Predictive** (Geraden allein aus den Priors, $N=0$), um deren Implikationen zu sehen. McElreaths „Sermon on priors“ in fünf Sätzen:

- Es gibt **keine korrekten** Priors, nur **wissenschaftlich begründbare**.
- **Weakly informative**: weiche wissenschaftliche Einschränkungen.
- Es ist leicht, **besser** zu sein als „flache“ Priors.
- Wer **nicht regularisiert**, übertreibt Effektgrößen.
- Das Vorurteil „Priors zum Schummeln“ ist falsch: in der Praxis **schrumpfen** Priors Effekte fast immer Richtung null → **Bayes ist konservativ**.

> [!note] Vertiefung — „regularisierender Prior“ ist exakt L2/Ridge (ML-Brücke) Das verbindet sich direkt mit dem, was du aus Bishop kennst. Ein Normal-Prior auf einen Koeffizienten ist **identisch** zur L2-Regularisierung. Der MAP-Schätzer maximiert $$\log p(\beta\mid D) ;\propto; \underbrace{\log p(D\mid\beta)}_{\text{Datenfit}} ;+; \underbrace{\log p(\beta)}_{\text{Prior}}, \qquad \beta\sim\mathcal{N}(0,\tau^2)\ \Rightarrow\ \log p(\beta) = -\tfrac{1}{2\tau^2}|\beta|^2 + \text{const}.$$ Der Prior-Term **ist** der Ridge-Strafterm $-\lambda|\beta|^2$ mit $\lambda = 1/(2\tau^2)$. Lies es als Sprache: ein engerer Prior (kleines $\tau$) = stärkere Strafe = stärkeres Schrumpfen Richtung null. (Ein **Laplace**-Prior gäbe entsprechend **L1/Lasso**.) „Bayes ist konservativ“ heißt also wörtlich: der Prior zieht $\beta$ zur Null, genau das Gegenteil von „Effekte aufblasen“.

---

## 2. Simulation-Based Validation

Der vollständige Workflow enthält eine **Validierungsschleife** (im Foliendiagramm rot): Generatives Modell → **synthetische Daten** → statistisches Modell → **synthetische Schätzungen** → prüfen, ob die _bekannten wahren_ Parameter zurückgewonnen werden.

McElreaths Punkte dazu:

- **Bare minimum**: das statistische Modell mit synthetischen Beobachtungen aus dem generativen Modell testen.
- **Beide Modelle könnten kaputt sein** (Code _oder_ Denkfehler).
- Selbst funktionierende Modelle erreichen vielleicht nicht das wissenschaftliche **Ziel**.
- **Ein generatives Modell zu schreiben hilft, das eigene Denken zu debuggen.**
- Stärkerer Test: **Simulation-Based Calibration (SBC).**

Im Foliendemo (vier Panels) wird $\beta$ für je 100 Wiederholungen geschätzt; die Posterior-Intervalle umschließen den gestrichelten wahren Wert. Vergleich der Panels: bei $N=100$ sind die Intervalle deutlich **enger** als bei $N=10$ — derselbe „mehr Daten → schmälerer Posterior“-Effekt wie in unseren früheren Beispielen, hier als Validierungs-Check sichtbar gemacht.

> [!info] Vertiefung — was genau wird hier validiert? Nicht die _Wissenschaft_, sondern **Mathematik + Code + Schätzer**. Die Schleife stellt die Frage: „_Wenn_ meine Modellannahmen exakt stimmen — findet mein Schätzer dann die Wahrheit zurück?“ Besteht er den Test nicht, liegt der Fehler bei dir (Implementierung/Logik), nicht bei der Natur. Es ist im Geiste eine **frequentistische Kalibrierungsprüfung eines bayesianischen Schätzers** — und genau deshalb so nützlich für Power-Analysen und Studiendesign _vor_ der Datenerhebung.

---

## 3. Das volle Kausalbild: DAG & unbeobachtete Ursachen

Das erweiterte Kausalmodell für Geschlecht $S$, Größe $H$, Gewicht $W$:

$$ \begin{aligned} H &= f_H(S, U) \ W &= f_W(H, S, V) \ S &= f_S(Y) \end{aligned} \qquad\qquad \begin{aligned} &U \to H \ &V \to W \ &S \to H,\quad S \to W,\quad H \to W,\quad Y \to S \end{aligned} $$

Dabei sind $U, V$ (eingekreist) **unbeobachtete Ursachen**, $Y$ eine unbeobachtete Ursache von $S$, beschriftet „**ignorable unless shared**“.

> [!important] Vertiefung — „ignorable unless shared“ ist die zentrale Kausallektion Warum stören $U$ (nur $\to H$), $V$ (nur $\to W$) und $Y$ (nur $\to S$) **nicht**? Weil eine unbeobachtete Ursache, die auf **genau eine** Variable zeigt, einfach in deren **Rausch-/Fehlerterm aufgeht** — sie ist der Grund, warum $W$ nicht deterministisch aus $H$ folgt, und steckt am Ende in $\sigma$. Kein Bias. Gefährlich wird eine unbeobachtete Ursache erst, wenn sie **geteilt** ist, also auf **zwei** Variablen zeigt (z. B. ein verstecktes $Z \to H$ _und_ $Z\to W$). Dann ist sie ein **Confounder**: sie öffnet einen Hinterpfad $H \leftarrow Z \to W$, der echte Assoziation vortäuscht. Die ganze Kunst kausaler Inferenz reduziert sich auf: _welche Ursachen sind geteilt?_ Alles andere darf man ignorieren.

---

## 4. Mediation & der do-Operator

$S$ wirkt auf $W$ über **zwei** Pfade: **direkt** ($S \to W$) und **indirekt** über die Größe ($S \to H \to W$). $H$ ist ein **Mediator** des Geschlechtseffekts. Daraus folgen zwei verschiedene Estimands.

**Der do-Operator** unterscheidet _Eingreifen_ von _Beobachten_. $p(W \mid \mathrm{do}(S{=}s))$ heißt: $S$ aktiv auf $s$ **setzen** (alle Pfeile _in_ $S$ durchschneiden), nicht bloß Personen mit $S{=}s$ beobachten. Der kausale Effekt ist ein **Kontrast** zweier Interventionen:

$$ \text{kausaler Effekt von }H = p\big(W \mid \mathrm{do}(H{=}h)\big) - p\big(W \mid \mathrm{do}(H{=}h')\big). $$

Für das Geschlecht:

- **Totaler Effekt** von $S$ auf $W$: Kontrast $p(W\mid\mathrm{do}(S{=}1))$ vs. $p(W\mid\mathrm{do}(S{=}0))$ — **beide** Pfade zusammen, also $H$ **nicht** konstant halten.
- **Direkter Effekt** von $S$ auf $W$: $p(W\mid\mathrm{do}(S), H)$ — die Größe **konstant halten** ($H$ stratifizieren), wodurch der indirekte Pfad blockiert wird.

> [!note] Vertiefung — der do-Operator als Graph-Chirurgie, und „it's not balance“ $\mathrm{do}(S{=}s)$ ist **wörtlich** das Löschen aller eingehenden Kanten von $S$ im DAG und das Festsetzen von $S=s$. Das ist der formale Grund, warum ein **randomisiertes Experiment** funktioniert: Randomisierung _ersetzt_ die natürlichen Ursachen von $S$ durch einen Münzwurf — sie **schneidet die Pfeile in $S$ durch**, exakt wie $\mathrm{do}$. McElreaths Spitze „it's not balance“ zielt genau darauf: Experimente wirken **nicht**, weil sie Kovariaten zwischen Gruppen ausbalancieren, sondern weil die Intervention **Hinterpfade kausal kappt**. Balance ist ein Symptom, nicht der Mechanismus.

---

## 5. „For-and-against“-Heuristiken

Gängige Faustregeln der angewandten Regression:

1. **Stratifiziere nach allem Prä-Treatment.**
2. **Stratifiziere nach nichts Post-Treatment.**

Die kausale Begründung (statt blindem Befolgen):

- **Prä-Treatment**-Variablen sind potenzielle **Confounder** → konditionieren blockiert ihren Hinterpfad. Gut.
- **Post-Treatment**-Variablen sind oft **Mediatoren** oder **Collider** → konditionieren öffnet entweder einen Bias oder zerstört den Effekt, den man messen will (z. B. den indirekten Pfad). Schlecht.

McElreath warnt aber: Diese Heuristiken **verbieten legitime Analysen** und können **endogenen Bias** riskieren — man sollte _verstehen, warum_ sie funktionieren, statt sie mechanisch anzuwenden. Konkret: Will man den **totalen** Effekt von $S$, darf man $H$ (post-treatment bzgl. $S$, da $S\to H$) **nicht** stratifizieren; will man den **direkten** Effekt, muss man es.

---

## 6. Kategorien: geordnet vs. ungeordnet

Zwei Typen diskreter Variablen:

- **Ungeordnet** (nominal): diskrete Typen ohne Reihenfolge — z. B. biologisches Geschlecht, Spezies.
- **Geordnet** (ordinal): diskrete Typen _mit_ Reihenfolge, aber ohne metrische Abstände — z. B. Bildungsgrad (Bachelor < Master < PhD).

Diese Vorlesung behandelt **ungeordnete** Kategorien; geordnete kommen später (sie brauchen eigene Maschinerie, da „Abstände“ unbekannt sind).

---

## 7. Index- vs. Dummy-Variablen — der Knackpunkt

Das ist der Teil, den du nochmal sauber haben wolltest. Beide kodieren dieselbe kategorielle Information für den Code, aber sie tun mathematisch **nicht** dasselbe.

### Die Dummy-Variante (Indikator, 0/1)

Für $S\in{\text{w},\text{m}}$ baut man eine 0/1-Variable, z. B. $D_i = 1$ falls männlich, sonst $0$. Frauen werden zur **Referenzkategorie**:

$$ \mu_i = \alpha + \beta_S,D_i \quad\Rightarrow\quad \begin{cases} \text{Frau } (D{=}0): & \mu = \alpha \ \text{Mann } (D{=}1): & \mu = \alpha + \beta_S \end{cases} $$

$\alpha$ ist also der **Mittelwert der Referenz** (Frauen), und $\beta_S$ die **Differenz** Mann $-$ Frau. (Im ML heißt die Variante mit _einer_ Indikatorspalte pro Kategorie statt $k-1$ „One-Hot“; statistisches Dummy-Coding nutzt klassisch $k-1$ Spalten mit Referenz.)

**Drei Probleme:**

1. **Willkürliche Asymmetrie.** Eine Kategorie ist die „Basis“ in $\alpha$, die andere nur eine Abweichung davon. Welche die Referenz ist, ist beliebig — aber es prägt die Parametrisierung.
2. **Ungleiche Prior-Unsicherheit** (der subtilste, eigentlich bayesianische Punkt). Setzt man $\alpha\sim\mathcal{N}(0,s)$ und $\beta_S\sim\mathcal{N}(0,s)$, dann hat der Mittelwert der Referenz a priori Varianz $s^2$, der Mittelwert der anderen Kategorie aber: $$ \mathrm{Var}(\alpha + \beta_S) = s^2 + s^2 = 2s^2. $$ Du behauptest also **unabsichtlich, über Männer a priori weniger zu wissen als über Frauen** — eine künstliche Asymmetrie, die niemand beabsichtigt hat.
3. **Schlechte Skalierung.** Bei $k$ Kategorien brauchst du $k-1$ Dummies und $k-1$ Koeffizienten, jeden **einzeln** in die Lineargleichung geschrieben. Bei Hunderten von Clustern (Länder, Individuen) wird das Modell unwartbar.

### Die Index-Variante (McElreaths Empfehlung)

Vergib **Ganzzahlen** als Kategorie-Codes, $S_i\in{1,2}$ (1 = Frau, 2 = Mann), und benutze den Index, um einen Parameter **auszuwählen**:

$$ \mu_i = \alpha_{S_i}, \qquad \alpha_j \sim \mathcal{N}(m, s)\ \text{ für jedes } j. $$

Es gibt jetzt einen **Vektor** von Achsenabschnitten $\alpha = (\alpha_1, \alpha_2)$ — einen pro Kategorie —, und für Person $i$ wählt man den passenden aus.

**Dieselben drei Punkte, gelöst:**

1. **Symmetrie.** $\alpha_1$ ist direkt der Frauen-Mittelwert, $\alpha_2$ direkt der Männer-Mittelwert. Keine Referenz, kein „Basis + Abweichung“.
2. **Gleiche Prior-Unsicherheit.** Jedes $\alpha_j$ bekommt **denselben** Prior unabhängig → identische A-priori-Unsicherheit für alle Kategorien. Die Asymmetrie verschwindet.
3. **Skaliert trivial.** Bei $k$ Kategorien ist $\alpha$ einfach ein Vektor der Länge $k$; der Modellcode `mu = a[S]` bleibt **unverändert**, egal ob $k=2$ oder $k=200$.

> [!important] Vertiefung — Index-Variable = Embedding-Lookup (deine ML-Brücke) Lies $\alpha_{S_i}$ als Operation: „nimm den Ganzzahl-Code der Kategorie von Person $i$ und **schlage damit die zugehörige Zeile** im Parametervektor nach“. Das ist **exakt** ein Embedding-Lookup — eine Embedding-Schicht holt per Integer-Token-ID die Zeile $S_i$ aus der Embedding-Matrix. Hier ist die Embedding-Dimension $1$ (ein Skalar pro Kategorie); mit mehreren Parametern pro Kategorie (Achsenabschnitt _und_ Steigung) wird es ein Embedding der Dimension $>1$. Und One-Hot? One-Hot $\times$ Gewichtsmatrix **ist** mathematisch derselbe Lookup, nur als Matrixmultiplikation geschrieben — $\mathbf{e}_{S_i}^\top A = A_{S_i}$. Die Dummy-Kodierung ist also dieselbe Operation in teurer, asymmetrischer Verkleidung (mit hineingebackener Referenz). Du kennst das Muster aus neuronalen Netzen längst.

> [!note] Vertiefung — die Differenz geht nicht verloren (summarize last!) Häufiger Einwand: „Aber bei Dummies bekomme ich $\beta_S$ = den Geschlechtsunterschied direkt — bei Index nicht.“ Doch: Du berechnest den Kontrast **nachträglich** als abgeleitete Größe aus den Posterior-Samples: $$\delta = \alpha_2 - \alpha_1 \quad\text{(pro Sample gezogen)}.$$ Das ist genau das **„summarize last“** aus A02 — und es ist sogar _korrekter_, weil $\alpha_1,\alpha_2$ aus dem **gemeinsamen** Posterior gezogen werden und ihre Kovarianz automatisch in $\delta$ einfließt. (Erinnerung an das „erste Gesetz“: Posterior-Parameter kovariieren; $\alpha$ und $\beta$ korrelieren bei unzentriertem $H$ oft stark negativ — der Grund, warum man $H$ zentriert, siehe A03.)

> [!info] Vertiefung — warum das das Tor zu Multilevel-Modellen ist Der eigentliche Grund, warum McElreath so auf Index-Variablen besteht: Sie sind die **natürliche Form für hierarchische Modelle**. Sobald du $\alpha_j \sim \mathcal{N}(m,s)$ stehen hast, kannst du $m$ und $s$ **selbst aus den Daten lernen** lassen (Hyperprior) — die Kategorien teilen sich dann eine gemeinsame Verteilung und werden **partiell gepoolt** (zueinander hingeschrumpft, eine Regularisierung _über Kategorien hinweg_). Mit $k=200$ Ländern ist das der Unterschied zwischen 200 verrauschten Einzelschätzungen und 200 stabilisierten. Mit Dummy-Coding ist dieser Schritt umständlich; mit Index-Variablen fällt er praktisch von selbst heraus.

---

## 8. Ausblick

Mit Index-Variablen und sauber definierten Estimands (totaler vs. direkter Effekt über den do-Operator) ist das Werkzeug bereit, um am Kalahari-Datensatz den **Geschlechtseffekt auf das Gewicht** zu zerlegen — direkt und über die Größe vermittelt. Die Mechanik des partiellen Poolings (gemeinsamer Hyperprior über die $\alpha_j$) ist der rote Faden in die Multilevel-Modelle der kommenden Vorlesungen.

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A04, abgeglichen mit deinen Folien-Screenshots. Korrekturen ggü. der Gemini-Fassung: (1) Estimands am do-Operator der Folien ausgerichtet — kausaler Effekt als Kontrast $p(W\mid\mathrm{do}(H{=}h)) - p(W\mid\mathrm{do}(H{=}h'))$ (Folie schreibt „casual“, gemeint ist **causal**); (2) Strukturgleichung $S=f_S(Y)$ statt $f_S(W)$, konsistent mit der DAG-Kante $Y\to S$ (ein $W\to S$ würde die Kausalrichtung umkehren); (3) One-Hot/Dummy-Nuance präzisiert ($k-1$ mit Referenz vs. $k$ Spalten). Die „Vertiefung“-Callouts sind meine Ergänzungen, nicht Teil des Vortrags._
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]