18-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# A03 – Geozentrische Modelle (Lineare Regression)

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 3_ Quelle: [YouTube – Lecture A03](https://www.youtube.com/watch?v=JX_UyidsQNg)

---

## 1. Kernidee

Die Vorlesung führt die **lineare Regression** aus genuin bayesianischer Sicht ein. Historisch ist sie _ursprünglich_ ein bayesianisches Verfahren — der Frequentismus kam später. Der konzeptionelle Leitsatz, der den ganzen Kurs trägt:

> Die Bedeutung der Schätzwerte einer Regression hängt **immer** von einem externen wissenschaftlichen (kausalen/mechanistischen) Modell ab, das _außerhalb_ der Regression liegt. **Das statistische Modell muss im Tandem mit einem Kausalmodell entwickelt werden.**

---

## 2. Geozentrische Modelle als Analogie

**Retrograde Bewegung.** Über dem Fixsternhintergrund wandern die Planeten (von lat. _Wanderer_) meist in eine Richtung, verlangsamen, **kehren scheinbar um**, und setzen dann den Pfad fort. Beim Mars (erdnah) ist das am auffälligsten.

**Ptolemäus' geozentrisches Modell** stellte die Erde ins Zentrum und ließ Planeten auf **Epizykeln** (Kreise auf Kreisen) laufen. Zwei Eigenschaften:

- **Strukturell falsch** — die Erde steht nicht im Zentrum.
- **Vorhersagepräzise** — über Jahrhunderte hervorragend. Ineinander rotierende Kreise können _jede_ stetige Kurve beliebig genau approximieren (im Kern eine Fourier-Reihe).

**Die Analogie:** Lineare Regressionen sind die _geozentrischen Modelle der Statistik_ — extrem nützliche Approximations- und Vorhersagewerkzeuge, deren additive Struktur aber **nicht** garantiert, dass der zugrunde liegende Mechanismus korrekt abgebildet wird. Sie sind **deskriptive Assoziationsmodelle**, keine mechanistischen. (Die _echte_ Erklärung des Zickzacks ist heliozentrisch: die innere, schnellere Erde überholt den Mars — der Zickzack ist ein reiner Perspektiveffekt.)

---

## 3. Gauß und das verlorene Objekt Ceres

Um 1800 war der Heliozentrismus akzeptiert, aber Entfernungs- und Positionsmessungen waren grob. Zwischen Mars und Jupiter wurde **Ceres** (heute als Zwergplanet klassifiziert, der größte Asteroid) gesichtet — und nach wenigen ungenauen Beobachtungen _wieder verloren_. Ein Wettbewerb suchte die Wiederkehrposition.

Der junge, noch arme **Carl Friedrich Gauß** wandte sich der (von ihm laut Tagebuch ungeliebten, weil „unreinen“) Astronomie zu. Aus den schlechten Datenpunkten erfand er praktisch das gesamte Regressionsgerüst — die **Methode der kleinsten Quadrate** (OLS) — und leitete sie in seinem Werk von 1809 **vollständig bayesianisch** her (eine andere Wahrscheinlichkeitstheorie gab es nicht). Er suchte einen schnellen Weg, den Posterior _ohne Computer_ zu approximieren — der direkte Vorläufer der quadratischen Approximation in §8. Er gewann, sagte Ceres' Wiederkehr exakt voraus und wurde berühmt.

---

## 4. Die Normalverteilung: zwei Argumente

Gauß leitete 1809 auch die Glockenkurve her — allein aus der Annahme, dass Beobachtungsfehler **symmetrisch** in alle Richtungen streuen. Warum die Normalverteilung überall auftaucht, hat zwei Begründungen:

**(1) Das generative Argument (Additivität).** Symmetrie ist nicht einmal nötig — entscheidend ist, dass sich **Fluktuationen addieren**. Gedankenexperiment Fußballfeld: Alle stehen auf der Mittellinie, jeder wirft wiederholt eine Münze (Kopf = Schritt links, Zahl = Schritt rechts). Die Endposition ist die **Summe** aller Schritte. Die meisten landen nahe der Mitte (Fluktuationen heben sich auf), wenige am Rand → im Aggregat **zwangsläufig eine Normalverteilung**.

**(2) Das inferentielle Argument.** Die Gauß-Verteilung hat nur **zwei** Parameter: Mittelwert und Streuung. Als Likelihood ist sie damit eine reine „Maschine zur Schätzung von Mittel und Streuung“. Sie liefert in einer Regression **korrekte Antworten für $\alpha,\beta$, selbst wenn die Daten gar nicht normalverteilt sind** (z. B. gleichverteilt) — denn geschätzt wird der bedingte _Mittelwert_, nicht die Form der Rohdaten.

> [!important] Vertiefung — Normalität sagt _nichts_ über den Mechanismus (Rückbezug auf Berry–Esseen & Robustheit) Das Fußballfeld **ist** der zentrale Grenzwertsatz als Bild: jede Endposition = Summe vieler unabhängiger ±1-Schritte → Normal. Genau diese Summe haben wir vorhin mit **Berry–Esseen** quantifiziert (Konvergenzrate $\propto \beta_3/\sqrt{n}$). Die stillschweigende Bedingung in McElreaths Kasten — _„unter endlicher Varianz“_ — ist **dieselbe**, die bei schweren Rändern (Cauchy/stabil) bricht; und das war wiederum der Grund, in der Ausreißer-Abbildung zur **Student-t** zu greifen. Ein roter Faden über drei Gespräche.
> 
> Die wissenschaftliche Pointe: Weil _viele verschiedene_ additive Mikroprozesse **denselben** Glockenausgang erzeugen, ist die Verteilung **nicht invertierbar** auf ihre Ursache. Aus „die Daten sind normal“ folgt **kein** biologischer Mechanismus — nur „viele kleine Effekte wirken additiv“. **Normalität ist maximale Ignoranz (Maximum-Entropie bei gegebenem Mittel und Varianz), kein mechanistischer Fingerabdruck.**

---

## 5. Der bayesianische Workflow

Vorab eine didaktische Wahrheit McElreaths: Man muss nicht alles _gleichzeitig_ verstehen. Es reicht, im **Flow** zu bleiben — Widerstand und leichte Verwirrung spüren, aber sich kontinuierlich weiterbewegen. Gute Wissenschaft ist möglich, bevor jedes Detail durchdrungen ist.

|Schritt|Beschreibung|
|---|---|
|1. Daten sichten|Datensatz und Variablen kennenlernen.|
|2. Estimand definieren|Festlegen, _was_ geschätzt wird.|
|3. Generatives Modell|Theoretisches Modell, das Daten synthetisieren kann.|
|4. Statistisches Modell|Struktur und Verteilungen formal festlegen.|
|5. Prior-Predictive-Sim.|Daten allein aus den Priors simulieren, Implikationen prüfen.|
|6. Posterior-Update|Echte Daten einpflegen, Posterior berechnen.|
|7. Posterior-Predictive|Vorhersagen erzeugen, Modell prüfen und interpretieren.|

---

## 6. Datensatz und Kausalmodell

Beispiel: der anthropometrische Datensatz von **Nancy Howell** (`Howell1` im `rethinking`-Paket) über die **!Kung San** der Kalahari (1960er). Fokus auf **Erwachsene ($\geq 18$ J.)**, weil das Wachstum abgeschlossen und der Zusammenhang dort annähernd linear ist.

Ziel: Einfluss der **Körpergröße $H$** auf das **Gewicht $W$**. Kausales Denken explizit: Eine Intervention auf $H$ (Beine kürzen) änderte $W$ drastisch; eine Intervention auf $W$ (mehr essen) ändert $H$ nicht. Der DAG:

$$ H \longrightarrow W \longleftarrow U $$

$U$ (eingekreist = unbeobachtet) bündelt alle ungemessenen Einflüsse auf $W$ (Muskelmasse, Knochendichte, Mageninhalt …) — denn $W$ ist nicht deterministisch durch $H$ festgelegt.

---

## 7. Die formale Modellsprache & das generative Modell

Die Notation ist **in beide Richtungen lesbar**: vorwärts (generativ, Daten erzeugen) und rückwärts (statistisch, Parameter schätzen). Links die Variablen, rechts ihre Definition — entweder deterministisch ($=$) oder distributionell ($\sim$). Der Index $i$ = Individuum / Datenzeile.

$$ \begin{aligned} W_i &= \beta,H_i + U_i \ U_i &\sim \text{Normal}(0,\sigma) \ H_i &\sim \text{Uniform}(130,,170) \end{aligned} $$

Symbol-Bedeutung als Sprache:

- $=$ — **strikt deterministische** Verknüpfung.
- $\sim$ — „ist verteilt als“: bringt **Zufall / Fluktuation** ins Modell.
- $\beta$ — Proportionalitätsfaktor (Steigung), einheitenabhängig (kg pro cm).
- $\sigma$ — Streuung der unbeobachteten Komponente $U$.

Voll generativ: In R erzeugt `sim_weight(H,b,sd)` mit `U <- rnorm(length(H),0,sd); W <- b*H + U` täuschend echte Kunstdaten — zur **Validierung** der Methode (kennt man die wahren $\beta,\sigma$, muss der Schätzer sie zurückgewinnen).

> [!important] Vertiefung — das generative Modell hat _keinen_ Achsenabschnitt Schau genau hin: Das generative Modell ist $W=\beta H + U$ — **ohne $\alpha$**. Es setzt also implizit $\alpha = 0$ (bei $H=0$ ist $W$ im Mittel exakt $0$). Das gleich folgende _statistische_ Modell (§8) fügt dagegen ein $\alpha$ hinzu. Das ist Absicht: Man **simuliert** Daten mit wahrem $\alpha=0$ und prüft, ob der Schätzer $\alpha\approx 0$ **zurückfindet**. Generatives Modell = Spezialfall des statistischen.
> 
> Zweite Feinheit, dieselbe Vorwärts/Rückwärts-Dualität wie in A02: Das generative Modell **zieht** $H_i \sim \text{Uniform}$, um fiktive Personen zu bauen. Das statistische Modell behandelt beobachtete $H_i$ als **gegeben** und konditioniert darauf ($\mathrm{E}(W\mid H)$). Dieselbe Notation, zwei Leserichtungen.

---

## 8. Statistisches Modell & quadratische Approximation

Geschätzt wird der **bedingte Erwartungswert** $\mathrm{E}(W\mid H)$:

$$ \begin{aligned} W_i &\sim \text{Normal}(\mu_i,\sigma) \ \mu_i &= \alpha + \beta,H_i \end{aligned} $$

$\mu_i$ = erwartetes Gewicht für Individuum $i$, aus **Achsenabschnitt $\alpha$** und **Steigung $\beta$**. Drei unbekannte Parameter, für die _simultan_ ein Posterior geschätzt wird: $\alpha,\beta,\sigma$.

**Die Posterior-Formel** (Folie):

$$ \underbrace{\Pr(\alpha,\beta,\sigma \mid H_i,W_i)}_{\text{Plausibilität einer konkreten Geraden}} ;=; \frac{\overbrace{\Pr(W_i\mid H_i,\alpha,\beta,\sigma)}^{\text{,,garden of forking data`` = Likelihood}};\overbrace{\Pr(\alpha,\beta,\sigma)}^{\text{Prior}}}{\underbrace{Z}_{\text{Normierung}}} $$

> [!note] Vertiefung — derselbe Apparat wie in A01/A02, nur 3-dimensional McElreath beschriftet die Likelihood bewusst mit _„garden of forking data“_ — es ist exakt das Pfadezählen aus A01, jetzt über einen **kontinuierlichen 3D-Parameterraum** $(\alpha,\beta,\sigma)$ statt über fünf Würfel-Hypothesen. **Jeder Punkt** $(\alpha,\beta,\sigma)$ ist eine konkrete Gerade-plus-Streuung; die Likelihood zählt, wie gut diese Gerade die Daten erzeugt. $Z=\iiint \text{Likelihood}\cdot\text{Prior},\mathrm{d}\alpha,\mathrm{d}\beta,\mathrm{d}\sigma$ ist dieselbe Konstante wie immer — die über alle Geraden gemittelte Likelihood, die nur die Fläche auf $1$ bringt.

**Quadratische Approximation (Laplace-Approximation, `quap`).** Statt langsamer Grid-Approximation nimmt man an, der gemeinsame Posterior sei näherungsweise eine **multivariate Normalverteilung**. Dank CLT ist das für lineare Modelle bei genug Daten „unvernünftig effektiv“, rechnet in Sekundenbruchteilen und stimmt mit der Maximum-Likelihood-Rechnung des Frequentismus überein (benannt nach **Laplace**, dem Urvater der angewandten Bayes-Statistik).

> [!note] Vertiefung — _warum_ „quadratisch“? (das kennst du als Laplace-Approx. aus Bishop) Der Name ist wörtlich zu nehmen. Sei $\ell(\theta)=\log p(\theta\mid D)$ die Log-Posterior-Dichte, $\hat\theta$ ihr **Modus** (Gipfel). Taylor um den Gipfel: $$\ell(\theta)\approx \ell(\hat\theta) + \underbrace{\nabla\ell(\hat\theta)^\top(\theta-\hat\theta)}_{=,0\ \text{am Gipfel}} + \tfrac{1}{2}(\theta-\hat\theta)^\top \mathbf{H},(\theta-\hat\theta).$$ Lies die Operatoren: $\nabla\ell = 0$ heißt „**oben auf dem Hügel** ist die Steigung null“. Der **lineare Term fällt weg**, der erste nicht-triviale Term ist **quadratisch** — daher der Name. Die Hesse-Matrix $\mathbf{H}$ (Krümmung) misst, _wie spitz_ der Gipfel ist. Exponenziert man eine Quadratform, bekommt man eine **Gauß-Glocke**: $$p(\theta\mid D)\approx \mathcal{N}!\big(\hat\theta,\ -\mathbf{H}^{-1}\big).$$ Lesart der Krümmung: **scharfer Gipfel** (große $|\mathbf H|$) → $-\mathbf H^{-1}$ klein → **schmaler Posterior, wenig Unsicherheit**. Und die Brücke zum Frequentismus: bei flachem Prior ist Posterior $\propto$ Likelihood, also $\hat\theta=$ **MLE** und $-\mathbf H^{-1}=$ inverse beobachtete **Fisher-Information** — exakt die asymptotische Kovarianz des MLE. Genau die Laplace-Approximation aus Bishop §4.4, hier als `quap`.

---

## 9. Priors und Prior-Predictive-Simulation

Priors sollen **weiches** wissenschaftliches Vorwissen ausdrücken (_weakly informative priors_) — nicht alles starr vorgeben, sondern nur **unmögliche** Resultate als unplausibel markieren und Raum für Fehlspezifikations-Entdeckung lassen. Die Randbedingungen und die konkreten Priors der Folie:

$$ \begin{aligned} W_i &\sim \text{Normal}(\mu_i,\sigma), \quad \mu_i=\alpha+\beta H_i\ \alpha &\sim \text{Normal}(0,10) \qquad &&\text{(bei } H=0 \Rightarrow W=0\text{: um 0 zentriert)}\ \beta &\sim \text{Uniform}(0,1) \qquad &&\text{(Gewicht steigt mit Größe; kg < cm)}\ \sigma &\sim \text{Uniform}(0,10) \qquad &&\text{(Streuung muss positiv sein)} \end{aligned} $$

**Prior-Predictive-Simulation.** Reine Zahlenlisten von Priors sind für den Verstand kaum interpretierbar. Weil das Modell generativ ist, zieht man $\alpha,\beta$ direkt aus den Priors — **ganz ohne Daten ($N=0$)**. Jede gezogene Kombination ist **eine Gerade** im $(H,W)$-Raum. Plottet man viele solcher Geraden, sieht man sofort, was das Modell _vor_ den Daten „glaubt“. Bei zu vagen Priors entstünden absurde Linien (größere Menschen _leichter_, oder irrwitzige Steigungen). Die Simulation zwingt einen, die Priors logisch zu justieren.

> [!info] Vertiefung — $\alpha\sim\text{Normal}(0,10)$ ist ein Platzhalter, kein „Gewicht bei Größe 0“ Die Begründung „$H=0\Rightarrow W=0$, also $\alpha$ um $0$“ ist plausibel, aber pass auf: Eine durch _Erwachsenen_-Daten gelegte Gerade bis $H=0$ zu verlängern ist eine **massive Extrapolation** weit außerhalb der Daten — $\alpha$ ist hier ein **mathematischer** Achsenabschnitt, keine biologisch sinnvolle Größe. Im Buch löst McElreath das später, indem er die **Größe zentriert** ($H-\bar H$): dann ist $\alpha$ das Gewicht bei _mittlerer_ Größe — interpretierbar, datennah, und es **entkoppelt** $\alpha$ und $\beta$ im Posterior (sonst stark korreliert). Merk dir das als Vorgriff; in dieser Vorlesung bleibt es beim rohen $H$.

---

## 10. Ausblick: das Bayes-Update

Sobald die echten !Kung-San-Daten Punkt für Punkt einfließen, schrumpft der plausible Parameterraum. Die multivariate Normalverteilung des Posteriors wird **schmaler und steiler**; im Datenraum stabilisieren sich die anfangs wild streuenden Prior-Geraden und legen sich **eng um den von den Daten diktierten Trend**. Die nächste Vorlesung pflegt genau hier die Daten ein, erzeugt Posterior-Vorhersagen und interpretiert sie wissenschaftlich.

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A03, abgeglichen mit deinen Folien-Screenshots. Korrekturen ggü. der Gemini-Fassung: (1) Höhen-Range im generativen Modell $\text{Uniform}(138,172)\to\text{Uniform}(130,170)$ laut Folie; (2) „weekly“ → **weakly** informative priors; (3) die konkreten Priors $\alpha\sim\text{Normal}(0,10),\ \beta\sim\text{Uniform}(0,1),\ \sigma\sim\text{Uniform}(0,10)$ und die beschriftete Posterior-Formel aus den Screenshots ergänzt. Die „Vertiefung“-Callouts sind meine Ergänzungen, nicht Teil des Vortrags._

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]