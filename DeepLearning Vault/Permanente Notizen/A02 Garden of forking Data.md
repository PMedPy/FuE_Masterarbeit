15-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended


# A02 – Garden of Forking Data

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 2_ Quelle: [YouTube – Lecture A02](http://www.youtube.com/watch?v=pGVkCWlXnlg)

---

## 1. Der Bayes'sche Workflow und die Logik des Aktualisierens

McElreaths zentrale Behauptung: In der Bayes-Statistik gibt es im Grunde **nur einen** Mechanismus, um aus Daten zu lernen — das **Bayes'sche Aktualisieren** (_Bayesian updating_). Man dreht immer dieselbe Kurbel.

Dem stellt er (als seine Sichtweise) die frequentistische Praxis gegenüber: Dort gibt es keine _eingebaute_ Logik, _wie_ ein Schätzer zu konstruieren ist. Man baut einen — Mittelwert, MLE, irgendein anderer — und untersucht **im Nachhinein** seine Eigenschaften (Bias, Varianz, Stichprobenverteilung). Konstruktion und Rechtfertigung sind getrennte Schritte.

**Das Ziel ist kein „wahrer" Punktwert.** Statt _den_ Wasseranteil $P$ zu finden, bestimmt man für **jeden** mathematisch möglichen Wert von $P$ seine Plausibilität _relativ_ zu allen anderen. Das Ergebnis ist eine ganze Verteilung über $P$.

---

## 2. Vom Zählen zur Wahrscheinlichkeit

Aufbauend auf A01 (4-seitiger Würfel-Globus): Man zählt im „Garten der verzweigenden Daten" die Pfade, die mit der beobachteten Sequenz verträglich sind. Bei wachsender Stichprobe werden die absoluten Pfadzahlen absurd groß (_„hella big numbers"_).

**Normalisierung.** Aus absoluten Zählungen werden Wahrscheinlichkeiten, indem man jede Zählung durch die **Summe aller Zählungen** teilt. Das Resultat ist die **Posterior-Verteilung** (_posterior distribution_):

$$ p(P \mid \text{Daten}) ;=; \frac{\text{Wege}(P)}{\sum_{P'} \text{Wege}(P')}. $$

Der Nenner ist reine Buchhaltung — er sorgt dafür, dass sich alles zu $1$ summiert, und ändert nichts an den _relativen_ Plausibilitäten.

---

## 3. Übergang zum kontinuierlichen Parameterraum

Ein echter Globus hat nicht 4 oder 20 Seiten, sondern unendlich viele mögliche Werte von $P \in [0,1]$. Beim Grenzübergang passiert genau eine Sache:

$$ \underbrace{\sum_{P'}}_{\text{diskret}} ;\longrightarrow; \underbrace{\int_0^1 ; \mathrm{d}P'}_{\text{stetig}}. $$

Aus der Summe über fünf Hypothesen wird ein Integral über ein Kontinuum. Die Posterior-Formel sieht sonst identisch aus; die **Normalisierungskonstante** im Nenner ist jetzt ein Integral und sorgt weiterhin nur dafür, dass die **Fläche unter der Kurve** exakt $1$ ergibt.

> [!note] Vertiefung — Bayes im Kontinuum als Sprache gelesen $$p(\theta \mid D) ;=; \frac{\overbrace{p(D \mid \theta)}^{\text{Likelihood}};\overbrace{p(\theta)}^{\text{Prior}}}{\underbrace{\int_0^1 p(D \mid \theta'),p(\theta'),\mathrm{d}\theta'}_{\text{Evidenz / Normierung}}}$$ Lies den Nenner nicht als Formel, sondern als Satz: er ist die **über alle Hypothesen gemittelte Likelihood** — „wie gut erklären die Daten sich _im Schnitt_, gewichtet mit dem Prior". Genau weil er nicht von $\theta$ abhängt (über $\theta'$ ist wegintegriert), ist er beim Vergleich der $\theta$-Werte eine **Konstante** und beeinflusst die _Form_ des Posteriors nicht — nur seine Höhe, damit die Fläche $1$ wird. Das ist der Grund, warum man so oft $p(\theta\mid D)\propto p(D\mid\theta),p(\theta)$ schreibt und das Integral erst ganz am Schluss ausrechnet (oder per Sampling umgeht, siehe §6).

### Warum die Glockenform _logisch_ entsteht

McElreath animiert, wie sich die typische „Hügelform" rein aus dem **Multiplizieren** der einzelnen Beobachtungen ergibt. Eine _einzelne_ Beobachtung liefert als Likelihood-Funktion über $\theta$ nur eine **Gerade**:

- eine Wasser-Beobachtung ($W$): die Linie $;\theta;$ (steigt von $0$ auf $1$),
- eine Land-Beobachtung ($L$): die Linie $;1-\theta;$ (fällt von $1$ auf $0$).

Mit jedem neuen Datenpunkt **wandert und verengt** sich der entstehende Hügel — mehr Evidenz, schmälere Verteilung.

> [!important] Vertiefung — „Falten" ist hier _Multiplikation_, nicht Faltung Im Transkript steht „Falten der Wahrscheinlichkeitsgeraden" — das ist sprachlich irreführend. Eine **Faltung** (_convolution_) kombiniert die Verteilungen _summierter_ Zufallsvariablen. Was hier passiert, ist etwas anderes und Einfacheres: an **jedem Punkt $\theta$** werden die Faktoren **punktweise multipliziert**. Aus den Geraden wird so ein Polynom: $$\prod_{\text{Würfe}} (\text{Linie}) ;=; \theta^{,W},(1-\theta)^{,L}.$$ Und das ist exakt der **Kern der Beta-Verteilung**. Damit schließt sich der Kreis zu zwei Dingen:
> 
> 1. zur diskreten A01-Formel $(4P)^W(4-4P)^L = 4^N,\theta^W(1-\theta)^L$ — derselbe Ausdruck, nur ohne den konstanten Faktor $4^N$;
> 2. zur Konjugiertheit aus unserem CRISPR-Fall: flacher Prior $\times$ diese Likelihood $=\text{Beta}(W+1,,L+1)$. Die „Hügelform" _ist_ die Beta-Dichte, und das Wandern/Verengen ist das Hochzählen ihrer Parameter.

---

## 4. Eigenschaften der Posterior-Verteilung

### Keine Mindeststichprobengröße

Im Bayes-Framework gibt es **kein** Mindest-$N$ (keine Faustregel wie das frequentistische $N=30$). Die Inferenz ist für **genau die Daten, die man hat**, mathematisch exakt. Sogar mit $N=0$ kann man arbeiten — dann ist der Posterior schlicht der **Prior**.

> [!info] Vertiefung — „exakt" heißt nicht „sicher" (Rückbezug auf die Berry–Esseen-Diskussion) Hier lohnt die Verbindung zu unserem Gespräch über das $N>30$-Folklore. Die frequentistische $30$ war eine **Approximationsschwelle**: ab wann ist die _Stichprobenverteilung eines Schätzers_ näherungsweise normal (CLT, mit Berry–Esseen-Rate $\propto \beta_3/\sqrt{n}$)? Das ist eine Aussage über ein _Näherungsverfahren_. Bayes macht keine solche Approximation — der Posterior ist immer die exakte logische Konsequenz aus Modell $+$ Prior $+$ Daten. Aber **exakt $\neq$ präzise**: bei wenig Daten ist der Posterior einfach **sehr breit**. Die Methode lügt nicht über kleine Stichproben, sie zeigt die Unsicherheit ehrlich als Breite. Du tauschst also die Approximationsannahme gegen die Notwendigkeit eines Priors — und bekommst dafür eine Antwort, die bei jedem $N$ wohldefiniert ist.

### Die Verteilung _ist_ der Schätzer

Ein Bayes-„Schätzer" liefert nie einen nackten Punktwert, sondern immer die **vollständige Unsicherheitsverteilung**. Punktschätzer (Mittelwert, Modus, Median) sind nur **Kommunikations-Abkürzungen** — man wirft Information weg, um eine Zahl nennen zu können.

### Intervalle ohne Magie

Man beschreibt die Form mit Intervallen, etwa einem zentralen 50%- oder 89%-Intervall. McElreath warnt davor, den Grenzen (besonders dem rituellen 95%-Intervall) **magische Eigenschaften** zuzuschreiben: Die Dichte ist stetig, an der Grenze passiert nichts Besonderes. Die berüchtigte 89% ist bewusst krumm gewählt — eine Primzahl ohne kulturellen Ballast —, gerade um zu betonen, dass die Zahl willkürlich ist.

> [!note] Vertiefung — Dichte vs. Masse (der Punkt aus Minute 33:54) Im Stetigen ist der Posterior eine **Wahrscheinlichkeitsdichte**, keine -masse. Konsequenz: die Wahrscheinlichkeit _eines einzelnen_ Werts ist $P(\theta = 0.5) = 0$. Sinn ergibt nur die **Fläche über einem Intervall**: $$P(a < \theta < b) = \int_a^b p(\theta \mid D),\mathrm{d}\theta.$$ Eine Dichte kann lokal sogar $>1$ sein (es ist eine Dichte, kein Anteil) — nur das _Integral_ ist auf $1$ normiert. Das ist auch der formale Grund für „keine Magie an der Grenze": Du verschiebst bloß die Integrationsgrenze um $\mathrm{d}\theta$, und die Fläche ändert sich glatt. Genau diese Logik stand schon hinter dem $89% $ -Glaubwürdigkeitsintervall, das du im CRISPR-Fall berichtet hast — _dort_ darf man „mit 89% liegt $\theta$ in …" sagen, beim frequentistischen CI nicht.

---

## 5. Posterior-Prädiktion (Vorhersage & Modellcheck)

Wissenschaftler interessieren sich meist nicht für den Parameter $\theta$ an sich, sondern für **beobachtbare Ereignisse** in der Welt. Dazu füttert man den fertigen Posterior **zurück** ins generative Modell und simuliert neue Daten. Das Ergebnis ist die **Posterior-Predictive-Distribution** — die Vorhersageverteilung, die die **gesamte** Unsicherheit des Modells trägt.

> [!important] Vertiefung — Posterior-Predictive bündelt _zwei_ Unsicherheiten Das ist der Kernpunkt, den man leicht unterschätzt. Die Vorhersage künftiger Daten $\tilde{y}$ ist $$p(\tilde{y} \mid D) = \int_0^1 \underbrace{p(\tilde{y} \mid \theta)}_{\text{(2) Stichprobenrauschen}};\underbrace{p(\theta \mid D)}_{\text{(1) Parameterunsicherheit}},\mathrm{d}\theta.$$ Es stecken **zwei** Quellen von Streuung drin:
> 
> 1. **Parameterunsicherheit** — wir kennen $\theta$ nicht exakt (Breite des Posteriors).
> 2. **Stichprobenrauschen** — selbst bei _bekanntem_ $\theta$ ist der einzelne Ausgang zufällig (Binomial-Streuung).
> 
> Wer stattdessen den **Punktschätzer** $\hat\theta$ einsetzt und $p(\tilde y\mid\hat\theta)$ rechnet, behält nur Quelle (2) und wirft (1) weg → systematisch **zu selbstsichere** Vorhersagen. Genau dieser Fehler ist der Grund für die nächste Regel.

---

## 6. Die goldene Regel: „Summarize Last"

**Niemals** einen Parameter vorab zu einem Punktwert kollabieren, um damit weiterzurechnen. Alle Berechnungen und Simulationen laufen über die **gesamte** Verteilung; zusammengefasst (Mittelwert, Intervall) wird **erst ganz am Schluss**.

In der Praxis erledigt man die Integrale meist nicht analytisch, sondern per **Sampling**: man zieht z. B. $10^4$ Werte $\theta^{(1)},\dots,\theta^{(N)}$ aus dem Posterior, schickt jeden durch das generative Modell und beschreibt erst die _resultierende Punktwolke_. Integrieren wird so zu **Zählen** — der gleiche Geist wie der „Garten" aus A01, nur numerisch.

> [!info] Vertiefung — warum „summarize last" und Posterior-Predictive dieselbe Münze sind Beides ist exakt eine Aussage über **Reihenfolge von Operationen**: Der Mittelwert einer nichtlinearen Funktion ist nicht die Funktion des Mittelwerts, $$\mathbb{E}[g(\theta)] \neq g(\mathbb{E}[\theta]) \quad(\text{Jensen, sobald } g \text{ gekrümmt ist}).$$ „Erst mitteln, dann einsetzen" vertauscht $\mathbb{E}$ und $g$ — und unterschätzt typischerweise die Streuung. „Summarize last" heißt schlicht: **erst $g$ auf die ganze Verteilung anwenden, dann $\mathbb{E}$**. Konkret im CRISPR-Fall war das der Grund, $P(\theta > 0.2)$ über die **ganze** $\text{Beta}(6,3)$ zu berechnen, statt nur zu prüfen, ob der Posterior-Mittelwert $0.67 > 0.2$ ist.

---

## 7. Ausblick

Die nächste Vorlesung behandelt **Prior Predictions** — das Gegenstück zur Posterior-Prädiktion: das Modell _vor_ dem Datenkontakt vorwärts laufen lassen, um zu prüfen, ob der Prior überhaupt wissenschaftlich plausible Daten erzeugt (genau der Prior-Predictive-Check, den ich im CRISPR-Fall von dir wollte).

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A02. Die als „Vertiefung" markierten Callouts sind ergänzende Erläuterungen und nicht Teil des Originalvortrags. Korrigiert wurde u. a. „Falten" → punktweise Multiplikation (§3)._

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]