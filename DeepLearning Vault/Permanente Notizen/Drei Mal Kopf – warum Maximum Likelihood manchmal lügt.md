22-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

---
# Drei Mal Kopf – warum Maximum Likelihood manchmal lügt

Aus "Deep Learning: Foundations and Concepts":

> ***"One aspect of the Bayesian viewpoint is that the inclusion of prior knowledge arises naturally. Suppose, for instance, that a fair-looking coin is tossed three times and lands heads each time. The maximum likelihood estimate of the probability of landing heads would give 1, implying that all future tosses will land heads! By contrast, a Bayesian approach with any reasonable prior will lead to a less extreme conclusion."***

Bishop wählt für eine seiner pointiertesten Aussagen ein lächerlich kleines Beispiel: Eine faire Münze, dreimal geworfen, dreimal Kopf. Was schätzt Maximum Likelihood ([[Likelihood Maximization]]) für die Kopf-Wahrscheinlichkeit $\mu$? Die Likelihood ist $L(\mu) = \mu^3$, und ihr Maximum liegt bei

$$ \hat{\mu}_{\text{ML}} = 1. $$

Die Schätzung sagt also: Mit Wahrscheinlichkeit eins wird jeder zukünftige Wurf Kopf zeigen. Wer eine Münze schon einmal in der Hand hatte, weiß, dass das absurd ist. Drei Würfe sind kein Beleg dafür, dass die Münze gar keine Zahl-Seite hat.

Die Pointe liegt nicht darin, dass Maximum Likelihood "schlechter" wäre als bayessche Inferenz. 

> [!abstract] Die Pointe ist subtiler:
> Maximum Likelihood hat _kein konzeptionelles Werkzeug_, um auszudrücken, was wir über Münzen _vor_ dem Experiment wissen. Das Vorwissen – Münzen sind in der Regel ungefähr fair, jedenfalls nicht deterministisch – existiert in der frequentistischen Formulierung nicht.

Die Methode tut so, **als würde sie das Problem voraussetzungsfrei angehen**, und produziert genau deswegen eine voraussetzungsreiche, nämlich unsinnige Antwort.

Eine bayessche Analyse mit einem milden Prior – etwa $\mu \sim \text{Beta}(2, 2)$, was leicht zu Fairness tendiert, aber alle Werte zulässt – kommt zu einem Posterior-Mittelwert von $\hat{\mu} = 5/7 \approx 0{,}71$. Das ist eine vernünftige Schätzung: Die Daten haben die Erwartung verschoben, aber nicht in eine Sicherheit gekippt, die sie nicht tragen können.

Was Bishop hier in einem Absatz zeigt, ist das gleiche Argument, das Cox' Theorem in dreißig Seiten beweist: Wer Vorwissen ignoriert, ignoriert es nicht – er versteckt es nur. Die Maximum-Likelihood-Schätzung _hat_ **implizit einen Prior**, nämlich den uniformen über $[0,1]$, der jeder Hypothese – auch der Hypothese, dass die Münze deterministisch Kopf produziert – die gleiche A-priori-Plausibilität zuweist. Diese Wahl ist nicht neutral; sie ist eine sehr spezifische Behauptung über die Welt, die in den meisten realistischen Situationen falsch ist.

Genau das macht das Beispiel didaktisch so wertvoll. Es ist kein Grenzfall, sondern ein Modellfall: Bei kleinen Stichproben ist Vorwissen nicht _zusätzlich_ hilfreich – es ist _konstitutiv_ für eine sinnvolle Schlussfolgerung. Die bayessche Methode macht das sichtbar. Die frequentistische lässt es als verschwiegene Voraussetzung im Hintergrund stehen.

Drei Münzwürfe sind selten genug, um daran zu erinnern.


# Nachtrag: Der uniforme Prior

**Was technisch stimmt:**

Wenn du Maximum a Posteriori (MAP) mit einem **uniformen Prior** $p(\mu) = 1$ auf $[0,1]$ rechnest, bekommst du genau das Maximum-Likelihood-Ergebnis. Denn der Posterior ist

$$ p(\mu \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mu) \cdot p(\mu), $$

und wenn $p(\mu)$ konstant ist, fällt es bei der Maximierung weg. Der **Modus** des Posteriors ist dann identisch mit dem Maximum-Likelihood-Schätzer. Insofern _kann_ man Maximum Likelihood als bayessche Punktschätzung mit uniformem Prior interpretieren – das ist eine legitime mathematische Lesart.

ABER:

Maximum Likelihood selbst _postuliert_ keinen Prior. Es ist eine Methode, die ohne Bayes-Theorem auskommt: Du nimmst die Likelihood-Funktion und maximierst sie, fertig. Frequentisten würden mit Recht protestieren, wenn man ihnen unterstellt, sie würden "heimlich" einen uniformen Prior benutzen – sie benutzen überhaupt keinen Prior, weil das Konzept in ihrem Framework nicht existiert.

Die korrektere Aussage ist also: _Falls_ man Maximum Likelihood in das bayessche Framework einbettet (um die Methoden zu vergleichen), **entspricht** es einer bayesschen Punktschätzung mit uniformem Prior. Das ist eine Aussage _über_ Maximum Likelihood aus bayesscher Sicht, nicht eine versteckte Voraussetzung _innerhalb_ von Maximum Likelihood.

**Warum die Pointe trotzdem hält:**

Das eigentliche Argument im Münzwurf-Beispiel überlebt diese Präzisierung. Es lautet sauberer formuliert: Maximum Likelihood hat _keinen Mechanismus_, um Vorwissen einzubeziehen. **Es behandelt jede Stichprobe so, als wäre sie die einzige Informationsquelle**. In Situationen, wo Vorwissen reichhaltig und die Stichprobe klein ist, führt das zu absurden Schätzungen – nicht, weil ein "schlechter Prior" am Werk wäre, sondern weil _gar kein_ Prior am Werk ist, der das Vorwissen einbringen könnte.

Den Satz in deinem Miniessay würde ich also so umformulieren:

> Eine Maximum-Likelihood-Schätzung _entspricht_ einer bayesschen Schätzung mit uniformem Prior – sie behandelt jede Hypothese, von "Münze ist fair" bis "Münze produziert deterministisch Kopf", a priori als gleich plausibel. **Das ist keine neutrale Ausgangsposition, sondern eine sehr spezifische Annahme über die Welt, die in den meisten realistischen Situationen das Vorwissen ignoriert**, das jeder vernünftige Forscher _eigentlich_ hätte.

## Ist Maximum Likelihood = frequentistische Wahrscheinlichkeit?

Nein. Das ist eine wichtige Unterscheidung, und es lohnt sich, sie sauber zu ziehen.

**Maximum Likelihood ist eine Schätzmethode.** Sie sagt: Wähle den Parameter, der die Wahrscheinlichkeit der beobachteten Daten maximiert. Mehr nicht. Sie macht für sich genommen keine Aussage darüber, was Wahrscheinlichkeit _bedeutet_.

**Frequentismus ist eine Interpretation von Wahrscheinlichkeit.** Er sagt: Wahrscheinlichkeit ist die langfristige Häufigkeit eines Ereignisses bei wiederholtem Experiment. Diese Interpretation prägt, wie man Konfidenzintervalle, Hypothesentests und p-Werte versteht, hat aber mit dem konkreten Algorithmus "Maximum Likelihood" erst mal nichts zu tun.

**Die Brücke:**

Maximum Likelihood ist in der _Praxis_ überwiegend mit der frequentistischen Schule assoziiert, weil:

Erstens entstand Maximum Likelihood historisch im frequentistischen Lager (R. A. Fisher, 1922). Fisher hat sich vehement von der bayesschen Tradition abgegrenzt und Maximum Likelihood als bewusste Alternative zum Bayes-Theorem entwickelt. Diese historische Allianz wirkt bis heute nach.

Zweitens passen die _Folgeprodukte_ von Maximum Likelihood – etwa asymptotische Konfidenzintervalle aus der Fisher-Information, Likelihood-Ratio-Tests, Konsistenz- und Effizienz-Aussagen – natürlich in den frequentistischen Begriffsrahmen. Sie machen Aussagen über das langfristige Verhalten des Schätzers bei wiederholter Stichprobenziehung, was genau die frequentistische Wahrscheinlichkeitskonzeption ist.

Drittens lässt sich Maximum Likelihood aber _auch_ bayesianisch lesen, wie oben gezeigt – als MAP mit uniformem Prior. In diesem Sinne ist es eine "Brückenmethode", die in beiden Schulen vorkommt, aber unterschiedlich gerahmt wird.

**Eine sauber gezeichnete Brücke sieht so aus:**

|Konzept|Was es ist|Lager|
|---|---|---|
|Frequentismus / Bayesianismus|Interpretation von Wahrscheinlichkeit|Philosophisch|
|Maximum Likelihood|Schätzalgorithmus für Punktschätzungen|Methodisch, beidseitig nutzbar|
|Konfidenzintervall|Intervallschätzung mit langfristiger Überdeckungsrate|frequentistisch|
|Credible Interval|Intervallschätzung über Posterior-Dichte|bayessch|
|MAP-Schätzer|Modus des Posteriors als Punktschätzung|bayessch|
|$t$-Test, p-Wert, $\alpha$-Niveau|Hypothesentest-Apparat|frequentistisch|

Wenn du in der Literatur "Maximum Likelihood" liest, kannst du in 95% der Fälle annehmen, dass der Autor frequentistisch denkt – aber das ist eine Konvention, keine logische Notwendigkeit. Die Methode selbst ist agnostisch.


# Referenzen
- [@bishopDeepLearningFoundations2024]
-  [Claude AI](https://claude.ai/chat/497afcff-8bd8-4925-8246-2e50d3cd2f88)