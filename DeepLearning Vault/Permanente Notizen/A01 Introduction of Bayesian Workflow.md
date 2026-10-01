14-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

> _Statistical Rethinking 2026 (Richard McElreath), Vorlesung 1_ Quelle: [YouTube – Lecture A01](http://www.youtube.com/watch?v=ztbYkBPDOgU)

---

## 1. Philosophie des Kurses: Wissenschaft vor Statistik

Der Titel _Statistical Rethinking_ ist Programm. Es geht **nicht** primär darum, frequentistische durch Bayes'sche Verfahren zu ersetzen, sondern darum, **das Verhältnis von Statistik und Wissenschaft umzudrehen.**

In der üblichen Ausbildung lernt man **Prozeduren**: ein Test hier, ein Konfidenzintervall dort, alles im Dienst der Publikation. Die Verbindung zur eigentlichen wissenschaftlichen Theorie bleibt diffus. McElreaths These:

> Ein statistisches Verfahren hat **für sich genommen keine Bedeutung**. Bedeutung entsteht erst, wenn eine externe wissenschaftliche Theorie — formal oder informell — den Output interpretierbar macht.

Daraus folgt die Leitidee des Kurses: Man startet bei einem **wissenschaftlichen Modell** und _leitet daraus_ das statistische Verfahren ab — nicht umgekehrt. Ziel der Ausbildung ist ein klarer, logischer **Workflow**, der Theorie und Verfahren so verknüpft, dass man der Interpretation der Ergebnisse vertrauen kann.

---

## 2. Der Bayes'sche Workflow

### A. Der Kern-Workflow

Jede Analyse durchläuft vier Stationen von links nach rechts:

| #   | Station                                        | Bedeutung                                                                                                                               |
| --- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Generatives Modell** (_generative model_)    | Ein wissenschaftliches Modell, das Daten _simulieren_ kann. Es beschreibt den **hypothetischen kausalen Entstehungsprozess** der Daten. |
| 2   | **Estimand**                                   | Lat. „das, was geschätzt werden soll" — die konkrete wissenschaftliche Zielgröße (z. B. eine Häufigkeit oder ein kausaler Effekt).      |
| 3   | **Statistisches Modell** (_statistical model_) | Die mathematische Übersetzung des generativen Modells, um aus echten Daten Information zu extrahieren.                                  |
| 4   | **Schätzung** (_estimate_)                     | Die Antwort des Modells auf die Frage — meist selbst eine _Verteilung_, kein Einzelwert.                                                |

### B. Der realistische, erweiterte Workflow

In der Praxis kommen hinzu:

- **Prior-Modellprüfung** (_prior predictive check_): Prüfung **bevor** echte Daten ins Spiel kommen. Ergibt das Modell wissenschaftlich Sinn? Läuft der Code?
- **Posterior-Modellprüfung** (_posterior predictive check_): Prüfung **nachdem** das Modell an die Daten angepasst wurde. Passt das Modell zu dem, was tatsächlich beobachtet wurde?
- **Nachbearbeitung der Schätzungen** (_post-processing_): Rohe Parameterwerte beantworten selten direkt die Forschungsfrage. Es braucht Folgeschritte:
    - **Marginale Effekte** (_marginal effects_): „Marginal" heißt in der Statistik „gemittelt". Präziser: man _integriert die übrigen Variablen heraus_ und betrachtet den Effekt im Durchschnitt über deren Verteilung. Kausale Effekte werden oft genau so berechnet.
    - **Poststratifizierung** (_poststratification_): Korrigiert Stichprobenverzerrungen, wenn man mit Populationsdaten statt mit perfekten Experimenten arbeitet.
    - **Sensitivitätsanalysen** (_sensitivity analyses_): Wie stabil sind die Ergebnisse, wenn man die Annahmen variiert?

McElreath nennt die volle Komplexität echter Spitzenforschung den _real dirty Bayesian workflow_: mehrere konkurrierende Modelle, lückenhafte Datensätze, Verzerrungen — Evidenz wird **trianguliert**. Der Kurs baut diese Komplexität schrittweise per _scaffolding_ (Gerüstbau) auf.

### C. Warum Bayes?

- Universelles Werkzeug zur Informationsextraktion.
- Bayes'sche Modelle sind **von Natur aus generativ**: vorwärts laufen lassen → synthetische Daten simulieren; rückwärts → Parameter aus echten Daten schätzen.
- Erweiterungen wie fehlende Daten (_missing data_), Messfehler (_measurement error_), latente Variablen (_latent variables_) und Regularisierung (_regularization_) fügen sich im Bayes-Framework natürlich ein.
- Die „Statistikkriege" (_stats wars_) des 20. Jahrhunderts sind — so McElreaths Einschätzung — entschieden: In moderner Forschung und KI-Entwicklung ist Bayes etabliert und unkontrovers.

> [!info] Vertiefung — Die generativ/invers-Dualität Der für dich vermutlich interessanteste Punkt: ein generatives Modell ist eine Abbildung **Parameter → Daten** ($\theta \mapsto x$), die man _vorwärts_ als Simulation laufen lässt. Inferenz ist die _Umkehrung_ dieser Abbildung, **Daten → Parameter** ($x \mapsto \theta$). Bayes' Theorem ist exakt der formale Apparat, der diese Inversion durchführt — es dreht $p(x \mid \theta)$ in $p(\theta \mid x)$ um. Das ist dieselbe Vorwärts/Rückwärts-Logik wie bei einem generativen vs. diskriminativen Modell im Deep Learning, nur dass Bayes _ein und dasselbe_ Modell in beide Richtungen benutzt.

---

## 3. Einstiegsbeispiel: Das Globus-Wurf-Experiment

**Frage:** Wie groß ist der Wasseranteil der Erde? Da wir die echte Erde nicht abtasten können, nehmen wir einen aufblasbaren Globus als Stellvertreter.

**Generatives Modell & Estimand**

- **Estimand:** der wahre Wasseranteil $P \in [0,1]$.
- **Datengenerierung:** Globus werfen, fangen, schauen, ob der rechte Zeigefinger auf Wasser ($W$) oder Land ($L$) liegt.
- **Annahmen:**
    1. Jeder Wurf ist **unabhängig**.
    2. Die Wahrscheinlichkeit, Wasser zu treffen, ist bei jedem Wurf exakt $P$ (keine systematische Verzerrung beim Fangen).

Ein kurzer R-Code simuliert diesen Prozess _vorwärts_ (_forward simulation_). Zwei Tests sichern den Code ab:

- **Extremwert-Test** (_sanity check_): Setze $P = 1$ (nur Wasser). Gibt die Simulation jemals Land aus, ist der Code fehlerhaft.
- **Asymptotischer Test:** Lässt man die Simulation sehr oft laufen ($\sim 10^4$), muss der Anteil simulierter Wasser-Treffer gegen den wahren Wert $P$ konvergieren.

---

## 4. Die Logik der Schätzung: Der Garten der verzweigenden Daten

Die Kernlogik von Bayes ist erstaunlich simpel: **Zählen.** Für jede mögliche Erklärung zählt man _alle Wege_, auf denen die beobachteten Daten hätten entstehen können. Erklärungen mit mehr passenden Wegen sind plausibler.

### Vereinfachung: der 4-seitige Globus (D4)

Ein runder Globus hat unendlich viele mögliche Wasseranteile. Zum Üben ersetzt McElreath ihn durch einen **4-seitigen Würfel** mit genau fünf möglichen Bauarten:

|Bauart|blaue (Wasser-) Seiten|$P$|
|---|---|---|
|reines Land|0 von 4|$0$|
||1 von 4|$0.25$|
||2 von 4|$0.5$|
||3 von 4|$0.75$|
|reines Wasser|4 von 4|$1$|

### Die Daten

Wir werfen diesen Würfel-Globus dreimal und beobachten die Sequenz **$W, L, W$**.

### Pfade im Garten zählen

Für jede Hypothese: Wie viele Pfade führen exakt zur Sequenz $W, L, W$? Pro Wurf multipliziert man mit der Anzahl der passenden Seiten.

- **$P=0$:** $0$ Wege (Wasser unmöglich).
- **$P=1$:** $0$ Wege (Land unmöglich).
- **$P=0.25$** (1 blau, 3 weiß): $1 \times 3 \times 1 = \mathbf{3}$
- **$P=0.5$** (2 blau, 2 weiß): $2 \times 2 \times 2 = \mathbf{8}$
- **$P=0.75$** (3 blau, 1 weiß): $3 \times 1 \times 3 = \mathbf{9}$

### Resultierende Plausibilitäten

|Hypothese $P$|Wege zu $W,L,W$|relative Plausibilität|
|:-:|:-:|:--|
|$0$|$0$|unmöglich|
|$0.25$|$3$|gering|
|$0.5$|$8$|hoch|
|$0.75$|$9$|höchste|
|$1$|$0$|unmöglich|

Diese Verteilung der relativen Zählergebnisse **ist die Posterior-Verteilung** (_posterior distribution_): die plausibelste Erklärung _gegeben_ die Daten.

> [!important] Vertiefung — Der „Garten" _ist_ Bayes' Theorem Das ist der Punkt, an dem das ganze Gebäude steht oder fällt, deshalb hier explizit. Für $N$ Würfe mit $W$ Wasser und $L$ Land ist die Anzahl der Pfade (siehe Formel unten) $$\text{Wege}(P) = (4P)^W ,(4-4P)^L = 4^{,N}, P^{,W}(1-P)^{,L}.$$ Der Faktor $4^N$ ist für **jede** Hypothese gleich groß — beim Vergleich der Hypothesen kürzt er sich heraus. Übrig bleibt $$\text{relative Plausibilität}(P);\propto; \underbrace{P^{,W}(1-P)^{,L}}_{\text{= Binomial-Likelihood}}.$$ Das Pfadezählen liefert also nichts anderes als die **Likelihood** $p(\text{Daten}\mid P)$. Hat man zusätzlich einen Prior (= Anzahl der Bauarten je Hypothese, falls manche häufiger sind), multipliziert man ihn dazu, und das Normieren auf Summe $1$ ergibt den Posterior: $$p(P\mid \text{Daten});=;\frac{\overbrace{p(\text{Daten}\mid P)}^{\text{Wege}};\overbrace{p(P)}^{\text{Prior}}}{\underbrace{\textstyle\sum_{P'} p(\text{Daten}\mid P'),p(P')}_{\text{Normierung}}}.$$ Mit anderen Worten: McElreaths „Garten" ist Bayes' Theorem, nur ohne die Formel — er lässt dich das Theorem _abzählen_, bevor er es dir als Gleichung zeigt.

> [!note] Vertiefung — Die Formel als Sprache lesen $(4P)^W (4-4P)^L$ ist keine mechanische Rechenvorschrift, sondern ein Satz:
> 
> - $4P$ = „wie viele der 4 Seiten sind Wasser" → die **Anzahl der Wege**, mit der _ein_ Wurf Wasser ergibt.
> - $(\cdot)^W$ = $W$ unabhängige Wasser-Würfe **logisch ge-UND-et**. Das Potenzieren ist hier die Konsequenz der Unabhängigkeitsannahme: _unabhängig_ heißt „die Verbundwahrscheinlichkeit faktorisiert", also wird aus dem UND ein **Produkt**.
> - analog $(4-4P)^L$ für die $L$ Land-Würfe.
> - Das Produkt beider = **Gesamtzahl distinkter Gartenpfade**, die mit der ganzen beobachteten Sequenz verträglich sind.
> 
> Genau hier hängt deine i.i.d.-Frage von vorhin dran: _independent_ erzeugt das Produkt, _identically distributed_ sorgt dafür, dass in jedem Faktor dasselbe $P$ steht.

---

## 5. Bayes'sche Aktualisierung (_Bayesian Updating_)

Ein großer Vorteil: bei neuen Daten muss man **nicht** alles neu rechnen. Man verarbeitet Datenpunkte nacheinander — der Schätzstand von gestern wird zum **Prior** für die Daten von morgen.

**Beispiel:** ein 4. Wurf, wieder **Wasser ($W$)**. Statt den Garten neu zu zeichnen, multipliziert man die bisherigen Wege mit der Anzahl neuer (blauer) Seiten:

|$P$|bisher|$\times$ blaue Seiten|neu|
|:-:|:-:|:-:|:-:|
|$0.25$|$3$|$\times 1$|$3$|
|$0.5$|$8$|$\times 2$|$16$|
|$0.75$|$9$|$\times 3$|$27$|

Mit mehr Daten werden die Abstände zwischen den Plausibilitäten immer deutlicher.

> [!info] Vertiefung — Warum die Reihenfolge egal ist Dass „Posterior von heute = Prior von morgen" funktioniert, ist kein Trick, sondern folgt direkt aus der Produktstruktur der Likelihood. Weil $$\prod_i p(x_i\mid P)$$ ein Produkt ist und Multiplikation **kommutativ und assoziativ** ist, liefert sequentielles Updaten (Punkt für Punkt) _exakt denselben_ Posterior wie das Verarbeiten aller Daten auf einmal — und das in jeder Reihenfolge. Die Daten-Reihenfolge ist für den Bayes'schen Endzustand irrelevant. Das ist die mathematische Garantie hinter dem „man kann einfach weiterzählen".

### Die Abkürzung für große Stichproben

Kombinatorische Pfadzahlen explodieren. Statt zu zählen, berechnet man sie direkt. Für einen 4-seitigen Würfel mit $W$-mal Wasser und $L$-mal Land:

$$\boxed{;\text{Wege} = (4P)^{W},(4 - 4P)^{L};}$$

Schon bei z. B. $20\times W$ und $10\times L$ liegt man bei mehreren **Milliarden** Wegen (für $P=0.5$ etwa $2^{30}\approx 1.07\cdot 10^9$, für $P=0.75$ sogar $\approx 3.5\cdot 10^9$).

---

## 6. Ausblick

Niemand möchte mit Milliarden hantieren. Man **normiert** die Zählergebnisse auf den Bereich $[0,1]$ — und genau das ist die Geburtsstunde der **Wahrscheinlichkeiten** (_probabilities_). Die nächste Vorlesung vollzieht außerdem den Schritt vom 4-seitigen Würfel zum **stetigen, runden Globus** mit unendlich vielen möglichen Werten von $P$ — aus der Summe über fünf Hypothesen wird dann ein Integral über ein Kontinuum.

---

_Bearbeitungshinweis: Inhalt nach McElreath, Lecture A01. Die als „Vertiefung" markierten Callouts sind ergänzende Erläuterungen, nicht Teil des Originalvortrags._


# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]