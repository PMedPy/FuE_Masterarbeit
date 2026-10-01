19-05-2026
Tags: #FuE #MachineLearning 
Status: #extended


> Bishop, _Deep Learning_ (DLFC), Kapitel 2 – Wahrscheinlichkeitsdichten, Gauß-Verteilung, Maximum Likelihood. Diese Notiz baut den Begriff der Likelihood von Null auf und erklärt, warum er das zentrale Bindeglied zwischen Wahrscheinlichkeitstheorie und ML ist.

---

## Der Ausgangspunkt: zwei verschiedene Fragen

Eine Wahrscheinlichkeitsverteilung $p(x \mid \theta)$ hat **zwei freie Größen**:

- $x$ – die Beobachtung (z.B. ein Messwert).
- $\theta$ – die Parameter der Verteilung (z.B. Mittelwert und Varianz einer Gauß-Verteilung).

Je nachdem, welche der beiden du als „bekannt" annimmst und welche du variierst, stellst du eine völlig andere Frage:

|Fixiert|Variabel|Bedeutung|Name|
|---|---|---|---|
|$\theta$|$x$|"Wie wahrscheinlich sind verschiedene Daten unter diesem Modell?"|Wahrscheinlichkeitsdichte|
|$x$|$\theta$|"Wie gut passen verschiedene Parameter zu diesen Daten?"|Likelihood|

**Dasselbe mathematische Objekt $p(x \mid \theta)$ – aber zwei verschiedene Interpretationsrichtungen.** Das ist der entscheidende konzeptionelle Sprung.

---

## Warum die Likelihood _keine_ Wahrscheinlichkeitsverteilung über $\theta$ ist

Ein häufiger Stolperstein: Die Likelihood $L(\theta) = p(x \mid \theta)$, betrachtet als Funktion von $\theta$, integriert **nicht** zu 1. Sie ist keine Verteilung über $\theta$ – sie ist eine **Funktion**, die jedem $\theta$ einen Zahlenwert zuordnet, der angibt, wie plausibel dieses $\theta$ ist, _gegeben_ die beobachteten Daten.

Erst wenn man via Bayes-Theorem einen Prior $p(\theta)$ einbringt, wird daraus eine echte Posterior-Verteilung $p(\theta \mid x)$:

$$p(\theta \mid x) = \frac{p(x \mid \theta) , p(\theta)}{p(x)}$$

Die Likelihood ist also die _Datenseite_ der Bayes-Gleichung – das, was die Daten dem Modell mitteilen, bevor Vorwissen einbezogen wird.

---

## Likelihood bei mehreren unabhängigen Beobachtungen

In der Praxis hat man selten _eine_ Beobachtung, sondern einen ganzen Datensatz $\mathcal{D} = {x_1, x_2, \ldots, x_N}$.

Unter der Annahme, dass die Beobachtungen **unabhängig und identisch verteilt** (i.i.d.) sind, faktorisiert die gemeinsame Dichte:

$$p(\mathcal{D} \mid \theta) = \prod_{n=1}^{N} p(x_n \mid \theta)$$

Das ist die **Likelihood-Funktion** des gesamten Datensatzes.

Die i.i.d.-Annahme ist die entscheidende strukturelle Vereinfachung: Ohne sie wäre die gemeinsame Dichte eine hochdimensionale Funktion mit Korrelationen zwischen allen Beobachtungen. Mit ihr wird sie zu einem schlichten Produkt.

---

## Warum Logarithmus? Die Log-Likelihood

Mit einem Produkt aus $N$ Faktoren – jeder davon eine Zahl zwischen 0 und 1 – kommt man praktisch nicht weit. Zwei Probleme:

1. **Numerisch:** Das Produkt unterläuft den Float-Bereich (für $N = 1000$ und typische Likelihoods landest du sofort bei Werten wie $10^{-3000}$ – nicht mehr darstellbar).
2. **Analytisch:** Die Ableitung eines Produkts ist hässlich (Produktregel, kettenartige Terme), die einer Summe ist sauber.

Beide Probleme verschwinden, wenn man den Logarithmus nimmt:

$$\ln p(\mathcal{D} \mid \theta) = \sum_{n=1}^{N} \ln p(x_n \mid \theta)$$

Da $\ln$ streng monoton steigt, hat $\ln L(\theta)$ **dieselben Maxima** wie $L(\theta)$. Maximierung der Log-Likelihood ist äquivalent zur Maximierung der Likelihood – nur viel angenehmer zu rechnen.

In der Praxis arbeitet man fast ausschließlich mit Log-Likelihoods. "Likelihood" als Wort meint oft implizit die Log-Variante.

---

## Maximum Likelihood Estimation (MLE)

Die fundamentale Idee:

> Wähle die Parameter $\theta$ so, dass die beobachteten Daten unter dem Modell am **wahrscheinlichsten** werden.

Formal:

$$\theta_{ML} = \arg\max_\theta ; p(\mathcal{D} \mid \theta) = \arg\max_\theta ; \sum_{n=1}^{N} \ln p(x_n \mid \theta)$$

Das ist eines der ältesten und wichtigsten Prinzipien der Statistik. Es liefert einen direkten Algorithmus, um Modellparameter aus Daten zu lernen – ohne zusätzliche Annahmen über Priors.

---

## Konkretes Beispiel: Gauß-Likelihood

Bishop führt das am Beispiel der eindimensionalen Gauß-Verteilung durch. Die Dichte ist:

$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Gegeben i.i.d. Daten $\mathcal{D} = {x_1, \ldots, x_N}$ ist die Likelihood:

$$p(\mathcal{D} \mid \mu, \sigma^2) = \prod_{n=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp!\left(-\frac{(x_n-\mu)^2}{2\sigma^2}\right)$$

Logarithmieren und vereinfachen:

$$\ln p(\mathcal{D} \mid \mu, \sigma^2) = -\frac{1}{2\sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 - \frac{N}{2} \ln(2\pi\sigma^2)$$

### Maximierung nach $\mu$

Ableitung nach $\mu$ gleich null setzen:

$$\frac{\partial}{\partial \mu} \ln p = \frac{1}{\sigma^2} \sum_{n=1}^{N} (x_n - \mu) \stackrel{!}{=} 0$$

Auflösen liefert:

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n$$

Der Maximum-Likelihood-Schätzer für den Mittelwert ist also schlicht der **empirische Mittelwert** der Daten. Dieses Ergebnis ist intuitiv – aber bemerkenswert ist, dass es als direkte Konsequenz des MLE-Prinzips herausfällt, nicht als Ad-hoc-Definition.

### Maximierung nach $\sigma^2$

Analog für die Varianz:

$$\sigma^2_{ML} = \frac{1}{N} \sum_{n=1}^{N} (x_n - \mu_{ML})^2$$

Auch das ist die empirische Varianz – mit einer wichtigen Eigenheit, siehe nächster Abschnitt.

---

## Eine Warnung: MLE kann _biased_ sein

Der Maximum-Likelihood-Schätzer für $\sigma^2$ ist **systematisch zu klein**. Im Erwartungswert gilt:

$$\mathbb{E}[\sigma^2_{ML}] = \frac{N-1}{N} , \sigma^2_{\text{wahr}}$$

Das heißt: Selbst bei beliebig vielen Wiederholungen des Experiments unterschätzt der MLE die Varianz im Mittel um den Faktor $(N-1)/N$. Erst der korrigierte Schätzer $\frac{1}{N-1}\sum_n (x_n - \mu_{ML})^2$ ist erwartungstreu (unbiased).

**Warum?** Anschaulich: Der MLE-Schätzer misst die Streuung relativ zum _empirischen_ Mittelwert $\mu_{ML}$, nicht zum _wahren_ Mittelwert $\mu$. Der empirische Mittelwert liegt aber definitionsgemäß genau dort, wo die Daten im Mittel landen – die Streuung um ihn herum ist immer ein bisschen kleiner als die Streuung um den wahren Mittelwert.

Für großes $N$ verschwindet der Effekt ($(N-1)/N \to 1$), aber er ist ein erstes Beispiel dafür, dass MLE nicht automatisch der „beste" Schätzer ist. Bishop nutzt das später, um zur Bayesschen Sichtweise überzuleiten, die diese Verzerrung systematisch behandelt.

---

## Likelihood in der ML-Praxis

Die Likelihood ist nicht nur ein statistisches Konzept – sie ist der **Ursprung der meisten Loss-Funktionen** im maschinellen Lernen:

| ML-Setting                   | Likelihood-Annahme                                                | Resultierender Loss       |
| ---------------------------- | ----------------------------------------------------------------- | ------------------------- |
| Regression mit Gauß-Rauschen | $p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)$ | Mean Squared Error        |
| Binäre Klassifikation        | $p(y \mid x, \theta) = \text{Bernoulli}(y \mid f_\theta(x))$      | Binary Cross-Entropy      |
| Mehrklassen-Klassifikation   | $p(y \mid x, \theta) = \text{Categorical}(y \mid f_\theta(x))$    | Categorical Cross-Entropy |

In allen drei Fällen ist die Loss-Funktion exakt die **negative Log-Likelihood** unter der jeweiligen Modellannahme. Das, was im Deep-Learning-Alltag als „der Loss" daherkommt, ist in Wahrheit fast immer ein verkleidetes Maximum-Likelihood-Problem.

Diese Brücke ist einer der Hauptgründe, warum Bishop so viel Aufwand in die Wahrscheinlichkeitstheorie steckt, bevor er zu neuronalen Netzen kommt.

---

## Verbindung zur Entropie

Die Log-Likelihood pro Beobachtung ist eng verwandt mit der **Kreuzentropie**:

$$-\frac{1}{N} \sum_{n=1}^{N} \ln p(x_n \mid \theta) ;\approx; \mathbb{E}_{x \sim p_{\text{wahr}}} [-\ln p(x \mid \theta)]$$

Die rechte Seite ist die Kreuzentropie zwischen der wahren Datenverteilung und dem Modell. Maximum Likelihood ist damit äquivalent zu Kreuzentropie-Minimierung – und über die KL-Divergenz auch äquivalent zu „mache das Modell so nah wie möglich an die wahre Verteilung".

So schließt sich der Kreis zur Entropie-Notiz: Information, Codierung, Boltzmann _und_ Likelihood sind alle Perspektiven auf denselben Grundbegriff.

---

## Kernaussagen

- Likelihood und Wahrscheinlichkeitsdichte sind dasselbe Objekt $p(x \mid \theta)$, aber mit unterschiedlicher Variable fixiert.
- Bei i.i.d. Daten faktorisiert die Likelihood; man arbeitet praktisch immer mit der Log-Likelihood.
- Maximum Likelihood Estimation liefert die Parameter, unter denen die Daten am wahrscheinlichsten sind.
- Der MLE-Schätzer für die Gauß-Varianz ist biased – ein erster Hinweis, dass MLE nicht das letzte Wort ist.
- Die meisten ML-Loss-Funktionen sind negative Log-Likelihoods unter spezifischen Modellannahmen.

# Referenzen
[@bishopDeepLearningFoundations2024]