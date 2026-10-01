Tags: #FuE #MachineLearning 
Status: #extended 

## Was hier wirklich gemeint ist mit den Erwartungswerten

$\mu_{ML}$ ist **nicht** "der Mittelwert von Likelihood". Es ist der **Maximum-Likelihood-Schätzer für den Parameter $\mu$**. Das ist ein wichtiger Unterschied.

Konkret: Wenn du annimmst, deine Daten $x_1, \dots, x_N$ kommen aus einer Gauß-Verteilung, und du fragst "welcher Wert von $\mu$ macht meine beobachteten Daten am wahrscheinlichsten?", dann fällt als Antwort heraus:

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n$$

Das ist einfach der Stichprobenmittelwert deiner Daten. Bishop hat das ein paar Seiten vorher hergeleitet (durch Ableiten der Log-Likelihood nach $\mu$ und Nullsetzen).

## Was Gleichung (2.59) also wirklich sagt

$$\mathbb{E}[\mu_{ML}] = \mu$$

In Worten: **Wenn du sehr viele verschiedene Datensätze der Größe $N$ aus derselben wahren Verteilung ziehen würdest, und für jeden Datensatz $\mu_{ML}$ ausrechnest, dann ist der Durchschnitt all dieser Schätzungen genau der wahre Parameter $\mu$.**

Das nennt man **Erwartungstreue** (engl. _unbiased_). Der Schätzer trifft "im Mittel" das Richtige.

Wichtig: Der Erwartungswert ist hier **über Datensätze**, nicht über Datenpunkte innerhalb eines Datensatzes. Stell dir vor:

- Datensatz 1: ${x_1^{(1)}, \dots, x_N^{(1)}}$ → liefert Schätzung $\mu_{ML}^{(1)}$
- Datensatz 2: ${x_1^{(2)}, \dots, x_N^{(2)}}$ → liefert Schätzung $\mu_{ML}^{(2)}$
- Datensatz 3: ... usw.

Jede einzelne Schätzung $\mu_{ML}^{(k)}$ wird vom wahren $\mu$ leicht abweichen (Stichprobenrauschen). Aber wenn du den Durchschnitt über unendlich viele solcher Schätzungen bildest, bekommst du genau $\mu$. Keine systematische Verzerrung.

## Und Gleichung (2.60) — der eigentliche Knackpunkt

$$\mathbb{E}[\sigma^2_{ML}] = \frac{N-1}{N} \sigma^2$$

Hier wird's interessant: Der ML-Schätzer für die **Varianz** ist _nicht_ erwartungstreu. Er **unterschätzt systematisch** die wahre Varianz um den Faktor $\frac{N-1}{N}$.

Anschaulich: Der ML-Schätzer für $\sigma^2$ misst die Streuung der Daten **um den geschätzten Mittelwert $\mu_{ML}$**, nicht um den wahren $\mu$. Da $\mu_{ML}$ aber so gewählt wurde, dass es möglichst gut zu _genau diesen_ Datenpunkten passt, sieht die Streuung um $\mu_{ML}$ herum immer etwas kleiner aus als die Streuung um den wahren $\mu$. Daher die Unterschätzung.

Für $N \to \infty$ verschwindet der Bias ($\frac{N-1}{N} \to 1$), aber bei kleinen $N$ ist er relevant. Genau deswegen verwendet man in der Statistik oft die "korrigierte" Stichprobenvarianz mit $\frac{1}{N-1}$ statt $\frac{1}{N}$ — die ist erwartungstreu.

## Warum das für dich (und die Enzymkinetik) wichtig ist

Wenn du später ML-Modelle für $k_{cat}$ oder $K_M$ trainierst, ist das genau die Art Frage, die immer wieder auftaucht: **Trifft mein Schätzer im Mittel das Richtige, oder hat er einen systematischen Bias?** Bishop verwendet dieses Beispiel als Aufhänger, um zu zeigen, dass Maximum Likelihood — so elegant es ist — eben nicht automatisch unverzerrt ist. Das motiviert später Bayessche Methoden und Regularisierung.

Soll ich dir die Herleitung von (2.59) oder (2.60) Schritt für Schritt durchgehen? Die von (2.59) ist eine Zeile, die von (2.60) ist die didaktisch lehrreichere.

## Was ein "Moment" ist

Ein **Moment** einer Verteilung ist einfach der Erwartungswert einer Potenz von $x$:

$$\text{$k$-tes Moment} = \mathbb{E}[x^k] = \int x^k , p(x) , dx$$

Das war's im Kern. Die Zahl $k$ heißt **Ordnung** des Moments.

- **Erstes Moment (first-order moment):** $\mathbb{E}[x^1] = \mathbb{E}[x]$ → das ist der **Mittelwert**.
- **Zweites Moment (second-order moment):** $\mathbb{E}[x^2]$ → hängt mit der Varianz zusammen, aber ist _nicht_ die Varianz selbst (gleich mehr dazu).
- **Drittes Moment:** $\mathbb{E}[x^3]$ → hängt mit Schiefe (Skewness) zusammen.
- **Viertes Moment:** $\mathbb{E}[x^4]$ → hängt mit Wölbung (Kurtosis) zusammen.

Die Idee dahinter: Jedes Moment fängt einen anderen Aspekt der Form einer Verteilung ein. Mittelwert → wo liegt sie. Varianz → wie breit ist sie. Schiefe → ist sie symmetrisch. Kurtosis → wie schwer sind die Tails. Eine Verteilung ist (unter milden Bedingungen) durch alle ihre Momente vollständig charakterisiert.

## Warum "Moment"? — eine kurze Anekdote

Der Name kommt aus der Mechanik. Wenn du eine Massendichte $\rho(x)$ entlang einer Achse hast, dann ist:

- $\int \rho(x) , dx$ = Gesamtmasse
- $\int x \cdot \rho(x) , dx$ = statisches Moment um den Ursprung → bestimmt den **Schwerpunkt**
- $\int x^2 \cdot \rho(x) , dx$ = Trägheitsmoment

Mit deinem Physik-Hintergrund ist dir das Trägheitsmoment vermutlich vertraut — genau dieselbe mathematische Struktur. Statistiker haben den Begriff schlicht übernommen, weil $p(x)$ formal wie eine "Wahrscheinlichkeitsdichte" funktioniert (Gesamtmasse = 1).

## Der Zusammenhang mit der Varianz

Hier ist der Punkt, der bei Bishop in Gleichung (2.54) genutzt wird. Die Varianz ist _definiert_ als:

$$\text{var}[x] = \mathbb{E}[(x - \mathbb{E}[x])^2]$$

Also: erwartete quadratische Abweichung vom Mittelwert. Wenn du das ausmultiplizierst:

$$\text{var}[x] = \mathbb{E}[x^2 - 2x\mathbb{E}[x] + \mathbb{E}[x]^2]$$

Mit der Linearität des Erwartungswerts (und weil $\mathbb{E}[x]$ eine Konstante ist):

$$\text{var}[x] = \mathbb{E}[x^2] - 2\mathbb{E}[x]\mathbb{E}[x] + \mathbb{E}[x]^2 = \mathbb{E}[x^2] - \mathbb{E}[x]^2$$

Das ist genau Gleichung (2.54). In Worten: **Varianz = zweites Moment minus (erstes Moment)²**. Diese Identität ist eine der nützlichsten Rechenformeln in der Statistik überhaupt — du wirst sie noch oft sehen.

## Zentrale vs. rohe Momente

Kleiner, aber wichtiger Begriff: Was ich bisher beschrieben habe ($\mathbb{E}[x^k]$), heißt genauer **rohes Moment** (oder "Moment um den Ursprung"). Daneben gibt es die **zentralen Momente**:

$$\text{$k$-tes zentrales Moment} = \mathbb{E}[(x - \mu)^k]$$

Hier wird also erst um den Mittelwert verschoben, dann potenziert. Damit:

- Erstes zentrales Moment = $0$ (per Konstruktion)
- **Zweites zentrales Moment = Varianz** ← das ist der Standardfall
- Drittes zentrales Moment → eng mit Schiefe verwandt
- Viertes zentrales Moment → eng mit Kurtosis verwandt

Wenn jemand umgangssprachlich "das zweite Moment" sagt, meint er meistens die Varianz (also das zentrale). Wenn Bishop sauber unterscheidet wie hier in (2.53), meint er das _rohe_ zweite Moment $\mathbb{E}[x^2]$, und die Varianz wird dann als Differenz zweier roher Momente konstruiert.

## Konkret bei der Gauß-Verteilung

Bishop sagt in (2.52) und (2.53):

- $\mathbb{E}[x] = \mu$
- $\mathbb{E}[x^2] = \mu^2 + \sigma^2$

Das zweite Moment ist also **nicht** $\sigma^2$, sondern $\mu^2 + \sigma^2$. Erst wenn du $\mathbb{E}[x]^2 = \mu^2$ abziehst, bleibt $\sigma^2$ übrig — die Varianz. Das ist genau, warum Bishop diese Identität (2.54) hier explizit aufschreibt: Sie verbindet die beiden rohen Momente mit der Varianz.

## Wozu das Ganze konzeptuell gut ist

Das Konzept der Momente kommt dir später in der Praxis an mehreren Stellen wieder über den Weg:

1. **Method of Moments** — eine alternative Schätzmethode zu Maximum Likelihood: man setzt empirische Stichprobenmomente gleich den theoretischen Momenten der Verteilung und löst nach den Parametern auf.
2. **Momenten-erzeugende Funktionen** — ein Werkzeug, mit dem man Momente bequem alle auf einmal ausrechnen kann.
3. **Batch Normalization in Deep Learning** — normalisiert Aktivierungen anhand des ersten und zweiten Moments (Mittelwert und Varianz) über einen Mini-Batch. Wenn du später bei Bishop oder in PyTorch BatchNorm siehst: das sind genau diese Momente, empirisch geschätzt.
4. **Adam-Optimizer** — der Name steht für "**Ada**ptive **M**oments". Adam pflegt während des Trainings Schätzungen des ersten und zweiten Moments der Gradienten, um die Lernrate adaptiv zu skalieren. Wenn du in PyTorch `torch.optim.Adam` aufrufst, sind die Hyperparameter `betas=(0.9, 0.999)` die Glättungsfaktoren für genau diese beiden Momente.

Spätestens beim letzten Punkt wirst du den Begriff also sehr konkret in deinem Code wiedersehen. Sag Bescheid, wenn du an der Stelle bist und ich das nochmal mit dem Bishop-Wissen verknüpfen soll.

# Referenzen
[@bishopDeepLearningFoundations2024]