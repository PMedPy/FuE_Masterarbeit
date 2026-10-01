12-05-2026
Tags: #FuE #MachineLearning 
Status: #extended


# Entropie – Herleitung aus der physikalischen Perspektive

> Bishop, _Deep Learning_ (DLFC), Kapitel 2 – Abschnitt zur „Physical Perspective" auf Entropie. Diese Notiz fokussiert auf die Boltzmann-Herleitung und schließt mit der Brücke zur Shannon-Entropie.

---

## Ausgangsfrage

Was passiert, wenn wir Entropie **nicht** über Information oder Codierung definieren, sondern rein kombinatorisch – wie Boltzmann es ursprünglich tat?

Setup: $N$ unterscheidbare Objekte werden auf eine Reihe von Behältern ("bins") verteilt. Im $i$-ten Behälter landen am Ende $n_i$ Objekte, mit der Nebenbedingung

$$\sum_i n_i = N.$$

---

## Mikrozustand vs. Makrozustand

Eine entscheidende Unterscheidung im Hintergrund:

- **Mikrozustand:** Konkrete Zuordnung jedes einzelnen Objekts zu einem Behälter ("Objekt 1 → Behälter 3, Objekt 2 → Behälter 7, ...").
- **Makrozustand:** Nur die Belegungszahlen $(n_1, n_2, \ldots)$ – ohne Information darüber, _welches_ Objekt wo liegt.

Mehrere Mikrozustände führen typischerweise zum selben Makrozustand. Genau diese Anzahl ist die zentrale Größe.

---

## Multiplizität $W$

Die Anzahl der Mikrozustände, die zu einem gegebenen Makrozustand $(n_1, n_2, \ldots)$ gehören, ist der **Multinomialkoeffizient**:

$$W = \frac{N!}{\prod_i n_i!}$$

**Herleitung in zwei Schritten:**

1. Es gibt $N!$ Möglichkeiten, $N$ unterscheidbare Objekte in eine Reihenfolge zu bringen.
2. Innerhalb eines Behälters spielt die Reihenfolge der $n_i$ Objekte für den Makrozustand keine Rolle. Diese internen Permutationen werden durch Division mit $\prod_i n_i!$ herausgeteilt.

$W$ heißt **Multiplizität** oder _statistisches Gewicht_. Boltzmann nannte es schlicht _Wahrscheinlichkeit_, daher der Buchstabe.

---

## Definition der Entropie

Bishop definiert die Entropie als logarithmische Multiplizität pro Objekt:

$$H = \frac{1}{N} \ln W = \frac{1}{N} \ln N! - \frac{1}{N} \sum_i \ln n_i!$$

Der Faktor $1/N$ macht $H$ zu einer **intensiven** Größe (pro Objekt, unabhängig von der Systemgröße).

---

## Übergang zum kontinuierlichen Limit – Stirling-Approximation

Für großes $N$ gilt die Stirling-Approximation:

$$\ln N! \approx N \ln N - N$$

Einsetzen in die Entropie-Definition:

$$H = \frac{1}{N}(N \ln N - N) - \frac{1}{N} \sum_i (n_i \ln n_i - n_i)$$

Die Terme $-N$ und $-\sum_i n_i = -N$ heben sich gegenseitig auf:

$$H = \ln N - \frac{1}{N} \sum_i n_i \ln n_i$$

Nun den Anteil $p_i = n_i / N$ einführen (das ist die empirische Wahrscheinlichkeit, in Behälter $i$ zu landen):

$$H = \ln N - \frac{1}{N} \sum_i N p_i \ln(N p_i) = \ln N - \sum_i p_i (\ln N + \ln p_i)$$

Mit $\sum_i p_i = 1$ fällt der $\ln N$-Term heraus, und es bleibt:

$$H = -\sum_i p_i \ln p_i$$

**Das ist exakt die Shannon-Entropie** – bis auf den Logarithmus, der hier zur Basis $e$ statt $2$ steht (Einheit: _nats_ statt _bits_; ein konstanter Umrechnungsfaktor $\ln 2$).

---

## Boltzmanns Originalformel

In der Physik wird die Entropie üblicherweise mit der Boltzmann-Konstante $k_B$ versehen:

$$S = k_B \ln W$$

Das ist die Formel auf Boltzmanns Grabstein in Wien. Sie ist mathematisch identisch zu Bishops $H = (1/N) \ln W$, nur mit anderer Normierung (Einheiten: Joule pro Kelvin statt einheitenlos).

---

## Welcher Makrozustand ist der wahrscheinlichste?

**Annahme:** Jeder _Mikro_zustand ist gleich wahrscheinlich (Prinzip der gleichen a-priori-Wahrscheinlichkeiten).

Dann ist der Makrozustand mit der größten Multiplizität $W$ der wahrscheinlichste – **also der mit der größten Entropie.**

Ohne weitere Nebenbedingungen ist das die **Gleichverteilung** $n_i = N/M$ (bei $M$ Behältern).

**Konsequenz:** Der zweite Hauptsatz der Thermodynamik ist statistisch, nicht mechanisch. Systeme laufen nicht _zwingend_ in den entropiemaximalen Zustand – sie tun es, weil dort schlicht die meisten Mikrozustände zu finden sind.

---

## Vergleich der drei Perspektiven auf Entropie

|Perspektive|$p_i$ ist...|Entropie misst...|Maximum bei...|
|---|---|---|---|
|Informationstheorie (Shannon)|Wahrscheinlichkeit eines Ereignisses|erwartete Überraschung|Gleichverteilung (maximale Unsicherheit)|
|Codierungstheorie|Wahrscheinlichkeit eines Symbols|minimale durchschnittliche Codelänge|Gleichverteilung (kein Code-Vorteil nutzbar)|
|Statistische Mechanik (Boltzmann)|Anteil $n_i / N$ in Behälter $i$|$\frac{1}{N} \ln W$ – Log der Mikrozustandszahl|Gleichverteilung (meiste Mikrozustände)|

Alle drei führen auf dieselbe Formel:

$$H = -\sum_i p_i \ln p_i$$

---

## Warum das für ML relevant ist

- **Maximum-Entropie-Verteilungen:** Gegeben Nebenbedingungen (fester Erwartungswert, feste Varianz, ...) liefert die Entropiemaximierung jeweils eine spezifische Verteilung – Gauß, Exponential, Boltzmann. Diese Verteilungen tauchen in Bayes-Modellen und als Priors überall auf.
- **Energiebasierte Modelle:** Boltzmann-Maschinen, RBMs, Energy-Based Models verwenden direkt die Boltzmann-Verteilung $p \propto e^{-E/T}$, die aus dieser Multiplizitätsbetrachtung mit Energie als Nebenbedingung folgt.
- **Free Energy / ELBO:** In Variational Inference (später bei VAEs relevant) wird die thermodynamische Sprache wörtlich – die ELBO ist mathematisch eine freie Energie.
- **Kreuzentropie als Loss / KL-Divergenz:** Beide sind direkte Konsequenzen des Codierungs-Bildes; die Boltzmann-Perspektive liefert dazu die physikalische Intuition, warum sie als „natürliche" Loss-Funktionen auftauchen.

Für den Enzymkinetik-Kontext: Die Boltzmann-Verteilung steckt in der Arrhenius-Gleichung, in Konformationsverteilungen und in jeder freien-Energie-Landschaft. Nicht direkt ML, aber Brücke zwischen physikalischer Realität und probabilistischen Modellen.

---

## Kernaussage in einem Satz

Entropie ist die Größe, in der drei Fragen dieselbe Antwort haben: **Wie überraschend ist die nächste Beobachtung? Wie effizient kann ich codieren? Wie viele mikroskopische Wege führen zu diesem makroskopischen Zustand?**

Dass alle drei auf $-\sum p_i \log p_i$ führen, ist kein Zufall – es ist die strukturelle Tiefe der Größe.

# Referenzen
[@bishopDeepLearningFoundations2024]