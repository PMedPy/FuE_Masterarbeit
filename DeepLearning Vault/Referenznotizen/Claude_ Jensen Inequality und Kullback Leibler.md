29-05-2026
Tags: #FuE #MachineLearning 
Status: #extended

## 1. Konvexe Funktionen – die Grundlage

Eine Funktion $f: \mathbb{R} \to \mathbb{R}$ heißt **konvex**, wenn für alle $x_1, x_2$ und $\lambda \in [0, 1]$ gilt:

$$ f(\lambda x_1 + (1 - \lambda) x_2) \leq \lambda f(x_1) + (1 - \lambda) f(x_2) $$

Geometrisch: Die Verbindungssehne zwischen zwei Punkten auf dem Graphen liegt **oberhalb oder auf** dem Graphen. Beispiele: $x^2$, $e^x$, $-\log x$ (für $x > 0$).

Eine Funktion ist **konkav**, wenn $-f$ konvex ist – die Ungleichung dreht sich um. Wichtigstes Beispiel: $\log x$.

## 2. Jensen-Ungleichung – Verallgemeinerung auf Erwartungswerte

Die obige Definition betrifft nur zwei Punkte. Die Jensen-Ungleichung ist im Kern dieselbe Aussage, aber auf beliebige Konvexkombinationen und damit auf Erwartungswerte verallgemeinert:

Für eine konvexe Funktion $f$ und eine Zufallsvariable $X$ gilt

$$ f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]. $$

Interpretation: **"Funktion vom Erwartungswert" ist kleiner gleich "Erwartungswert der Funktion"** – wenn $f$ konvex ist.

Für konkave $f$ kehrt sich die Ungleichung um:

$$ f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]. $$

Gleichheit gilt genau dann, wenn $X$ fast sicher konstant ist oder $f$ auf dem Träger von $X$ affin-linear ist.

### Intuition

Konvexe Funktionen "krümmen Werte nach oben". Mittelst du erst und transformierst dann ($f(\mathbb{E}[X])$), bekommst du den Wert am Schwerpunkt. Transformierst du erst und mittelst dann ($\mathbb{E}[f(X)]$), zählen die nach oben gekrümmten Außenbereiche stärker – das Ergebnis ist größer.

## 3. KL-Divergenz – Definition

Die **Kullback-Leibler-Divergenz** misst, wie stark eine Verteilung $q$ von einer Referenzverteilung $p$ abweicht:

$$ D_{KL}(p | q) = \mathbb{E}_p!\left[\log \frac{p(X)}{q(X)}\right] = \sum_x p(x) \log \frac{p(x)}{q(x)} $$

(stetig analog mit Integral). Sie ist **nicht symmetrisch** ($D_{KL}(p|q) \neq D_{KL}(q|p)$ im Allgemeinen) und damit keine echte Metrik. Sie hat aber eine fundamentale Eigenschaft:

$$ D_{KL}(p | q) \geq 0, \quad \text{mit Gleichheit} \iff p = q. $$

Das ist die **Gibbs-Ungleichung** – und genau hier kommt Jensen ins Spiel.

## 4. Beweis von $D_{KL} \geq 0$ via Jensen

Ziehe das Minus aus dem Logarithmus heraus, um den Bruch umzudrehen:

$$ D_{KL}(p | q) = -\mathbb{E}_p!\left[\log \frac{q(X)}{p(X)}\right]. $$

Da $\log$ **konkav** ist, gilt nach Jensen:

$$ \mathbb{E}_p!\left[\log \frac{q(X)}{p(X)}\right] \leq \log \mathbb{E}_p!\left[\frac{q(X)}{p(X)}\right]. $$

Die rechte Seite lässt sich explizit ausrechnen:

$$ \mathbb{E}_p!\left[\frac{q(X)}{p(X)}\right] = \sum_x p(x) \cdot \frac{q(x)}{p(x)} = \sum_x q(x) = 1. $$

Damit ist

$$ \mathbb{E}_p!\left[\log \frac{q(X)}{p(X)}\right] \leq \log 1 = 0, $$

und nach Multiplikation mit $-1$ folgt

$$ D_{KL}(p | q) \geq 0. $$

Gleichheit nur, wenn $q(x)/p(x)$ fast sicher konstant ist – und da beide Wahrscheinlichkeitsverteilungen sind, muss diese Konstante $1$ sein, also $p = q$. ∎

**Kernpunkt:** Jensen ist hier kein technisches Detail, sondern _der_ Schritt, der die ganze Aussage trägt. Ohne Jensen gibt es keinen direkten Weg, $D_{KL} \geq 0$ zu zeigen.

## 5. Warum das in ML wichtig ist

Die Nicht-Negativität der KL-Divergenz ist das Rückgrat vieler ML-Resultate:

- **Maximum-Likelihood:** Minimieren der Kreuzentropie ist äquivalent zu Minimieren von $D_{KL}(p_{\text{data}} | p_{\text{model}})$ – das Modell wird optimal bei $p_{\text{model}} = p_{\text{data}}$, weil $D_{KL}$ dort und nur dort null wird.
- **Variational Inference / ELBO:** Die Evidence Lower Bound entsteht durch Anwenden von Jensen auf $\log p(x) = \log \int p(x, z), dz$ – die Ungleichung _ist_ die Schranke.
- **EM-Algorithmus:** Konvergenz folgt, weil eine Jensen-Schranke iterativ angehoben wird.
- **Informationstheorie:** Mutual Information, Cross-Entropy etc. bauen alle auf KL auf; ihre Nicht-Negativität geht überall auf denselben Jensen-Trick zurück.

In Bishops _Deep Learning_ (DLFC) wird die Grundlage in Kapitel 2 (Wahrscheinlichkeit & Informationstheorie) gelegt. EM und ELBO erscheinen später, nutzen aber genau diese Mechanik.

## 6. Zwei Sätze zum Mitnehmen

Die Jensen-Ungleichung ist die natürliche Verallgemeinerung der Konvexitätsdefinition auf Erwartungswerte. Sie liefert den entscheidenden Schritt, um die Nicht-Negativität der KL-Divergenz zu beweisen – und damit das Fundament für fast jede Anwendung der KL-Divergenz im Machine Learning.
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]