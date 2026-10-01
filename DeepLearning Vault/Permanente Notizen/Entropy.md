23-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
In der Informationstechnologie spricht man von Entropie als

> Grad der Überraschung, Bzw wie Überraschend bestimmte Ereignisse sind.

Je höher die Entropie, umso weniger überraschend ist ein Ereignis.
Über den Informationsgehalt einer Wahrscheinlichkeitsverteilung kann die Entropie eine Aussage machen: So hat eine Uniforme Verteilung (Alle Ereignisse mit gleicher Wahrscheinlichkeit) eine höhere Entropie, als z.B eine schmale Gaussverteilung mit geringer Varianz.

# Mathematische Definiton
Mathematisch gesehen ist die Entropie die Summe des Produktes aus der diskreten Verteilung $p(x)$ und deren Informationsgehalt über alle möglichen Werte $x$ einer Verteilung:
$$H[x] = -\sum_x p(x)\log_{2}p(x)$$

Wobei:
$$h(x)=-\log_{2}p(x)$$der Informationsgehalt der Verteilung $p(x)$ ist. Damit ist die Entropie das **gewichtete Mittel des Informationsgehaltes einer Verteilung**

# Shannon Theorem
[[Shannon's Theorem]], Auch genannt "noiseless coding theorem" besagt, dass die Entropie $H(x)$ bei gegebener Verteilung $p(x)$ die unterste Schranke für mittlere Codelänge darstellt, um eine Zufallsvariable $x$ auszudrücken. 
# Die physikalische Betrachtung

Was ist, wenn man die Entropie nicht unter Informationstechnischen Gesichtspunkten betrachtet, sondern rein kombinatorisch - wie Boltzmann?
#### Ausgangslage
$N$ unterscheidbare Objekte werden auf eine Reihe von Behältern ("bins") verteilt. Im $i$-ten Behälter landen am Ende $n_i$ Objekte, mit der Nebenbedingung

$$\sum_i n_i = N.$$
#### Mikrozustand vs Makrozustand
Bei unterscheidbaren Objekte, gibt es mehrere Mikrozustände um einen Makrozustand zu erreichen, wenn gilt:
- **Mikrozustand:** Konkrete Zuordnung jedes einzelnen Objekts zu einem Behälter ("Objekt 1 → Behälter 3, Objekt 2 → Behälter 7, ...").
- **Makrozustand:** Nur die Belegungszahlen $(n_1, n_2, \ldots)$ – ohne Information darüber, _welches_ Objekt wo liegt.
Die Anzahl an Mikrozuständen pro Makrozuständen ist die entscheidende Größe: An ihr wird die Entropie ermessen.
#### Multiplizität
Die Anzahl an möglichen Mikrozuständen die zu einem Makrozustand führen wird durch die Multinomialkoeffizienten oder $Multiplizität$ berechnet:
$$W = \frac{N!}{\prod_i n_i!}$$
Es gibt $N$-Möglichkeiten $N$ Objekte in eine Reihenfolge zu bringen (Permutation) ist $N!$ Um nun die Reihenfolge der Objekte in $n_{i}$ Behälter herauszurechnen folgt die Teilung durch den Nenner (Reihenfolge in den Behälter für Makrozustand irrelevant)

#### Aus der Multiplizität folgt die Entropie
Bishop definiert die Entropie als **logarithmische Multiplizität pro Objekt**:

$$H = \frac{1}{N} \ln W = \frac{1}{N} \ln N! - \frac{1}{N} \sum_i \ln n_i!$$
Über die Stirling-Approximation und den Anteil $p_i = n_i / N$ kommt man auf die Definition aus der Informationstheorie. (siehe Oben)

#### Der Wahrscheinlichste Makrozustand
Unter der Annahme, alle Mikrozustände seien gleich Wahrscheinlich, ist der Makrozustand am wahrscheinlichsten mit der größten Entropie (Man denke: Entropie = logarithmische Multiplizität pro Objekt)

# Weiterführend
[[Differential Entropy]]

# Referenzen
[@bishopDeepLearningFoundations2024]