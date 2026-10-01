22-05-2026
Tags: #Python  #FuE 
Status: #unextended

# Allgemein
Die Likelihood sagt aus, wie gut die beobachteten Daten unter Annahme der Parameterwahl zu einem Modell passen. 

Die Likelihood kann in Anbetracht  einer Wahrscheinlichkeitsverteilung bedeuten:
> Wie gut passen unterschiedliche Parameter $\theta$ (Z.b Mittelwert, Varianz) zu den gemessenen Daten $x$?

Eine solche Wahrscheinlichkeitsverteilung kann dargestellt werden als:
$$p(x\mid \theta)$$

>[!caution] 
>Likelihood ist keine Wahrscheinlichkeitsverteilung, sondern eine Funktion. für jedes $\theta$ fragt sie, wie gut die vorhandenen Daten repräsentiert werden.

# Bayesische Betrachtung
Erst wenn ein [[Prior]] $p(\theta)$ hinzugefügt wird, wird daraus eine Posterior-Verteilung, also eine aktualisierte Prior-Annahme. 
$$p(\theta \mid x) = \frac{p(x \mid \theta)  p(\theta)}{p(x)}$$
>***Die Likelihood ist also eine reine Betrachtung der Daten, ohne Einbezug von Vorkenntnissen***

# Likelihood bei Sätze von Daten

Wenn $\mathcal{D} = {x_1, x_2, \ldots, x_N}$. eine Reihe von Daten ist, gilt, wenn sie unabhängig und identisch verteilt sind ([[I.I.D]]) die Produktregel:
$$p(\mathcal{D} \mid \theta) = \prod_{n=1}^{N} p(x_n \mid \theta)$$
Ohne [[I.I.D]] wäre die Likelihood eine hochdimensionale korrelative Funktion.

# Likelihood maximieren

Um Schätzer für eine gegebene Datenverteilung zu berechnen, wir die Likelehood funktion nach den jeweiligen Schätzer abgeleitet und Null gesetzt.

Wegen numerischen Schwierigkeiten ($N= 1000$) kommen Computer an die Grenzen des [[Floats (Fließkommzahlen)]]-Bereich. Daher wird zur Maximierung nicht $p(\mathcal{D} \mid \theta)$ maximiert, sondern der Logarithmus. Aus dem Produkt wird eine Summe:
$$\ln p(\mathcal{D} \mid \theta) = \sum_{n=1}^{N} \ln p(x_n \mid \theta)$$
Zudem lassen sich Summen einfacher Ableiten. 
Vorteil es ist keine [[Density Transformation]] nötig, da der natürliche Logarithmus **streng monoton steigt**

Siehe [[Likelihood Maximization]]

# Referenzen
