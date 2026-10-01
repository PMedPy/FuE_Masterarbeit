29-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Die Jensen Ungleichheit ist die allgemeine Ausführung für die Prüfung auf [[Convex Function]]s, nicht nur mit zwei Punkten, sondern mit beliebig vielen $x_{i}$:
$$f\left(\sum_{i=1}^{M}\lambda_{i}x_{i}\right)\leq \sum_{i=1}^{M}\lambda_{i}f(x_{i})$$ Mit $\lambda_{i}$ und der Normierungsbedingung $\sum_{i}\lambda_{i}=1$ von $0 \leq \lambda \leq 1$
 
# Wahrscheinlichkeitsinterpretation
Interpretiert man $\lambda_{i}$ als EINE Wahrscheinlichkeit von $x_{i}$, so ergeben sich Erwartungswerte: 
$$ f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]. $$
Heißt:
> Die Funktion eines Erwartungswertes ist kleiner gleich des Erwartungswertes der Funktion. 

Für stetige Werte $x$ erhält man:
$$f\!\left(\int x\, p(x)\, dx\right) \leqslant \int f(x)\, p(x)\, dx$$

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]