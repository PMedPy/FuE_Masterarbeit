27-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Eine Funktion ist konvex, wenn sie Schüsselförmig ist.
D.h. wenn die Funktion bzw die Kurve unterhalb einer gespannten  Sekante ist.

# Mathematisch
Eine Funktion ist convex, wenn sie folgende Ungleichung erfüllt:
$$f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)$$
Wobei Jeder Wert von $x$ im Intervall $[a|b]$  als  $\lambda a +(\lambda +1)b$ mit $0 \leqslant \lambda \leqslant 1$  geschrieben werden kann und die zugehörigen Werte von $f(x)$ mitbringen Wobei $\lambda$ Interpolationsparameter ist (Wird noch später als Wahrscheinlichkeit wichtig.) . Das ist Äquivalent zu der Bedingung, dass 
> *jeder Punkt der zweiten Ableitung $f''(x)$ positiv sein muss.* 

Für concave Funktionen gelten entgegengesetzte Eigenschaften und es gilt: Wenn $f(x)$ convex ist, dann ist $-f(x)$ concav. Die Ungleichung dreht sich um
# Strictly convex
Eine Funktion ist "strikt convex", wenn die obere Ungleichung nur bei $\lambda =0$ und $\lambda =1$ erfüllt wird. 

# Jensen Ungleichheit
Für jede mögliche Sammlung an Punkten $x_{i}$ ist die Jensen Ungleichheit eine allgemeinere Formulierung (siehe [[Jensen Inequality]])

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]