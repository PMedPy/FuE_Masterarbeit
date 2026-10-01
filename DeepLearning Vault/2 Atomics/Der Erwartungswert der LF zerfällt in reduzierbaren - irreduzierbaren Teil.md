08-07-2026
Tags: #FuE #MachineLearning 
Status: #Atomic

Der Erwartungswert der Lossfunktion lässt sich auf zwei Terme aufteilen, wobei der erste sich auf 0 reduziert wenn $f(x) = E[\mathbf{t}\mid \mathbf{x}]$ und der Zweite die mittlere Varianz $var(x \mid t)$ über die Auftrittswahrscheinlichkeit $p(x)$ ist (Hier würde ich dann die Gleichung einbauen)

$$\mathbb{E}[L] = \underbrace{\int \{f(x) - \mathbb{E}[t\mid x]\}^2\, p(x)\, dx}_{\to 0 \text{ wenn } f(x)=\mathbb{E}[t|x]} + \underbrace{\int \operatorname{Var}[t\mid x]\, p(x)\, dx}_{\text{irreduzibel}}$$
# Weiterführung
[[Loss function]]
[[Die wahl der Lossfunktion (LF) ist unabhängig vom Datenrauschen]]

# Referenzen
[@bishopDeepLearningFoundations2024]