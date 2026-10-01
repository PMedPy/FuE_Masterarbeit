29-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Auch auf Deutsch "Gegenseitige Information" oder "Transinformation"

Wenn zwei Variablen $x$ und $y$ unabhängig voneinander sind, gilt: $p(x,y) =p(x)p(y)$ 

Falls diese Variablen voneinander Abhängen, kann die [[Kullback-Leibler Divergence]] als Maß für die Intensität der Abhängigkeit verwendet werden:
$$ \begin{aligned}
\mathrm{I}[\mathbf{x}, \mathbf{y}] &\equiv \mathrm{KL}(p(\mathbf{x}, \mathbf{y}) \,\|\, p(\mathbf{x}) p(\mathbf{y})) \\
&= -\iint p(\mathbf{x}, \mathbf{y}) \ln\!\left( \frac{p(\mathbf{x}) p(\mathbf{y})}{p(\mathbf{x}, \mathbf{y})} \right) \mathrm{d}\mathbf{x} \, \mathrm{d}\mathbf{y}
\end{aligned}$$
Die Größe $\mathrm{I}[\mathbf{x}, \mathbf{y}]$ ist die Transinformation zwischen  $x$ und $y$  
Über die [[Sum Rule]] und [[Product Rule]] ist die [[Conditional Entropy]] mit  $\mathbf{I}$ verknüpft:
$$\mathrm{I}[\mathbf{x}, \mathbf{y}] = \mathrm{H}[\mathbf{x}] - \mathrm{H}[\mathbf{x} | \mathbf{y}] = \mathrm{H}[\mathbf{y}] - \mathrm{H}[\mathbf{y} | \mathbf{x}]$$
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]