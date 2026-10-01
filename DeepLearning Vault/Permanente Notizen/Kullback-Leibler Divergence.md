25-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Die Kullback-Leibler-Divergenz oder "KL-Divergenz) oder auch "relative Entropie", kann zum Vergleich von zwei Verteilungen verwendet werden. 
Angenommen, es gibt eine wahre Verteilung $p(x)$ und eine approximierte $q(x)$ der Variablen $x$, die mittlere minimale codelänge wird durch:
$$\begin{align}
\mathrm{KL}(p \| q) &= -\int p(\mathbf{x}) \ln q(\mathbf{x})\, \mathrm{d}\mathbf{x} - \left( -\int p(\mathbf{x}) \ln p(\mathbf{x})\, \mathrm{d}\mathbf{x} \right) \\
&= -\int p(\mathbf{x}) \ln \left\{ \frac{q(\mathbf{x})}{p(\mathbf{x})} \right\}\, \mathrm{d}\mathbf{x}.
\end{align}$$
Gegeben, wobei $-\int p(\mathbf{x}) \ln q(\mathbf{x})\, \mathrm{d}\mathbf{x}$ die [[Entropy]] von $p(x)$ ist und $- \left( -\int p(\mathbf{x}) \ln p(\mathbf{x})\, \mathrm{d}\mathbf{x} \right)$ die Kreuzentropie zwischen $p(x)$ und $q(x)$ ist. Diese Differenz zeigt den nötigen Informationsgehalt (in [[Nats]]), um werte aus $p(x)$ zu kodieren, anstatt $q(x)$.

>Merke: Es gilt $\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$, da die Funktion nicht Symmetrisch ist.

# Der Test mit KL-Divergenz
Es gilt $0 \leqslant \mathrm{KL}(p \| q)$, und nur wenn $p(x)=q(x)$ gilt.
Das schließt sich aus der [[Jensen Inequality]], wenn diese in die KL-Divergenz eingesetzt wird.

So kann die KL-Divergenz als Maß für die Unähnlichkeit zweier Verteilungen gesehen werden. 

# Eine Anwendung und Maxmimum log likelihood
Wir nehmen an eine wahre Verteilung $p(x)$ mit $q(x|\theta)$ zu approximieren, wobei $\theta$ anzupassende Parameter darstellen. Die KL-Divergenz wird innerhalb des Verfahrens minimiert. Da $p(x)$ unbekannt ist, muss man diese aus Daten annähern. Es gilt die Näherung: 
$$\mathrm{KL}(p \| q) \simeq \frac{1}{N} \sum_{n=1}^{N} \left\{ -\ln q(\mathbf{x}_n | \boldsymbol{\theta}) + \ln p(\mathbf{x}_n) \right\}$$
 $\ln p(\mathbf{x}_n)$ hängt nicht von $\theta$ ab, wobei der erste Term die negative log [[Likelihood]] Funktion ist. 
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]