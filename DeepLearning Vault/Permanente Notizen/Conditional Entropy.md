29-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Bein einer gemeinsamen Wahrscheinlichkeitsverteilung $p(x,y)$ kann eine bedingte Entropie berechnet werden, wenn z.B $x$ bereits bekannt ist. 
$$\mathrm{H}[\mathbf{y} | \mathbf{x}] = -\iint p(\mathbf{y}, \mathbf{x}) \ln p(\mathbf{y} | \mathbf{x}) \, \mathrm{d}\mathbf{y} \, \mathrm{d}\mathbf{x}$$
Bedeutet: Die [[Entropy]] von $y$ unter allen beobachteten Werten von $x$.
Nicht zufällig gilt beim anlegen der [[Product Rule]], dass die [[Entropy]] von $(y | x)$ die Bedingte Entropie - Die marginale [[Entropy]] von $x$ ist:
$$\mathrm{H}[\mathbf{y} , \mathbf{x}]= \mathrm{H}[\mathbf{y} | \mathbf{x}]+ \mathrm{H}[ \mathbf{x}]$$
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]