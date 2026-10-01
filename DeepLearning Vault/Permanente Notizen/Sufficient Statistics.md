09-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
> Suffiziente Statistik ist eine Funktion der Daten, die alle Informationen der Daten zur **Schätzung** des einen Parameters einer Verteilung benötigt wird. 

# Generell
Gegeben ist ein Datensatz $X =[x_{1},x_{2}\dots x_{n}]$ und ein Parameter $\eta$. Eine Statistik $u(x)$ (irgendeine Funktion von x) ist suffizient zu $\eta$, wenn die [[Likelihood]] nur über diese Statistik abhängt:
$$p(\mathbf{X} \mid \mathbf{u}(\mathbf{X}), \boldsymbol{\eta}) = p(\mathbf{X} \mid \mathbf{u}(\mathbf{X}))$$
Die
### Vorstellung:
Die Suffiziente Statistik beinhaltet nur die Infos, die wichtig sind um die Parameter für die Verteilung der Daten zu bestimmen:
$$\underbrace{\mathbf{X}}_{\text{volle Daten}} \;\longrightarrow\; \underbrace{\mathbf{u}(\mathbf{X})}_{\text{Info über }\boldsymbol{\eta}} \;+\; \underbrace{\text{Rest}}_{\text{parameter-frei, wird verworfen}}$$

Der "Rest" sind informationen über Reihenfolge bestimmter Ergebnisse
# Weiterführung
[[Multivariant Gaussian]]
[[Exponential Family]]
# Referenzen
[@bishopDeepLearningFoundations2024]