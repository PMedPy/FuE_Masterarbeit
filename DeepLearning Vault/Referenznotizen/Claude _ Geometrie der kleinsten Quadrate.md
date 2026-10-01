
24-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

#  (Bishop, Abschnitt 4.1.4)

## Kernmissverständnis vorab: Was ist orthogonal zu was?

> [!important] Häufige Verwechslung **Nicht** $\mathbf{y}$ steht orthogonal auf $\mathbf{t}$ — das wäre geometrisch zufällig und ohne inhaltliche Bedeutung.
> 
> **Richtig:** Der **Residuenvektor** $(\mathbf{t} - \mathbf{y}_\mathrm{ML})$ steht orthogonal auf den **Unterraum** $\mathcal{S}$.
> 
> $$(\mathbf{t} - \mathbf{y}_\mathrm{ML}) \perp \mathcal{S}$$
> 
> Das ist keine geometrische Zufälligkeit, sondern äquivalent zur Minimierung des quadratischen Fehlers — und exakt der Inhalt der Normalengleichung.

---

## 1. Der N-dimensionale Zielraum

Man betrachtet einen $N$-dimensionalen Raum, dessen Achsen durch die Zielwerte $t_1, \ldots, t_N$ aufgespannt werden. Der Datenvektor

$$\mathbf{t} = (t_1, \ldots, t_N)^\top$$

ist ein fester Vektor in diesem Raum — die Messdaten, unveränderlich.

## 2. Basisvektoren und der Unterraum $\mathcal{S}$

Jede Basisfunktion $\phi_j$, ausgewertet an allen $N$ Datenpunkten, ergibt einen Vektor

$$\boldsymbol{\varphi}_j = \bigl(\phi_j(\mathbf{x}_1),, \ldots,, \phi_j(\mathbf{x}_N)\bigr)^\top \in \mathbb{R}^N$$

Dieser Vektor $\boldsymbol{\varphi}_j$ entspricht der $j$-ten **Spalte** der Designmatrix $\boldsymbol{\Phi}$.

Falls die Anzahl der Basisfunktionen $M$ kleiner ist als die Anzahl der Datenpunkte $N$, spannen die $M$ Vektoren $\boldsymbol{\varphi}_j$ einen $M$-dimensionalen **linearen Unterraum** $\mathcal{S} \subset \mathbb{R}^N$ auf.

## 3. Der Modellvektor y

Der Modellvektor $\mathbf{y}$ enthält die Vorhersagen des Modells an allen $N$ Datenpunkten:

$$y_n = y(\mathbf{x}_n, \mathbf{w}), \quad n = 1, \ldots, N$$

Da $\mathbf{y}$ eine Linearkombination der Basisvektoren ist, liegt er zwingend in $\mathcal{S}$:

$$\mathbf{y} = \sum_j w_j \boldsymbol{\varphi}_j = \boldsymbol{\Phi}\mathbf{w} \in \mathcal{S}$$

Durch Variation von $\mathbf{w}$ kann $\mathbf{y}$ **jeden Punkt in $\mathcal{S}$** annehmen.

> [!note] y als Zielgröße $\mathbf{y}$ und $\mathbf{t}$ haben dieselbe Dimension $N$ und leben im selben Raum. $\mathbf{t}$ sind die gemessenen Daten (fest), $\mathbf{y}$ ist die steuerbare Modellvorhersage.

## 4. Least-Squares als orthogonale Projektion

Der quadratische Fehler ist (bis auf Faktor $\tfrac{1}{2}$) das quadrierte euklidische Abstandsmaß zwischen $\mathbf{t}$ und $\mathbf{y}$:

$$E(\mathbf{w}) = \frac{1}{2}|\mathbf{t} - \mathbf{y}|^2$$

Die ML-Lösung wählt das $\mathbf{w}$, das $\mathbf{y}$ zum **Fußpunkt der Senkrechten** von $\mathbf{t}$ auf $\mathcal{S}$ macht — also zur orthogonalen Projektion von $\mathbf{t}$ auf $\mathcal{S}$.

> [!note] Intuition ML-Schätzung bedeutet hier: Welche Parameterwahl $\mathbf{w}$ bringt mich innerhalb von $\mathcal{S}$ am nächsten an $\mathbf{t}$? Das Minimum des Abstands ist genau dann erreicht, wenn der Residuenvektor senkrecht auf $\mathcal{S}$ steht.

### Orthogonalitätsbedingung

Der Residuenvektor muss senkrecht auf jedem Basisvektor stehen:

$$\boldsymbol{\varphi}_j^\top (\mathbf{t} - \mathbf{y}_\mathrm{ML}) = 0 \quad \forall, j$$

In Matrixform ist das exakt die **Normalengleichung**:

$$\boldsymbol{\Phi}^\top(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}_\mathrm{ML}) = 0$$

Aufgelöst nach $\mathbf{w}_\mathrm{ML}$:

$$\mathbf{w}_\mathrm{ML} = (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \mathbf{t}$$

Die Normalengleichung ist also keine algebraische Spielerei — sie ist direkt die geometrische Bedingung, dass das Residuum senkrecht auf $\mathcal{S}$ steht.

## 5. Numerische Probleme: Singularität von $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$

$\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ ist eine $M \times M$-Matrix. Sie wird **singulär** (nicht invertierbar), wenn zwei oder mehr der Spaltenvektoren $\boldsymbol{\varphi}_j$ **kollinear** sind — d.h. einer ist ein Vielfaches des anderen.

**Geometrische Bedeutung:** Zwei Basisvektoren zeigen in dieselbe Richtung im $\mathbb{R}^N$. Der aufgespannte Unterraum hat dann effektiv weniger als $M$ Dimensionen. Die orthogonale Projektion von $\mathbf{t}$ auf $\mathcal{S}$ ist weiterhin eindeutig — aber **welche Gewichtskombination $\mathbf{w}$ sie erzeugt, ist es nicht**. Unendlich viele $\mathbf{w}$ liefern dasselbe $\mathbf{y}$.

Dies führt dazu, dass $\det(\boldsymbol{\Phi}^\top \boldsymbol{\Phi}) = 0$ und die Inverse nicht existiert. Solche Beinahe-Kollinearitäten sind bei realen Datensätzen häufig.

### Abhilfen

|Methode|Idee|
|---|---|
|**SVD** (Singulärwertzerlegung)|Liefert trotz Singularität die Minimum-Norm-Lösung, numerisch stabil|
|**Regularisierung** ($\ell_2$ / Ridge)|Ersetzt $\boldsymbol{\Phi}^\top \boldsymbol{\Phi}$ durch $\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I}$, immer positiv definit|

$$(\boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I})^{-1} \boldsymbol{\Phi}^\top \mathbf{t}$$

Die Regularisierung hat dabei den schönen Nebeneffekt, dass die Matrix selbst bei exakter Kollinearität invertierbar bleibt — der Term $\lambda \mathbf{I}$ "hebt" alle Singulärwerte um $\lambda$ an.

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]