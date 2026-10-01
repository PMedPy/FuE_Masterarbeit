22-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Gaussverteilung abhängig von mehreren Variablen

Anders als die Multinomial-Verteilung, wo nur bestimmte Zustände möglich sind (z.B eine Range von $[0,1]$), ist der Gauss eine Verteilung stetiger Variablen:
$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\}$$
Mit:
- $\mathbf{x} \in \mathbb{R}^D$ – ein Punkt im $D$-dimensionalen Raum, an dem wir die Dichte auswerten
- $\boldsymbol{\mu} \in \mathbb{R}^D$ – der Mittelwertvektor, das Zentrum der Verteilung
- $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ – die Kovarianzmatrix (Siehe [[Covariance matrix]]), symmetrisch und positiv definiert. Sie kodiert wie die Verteilung im Raum "geformt" und "orientiert" ist
- $|\boldsymbol{\Sigma}|$ – Determinante von $\boldsymbol{\Sigma}$, ein Maß für das "Volumen" der Verteilung

Die multivariate Version ist strukturell genau das Gleiche, nur mit Matrizen statt Skalaren:

| 1D                    | $D$-dimensional                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| $\sigma^2$ (Varianz)  | $\boldsymbol{\Sigma}$ (Kovarianzmatrix)                                                           |
| $1/\sigma^2$          | $\boldsymbol{\Sigma}^{-1}$ (Präzisionsmatrix)                                                     |
| $(x-\mu)^2$           | $(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |
| $\sqrt{2\pi\sigma^2}$ | $(2\pi)^{D/2}$                                                                                    |

### 1. Der Exponent – die Form

$$-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$$

Der Ausdruck $(\mathbf{x}-\boldsymbol{\mu})^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ ist eine skalare Größe und heißt **quadrierte [[Mahalanobis-Distance]] $\Delta^2$ von $\mathbf{x}$ zu $\boldsymbol{\mu}$.

Das ist der konzeptionell wichtigste Teil. Eine "normale" euklidische Distanz $|\mathbf{x}-\boldsymbol{\mu}|^2$ behandelt alle Richtungen gleich. Mahalanobis-Distanz aber **gewichtet die Richtungen** nach der inversen Kovarianz: In Richtungen großer Varianz ist ein Schritt "billig", in Richtungen kleiner Varianz "teuer".

Geometrisch: Setzt man $\Delta^2 = \text{const}$, bekommt man die **Höhenlinien** der Verteilung. Im 2D sind das Ellipsen, im 3D Ellipsoide, allgemein Hyperellipsoide. Die Hauptachsen dieser Ellipsen zeigen entlang der **Eigenvektoren von $\boldsymbol{\Sigma}$**, und die Halbachsenlängen sind proportional zu $\sqrt{\lambda_i}$ (Wurzel der Eigenwerte). 

### 2. Der Vorfaktor – die Normalisierung

$$\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}$$

Dieser Faktor sorgt dafür, dass das Integral über den ganzen $\mathbb{R}^D$ gleich 1 ist – Voraussetzung dafür, dass es eine gültige Wahrscheinlichkeitsdichte ist.

- $(2\pi)^{D/2}$ kommt vom Gauß-Integral, einmal pro Dimension: $\int e^{-x^2/2},dx = \sqrt{2\pi}$
- $|\boldsymbol{\Sigma}|^{1/2}$ skaliert nach dem "Volumen" der Verteilung. Eine breite Verteilung muss niedriger sein, damit das Gesamtvolumen unter der Dichte 1 bleibt

## Spezialfall: Sigma diagonal

Setzt man $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_D^2)$ (also keine Korrelationen zwischen Dimensionen), faktorisiert die multivariate Gauß in ein Produkt aus $D$ unabhängigen 1D-Gauß-Verteilungen. Das ist ein guter Sanity-Check für die Formel – sie reduziert sich korrekt auf den bekannten Fall.

## Warum das für dich relevant ist

Die multivariate Gauß taucht in deinem Projekt an mehreren Stellen wieder auf:

- als Likelihood-Modell in der Regression (Bishop Kap. 4, beim Übergang zu vektorwertigen Targets)
- als Prior in der Bayesian Linear Regression
- als Annahme über latente Variablen in vielen generativen Modellen
- und konkret bei Enzymkinetik: wenn du z.B. $K_M$ und $V_\text{max}$ gleichzeitig vorhersagst, sind das zwei korrelierte Targets – die Kovarianzstruktur ist dann nicht ignorierbar


---

# Referenzen
[@bishopDeepLearningFoundations2024]
[Claude Ai](https://claude.ai/chat/df60df95-e614-4b03-986c-e0d4e5597054)