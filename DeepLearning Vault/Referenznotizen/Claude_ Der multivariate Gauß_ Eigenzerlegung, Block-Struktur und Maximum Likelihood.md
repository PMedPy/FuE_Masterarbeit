03-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Topic

> Zusammenfassung einer Lern-Session zu Bishop _Deep Learning_, Kapitel 3.2. Roter Faden: Fast alles, was hier passiert, ist im Kern **eine quadratische Form** — der Exponent des Gauß — und ihre Zerlegung in Eigenrichtungen. Wer diese eine Idee verinnerlicht hat, sieht sie in der Mahalanobis-Distanz, in den Ellipsen-Konturen und in der Block-Struktur immer wieder auftauchen.

---

## 1. Die quadratische Form und ihr eigentlicher Trick

Im Exponenten der Gauß-Dichte steht die **Mahalanobis-Distanz**:

$$\Delta^2 = (\mathbf{x}-\boldsymbol{\mu})^T ,\Sigma^{-1}, (\mathbf{x}-\boldsymbol{\mu})$$

Sie ist nicht die euklidische Distanz — der Faktor $\Sigma^{-1}$ gewichtet die Richtungen unterschiedlich. Der zentrale Umschreib-Trick beruht darauf, dass $\Sigma$ symmetrisch ist und deshalb eine **Orthonormalbasis aus Eigenvektoren** ${\mathbf{u}_1, \dots, \mathbf{u}_D}$ besitzt mit $\mathbf{u}_i^T\mathbf{u}_j = \delta_{ij}$.

Daraus folgt die **Vollständigkeitsrelation** (resolution of identity):

$$\sum_{i=1}^{D} \mathbf{u}_i \mathbf{u}_i^T = \mathbf{I}$$

Jedes $\mathbf{u}_i\mathbf{u}_i^T$ ist eine $D\times D$-Matrix (ein Projektor auf die Richtung $\mathbf{u}_i$); ihre Summe ergibt die Einheitsmatrix.

> **Wo es gehakt hat — die häufigste Verwechslung:** Man schreibt $\Delta^2$ über _einen_ Eigenvektor $\mathbf{u}^T(\mathbf{x}-\boldsymbol{\mu})$, gemeint ist aber immer die **Summe über alle** $\mathbf{u}_i$. Die Kurzschreibweise mit einem einzigen $\mathbf{u}$ verschleiert genau das. Hinter dem Umschreiben steckt nicht ein ausgewählter Eigenvektor, sondern die vollständige Basis.

Schiebt man die Identität in die euklidische Form ein und nutzt, dass $\mathbf{u}_i^T\mathbf{z}$ ein **Skalar** ist (Transponieren ändert an einer Zahl nichts), erhält man mit $\mathbf{z} = \mathbf{x}-\boldsymbol{\mu}$:

$$(\mathbf{x}-\boldsymbol{\mu})^T(\mathbf{x}-\boldsymbol{\mu}) = \sum_{i=1}^{D}\big(\mathbf{u}_i^T(\mathbf{x}-\boldsymbol{\mu})\big)^2$$

Für die echte Mahalanobis-Distanz kommt nur ein Gewicht hinzu. Aus $\Sigma\mathbf{u}_i = \lambda_i\mathbf{u}_i$ folgt für die Inverse dieselbe Basis mit invertierten Eigenwerten:

$$\Sigma^{-1} = \sum_{i=1}^{D}\frac{1}{\lambda_i},\mathbf{u}_i\mathbf{u}_i^T \qquad\Longrightarrow\qquad \Delta^2 = \sum_{i=1}^{D}\frac{\big(\mathbf{u}_i^T(\mathbf{x}-\boldsymbol{\mu})\big)^2}{\lambda_i}$$

**Intuition:** Definiert man die gedrehten Koordinaten $y_i = \mathbf{u}_i^T(\mathbf{x}-\boldsymbol{\mu})$, wird die quadratische Form diagonal:

$$\Delta^2 = \sum_i \frac{y_i^2}{\lambda_i}$$

Das ist eine Rotation in das System der Hauptachsen der Verteilung — und es ist bereits die Gleichung eines Ellipsoids (siehe Abschnitt 5).

### Durchgerechnetes Beispiel

Für $\Sigma = \begin{pmatrix} 2 & 1 \ 1 & 2 \end{pmatrix}$ liefert das charakteristische Polynom $\lambda_1 = 3,\ \lambda_2 = 1$ mit

$$\mathbf{u}_1 = \tfrac{1}{\sqrt{2}}\begin{pmatrix}1\1\end{pmatrix}, \qquad \mathbf{u}_2 = \tfrac{1}{\sqrt{2}}\begin{pmatrix}1\-1\end{pmatrix}$$

Die Projektoren addieren sich zur Identität — die Diagonalelemente ergänzen sich, die Außerdiagonalen heben sich auf (kein Zufall, sondern Folge der Orthogonalität):

$$\tfrac12\begin{pmatrix}1&1\1&1\end{pmatrix} + \tfrac12\begin{pmatrix}1&-1\-1&1\end{pmatrix} = \begin{pmatrix}1&0\0&1\end{pmatrix}$$

Für $\mathbf{z} = (2, 0)^T$ stimmen beide Rechenwege überein:

$$\mathbf{z}^T\Sigma^{-1}\mathbf{z} = \tfrac{8}{3} \qquad\text{und}\qquad \sum_i\frac{(\mathbf{u}_i^T\mathbf{z})^2}{\lambda_i} = \frac{2}{3} + \frac{2}{1} = \tfrac{8}{3}$$

Schön sichtbar wird hier der **Effekt der Gewichtung**: Die Komponente entlang $\mathbf{u}_2$ (kleiner Eigenwert, schmale Richtung) zählt voll, die entlang $\mathbf{u}_1$ (großer Eigenwert, breite Richtung) nur mit $\tfrac13$. Abweichungen in Richtungen großer Varianz sind „billiger", weil sie dort statistisch unauffälliger sind.

---

## 2. Woher kommen die Eigenwerte? — die fehlende Theorie

> **Schwerpunkt.** Hier lag die größte offene Frage: _Warum_ gibt es Eigenwerte, _warum genau $D$ Stück_, und woher kommt der Zusammenhang „$D$ = Anzahl Eigenwerte"? Deshalb dieser Abschnitt ausführlicher.

**Was ein Eigenvektor ist.** Eine Matrix dreht und streckt Vektoren normalerweise gleichzeitig. Ein Eigenvektor ist eine ausgezeichnete Richtung, in der die Matrix _nicht dreht_, sondern nur skaliert. Der Eigenwert $\lambda$ ist dieser Streckfaktor:

$$\Sigma,\mathbf{u} = \lambda,\mathbf{u}, \qquad \mathbf{u} \neq \mathbf{0}$$

**Wie man sie findet.** Umgestellt:

$$(\Sigma - \lambda I),\mathbf{u} = \mathbf{0}$$

Das ist ein homogenes Gleichungssystem. Wäre $(\Sigma - \lambda I)$ invertierbar, gäbe es nur die triviale Lösung $\mathbf{u}=\mathbf{0}$. Eine _nichttriviale_ Richtung existiert also nur, wenn die Matrix singulär ist — wenn ihre Determinante verschwindet:

$$\det(\Sigma - \lambda I) = 0$$

Das ist die **charakteristische Gleichung**, der einzige Hebel, um die $\lambda$ überhaupt herauszubekommen.

**Warum genau $D$ Eigenwerte.** Entscheidend ist, _was für ein Objekt_ diese Determinante als Funktion von $\lambda$ ist. Für $2\times2$:

$$\det\begin{pmatrix} a-\lambda & b \ b & d-\lambda \end{pmatrix} = \lambda^2 - (a+d)\lambda + (ad - b^2)$$

Ein **Polynom 2. Grades** — und ein quadratisches Polynom hat (Fundamentalsatz der Algebra) genau 2 Nullstellen. Daher zwei Eigenwerte. Nicht weil „$D=2$" eine eigene Regel wäre, sondern weil die Determinante einer $2\times2$-Matrix ein quadratisches Polynom erzeugt.

Allgemein liefert das Diagonalprodukt $(a_{11}-\lambda)\cdots(a_{DD}-\lambda)$ den führenden Term $(-\lambda)^D$. Das charakteristische Polynom hat also **Grad $D$** und damit genau $D$ Nullstellen (mit Vielfachheit gezählt). Das ist die ganze Herkunft von „$D$ = Anzahl Eigenwerte": Sie steckt nicht als Annahme in der Matrix, sondern fällt aus dem Grad des Polynoms heraus.

**Der Feinschliff — Spektralsatz.** Im Allgemeinen könnten diese $D$ Wurzeln komplex oder mehrfach sein. Hier rettet die Symmetrie: Eine Kovarianzmatrix ist reell und symmetrisch ($\Sigma = \Sigma^T$). Für solche Matrizen garantiert der **Spektralsatz**:

- alle $D$ Eigenwerte sind **reell**,
- die Eigenvektoren bilden eine **Orthonormalbasis**.

Genau das ist der tiefere Grund, warum die Beispiele so glatt aufgehen — und warum die Vollständigkeitsrelation aus Abschnitt 1 überhaupt gilt. Sie ist kein Zufall der gewählten Zahlen, sondern Konsequenz der Symmetrie von $\Sigma$.

---

## 3. Die Block-Struktur des Gauß

### Wie die vier Terme entstehen (Bishop 3.54)

Bei der bedingten Verteilung zerlegt man Vektor und **Präzisionsmatrix** $\Lambda = \Sigma^{-1}$ in Blöcke gemäß der Aufteilung in $a$- und $b$-Komponenten:

$$\mathbf{x}-\boldsymbol{\mu} = \begin{pmatrix} \mathbf{z}_a \ \mathbf{z}_b \end{pmatrix}, \qquad \Lambda = \begin{pmatrix} \Lambda_{aa} & \Lambda_{ab} \ \Lambda_{ba} & \Lambda_{bb} \end{pmatrix}$$

Die quadratische Form $\mathbf{z}^T\Lambda\mathbf{z}$ ist dann **reine Block-Matrix-Multiplikation**. Man kombiniert jeden der zwei Blöcke von links ($a$ oder $b$) mit jedem der zwei von rechts — ein $2\times2$-Raster, also $2\cdot2 = 4$ Terme:

$$\mathbf{z}_a^T\Lambda_{aa}\mathbf{z}_a + \mathbf{z}_a^T\Lambda_{ab}\mathbf{z}_b + \mathbf{z}_b^T\Lambda_{ba}\mathbf{z}_a + \mathbf{z}_b^T\Lambda_{bb}\mathbf{z}_b$$

Exakt wie $(a+b)(a+b) = aa + ab + ba + bb$. Bei Skalaren würde man $ab$ und $ba$ zu $2ab$ zusammenfassen; hier _darf_ man das erst über die Symmetrie $\Lambda_{ab} = \Lambda_{ba}^T$ erkennen.

### Präzisions- vs. Kovarianzmatrix (3.57 vs. 3.66)

Die bedingte Verteilung $p(\mathbf{x}_a \mid \mathbf{x}_b)$ ist wieder gaußförmig. Ihre Kovarianz lässt sich auf zwei Weisen schreiben:

|Darstellung|Kovarianz von $p(\mathbf{x}_a\mid\mathbf{x}_b)$|
|---|---|
|über Präzisionsblöcke (einfach)|$\Lambda_{aa}^{-1}$|
|über Kovarianzblöcke (sperrig)|$\Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$|

Beide Ausdrücke sind **identisch** — es gilt nämlich genau

$$\Lambda_{aa}^{-1} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$$

Das ist das **Schur-Komplement** von $\Sigma_{bb}$. Merksatz: $\Lambda_{aa}$ ist **nicht** $(\Sigma_{aa})^{-1}$ — die Blöcke der Inversen sind nicht die Inversen der Blöcke.

Genau das ist Bishops Pointe: Wer die _volle_ Matrix $\Sigma$ einmal invertiert, um $\Lambda$ zu erhalten, hat den Schur-Komplement-Aufwand bereits miterledigt — er steckt fertig im Block $\Lambda_{aa}$. Die Präzisionsmatrix „kennt" die bedingte Abhängigkeitsstruktur direkt. (Deshalb spielt sie auch in graphischen Modellen die Hauptrolle: eine Null in $\Lambda_{ab}$ bedeutet bedingte Unabhängigkeit.)

---

## 4. Maximum Likelihood beim multivariaten Gauß

### Notation: ein Punkt vs. der ganze Datensatz

|Symbol|Bedeutung|
|---|---|
|$\mathbf{x}_n$|**ein** Datenpunkt — ein Vektor mit $D$ Komponenten|
|${\mathbf{x}_n}$|die **Menge aller** Punkte: $\mathbf{x}_1, \dots, \mathbf{x}_N$|
|$N$|Anzahl der Beobachtungen, $D$ = Dimension|

Beispiel (Enzymkinetik, $D=2$): Jeder Punkt $\mathbf{x}_n = (\log K_M,\ \log k_{\text{cat}})^T$ einer Enzymvariante. Gestapelt ergibt das die Datenmatrix $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)^T$ der Größe $N\times D$, in der **jede Zeile ein $\mathbf{x}_n^T$** ist (daher das Transponieren).

Die Annahme „independently drawn" ist genau der Grund, warum in der Likelihood ein **Produkt** über alle $n$ steht — und nach dem Logarithmieren eine **Summe** $\sum_{n=1}^N$. Die ML-Lösungen sind Stichproben-Mittelwert und Stichproben-Kovarianz:

$$\boldsymbol{\mu}_{\text{ML}} = \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n, \qquad \Sigma_{\text{ML}} = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}_{\text{ML}})(\mathbf{x}_n - \boldsymbol{\mu}_{\text{ML}})^T$$

### Erwartungswert, Bias und die N−1-Korrektur

> **Schwerpunkt.** Die wiederkehrende Frage: Was bedeutet $\mathbb{E}[\boldsymbol{\mu}_{\text{ML}}]$ bzw. $\mathbb{E}[\Sigma_{\text{ML}}]$ eigentlich? Deshalb hier präzise.

Der Kern: $\boldsymbol{\mu}_{\text{ML}}$ ist selbst eine **Zufallsgröße**, weil sie von den zufällig gezogenen Daten abhängt. Ein anderer Datensatz → ein anderes $\boldsymbol{\mu}_{\text{ML}}$.

$\mathbb{E}[\boldsymbol{\mu}_{\text{ML}}]$ ist der Mittelwert dieser Schätzung **über unendlich viele hypothetische Datensätze** der Größe $N$, jeder frisch aus der wahren Verteilung gezogen. _Nicht_ der Mittelwert innerhalb eines Datensatzes, sondern über die gedachte Wiederholung des ganzen Experiments.

$$\mathbb{E}[\boldsymbol{\mu}_{\text{ML}}] = \boldsymbol{\mu}$$

Der ML-Mittelwert streut um $\boldsymbol{\mu}$, trifft aber _im Mittel_ exakt — **erwartungstreu** (unbiased).

$$\mathbb{E}[\Sigma_{\text{ML}}] = \frac{N-1}{N},\Sigma$$

Hier ist das Ergebnis um den Faktor $\tfrac{N-1}{N} < 1$ **zu klein** — der ML-Kovarianzschätzer unterschätzt die wahre Streuung systematisch. Grund: $\Sigma_{\text{ML}}$ wird mit $\boldsymbol{\mu}_{\text{ML}}$ gebildet, nicht mit dem wahren $\boldsymbol{\mu}$. Die Punkte liegen per Konstruktion näher an _ihrem eigenen_ Schwerpunkt als am unbekannten wahren Mittelwert. Man misst die Streuung gegen einen Bezugspunkt, der „zu gut passt", und bekommt eine zu kleine Varianz.

Die **Korrektur** (Bessel): Teilen durch $N-1$ statt $N$ macht den Schätzer erwartungstreu.

$$\widetilde{\Sigma} = \frac{1}{N-1}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}_{\text{ML}})(\mathbf{x}_n - \boldsymbol{\mu}_{\text{ML}})^T, \qquad \mathbb{E}[\widetilde{\Sigma}] = \Sigma$$

Anschaulich: ein Freiheitsgrad ist bereits für die Schätzung von $\boldsymbol{\mu}_{\text{ML}}$ „verbraucht". Für großes $N$ wird der Unterschied vernachlässigbar; bei kleinen Datensätzen (z. B. fünf Enzymvarianten) ist er spürbar.

---

## 5. Geometrische Synthese: die Ellipsen-Konturen (Bishop, Abb. 3.6)

Hier schließt sich der Kreis zu Abschnitt 1. Eine bivariate Gauß-Verteilung lebt in 3D: über jedem Punkt der Ebene eine Höhe $p(\mathbf{x})$, eine Glocke. Schneidet man sie auf konstanten Höhen durch, ergeben die projizierten Schnittlinien die **roten Ringe** — Höhenlinien konstanter Dichte, wie auf einer Wanderkarte.

**Warum Ellipsen.** „Gleiche Höhe" heißt gleiche Mahalanobis-Distanz:

$$(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \text{const} \quad\Longleftrightarrow\quad \sum_i \frac{y_i^2}{\lambda_i} = \text{const}$$

Das ist exakt die Ellipsengleichung aus Abschnitt 1. Damit lassen sich alle Bildmerkmale ablesen:

- **Zentrum** aller Ringe bei $\boldsymbol{\mu}$.
- **Achsenrichtungen** = Eigenvektoren $\mathbf{u}_i$ von $\Sigma$. Schräg liegende Ellipsen bedeuten Verdrehung gegen die Koordinatenachsen — also **Korrelation** zwischen den beiden Größen.
- **Halbachsenlängen** = $\sqrt{\lambda_i}$. Lange Achse = großer Eigenwert (große Varianz), kurze Achse = kleiner Eigenwert.
- **Mehrere Ringe** = dieselbe Ellipse, nur skaliert: gleiches Zentrum, gleiche Achsen, nach außen wachsend (innen hohe Dichte, außen niedrige). Analog zur 1σ/2σ/3σ-Logik der 1D-Glocke.

**Die Pointe der Abbildung.** Ein einzelner Gauß hat nur _ein_ Zentrum und _einen_ elliptischen Fleck. Zerfallen die Daten in zwei Klumpen, legt er viel Wahrscheinlichkeitsmasse in die spärlich besetzte Region dazwischen. Ein **Mixture of Gaussians** (mehrere Glocken mit eigenem $\boldsymbol{\mu}$, $\Sigma$ und Gewicht) passt dann viel besser — der Übergang, auf den Bishop hier hinarbeitet. Für die Enzymkinetik relevant, falls die Parameter in Unterpopulationen zerfallen (z. B. verschiedene Enzymklassen mit eigenem $K_M$-Regime).

---

## Roter Faden in einem Satz

Der Exponent des Gauß ist eine quadratische Form; ihre Eigenzerlegung dreht das Problem in die Hauptachsen ($\sum_i y_i^2/\lambda_i$), und genau diese eine Idee erklärt die Mahalanobis-Distanz, die elliptischen Dichtekonturen und — über die Block-Inverse $\Lambda$ — auch die bedingte Verteilung.


# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]