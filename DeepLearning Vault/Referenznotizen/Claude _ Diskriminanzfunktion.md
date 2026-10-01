17-07-2026
Tags: #FuE #MachineLearning 
Status: #extended

## title: "Bishop 5.1 – Lineare Diskriminanzfunktionen" quelle: "Bishop, Deep Learning: Foundations and Concepts, Kapitel 5.1" type: concept

# Lineare Diskriminanzfunktionen (Bishop 5.1)

> [!info] Kernaussage des Kapitels Klassifikation wird zu Geometrie. Ein linearer Ausdruck $y(\mathbf{x}) = \mathbf{w}^\mathrm{T}\mathbf{x} + w_0$ zerschneidet den Eingaberaum mit einer Hyperebene. Dabei gilt:
> 
> - $\mathbf{w}$ bestimmt die **Orientierung** der Trennfläche (Normalenvektor),
> - $w_0$ bestimmt die **Verschiebung** vom Ursprung,
> - $y(\mathbf{x})/|\mathbf{w}|$ ist der **vorzeichenbehaftete Abstand** von der Trennfläche.

---

## 0. Der Lineare-Algebra-Baukasten

Alles in diesem Kapitel steht auf vier Bausteinen. Wer die hat, braucht nichts weiter.

### 0.1 Skalarprodukt

Für $\mathbf{a}, \mathbf{b} \in \mathbb{R}^D$:

$$\mathbf{a}^\mathrm{T}\mathbf{b} = \sum_{i=1}^{D} a_i b_i = |\mathbf{a}|,|\mathbf{b}|\cos\theta$$

Die zweite Gleichheit ist die geometrische Lesart: Das Skalarprodukt misst, **wie stark $\mathbf{b}$ in Richtung $\mathbf{a}$ zeigt**, skaliert mit beiden Längen.

Wichtigste Konsequenz:

$$\mathbf{a}^\mathrm{T}\mathbf{b} = 0 \iff \mathbf{a} \perp \mathbf{b} \quad (\text{für } \mathbf{a},\mathbf{b} \neq \mathbf{0})$$

Denn $\cos 90° = 0$.

### 0.2 Norm

$$|\mathbf{a}| = \sqrt{\mathbf{a}^\mathrm{T}\mathbf{a}} = \sqrt{\sum_i a_i^2}$$

Daraus die Identität, die weiter unten dreimal gebraucht wird:

$$\mathbf{a}^\mathrm{T}\mathbf{a} = |\mathbf{a}|^2$$

### 0.3 Einheitsvektor

$$\hat{\mathbf{a}} = \frac{\mathbf{a}}{|\mathbf{a}|}, \qquad |\hat{\mathbf{a}}| = 1$$

Ein Einheitsvektor trägt **nur die Richtungsinformation**, keine Längeninformation.

### 0.4 Projektion auf eine Richtung

Die (vorzeichenbehaftete) Länge des Schattens von $\mathbf{x}$ auf die Richtung $\hat{\mathbf{a}}$:

$$\text{proj}_{\hat{\mathbf{a}}}(\mathbf{x}) = \hat{\mathbf{a}}^\mathrm{T}\mathbf{x} = \frac{\mathbf{a}^\mathrm{T}\mathbf{x}}{|\mathbf{a}|}$$

Das ist ein **Skalar**, keine Vektorgröße. Vorzeichen negativ = $\mathbf{x}$ zeigt entgegen $\hat{\mathbf{a}}$.

> [!note] Merksatz Sobald irgendwo $\dfrac{\mathbf{w}^\mathrm{T}(\cdots)}{|\mathbf{w}|}$ steht, lies: _"Abstand, gemessen entlang der Normalenrichtung."_ Die Division durch $|\mathbf{w}|$ ist immer eine Normierung auf Einheitslänge.

---

## 1. Was eine Diskriminanzfunktion ist

Eine Diskriminanzfunktion nimmt einen Eingabevektor $\mathbf{x}$ und weist ihn einer von $K$ Klassen $\mathcal{C}_k$ zu. Bei zwei Klassen genügt eine einzige Funktion:

$$y(\mathbf{x}) = \mathbf{w}^\mathrm{T}\mathbf{x} + w_0 \tag{5.2}$$

Entscheidungsregel:

$$\mathbf{x} \in \mathcal{C}_1 \iff y(\mathbf{x}) \geq 0, \qquad \mathbf{x} \in \mathcal{C}_2 \iff y(\mathbf{x}) < 0$$

Begriffe:

|Symbol|Name|Rolle|
|---|---|---|
|$\mathbf{w}$|Gewichtsvektor (weight vector)|Orientierung der Trennfläche|
|$w_0$|Bias|Verschiebung der Trennfläche|
|$y(\mathbf{x}) = 0$|Entscheidungsgrenze (decision boundary)|die Trennfläche selbst|

> [!warning] Terminologie-Falle Der Bias $w_0$ hat **nichts** mit dem statistischen Bias aus der Bias-Varianz-Zerlegung zu tun. Gleiches Wort, völlig anderer Begriff. Bishop weist im Text extra darauf hin.

### Abgrenzung zur Regression

Die algebraische Form ist mit der linearen Regression identisch. Der Unterschied liegt in der **Verwendung des Outputs**:

|                       | Lineare Regression                               | Lineare Diskriminante                |
| --------------------- | ------------------------------------------------ | ------------------------------------ |
| Target                | kontinuierlich, $t \in \mathbb{R}$               | Klassenlabel $\mathcal{C}_k$         |
| Output-Interpretation | der Wert ist die Vorhersage                      | das Vorzeichen ist die Vorhersage    |
| Geometrie             | Fläche im $(\mathbf{x},t)$-Raum                  | Trennfläche **im** $\mathbf{x}$-Raum |
| Optimalitätskriterium | $\mathbb{E}[t\mid\mathbf{x}]$ unter Squared Loss | Fehlklassifikation unter 0/1-Loss    |

"Linear" bedeutet auch hier: **linear in den Parametern $\mathbf{w}$**, nicht zwingend in $\mathbf{x}$. Ersetzt man $\mathbf{x}$ durch Basisfunktionen $\boldsymbol{\phi}(\mathbf{x})$, ist die Trennfläche im $\boldsymbol{\phi}$-Raum eine Hyperebene, im ursprünglichen $\mathbf{x}$-Raum aber gekrümmt.

> [!important] Was in 5.1 noch NICHT passiert Abschnitt 5.1 beschreibt nur das **Modell** und seine Geometrie — nicht, wie $\mathbf{w}$ gelernt wird. Die Lernverfahren (Least Squares, Perceptron, Fisher-Diskriminante) folgen erst danach. Pointe des Kapitels: das Modell ist dasselbe wie bei der Regression, das **Trainingskriterium** darf es aber nicht sein — naives Least Squares auf Klassenlabels funktioniert schlecht, weil weit entfernte, korrekt klassifizierte Punkte großen quadratischen Fehler erzeugen und die Grenze verziehen.

---

## 2. Warum die Grenze eine Hyperebene ist

Die Entscheidungsgrenze ist die Lösungsmenge von

$$\mathbf{w}^\mathrm{T}\mathbf{x} + w_0 = 0$$

Das ist die Definitionsgleichung einer **affinen Hyperebene** im $\mathbb{R}^D$:

|$D$|Hyperebene ist|Dimension|
|---|---|---|
|2|eine Gerade|1|
|3|eine Ebene|2|
|$D$|Hyperebene|$D-1$|

Allgemein: Die Trennfläche ist $(D-1)$-dimensional. Eine skalare Gleichung entfernt genau einen Freiheitsgrad.

---

## 3. Herleitung 1: $\mathbf{w}$ steht senkrecht auf der Grenze

**Behauptung:** $\mathbf{w}$ ist der Normalenvektor der Entscheidungsgrenze.

**Beweis (drei Zeilen):**

Seien $\mathbf{x}_\mathrm{A}$ und $\mathbf{x}_\mathrm{B}$ zwei beliebige Punkte **auf** der Grenze. Dann gilt nach Definition der Grenze:

$$y(\mathbf{x}_\mathrm{A}) = 0 \quad \text{und} \quad y(\mathbf{x}_\mathrm{B}) = 0$$

Subtrahiere die beiden Gleichungen voneinander:

$$(\mathbf{w}^\mathrm{T}\mathbf{x}_\mathrm{A} + w_0) - (\mathbf{w}^\mathrm{T}\mathbf{x}_\mathrm{B} + w_0) = 0 - 0$$

Die $w_0$ heben sich weg, das Skalarprodukt ist linear:

$$\mathbf{w}^\mathrm{T}(\mathbf{x}_\mathrm{A} - \mathbf{x}_\mathrm{B}) = 0$$

**Interpretation (das ist der eigentliche Denkschritt):** Der Vektor $(\mathbf{x}_\mathrm{A} - \mathbf{x}_\mathrm{B})$ verbindet zwei Punkte, die beide auf der Grenze liegen — er zeigt also **entlang** der Grenzfläche. Da $\mathbf{x}_\mathrm{A}, \mathbf{x}_\mathrm{B}$ beliebig waren, ist jeder Richtungsvektor innerhalb der Grenzfläche von dieser Form. Nach Baustein 0.1 bedeutet das Skalarprodukt null: $\mathbf{w}$ steht auf **jeder** Richtung innerhalb der Fläche senkrecht.

$\Rightarrow$ $\mathbf{w}$ ist der Normalenvektor. $\mathbf{w}$ legt die Orientierung fest. $\blacksquare$

> [!note] Strukturelle Parallele Dasselbe Argumentmuster wie bei der Geometrie der kleinsten Quadrate: dort steht der Residuenvektor senkrecht auf dem von den Basisfunktionen aufgespannten Unterraum. Hier steht $\mathbf{w}$ senkrecht auf dem Richtungsraum der Trennfläche. In beiden Fällen ist "Skalarprodukt verschwindet für alle Vektoren eines Unterraums" die Definition von Orthogonalität zum Unterraum.

---

## 4. Herleitung 2: Der Abstand vom Ursprung (Gl. 5.3)

$\mathbf{w}$ gibt die Ausrichtung — aber **wo** liegt die Grenze? Das regelt $w_0$.

Sei $\mathbf{x}$ ein Punkt auf der Grenze, also $\mathbf{w}^\mathrm{T}\mathbf{x} + w_0 = 0$, umgestellt:

$$\mathbf{w}^\mathrm{T}\mathbf{x} = -w_0$$

Teile beide Seiten durch $|\mathbf{w}|$:

$$\frac{\mathbf{w}^\mathrm{T}\mathbf{x}}{|\mathbf{w}|} = -\frac{w_0}{|\mathbf{w}|} \tag{5.3}$$

**Interpretation:** Die linke Seite ist nach Baustein 0.4 exakt die Projektion von $\mathbf{x}$ auf die Einheits-Normalenrichtung $\mathbf{w}/|\mathbf{w}|$. Bemerkenswert: Diese Projektion hat für **jeden** Punkt der Grenze denselben Wert — nämlich $-w_0/|\mathbf{w}|$. Das muss so sein, denn die Grenze steht ja senkrecht zu $\mathbf{w}$; verschiebt man sich innerhalb der Grenze, ändert sich der Schatten auf die Normalenrichtung nicht.

$\Rightarrow$ Dieser gemeinsame Wert **ist** der vorzeichenbehaftete Abstand des Ursprungs zur Grenze, gemessen entlang der Normalen:

$$d_{\text{Ursprung}} = -\frac{w_0}{|\mathbf{w}|}$$

---

## 5. Herleitung 3: $y(\mathbf{x})$ misst den Abstand (Gl. 5.4–5.5)

Jetzt für einen **beliebigen** Punkt $\mathbf{x}$, nicht nur einen auf der Grenze.

**Schritt 1 — Zerlegung.** Sei $\mathbf{x}_\perp$ die orthogonale Projektion (der "Fußpunkt") von $\mathbf{x}$ auf die Grenze. Dann lässt sich $\mathbf{x}$ zerlegen in "Fußpunkt plus ein Schritt der Länge $r$ entlang der Einheitsnormalen":

$$\mathbf{x} = \mathbf{x}_\perp + r,\frac{\mathbf{w}}{|\mathbf{w}|} \tag{5.4}$$

Dabei ist $r$ genau die gesuchte Größe: der vorzeichenbehaftete Abstand. Diese Zerlegung ist keine Annahme, sondern eine Definition von $r$ — der Vektor $\mathbf{x} - \mathbf{x}_\perp$ steht per Konstruktion senkrecht auf der Grenze, ist also ein Vielfaches von $\mathbf{w}/|\mathbf{w}|$; $r$ ist dieses Vielfache.

**Schritt 2 — $\mathbf{w}^\mathrm{T}$ anwenden und $w_0$ addieren.** Auf beide Seiten von (5.4):

$$\mathbf{w}^\mathrm{T}\mathbf{x} + w_0 = \mathbf{w}^\mathrm{T}\mathbf{x}_\perp + w_0 + r,\frac{\mathbf{w}^\mathrm{T}\mathbf{w}}{|\mathbf{w}|}$$

**Schritt 3 — die drei Vereinfachungen.**

1. Linke Seite ist per Definition $y(\mathbf{x})$.
2. $\mathbf{w}^\mathrm{T}\mathbf{x}_\perp + w_0 = y(\mathbf{x}_\perp) = 0$, denn $\mathbf{x}_\perp$ liegt auf der Grenze.
3. $\mathbf{w}^\mathrm{T}\mathbf{w} = |\mathbf{w}|^2$ (Baustein 0.2), also $\dfrac{\mathbf{w}^\mathrm{T}\mathbf{w}}{|\mathbf{w}|} = |\mathbf{w}|$.

Es bleibt:

$$y(\mathbf{x}) = 0 + r,|\mathbf{w}|$$

**Schritt 4 — auflösen.**

$$r = \frac{y(\mathbf{x})}{|\mathbf{w}|} \tag{5.5}$$

> [!important] Die eigentliche Aussage des Abschnitts $y(\mathbf{x})$ ist nicht nur ein Vorzeichen-Orakel.
> 
> - Das **Vorzeichen** sagt: auf welcher Seite.
> - Der **Betrag**, geteilt durch $|\mathbf{w}|$, sagt: wie weit weg.
> 
> Großes $|y(\mathbf{x})|$ bedeutet: weit von der Grenze entfernt, also "sichere" Entscheidung. Diese Größe ist der Hebel für alles Weitere.

### Wozu das gebraucht wird

- **Logistische Regression** setzt $p(\mathcal{C}_1\mid\mathbf{x}) = \sigma(y(\mathbf{x}))$ mit der Sigmoid-Funktion. Der Abstand zur Grenze wird zur Konfidenz: nahe an der Grenze $\Rightarrow p \approx 0{,}5 \Rightarrow$ "weiß nicht".
- **Support Vector Machine (SVM)** maximiert direkt den kleinsten Abstand der Datenpunkte zur Grenze (den "Margin"). Ohne Gl. (5.5) lässt sich dieses Optimierungsproblem nicht einmal hinschreiben.
- **Neuronale Netze:** Ein einzelnes Neuron ist exakt $\mathbf{w}^\mathrm{T}\mathbf{x} + w_0$, gefolgt von einer Nichtlinearität. Ein tiefes Netz ist eine Verschachtelung genau dieser Bausteine.
- **Skalierungs-Redundanz:** $(\mathbf{w}, w_0)$ und $(\lambda\mathbf{w}, \lambda w_0)$ für $\lambda > 0$ beschreiben **dieselbe** Grenze, liefern aber unterschiedliche $y$-Werte. Die Division durch $|\mathbf{w}|$ rechnet genau diese Redundanz heraus. Das ist der Grund, warum $r$ eine wohldefinierte geometrische Größe ist und $y$ allein nicht — und warum Regularisierung über $|\mathbf{w}|$ überhaupt eine geometrische Bedeutung hat.

---

## 6. Die kompakte Notation (Gl. 5.6)

Buchhaltungstrick zur Beseitigung des Sonderfalls $w_0$. Führe eine Dummy-Eingabe $x_0 = 1$ ein und definiere:

$$\tilde{\mathbf{w}} = (w_0, \mathbf{w}), \qquad \tilde{\mathbf{x}} = (x_0, \mathbf{x}) = (1, \mathbf{x})$$

Dann ist

$$y(\mathbf{x}) = \tilde{\mathbf{w}}^\mathrm{T}\tilde{\mathbf{x}} \tag{5.6}$$

Nachrechnen: $\tilde{\mathbf{w}}^\mathrm{T}\tilde{\mathbf{x}} = w_0 \cdot 1 + \mathbf{w}^\mathrm{T}\mathbf{x}$. Stimmt.

Das ist dieselbe Konstruktion wie die Einser-Spalte in der Designmatrix bei der linearen Regression.

**Der Preis:** Im erweiterten $(D+1)$-dimensionalen Raum gibt es kein $w_0$ mehr, das die Fläche verschiebt. Die Trennfläche geht dort zwangsläufig **durch den Ursprung** und ist $D$-dimensional. Die affine Hyperebene im $\mathbb{R}^D$ wird zur linearen Hyperebene (Untervektorraum) im $\mathbb{R}^{D+1}$ — geschnitten mit der Ebene $x_0 = 1$ ergibt sich wieder die ursprüngliche affine Fläche.

---

## 7. Übungsaufgabe (2D, zum Selbstrechnen)

Gegeben:

$$\mathbf{w} = \begin{pmatrix} 3 \ 4 \end{pmatrix}, \qquad w_0 = -10$$

1. **Grenze zeichnen.** Setze $3x_1 + 4x_2 - 10 = 0$ und bestimme beide Achsenschnittpunkte.
2. **Normalenabstand vom Ursprung.** Berechne $|\mathbf{w}|$ und daraus $-w_0/|\mathbf{w}|$. In der Zeichnung nachmessen.
3. **Klassifizieren.** Bestimme für $\mathbf{x}_\mathrm{A} = (2,2)^\mathrm{T}$ und $\mathbf{x}_\mathrm{B} = (0,0)^\mathrm{T}$ jeweils Klasse und Abstand $r$.
4. **Skalierungstest.** Setze $\mathbf{w}' = 2\mathbf{w}$, $w_0' = 2w_0$. Ändert sich die Grenze? Ändert sich $y(\mathbf{x}_\mathrm{A})$? Ändert sich $r$?
5. **Fußpunkt.** Berechne $\mathbf{x}_\perp = \mathbf{x}_\mathrm{A} - r,\mathbf{w}/|\mathbf{w}|$ und prüfe, ob $y(\mathbf{x}_\perp) = 0$ herauskommt. Damit ist Gl. (5.4) selbst verifiziert.

> Schritt 4 ist der lehrreichste: Er zeigt, warum $|\mathbf{w}|$ im Nenner **stehen muss**.

---

## 8. Lösung

### Vorbereitung

$$|\mathbf{w}| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5, \qquad \frac{\mathbf{w}}{|\mathbf{w}|} = \begin{pmatrix} 0{,}6 \ 0{,}8 \end{pmatrix}$$

### Aufgabe 1 — Grenze

$3x_1 + 4x_2 - 10 = 0$

- $x_2 = 0 \Rightarrow x_1 = 10/3 \approx 3{,}33$ → Schnittpunkt $(10/3,\ 0)$
- $x_1 = 0 \Rightarrow x_2 = 10/4 = 2{,}5$ → Schnittpunkt $(0,\ 2{,}5)$

Gerade durch diese beiden Punkte. Der Vektor $\mathbf{w} = (3,4)^\mathrm{T}$ steht senkrecht darauf und zeigt vom Ursprung weg — Kontrolle: Richtungsvektor der Geraden ist $(0,2{,}5) - (10/3,0) = (-10/3,\ 2{,}5)$, und $3\cdot(-10/3) + 4\cdot 2{,}5 = -10 + 10 = 0$. ✓

### Aufgabe 2 — Abstand vom Ursprung

$$-\frac{w_0}{|\mathbf{w}|} = -\frac{-10}{5} = 2$$

Der Ursprung liegt also 2 Einheiten von der Grenze entfernt, und zwar auf der **negativen** Seite (die Grenze liegt in Richtung $+\mathbf{w}$ vom Ursprung aus).

### Aufgabe 3 — Klassifikation

**Punkt $\mathbf{x}_\mathrm{A} = (2,2)^\mathrm{T}$:**

$$y(\mathbf{x}_\mathrm{A}) = 3\cdot 2 + 4\cdot 2 - 10 = 6 + 8 - 10 = 4$$

$$r_\mathrm{A} = \frac{4}{5} = 0{,}8$$

$y \geq 0 \Rightarrow \mathbf{x}_\mathrm{A} \in \mathcal{C}_1$, Abstand $0{,}8$ auf der positiven Seite.

**Punkt $\mathbf{x}_\mathrm{B} = (0,0)^\mathrm{T}$:**

$$y(\mathbf{x}_\mathrm{B}) = 0 + 0 - 10 = -10, \qquad r_\mathrm{B} = \frac{-10}{5} = -2$$

$y < 0 \Rightarrow \mathbf{x}_\mathrm{B} \in \mathcal{C}_2$, Abstand $2$ auf der negativen Seite.

> [!note] Konsistenzcheck $|r_\mathrm{B}| = 2$ stimmt exakt mit dem Ergebnis aus Aufgabe 2 überein — der Ursprung _ist_ ja $\mathbf{x}_\mathrm{B}$. Vorzeichenlogik: $r_\mathrm{B} = -2$ heißt "Ursprung liegt 2 Einheiten auf der negativen Seite"; $-w_0/|\mathbf{w}| = +2$ heißt "die Grenze liegt 2 Einheiten in $+\mathbf{w}$-Richtung vom Ursprung". Dieselbe Aussage aus entgegengesetzter Perspektive.

### Aufgabe 4 — Skalierungstest

$\mathbf{w}' = (6,8)^\mathrm{T}$, $w_0' = -20$, $|\mathbf{w}'| = \sqrt{36+64} = 10$.

|Größe|Original|Skaliert|Geändert?|
|---|---|---|---|
|Grenzgleichung|$3x_1 + 4x_2 - 10 = 0$|$6x_1 + 8x_2 - 20 = 0$|**nein** (Faktor 2 kürzt sich)|
|$y(\mathbf{x}_\mathrm{A})$|$4$|$12 + 16 - 20 = 8$|**ja**, verdoppelt|
|$\|\mathbf{w}\|$|$5$|$10$|ja, verdoppelt|
|$r_\mathrm{A}$|$4/5 = 0{,}8$|$8/10 = 0{,}8$|**nein**|

**Das ist die Pointe:** $y$ ist von der Skalierung abhängig und damit **keine** geometrische Größe. $r = y/|\mathbf{w}|$ ist skalierungsinvariant und damit die echte Geometrie. Die Norm im Nenner ist kein kosmetischer Faktor, sondern das, was die Redundanz herausdividiert.

### Aufgabe 5 — Fußpunkt

$$\mathbf{x}_\perp = \mathbf{x}_\mathrm{A} - r_\mathrm{A}\frac{\mathbf{w}}{|\mathbf{w}|} = \begin{pmatrix} 2 \ 2 \end{pmatrix} - 0{,}8\begin{pmatrix} 0{,}6 \ 0{,}8 \end{pmatrix} = \begin{pmatrix} 2 - 0{,}48 \ 2 - 0{,}64 \end{pmatrix} = \begin{pmatrix} 1{,}52 \ 1{,}36 \end{pmatrix}$$

Probe:

$$y(\mathbf{x}_\perp) = 3\cdot 1{,}52 + 4\cdot 1{,}36 - 10 = 4{,}56 + 5{,}44 - 10 = 0 \quad \checkmark$$

Der Fußpunkt liegt tatsächlich auf der Grenze — Gl. (5.4) ist damit an einem konkreten Beispiel verifiziert.

**Zusatz (empfohlen):** Dasselbe für den Ursprung: $\mathbf{x}_\perp = (0,0)^\mathrm{T} - (-2)\cdot(0{,}6,\ 0{,}8)^\mathrm{T} = (1{,}2,\ 1{,}6)^\mathrm{T}$. Probe: $3{,}6 + 6{,}4 - 10 = 0$ ✓. Und $|(1{,}2,\ 1{,}6)^\mathrm{T}| = \sqrt{1{,}44 + 2{,}56} = \sqrt{4} = 2$ — passt zu Aufgabe 2.

---

## 9. Zusammenfassung als Kausalkette

$$\underbrace{\mathbf{w}}_{\text{Orientierung}} ;\longrightarrow; \underbrace{w_0}_{\text{Verschiebung}} ;\longrightarrow; \underbrace{y(\mathbf{x})}_{\text{skalierter Abstand}} ;\longrightarrow; \underbrace{y(\mathbf{x})/|\mathbf{w}|}_{\text{echter Abstand}}$$

|Schritt|Aussage|Beweisidee|
|---|---|---|
|$\mathbf{w} \perp$ Grenze|$\mathbf{w}$ ist Normalenvektor|Differenz zweier Grenzpunkte, $w_0$ hebt sich weg|
|$-w_0/\|\mathbf{w}\|$|Abstand Ursprung–Grenze|Grenzgleichung durch $\|\mathbf{w}\|$ teilen = Projektion|
|$r = y(\mathbf{x})/\|\mathbf{w}\|$|$y$ misst Abstand|Zerlegung $\mathbf{x} = \mathbf{x}_\perp + r,\hat{\mathbf{w}}$, dann $\mathbf{w}^\mathrm{T}$ anwenden|
|$y = \tilde{\mathbf{w}}^\mathrm{T}\tilde{\mathbf{x}}$|kompakte Notation|Dummy-Input $x_0 = 1$, Grenze durch Ursprung im $\mathbb{R}^{D+1}$|

## Offene Anschlusspunkte

- Wie wird $\mathbf{w}$ gelernt? (Least Squares, Perceptron, Fisher — folgt in 5.1.x ff.)
- Warum versagt Least Squares bei Klassifikation?
- Erweiterung auf $K > 2$ Klassen (Bishop 5.1.2): one-versus-rest und one-versus-one erzeugen mehrdeutige Regionen; Lösung sind $K$ Diskriminanzfunktionen mit $\arg\max_k y_k(\mathbf{x})$.
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]