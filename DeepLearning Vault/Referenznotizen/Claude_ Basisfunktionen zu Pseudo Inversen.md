17-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

---
# Bishop 4.1 — Von Basisfunktionen zur Pseudo-Inversen

> [!abstract] Der rote Faden Diese Notiz verfolgt **eine einzige Idee** durch ihre Verkleidungen: Lineare Regression ist ein _probabilistisches Modell_. Aus „Kurve durch Punkte" wird „bedingte Verteilung $p(t\mid x)$", aus „Fehler minimieren" wird „Likelihood maximieren", und am Ende fällt die geschlossene Lösung $\mathbf{w}_{ML}$ heraus. Jeder Schritt erzwingt den nächsten.

---

## 1 — Warum nichtlineare Basisfunktionen?

Das Modell ist eine **gewichtete Summe fester Bausteine**:

$$ y(x, \mathbf{w}) = \sum_{j} w_j, \phi_j(x) $$

> [!info] Math als Sprache Lies $\sum_j w_j \phi_j$ als **gewichteten Durchschnitt von Bausteinen**: Die $\phi_j$ sind feste, nichtlineare Formen, $\mathbf{w}$ entscheidet nur, _wie stark_ jeder Baustein zählt.

Der Clou — **linear in $\mathbf{w}$, nichtlinear in $x$**:

- **Ausdruckskraft:** Mit $\phi(x)=x$ gibt es nur Hyperebenen. Mit nichtlinearen $\phi$ (Gauß, Sigmoide, Polynome) lassen sich gekrümmte Zusammenhänge modellieren.
- **Trotzdem geschlossene Lösung:** Weil linear in $\mathbf{w}$, bleibt das Fehlergebirge konvex (eine Talsohle, kein Verirren) → analytische Normalengleichung.
- **Preis:** Die $\phi$ müssen vorab festgelegt werden; bei hochdimensionalem $x$ explodiert ihre Zahl (_curse of dimensionality_). → Genau das motiviert später neuronale Netze, die die $\phi$ **selbst lernen**.

> [!note] Lokalisierung kostet ein Frequenzspektrum Eine **Fourier**-Basis hat eine scharfe Frequenz, ist aber über den ganzen Raum ausgedehnt. Eine **räumlich lokalisierte** Basis (Gauß-Glocke) braucht _zwangsläufig_ ein ganzes Spektrum von Frequenzen, um ihren scharfen Abfall darzustellen. Das ist die **Fourier-Unschärferelation** $\Delta x,\Delta k \geq \tfrac12$ — dieselbe Mathematik wie Heisenberg ($\Delta x,\Delta p \geq \tfrac{\hbar}{2}$), weil Ort und Impuls ein Fourier-Paar sind. **Wavelets** sind der Kompromiss: lokalisiert in _beiden_ Achsen. Die Gauß-Glocke erfüllt die Ungleichung mit Gleichheit (sie ist ihr eigenes Fourier-Transformiertes).

---

## 2 — Die Likelihood: Regression wird zur Verteilung

Annahme (4.7): Ziel = deterministische Funktion + Rauschen.

$$ t = y(\mathbf{x}, \mathbf{w}) + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \sigma^2) $$

Daraus (4.8):

$$ p(t \mid \mathbf{x}, \mathbf{w}, \sigma^2) = \mathcal{N}!\left(t \mid y(\mathbf{x},\mathbf{w}),, \sigma^2\right) $$

> [!warning] Stolperstein: _zero-mean_ — wessen Mittelwert? Es gibt **zwei verschiedene Mittelwerte**, die leicht durcheinandergehen:
> 
> - Das **Rauschen** $\epsilon$ ist zero-mean — es schubst $t$ im Schnitt weder hoch noch runter.
> - Die **Zielgröße** $t$ ist **nicht** zero-mean. Addiert man eine Konstante ($y$) zu einer zero-mean-Variable, verschiebt sich der ganze Mittelwert: $$\mathbb{E}[t] = \mathbb{E}[y + \epsilon] = y + \underbrace{\mathbb{E}[\epsilon]}_{=0} = y(\mathbf{x},\mathbf{w})$$ **Bild:** Die Glocke des Rauschens sitzt _immer_ auf null. (4.8) nimmt diese Glocke und **schiebt sie an die Stelle $y$** — für jedes $x$ woanders. Form bleibt ($\sigma^2$), nur das Zentrum wandert mit der Vorhersage mit. Genau weil das Rauschen zero-mean ist, sitzt die Masse _symmetrisch_ um $y$ → das macht ML und Least-Squares identisch.

> [!tip] Den Strich richtig lesen — zwei Bedeutungen Im Ausdruck $p(t \mid \mathbf{x},\mathbf{w},\sigma^2) = \mathcal{N}(t \mid y, \sigma^2)$ bedeutet der Strich **zweimal etwas anderes**:
> 
> - **Links** ($p(t\mid\dots)$): Strich = „bedingt auf / gegeben" (Konditionierung).
> - **Rechts** ($\mathcal{N}(t\mid\dots)$): Strich = bloße Konvention „die Größe $t$ ist verteilt mit den Parametern rechts vom Strich".
> 
> Also **nicht** „abhängig von $t$", sondern: die Glocke wird **an der Stelle $t$ ausgewertet**, geformt durch Zentrum $y$ und Breite $\sigma^2$.

> [!example] Geschärfte Lesart von (4.8) Die Wahrscheinlichkeits**dichte** der Zielvariable $t$ — gegeben Input $\mathbf{x}$, Gewichte $\mathbf{w}$ und Varianz $\sigma^2$ — ist eine Normalverteilung, deren **Schwerpunkt** die Regressionsfunktion $y(\mathbf{x},\mathbf{w})$ vorgibt und deren **Breite** $\sigma^2$ ist. Das Modell sagt keinen Punkt vorher, sondern eine **ganze Verteilung möglicher $t$** — Vorhersage als Glocke statt als Punkt.
> 
> _Mikro-Notiz:_ $p$ ist eine **Dichte**, keine Wahrscheinlichkeit. Die Wahrscheinlichkeit, exakt einen Punkt zu treffen, ist null; erst $\int_a^b p(t),dt$ gibt eine echte Wahrscheinlichkeit.

### Das 3D-Bild

Auf drei Achsen $(x, t, p)$ wird (4.8) wörtlich: die Regressionskurve ist der **Kamm** eines Wahrscheinlichkeitsgebirges.

- Die **rote Kurve** läuft über den Grat — weil $\mathcal{N}(t\mid y,\sigma^2)$ bei $t=y$ maximal ist. „Regression" und „Dichte" sind **dasselbe Objekt aus zwei Blickrichtungen**.
- **Vertikale Glocken** = die aufploppenden Verteilungen, alle gleich breit (konstante Varianz = Homoskedastizität).
- **Punkte am Boden** ($p=0$) = echte Ziehungen $t_n \sim \mathcal{N}(y(x_n),\sigma^2)$.
- Die vertraute **2D-Ansicht** „Linie durch Punkte" ist nur der **Schatten** dieses 3D-Objekts auf den Boden.

> [!note] Die Glocke ploppt überall, nicht nur an Datenpunkten Das Modell definiert für **jedes** $x$ eine Vorhersageverteilung — auch an ungesehenen Stellen. Die Trainingspunkte sind nur die Orte, an denen eine _Ziehung_ beobachtet wurde. Genau deshalb kann man später an neuen $x$ vorhersagen.

---

## 3 — Maximum Likelihood → die Fehlerfunktion

Log-Likelihood (4.10), durch Einsetzen der Gauß-Formel und Aufsummieren:

$$ \ln p(\mathbf{t}\mid\mathbf{X},\mathbf{w},\sigma^2) = -\frac{N}{2}\ln\sigma^2 - \frac{N}{2}\ln(2\pi) - \frac{1}{\sigma^2}E_D(\mathbf{w}) $$

> [!warning] Stolperstein: Warum „taucht" $E_D$ auf? $E_D$ ist **kein neuer Term** — nur ein **Name** für das, was vom Gauß-Exponential übrig bleibt, sobald man $\ln$ zieht.
> 
> 1. $\ln$ trennt Vorfaktor und Exponential: $\ln\mathcal{N} = -\tfrac12\ln\sigma^2 - \tfrac12\ln(2\pi) - \tfrac{(t_n - y)^2}{2\sigma^2}$.
> 2. $\sum_n$ über die ersten zwei (von $n$ unabhängigen) Terme gibt das $N$-fache.
> 3. Der Rest ist per **Definition**: $$E_D(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \big(t_n - \mathbf{w}^\top\phi(\mathbf{x}_n)\big)^2 \quad \text{(Sum-of-Squares)}$$
> 
> **Pointe (Vorzeichen!):** Vor $E_D$ steht ein **Minus**. Log-Likelihood _maximieren_ = $E_D$ _minimieren_. → ML unter Gauß-Rauschen **ist** Least-Squares. Andere Rauschannahmen geben andere Fehler (Laplace → L1).

> [!info] Math als Sprache $(t_n - \mathbf{w}^\top\phi(\mathbf{x}_n))^2$ lesen als **quadrierter Abstand** zwischen Wahrheit und Vorhersage. Das $|\cdot|^2$ = Energie/Fehler, vorzeichenlos. $E_D$ summiert diese Fehler-Energie über den Datensatz.

---

## 4 — Die geschlossene Lösung

Gradient (4.12), null gesetzt (4.13), nach $\mathbf{w}$ aufgelöst (4.14):

$$ \mathbf{w}_{ML} = \left(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\top \mathbf{t} $$

### Woher kommt die Design-Matrix $\boldsymbol{\Phi}$?

> [!warning] Stolperstein: Umstellen ≠ Matrix-Entstehung Das sind **zwei getrennte Schritte**, die leicht verschmelzen:
> 
> - **Schritt A (Umstellen):** Den $\mathbf{w}$-Term auf eine Seite bringen, invertieren. Das löst nach $\mathbf{w}$ auf. _Hier ist noch keine Design-Matrix im Spiel_ — nur Summen über äußere Produkte.
> - **Schritt B (Bündeln):** Erkennen, dass eine **Summe von äußeren Produkten = ein Matrixprodukt** ist. _Das_ erzeugt $\boldsymbol{\Phi}$ — rein notationell, kein Rechnen.

> [!tip] Der Merksatz **Ein $\sum_n$ über ein Produkt von Dingen, die mit dem Datenindex $n$ indiziert sind, ist immer ein verstecktes Matrixprodukt.** $$\sum_n \phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^\top = \boldsymbol{\Phi}^\top\boldsymbol{\Phi}, \qquad \sum_n t_n,\phi(\mathbf{x}_n) = \boldsymbol{\Phi}^\top\mathbf{t}$$ Die Summe verschwindet nicht — sie wird von der **Multiplikationsregel verschluckt**. $\boldsymbol{\Phi}$ stapelt die $\phi(\mathbf{x}_n)^\top$ zeilenweise (Zeile = Datenpunkt, Spalte = Basisfunktion).

> [!note] Dimensions-Check (verankert das Verständnis) $\boldsymbol{\Phi}$ ist $N\times M$ ($N$ Daten, $M$ Basisfunktionen). Dann:
> 
> - $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$ → $M\times M$ (Feature-Korrelation, über Daten summiert)
> - $\boldsymbol{\Phi}^\top\mathbf{t}$ → $M$-Vektor
> - $\mathbf{w}_{ML}$ → $M$-dimensional (ein Gewicht pro Basisfunktion)
> 
> Über $n$ (Daten) wird **wegsummiert**; übrig bleibt die $M$-Welt (Features). Das ist die Signatur „Daten verarbeitet, nur ihr Einfluss zählt".

---

## 5 — Die Moore–Penrose-Pseudo-Inverse

$$ \boldsymbol{\Phi}^\dagger \equiv \left(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\top \qquad\Longrightarrow\qquad \mathbf{w}_{ML} = \boldsymbol{\Phi}^\dagger,\mathbf{t} $$

> [!warning] Stolperstein: Das ist KEINE Division $\boldsymbol{\Phi}^\dagger$ ist **nicht** „Gewichte geteilt durch Targets". Die Richtung ist umgekehrt: Sie ist die **Maschine, die Targets in Gewichte übersetzt** ($\mathbf{t} \mapsto \mathbf{w}$).
> 
> - Bei Matrizen gibt es keine Division — nur „mit der Inversen multiplizieren".
> - Modell: $\mathbf{t} \approx \boldsymbol{\Phi},\mathbf{w}$ („Gewichte durch die Design-Matrix → Targets").
> - $\boldsymbol{\Phi}^\dagger$ macht das **rückgängig**: $\mathbf{w} = \boldsymbol{\Phi}^\dagger\mathbf{t}$. Sie kehrt die Vorhersage-Abbildung um.

> [!info] Warum „pseudo"? — die schöne Geometrie $\boldsymbol{\Phi}$ ist $N\times M$, also **nicht quadratisch** ($N \gg M$). Eine echte Inverse existiert nicht. Das System $\mathbf{t} = \boldsymbol{\Phi}\mathbf{w}$ ist **überbestimmt**: $N$ Gleichungen, $M$ Unbekannte. Bei Rauschen gibt es **kein** $\mathbf{w}$, das alle exakt erfüllt — $\mathbf{t}$ liegt nicht im Bildraum von $\boldsymbol{\Phi}$. Was $\boldsymbol{\Phi}^\dagger$ stattdessen tut: liefert das $\mathbf{w}$, das $|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}|^2$ **minimiert**. Geometrisch **projiziert** $\boldsymbol{\Phi}\boldsymbol{\Phi}^\dagger$ den Targetvektor senkrecht auf den von den Basisfunktionen aufgespannten Unterraum → der nächstgelegene Punkt. Das _ist_ Least-Squares in geometrischer Sprache.

> [!example] Spezialfall: echte Inverse als Grenzfall Wäre $\boldsymbol{\Phi}$ quadratisch und invertierbar ($N = M$, exakt bestimmt), dann: $$\boldsymbol{\Phi}^\dagger = (\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top = \boldsymbol{\Phi}^{-1}\boldsymbol{\Phi}^{-\top}\boldsymbol{\Phi}^\top = \boldsymbol{\Phi}^{-1}$$ Die Pseudo-Inverse ist also eine **echte Verallgemeinerung** der Inversen — fällt im Spezialfall darauf zurück, kann aber zusätzlich den überbestimmten Normalfall behandeln.

---

## 6 — Ausblick: frequentistisch vs. bayesianisch

> [!abstract] Die offene Achse Alles bisher liefert **einen** Punktschätzer $\mathbf{w}_{ML}$ — ein einziger Kamm. Das zeigt nur die **aleatorische** Unsicherheit (das Rauschen $\sigma^2$), **nicht** die epistemische („welche Kurve ist überhaupt richtig?").
> 
> - **Frequentistisch:** fixiert ein $\mathbf{w}$, fragt „wie streut meine Schätzung über viele hypothetische Datensätze?"
> - **Bayesianisch:** behält $\mathbf{w}$ als Zufallsgröße, fragt direkt „welche Kurven sind angesichts _dieser_ Daten wie plausibel?" → $p(\mathbf{w}\mid\text{Daten})$.
> 
> Im 3D-Bild: aus dem **einen Kamm** wird ein **Bündel plausibler Grate**, das an datenarmen Stellen ausfranst. → nächster Abschnitt bei Bishop.

---

## Zusammenfassung in einer Zeile

> [!quote] Gauß-Rauschen + Maximum Likelihood **erzwingt** Least-Squares; die Lösung $\mathbf{w}_{ML} = \boldsymbol{\Phi}^\dagger\mathbf{t}$ ist eine **Projektion** des Targetvektors auf den Feature-Unterraum — keine Division, sondern die bestmögliche Umkehrung einer überbestimmten Vorhersage-Maschine.

### Offene To-dos

- [ ] Matrizen-Grundlagen auffrischen (äußeres Produkt, $\boldsymbol{\Phi}^\top\boldsymbol{\Phi}$, Projektionsmatrizen)
- [ ] Bayesianische Erweiterung: Bündel von Kurven aus $p(\mathbf{w}\mid\text{Daten})$ plotten
- [ ] Heteroskedastizität: $\sigma^2 \to \sigma^2(x)$ — relevant für $k_{cat}$-Vorhersage

### Verwandte Notizen

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]