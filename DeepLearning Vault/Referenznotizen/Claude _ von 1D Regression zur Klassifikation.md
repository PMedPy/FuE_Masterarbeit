05-08-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Von der 1D-Regression zur Klassifikations-Lossfunktion

> [!abstract] Worum es geht Dieselbe Lossfunktion, dreimal in wachsender Allgemeinheit:
> 
> |Stufe|Modell|Loss|Lösung|
> |---|---|---|---|
> |0|$y = wx + b$, Skalare|$\tfrac12\sum_n (y_n - t_n)^2$|$w = \mathrm{Cov}/\mathrm{Var}$|
> |1|$y = \mathbf w^T\mathbf x$, $D$ Features|$\tfrac12\lVert \mathbf X\mathbf w - \mathbf t\rVert^2$|$\mathbf w = (\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf t$|
> |2|$\mathbf y = \mathbf W^T\mathbf x$, $K$ Outputs|$\tfrac12\mathrm{Tr}{(\mathbf X\mathbf W - \mathbf T)^T(\mathbf X\mathbf W - \mathbf T)}$|$\mathbf W = (\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf T$|
> 
> Klassifikation via Least Squares ist **kein neuer Loss** — es ist Stufe 2 mit 1-of-$K$-kodiertem $\mathbf T$. Der gesamte mathematische Aufwand steckt in der Buchhaltung: _welcher Index wird aufsummiert, welcher überlebt._


---

## 1. Werkzeugkasten: Shapes lesen

### 1.1 Die Grundkonvention

Ein nackter Vektor ist **immer eine Spalte**. $\mathbf x \in \mathbb R^D$ hat Shape $D\times 1$, $\mathbf x^T$ hat Shape $1\times D$.

### 1.2 Die Kontraktionsregel

$$(\mathbf A\mathbf B)_{ik} = \sum_j A_{ij}B_{jk}$$

Der Index $j$ steht beim Aneinanderschreiben **innen** und wird aufsummiert. Er verschwindet. Die äußeren Indizes $i$ und $k$ überleben und werden zu den Indizes des Ergebnisses.

> [!important] Der Merksatz **Innen = wird aufsummiert. Außen = überlebt.** Transponieren ist das einzige Werkzeug, um einen Index von außen nach innen zu holen.
> 
> Wenn du dich fragst „warum steht hier ein $T$?", lautet die Antwort immer: _damit der Index, über den summiert werden soll, an der inneren Position steht._

### 1.3 Die Arbeitsweise

Bei jeder Matrixgleichung, die du liest, gehst du drei Schritte:

1. **Shapes annotieren.** Über jedes Symbol die Dimensionen schreiben. Buchstäblich, mit Bleistift.
2. **Kontraktion identifizieren.** Welche Dimension berührt sich? Die wird aufsummiert.
3. **Semantik zuordnen.** Was _bedeutet_ die aufsummierte Dimension — Daten? Features? Klassen?

Schritt 3 ist der, der aus Buchhaltung Verständnis macht. Beispiel: $\mathbf X^T\mathbf X$ kontrahiert den Datenindex $n$, übrig bleiben zwei Feature-Indizes → das Ergebnis _muss_ etwas über die Beziehung von Features untereinander aussagen, gemittelt über die Daten. Und genau das tut es.

### 1.4 Das entscheidende Gegensatzpaar

$$\mathbf a^T\mathbf b = \sum_i a_i b_i \quad (\text{Skalar}), \qquad \mathbf a\mathbf b^T = [a_i b_j]_{ij} \quad (\text{Matrix})$$

Bei $\mathbf a^T\mathbf b$ berühren sich $D$ und $D$ → kontrahiert, alles kollabiert zu einer Zahl. Bei $\mathbf a\mathbf b^T$ berühren sich $1$ und $1$ → nichts wird summiert, es bläht auf zu $D\times D$.

Inneres versus äußeres Produkt. Alles Folgende ist eine Variation davon.

---

## 2. Stufe 0 — Eine Dimension, ein Output

### 2.1 Modell und Loss

Daten: ${(x_n, t_n)}$, $n = 1,\dots,N$, beides Skalare.

$$y_n = w x_n + b$$

$$E(w,b) = \frac12\sum_{n=1}^N (y_n - t_n)^2 = \frac12\sum_{n=1}^N (w x_n + b - t_n)^2$$

Der Faktor $\tfrac12$ ist reine Kosmetik — er kürzt sich beim Ableiten gegen die 2 aus der Kettenregel.

### 2.2 Ableiten

$$\frac{\partial E}{\partial b} = \sum_n (w x_n + b - t_n) \overset{!}{=} 0$$

$$\frac{\partial E}{\partial w} = \sum_n (w x_n + b - t_n), x_n \overset{!}{=} 0$$

> [!note] Was hier passiert Beide Gleichungen haben die Form „**Residuum, gewichtet mit der Ableitung des Modells nach dem Parameter, summiert über die Daten, gleich null**". Beim Bias ist die Ableitung $\partial y_n/\partial b = 1$, beim Gewicht ist sie $\partial y_n/\partial w = x_n$.
> 
> Das ist bereits die Normalengleichung in Rohform. Alles, was in den höheren Stufen kommt, ist dasselbe Muster — nur mit mehr Indizes.

### 2.3 Auflösen

Aus der ersten Gleichung: $b = \bar t - w\bar x$ mit $\bar x = \tfrac1N\sum_n x_n$. Einsetzen in die zweite und umsortieren liefert

$$w = \frac{\sum_n (x_n - \bar x)(t_n - \bar t)}{\sum_n (x_n - \bar x)^2} = \frac{\mathrm{Cov}[x,t]}{\mathrm{Var}[x]}$$

> [!important] Die Lesart, die du behalten sollst **Gewicht = Kreuzkorrelation zwischen Input und Ziel, geteilt durch die Autokorrelation des Inputs.**
> 
> Diese Struktur überlebt jede Verallgemeinerung. In Stufe 1 wird aus dem Bruch eine Matrixinversion — der Zähler wird zu $\mathbf X^T\mathbf t$, der Nenner zu $\mathbf X^T\mathbf X$. Mehr passiert nicht.

### 2.4 Der Bias-Trick

Setze $\tilde{\mathbf x}_n = (1, x_n)^T$ und $\tilde{\mathbf w} = (b, w)^T$. Dann ist

$$y_n = \tilde{\mathbf w}^T\tilde{\mathbf x}_n$$

Der Bias wird zu einem gewöhnlichen Gewicht auf einem Feature, das konstant 1 ist. Ab jetzt lassen wir die Tilde weg und tun so, als wäre der Bias schon in $\mathbf x$ und $\mathbf w$ enthalten. $D$ meint dann implizit $D+1$.

> [!info] FuE-Anker `torch.nn.Linear(D, K)` hält den Bias intern separat (`layer.bias`), rechnet also _nicht_ mit augmentierter Designmatrix. Mathematisch identisch, numerisch etwas sauberer, weil die Spalte aus Einsen die Konditionierung von $\mathbf X^T\mathbf X$ verschlechtern kann, wenn die Features nicht zentriert sind.

---

## 3. Stufe 1 — D Features, ein Output

### 3.1 Das Kontobuch

|Objekt|Shape|Zeile $n$|Spalte $j$|
|---|---|---|---|
|$\mathbf x_n$|$D\times 1$|—|—|
|$\mathbf X$ (Designmatrix)|$N\times D$|$\mathbf x_n^T$|Feature $j$ über alle Daten|
|$\mathbf w$|$D\times 1$|—|—|
|$\mathbf t$|$N\times 1$|—|—|
|$\mathbf X\mathbf w$|$N\times 1$|$y_n$|—|

> [!note] Warum die Designmatrix so orientiert ist Zeilenindex = Daten, Spaltenindex = Features. Dieselbe Konvention wie ein Pandas-DataFrame (Zeilen = Beobachtungen, Spalten = Variablen) und wie ein PyTorch-Batch-Tensor der Form `(batch, features)`.
> 
> Beachte den Transpose in „die $n$-te Zeile ist $\mathbf x_n^T$": $\mathbf x_n$ ist per Konvention eine Spalte, zum Hinlegen als Zeile muss transponiert werden. Reine Typografie, kein Inhalt.

### 3.2 Vom Skalarprodukt zur Matrixmultiplikation

Für einen einzelnen Datenpunkt:

$$y_n = \mathbf w^T\mathbf x_n = \sum_{j=1}^D w_j x_{nj}$$

Shapes: $(1\times D)(D\times 1) \to 1\times 1$. Kontrahiert wird $D$ — der Featureindex. Richtig so: über Features wird summiert, jeder Datenpunkt bleibt für sich.

Jetzt **alle** $N$ Datenpunkte gleichzeitig. Wir wollen einen Vektor der Länge $N$, also muss $N$ außen stehen und $D$ innen:

$$\mathbf y = \mathbf X\mathbf w, \qquad (N\times D)(D\times 1) \to N\times 1 \quad\checkmark$$

$$y_n = (\mathbf X\mathbf w)_n = \sum_{j=1}^D X_{nj}w_j$$

> [!important] Erste Anwendung der Innen/Außen-Regel Warum $\mathbf X\mathbf w$ und nicht $\mathbf w^T\mathbf X$?
> 
> - Gewünscht: Index $n$ überlebt, Index $j$ wird summiert.
> - $n$ muss also außen, $j$ innen.
> - In $\mathbf X$ ($N\times D$) steht $n$ links, $j$ rechts → $\mathbf X$ gehört nach **links**, dann ist $j$ innen. ✓
> - $\mathbf w^T\mathbf X$ wäre $(1\times D)(N\times D)$ — passt nicht einmal.

### 3.3 Der Loss

$$E(\mathbf w) = \frac12\sum_{n=1}^N (y_n - t_n)^2$$

Mit dem Residuenvektor $\mathbf r = \mathbf X\mathbf w - \mathbf t$ (Shape $N\times 1$):

$$\sum_n r_n^2 = \mathbf r^T\mathbf r$$

Hier greift wieder die Kontraktionsregel: $\mathbf r^T\mathbf r$ kontrahiert $N$ — den Datenindex. Genau das wollen wir, denn der Loss ist eine Summe über die Daten.

$$\boxed{;E(\mathbf w) = \frac12(\mathbf X\mathbf w - \mathbf t)^T(\mathbf X\mathbf w - \mathbf t) = \frac12\lVert\mathbf X\mathbf w - \mathbf t\rVert^2;}$$

> [!warning] Die häufigste Verwechslung $\mathbf r^T\mathbf r$ ist ein **Skalar** ($1\times1$), Summe über alle Daten. $\mathbf r\mathbf r^T$ ist eine $N\times N$-**Matrix**, die nichts summiert. Der Loss braucht die erste Form. Bei jedem Quadrat, das du in Matrixform bringen willst, ist die Frage: _Über welchen Index soll summiert werden?_ Der muss innen.

### 3.4 Der Gradient — komponentenweise hergeleitet

Das ist die Stelle, an der der Transpose in $\mathbf X^T$ „geboren" wird. Rechne es einmal per Index, dann siehst du es für immer.

$$\frac{\partial E}{\partial w_j} = \sum_{n=1}^N (y_n - t_n)\frac{\partial y_n}{\partial w_j} = \sum_{n=1}^N r_n X_{nj}$$

Jetzt die Frage: Wie schreibt man $\sum_n r_n X_{nj}$ als Matrixprodukt?

- Summiert wird über $n$ → $n$ muss **innen**.
- Überleben soll $j$ → $j$ muss **außen**.
- In $\mathbf X$ steht $n$ links und $j$ rechts. Falsch herum. Also transponieren: in $\mathbf X^T$ ($D\times N$) steht $j$ links, $n$ rechts.
- $\mathbf X^T\mathbf r$: $(D\times N)(N\times 1) \to D\times 1$. ✓

$$\nabla_{\mathbf w}E = \mathbf X^T(\mathbf X\mathbf w - \mathbf t)$$

> [!important] Die Kernaussage Der Transpose in $\mathbf X^T$ ist **nicht dekorativ**. Er steht dort, weil der Gradient einen Feature-Index trägt, die Summe aber über Daten läuft. Datenindex nach innen, Featureindex nach außen — das erzwingt genau eine Schreibweise.
> 
> Faustregel für Gradienten: _Der Gradient nach einem Parameter hat immer die Shape des Parameters._ $\mathbf w$ ist $D\times1$, also muss $\nabla_{\mathbf w}E$ auch $D\times1$ sein. Das ist ein kostenloser Shape-Check für jede Ableitungsformel.

### 3.5 Zwei Matrizen, die man auseinanderhalten muss

$$\mathbf X^T\mathbf X ;:; (D\times N)(N\times D) \to D\times D, \qquad \mathbf X\mathbf X^T ;:; (N\times D)(D\times N) \to N\times N$$

|                 | $\mathbf X^T\mathbf X$            | $\mathbf X\mathbf X^T$                |
| --------------- | --------------------------------- | ------------------------------------- |
| Shape           | $D\times D$                       | $N\times N$                           |
| Kontrahiert     | Daten $n$                         | Features $j$                          |
| Eintrag $(a,b)$ | $\sum_n X_{na}X_{nb}$             | $\sum_j X_{aj}X_{bj}$                 |
| Bedeutung       | Ähnlichkeit **zwischen Features** | Ähnlichkeit **zwischen Datenpunkten** |
| Name            | Moment-/Streumatrix               | Gram-Matrix                           |
| Taucht auf in   | Normalengleichung                 | Kernel-Methoden, Dual-Formulierung    |

Eine Identität, die viel erklärt:

$$\mathbf X^T\mathbf X = \sum_{n=1}^N \mathbf x_n\mathbf x_n^T$$

Lies das als: _Summe über die Daten von äußeren Produkten._ Jeder Datenpunkt trägt eine $D\times D$-Matrix bei, alle werden aufaddiert. Bei zentrierten Daten ist $\tfrac1N\mathbf X^T\mathbf X$ genau die empirische Kovarianzmatrix der Features.

---

## 4. Stufe 2 — D Features, K Outputs

### 4.1 Der einzige neue Gedanke

Statt eines Zielwerts pro Datenpunkt gibt es jetzt $K$. Das Gewicht wird vom Vektor zur Matrix.

$$\mathbf W = [,\mathbf w_1 \mid \mathbf w_2 \mid \cdots \mid \mathbf w_K,], \qquad \text{Shape } D\times K$$

Die $k$-te **Spalte** ist der Gewichtsvektor für Output $k$. Für einen Datenpunkt:

$$\mathbf y(\mathbf x) = \mathbf W^T\mathbf x, \qquad (K\times D)(D\times 1) \to K\times 1 \quad\checkmark$$

Komponente $k$: Zeile $k$ von $\mathbf W^T$ = Spalte $k$ von $\mathbf W$ = $\mathbf w_k$, also $y_k = \mathbf w_k^T\mathbf x$. Konsistent mit Stufe 1.

### 4.2 Das erweiterte Kontobuch

|Objekt|Shape|Zeile $n$|Spalte $k$|
|---|---|---|---|
|$\mathbf X$|$N\times D$|$\mathbf x_n^T$|Feature $j$ über alle Daten|
|$\mathbf W$|$D\times K$|Gewichte von Feature $j$|$\mathbf w_k$|
|$\mathbf T$|$N\times K$|$\mathbf t_n^T$|Ziel $k$ über alle Daten|
|$\mathbf X\mathbf W$|$N\times K$|$\mathbf y(\mathbf x_n)^T$|Output $k$ über alle Daten|
|$\mathbf R = \mathbf X\mathbf W - \mathbf T$|$N\times K$|Residuen von Punkt $n$|Residuen von Output $k$|

$$(\mathbf X\mathbf W)_{nk} = \sum_{j=1}^D X_{nj}W_{jk} = \mathbf x_n^T\mathbf w_k = y_k(\mathbf x_n)$$

Kontrahiert wird $j$ (Features), überleben $n$ und $k$. In Worten: **$\mathbf X\mathbf W$ ist die vollständige Vorhersagetabelle, alle Daten mal alle Outputs auf einen Schlag.**

> [!info] FuE-Anker Das ist buchstäblich `X @ W` in NumPy und der Forward-Pass eines `nn.Linear`-Layers in PyTorch. Die Batch-Dimension bleibt vorne stehen und wird nicht angefasst — das ist der Grund, warum PyTorch die Konvention `(batch, features)` und Gewichte in der Form `(out_features, in_features)` verwendet: intern rechnet es $\mathbf X\mathbf W^{\text{torch},T}$.

### 4.3 Der Loss — jetzt wird die Spur gebraucht

Gewünscht ist die Summe über **alle** Einträge der Residuenmatrix, quadriert:

$$E_D(\mathbf W) = \frac12\sum_{n=1}^N\sum_{k=1}^K R_{nk}^2$$

Das ist die quadrierte Frobenius-Norm $\tfrac12\lVert\mathbf R\rVert_F^2$. Wie bringt man eine **doppelte** Summe in Matrixform? Ein Matrixprodukt kann nur _einen_ Index kontrahieren. Also braucht man zwei Schritte.

**Schritt 1 — Produkt kontrahiert einen Index.**

$$\mathbf R^T\mathbf R ;:; (K\times N)(N\times K)\to K\times K, \qquad (\mathbf R^T\mathbf R)_{kk'} = \sum_{n=1}^N R_{nk}R_{nk'}$$

Kontrahiert: $n$. Übrig: zwei Outputindizes. Auf der Diagonale ($k = k'$) steht $\sum_n R_{nk}^2$ — die Quadratsumme von Output $k$. Auf der Nebendiagonale stehen Kreuzterme zwischen verschiedenen Outputs, die wir **nicht** wollen.

**Schritt 2 — Die Spur kontrahiert den zweiten Index.**

$$\mathrm{Tr}{\mathbf R^T\mathbf R} = \sum_{k=1}^K(\mathbf R^T\mathbf R)_{kk} = \sum_{k=1}^K\sum_{n=1}^N R_{nk}^2 \quad\checkmark$$

$$\boxed{;E_D(\mathbf W) = \frac12\mathrm{Tr}\Big{(\mathbf X\mathbf W - \mathbf T)^T(\mathbf X\mathbf W - \mathbf T)\Big};}$$

> [!important] Die Rollenverteilung Der **Transpose** sorgt dafür, dass über die Daten summiert wird. Die **Spur** wirft die Output-Kreuzterme weg und summiert die Diagonale. Zusammen: „quadriere alles, summiere alles". Mehr steht in der Formel nicht.

> [!note] Warum nicht $\mathrm{Tr}{\mathbf R\mathbf R^T}$? Ginge auch — wegen $\mathrm{Tr}(\mathbf A\mathbf B) = \mathrm{Tr}(\mathbf B\mathbf A)$ sind beide Spuren **identisch**. Der Unterschied liegt nur in der Zwischenmatrix:
> 
> - $\mathbf R^T\mathbf R$ ist $K\times K$, kontrahiert Daten, Diagonale = Fehler pro Output.
> - $\mathbf R\mathbf R^T$ ist $N\times N$, kontrahiert Outputs, Diagonale = Fehler pro Datenpunkt.
> 
> Beide Diagonalen summieren sich zur gleichen Zahl, aber sie zerlegen den Fehler unterschiedlich. Bishop wählt $\mathbf R^T\mathbf R$, weil es zur Ableitung nach $\mathbf W$ ($D\times K$) besser passt und bei $N \gg K$ die kleinere Matrix ist.

### 4.4 Die entscheidende Beobachtung

$$E_D(\mathbf W) = \frac12\sum_{k=1}^K\underbrace{\lVert\mathbf X\mathbf w_k - \mathbf t_{:,k}\rVert^2}_{\text{Loss aus Stufe 1}}$$

Es gibt **keinen einzigen Term**, der $\mathbf w_k$ und $\mathbf w_{k'}$ koppelt. Das war ja gerade der Zweck der Spur: die Nebendiagonalen wegzuwerfen.

> [!important] Konsequenz Least Squares mit $K$ Outputs sind **$K$ vollständig unabhängige Einzelregressionen**, die sich nur die Designmatrix teilen. Deshalb sieht die Lösung gleich aus wie in Stufe 1, nur mit $\mathbf t \to \mathbf T$.
> 
> Für Regression ist das harmlos. Für Klassifikation ist es genau das Problem — siehe Abschnitt 5.3.

---

## 5. Stufe 3 — Klassifikation

### 5.1 Was sich ändert: nur die Bedeutung von T

1-of-$K$-Kodierung: $\mathbf t_n \in {0,1}^K$ mit genau einer Eins.

$$\mathbf T = \begin{pmatrix} 0 & 1 & 0 \ 1 & 0 & 0 \ 0 & 0 & 1 \ \vdots & & \end{pmatrix} \quad (N\times K)$$

Formal ändert sich **nichts**. Loss, Gradient und Lösung sind identisch zu Stufe 2. Klassifiziert wird über $\arg\max_k y_k(\mathbf x)$.

### 5.2 Die Rechtfertigung: Erwartungswert = Wahrscheinlichkeit

Die Komponente $t_k$ ist eine Bernoulli-Variable. Also:

$$\mathbb E[t_k\mid\mathbf x] = 1\cdot p(t_k=1\mid\mathbf x) + 0\cdot p(t_k=0\mid\mathbf x) = p(\mathcal C_k\mid\mathbf x)$$

$$\mathbb E[\mathbf t\mid\mathbf x] = \big(p(\mathcal C_1\mid\mathbf x),\dots,p(\mathcal C_K\mid\mathbf x)\big)^T$$

Das ist der _Vektor der Posterior-Klassenwahrscheinlichkeiten_. Aus der Entscheidungstheorie weiß man, dass der Minimierer des erwarteten quadratischen Verlusts die bedingte Erwartung ist:

$$y^\star(\mathbf x) = \mathbb E[t\mid\mathbf x]$$

Least Squares zielt also automatisch auf die Posteriors. Die Zielscheibe ist richtig — nur der Pfeil taugt nichts.

### 5.3 Warum es trotzdem scheitert

**Problem 1 — kein zulässiger Wertebereich.** Das Modell ist linear in $\mathbf x$, nichts zwingt die Outputs in $[0,1]$. Kurioserweise gilt mit Bias-Term exakt $\sum_k y_k(\mathbf x) = 1$ für jedes $\mathbf x$ — die Outputs summieren sich zu eins, können aber einzeln negativ oder größer eins sein. Es _sieht aus_ wie eine Verteilung und ist keine.

**Problem 2 — die Entkopplung aus 4.4.** Jede Klasse rechnet für sich, niemand koordiniert. Es gibt keinen Mechanismus, der die Klassen gegeneinander abwägt.

**Problem 3 — Robustheit.** Der quadratische Loss bestraft Punkte, die „zu weit auf der richtigen Seite" liegen. Weit entfernte Punkte einer korrekt klassifizierten Klasse ziehen die Entscheidungsgrenze zu sich. Bei Gauß-verteiltem Rauschen ist das korrekt, bei binären Targets ist es unsinnig.

> [!important] Der konzeptionelle Ausweg **Softmax koppelt die Outputs** über den gemeinsamen Nenner: $$p(\mathcal C_k\mid\mathbf x) = \frac{\exp(a_k)}{\sum_{k'}\exp(a_{k'})}, \qquad a_k = \mathbf w_k^T\mathbf x$$ Damit ist der zulässige Bereich per Konstruktion erzwungen (Problem 1), und die Klassen _müssen_ sich gegeneinander behaupten (Problem 2). Die Cross-Entropy als passender Loss löst Problem 3.
> 
> Der Preis: keine geschlossene Lösung mehr. Ab hier braucht man iterative Optimierung — und damit ist man beim Trainingsloop, also bei PyTorch.

---

## 6. Die Normalengleichung: algebraisch

### 6.1 Weg A — komponentenweise (empfohlen für die erste Herleitung)

$$\frac{\partial E_D}{\partial W_{jk}} = \sum_{n=1}^N R_{nk}\frac{\partial y_k(\mathbf x_n)}{\partial W_{jk}} = \sum_{n=1}^N R_{nk}X_{nj}$$

Innen/außen: summiert wird $n$, überleben sollen $j$ und $k$. In $\mathbf X$ steht $n$ links — falsch herum, also $\mathbf X^T$ nach vorne:

$$\nabla_{\mathbf W}E_D = \mathbf X^T\mathbf R = \mathbf X^T(\mathbf X\mathbf W - \mathbf T)$$

Shape-Check: $(D\times N)(N\times K) \to D\times K$. Das ist die Shape von $\mathbf W$. ✓

Nullsetzen:

$$\boxed{;\mathbf X^T\mathbf X,\mathbf W = \mathbf X^T\mathbf T;}$$

$$\mathbf W = (\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf T = \mathbf X^\dagger\mathbf T$$

Shape-Check: $(D\times D)(D\times K)\to D\times K$. ✓

### 6.2 Weg B — Matrixkalkül mit der Spur

Ausmultiplizieren:

$$2E_D = \mathrm{Tr}{\mathbf W^T\mathbf X^T\mathbf X\mathbf W} - \mathrm{Tr}{\mathbf W^T\mathbf X^T\mathbf T} - \mathrm{Tr}{\mathbf T^T\mathbf X\mathbf W} + \mathrm{Tr}{\mathbf T^T\mathbf T}$$

Die beiden mittleren Terme sind Transponierte voneinander; wegen $\mathrm{Tr}(\mathbf A) = \mathrm{Tr}(\mathbf A^T)$ sind sie gleich:

$$2E_D = \mathrm{Tr}{\mathbf W^T\mathbf A\mathbf W} - 2,\mathrm{Tr}{\mathbf W^T\mathbf B} + \text{const}, \qquad \mathbf A = \mathbf X^T\mathbf X,; \mathbf B = \mathbf X^T\mathbf T$$

Mit den beiden Ableitungsregeln (siehe Werkzeugkasten) und $\mathbf A = \mathbf A^T$:

$$\nabla_{\mathbf W}(2E_D) = 2\mathbf A\mathbf W - 2\mathbf B \overset{!}{=} 0 \quad\Longrightarrow\quad \mathbf X^T\mathbf X\mathbf W = \mathbf X^T\mathbf T$$

Identisch zu Weg A. Da $E_D$ quadratisch mit positiv semidefiniter Hesse-Matrix $\mathbf X^T\mathbf X$ ist, ist der stationäre Punkt das **globale** Minimum.

### 6.3 Die inhaltliche Lesart

$$\underbrace{\mathbf X^T\mathbf X}_{\textstyle \sum_n \mathbf x_n\mathbf x_n^T};\mathbf W = \underbrace{\mathbf X^T\mathbf T}_{\textstyle \sum_n \mathbf x_n\mathbf t_n^T}$$

Links die **Autokorrelation der Features**, rechts die **Kreuzkorrelation zwischen Input und Ziel**. Also:

> Die Gewichte sind die Kreuzkorrelation mit dem Ziel, entzerrt um die gegenseitige Überlappung der Features.

Bei orthonormalen Features wäre $\mathbf X^T\mathbf X = \mathbf I$ und die Gewichte _wären_ schlicht die Korrelationen mit dem Ziel. Die Inverse korrigiert genau die Tatsache, dass reale Features sich überlappen. Und das ist exakt die Struktur aus Stufe 0: $w = \mathrm{Cov}/\mathrm{Var}$.

---

## 7. Die Normalengleichung: geometrisch

Der geometrische Weg liefert dieselbe Gleichung **ohne jede Ableitung** — und erklärt nebenbei den Namen.

### 7.1 Der Spaltenraum

Betrachte eine Spalte $k$ separat (wegen 4.4 dürfen wir das). Gesucht ist $\mathbf w$ mit $\mathbf X\mathbf w \approx \mathbf t$, beides Vektoren in $\mathbb R^N$.

Schreibe das Produkt anders:

$$\mathbf X\mathbf w = w_1\mathbf X_{:,1} + w_2\mathbf X_{:,2} + \cdots + w_D\mathbf X_{:,D}$$

> [!important] Die zentrale Umdeutung $\mathbf X\mathbf w$ ist eine **Linearkombination der Spalten von $\mathbf X$**. Die Gewichte $w_j$ sind die Koeffizienten.
> 
> Egal welches $\mathbf w$ du wählst — das Ergebnis liegt **immer** im Spaltenraum $$\mathcal S = \mathrm{span}{\mathbf X_{:,1},\dots,\mathbf X_{:,D}}\subseteq\mathbb R^N,$$ einem höchstens $D$-dimensionalen Unterraum. Bei $N \gg D$ ist das ein winziger Unterraum eines riesigen Raums.
> 
> Der Zielvektor $\mathbf t$ liegt im Allgemeinen **nicht** in $\mathcal S$. Das ist der ganze Grund, warum es überhaupt einen Fehler gibt: Das Gleichungssystem $\mathbf X\mathbf w = \mathbf t$ ist überbestimmt ($N$ Gleichungen, $D$ Unbekannte) und hat keine exakte Lösung.

### 7.2 Die Projektion

Umformulierte Frage: **Welcher Punkt in $\mathcal S$ liegt $\mathbf t$ am nächsten (in der 2-Norm)?**

Antwort aus der Schulgeometrie: der **Lotfußpunkt**. Und der ist genau dadurch charakterisiert, dass der Verbindungsvektor — das Residuum $\mathbf r = \mathbf X\mathbf w - \mathbf t$ — **senkrecht auf $\mathcal S$** steht.

Warum das das Minimum ist, in einer Zeile: Für jeden anderen Punkt $\mathbf v\in\mathcal S$ gilt mit Pythagoras $$\lVert\mathbf t - \mathbf v\rVert^2 = \lVert\mathbf t - \mathbf p\rVert^2 + \lVert\mathbf p - \mathbf v\rVert^2 \geq \lVert\mathbf t - \mathbf p\rVert^2,$$ wobei $\mathbf p$ der Lotfußpunkt ist. Der zweite Term ist genau dann null, wenn $\mathbf v = \mathbf p$.

### 7.3 Orthogonalität als Matrixgleichung

„Senkrecht auf $\mathcal S$" heißt: senkrecht auf **jeder** Spalte von $\mathbf X$:

$$\mathbf X_{:,j}^T,\mathbf r = 0 \quad \text{für alle } j = 1,\dots,D$$

Das sind $D$ Skalarprodukte. Und alle $D$ gleichzeitig sind genau — Innen/Außen-Regel: der Datenindex $n$ soll kontrahiert werden, der Spaltenindex $j$ überleben, also $\mathbf X$ transponieren —

$$\boxed{;\mathbf X^T(\mathbf X\mathbf W - \mathbf T) = \mathbf 0;}$$

**Das ist die Normalengleichung.** Ohne Ableitung, ohne Spur, ohne Matrixkalkül.

> [!important] Der Name _Normale_ = Senkrechte. Die Gleichung heißt Normalengleichung, weil sie besagt, dass das Residuum **normal (senkrecht) auf dem Spaltenraum** steht. Nichts „Normales" im Sinne von gewöhnlich.

### 7.4 Die Projektionsmatrix

Einsetzen der Lösung liefert den Lotfußpunkt explizit:

$$\hat{\mathbf t} = \mathbf X\mathbf w^\star = \underbrace{\mathbf X(\mathbf X^T\mathbf X)^{-1}\mathbf X^T}_{\textstyle \mathbf P},\mathbf t$$

$\mathbf P$ ist die orthogonale Projektion auf $\mathcal S$ (in der Statistik: _Hat-Matrix_, weil sie $\mathbf t$ den Hut aufsetzt). Ihre beiden charakteristischen Eigenschaften:

$$\mathbf P^2 = \mathbf P \quad(\text{idempotent}), \qquad \mathbf P^T = \mathbf P \quad(\text{symmetrisch})$$

Idempotenz in Worten: _Zweimal projizieren ändert nichts mehr_ — nach dem ersten Mal liegt man ja schon im Unterraum. Ein Projektor hat nur die Eigenwerte 0 und 1, und $\mathrm{Tr}(\mathbf P) = \mathrm{rank}(\mathbf X) = D$ (bei vollem Rang) — die Anzahl der effektiven Parameter. Diese Zahl taucht später bei den effektiven Freiheitsgraden und bei der Bias-Varianz-Zerlegung wieder auf.

### 7.5 Die zwei Bilder nebeneinander

| Parameterraum $\mathbb R^D$ | Datenraum $\mathbb R^N$                        |                                                      |
| --------------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Objekt                      | $E(\mathbf w)$ als Paraboloid über $\mathbf w$ | Vektoren $\mathbf t$, $\hat{\mathbf t}$, $\mathbf r$ |
| Lösung ist                  | Talsohle, $\nabla E = 0$                       | Lotfußpunkt, $\mathbf r\perp\mathcal S$              |
| Werkzeug                    | Ableiten                                       | Pythagoras                                           |
| Krümmung                    | Hesse-Matrix $\mathbf X^T\mathbf X$            | Winkel zwischen Spalten                              |

Beide Bilder beschreiben dasselbe. Das Parameterraum-Bild trägt weiter, wenn keine geschlossene Lösung existiert (Softmax, neuronale Netze): dann rutscht man den Hang per Gradientenabstieg hinunter, statt die Talsohle direkt auszurechnen.

---

## 8. Konditionierung und die Brücke zu Ridge

Die Lösung existiert nur, wenn $\mathbf X^T\mathbf X$ invertierbar ist, also wenn die Spalten von $\mathbf X$ linear unabhängig sind. Bei **stark korrelierten Features** — Proteinexpressionen sind das Paradebeispiel — sind sie es _fast_ nicht:

- Spalten fast parallel → $\mathcal S$ wird von einem „schlechten Gerüst" aufgespannt.
- $\mathbf X^T\mathbf X$ hat sehr kleine Eigenwerte → Konditionszahl $\kappa$ riesig.
- Kleine Änderungen in $\mathbf t$ → große Änderungen in $\mathbf w$.
- Im Parameterraum: das Paraboloid hat ein langgestrecktes, fast flaches Tal.

**Ridge** addiert einen Term auf die Diagonale:

$$\mathbf w = (\mathbf X^T\mathbf X + \lambda\mathbf I)^{-1}\mathbf X^T\mathbf t$$

Jeder Eigenwert wird um $\lambda$ angehoben, die Konditionszahl sinkt, das Tal bekommt Wände. Bayesianisch gelesen: das entspricht exakt einem Gauß-Prior $p(\mathbf w) = \mathcal N(\mathbf 0, \tau^2\mathbf I)$ mit $\lambda = \sigma^2/\tau^2$, und $\mathbf w$ ist dann der MAP-Schätzer.

> [!info] FuE-Anker Das ist `weight_decay` in `torch.optim.Adam(..., weight_decay=λ)`. Derselbe Term, andere Sprache: Regularisierung, Prior, Konditionierung, Weight Decay — vier Namen für eine Diagonalverschiebung.

---

## 9. Selbstcheck

Jeweils erst selbst rechnen, dann aufklappen.

**A) Shapes.** Bestimme die Shapes von $\mathbf W^T\mathbf X^T\mathbf X\mathbf W$, $\mathbf X\mathbf W\mathbf W^T\mathbf X^T$ und $\mathbf T^T\mathbf X\mathbf W$. Welcher Index wird jeweils zuerst kontrahiert?

> [!note]- Lösung A
> 
> - $\mathbf W^T\mathbf X^T\mathbf X\mathbf W$: $(K{\times}D)(D{\times}N)(N{\times}D)(D{\times}K) \to K\times K$. Von innen nach außen: erst $\mathbf X^T\mathbf X$ (kontrahiert $n$), dann links und rechts $j$.
> - $\mathbf X\mathbf W\mathbf W^T\mathbf X^T$: $\to N\times N$. Erst $\mathbf W\mathbf W^T$ (kontrahiert $k$), dann außen $j$.
> - $\mathbf T^T\mathbf X\mathbf W$: $(K{\times}N)(N{\times}D)(D{\times}K) \to K\times K$. Erst $n$, dann $j$.

**B) Der Transpose-Test.** Warum steht in $\nabla_{\mathbf W}E = \mathbf X^T\mathbf R$ der Transpose bei $\mathbf X$ und nicht bei $\mathbf R$? Begründe rein über Innen/Außen.

> [!note]- Lösung B Der Gradient muss die Shape von $\mathbf W$ haben, also $D\times K$. Die Summe im Gradienten läuft über $n$. Also: $n$ innen, $j$ und $k$ außen. In $\mathbf R$ ($N\times K$) steht $n$ bereits links und $k$ rechts — passt für die rechte Position. In $\mathbf X$ ($N\times D$) steht $n$ links, gebraucht wird $n$ rechts → transponieren. $\mathbf R^T\mathbf X$ wäre $K\times D$, also die Transponierte des Gesuchten.

**C) Einzelner Datenpunkt.** Zeige, dass $\mathbf X^T\mathbf R = \sum_n \mathbf x_n\mathbf r_n^T$, wobei $\mathbf r_n$ das Residuum von Punkt $n$ ist ($K\times1$).

> [!note]- Lösung C $(\mathbf X^T\mathbf R)_{jk} = \sum_n X_{nj}R_{nk} = \sum_n (\mathbf x_n)_j(\mathbf r_n)_k = \sum_n(\mathbf x_n\mathbf r_n^T)_{jk}$. Allgemeines Muster: _Ein Produkt der Form $\mathbf A^T\mathbf B$ mit gemeinsamem Zeilenindex ist immer die Summe der äußeren Produkte der jeweiligen Zeilen._

**D) Geometrie.** Was passiert mit $\hat{\mathbf t} = \mathbf P\mathbf t$, wenn $\mathbf t$ bereits in $\mathcal S$ liegt? Und wenn $\mathbf t\perp\mathcal S$?

> [!note]- Lösung D Liegt $\mathbf t$ in $\mathcal S$: $\mathbf P\mathbf t = \mathbf t$, Residuum null, exakte Lösung. Steht $\mathbf t$ senkrecht auf $\mathcal S$: $\mathbf P\mathbf t = \mathbf 0$, alle Gewichte null, das Modell erklärt nichts. Das sind die Eigenwerte 1 und 0 der Projektionsmatrix.

**E) Numerisch.** Erzeuge eine $\mathbf X$ mit zwei fast identischen Spalten, berechne `np.linalg.cond(X.T @ X)` und vergleiche die Lösung über `np.linalg.solve(X.T@X, X.T@t)` mit `np.linalg.lstsq(X, t)`. Dann mit Ridge. (Regel für später: die Normalengleichung _explizit_ zu invertieren ist numerisch die schlechteste Variante — sie quadriert die Konditionszahl. In der Praxis: QR- oder SVD-basiert, also `lstsq`.)

---

## 10. Referenztabellen

### 10.1 Innen/Außen — Entscheidungshilfe

|Ich will summieren über …|… und behalten|Also|Ergebnis-Shape|
|---|---|---|---|
|Features $j$|Daten $n$|$\mathbf X\mathbf w$|$N\times1$|
|Daten $n$|Features $j$|$\mathbf X^T\mathbf r$|$D\times1$|
|Daten $n$|Feature $\times$ Feature|$\mathbf X^T\mathbf X$|$D\times D$|
|Features $j$|Datum $\times$ Datum|$\mathbf X\mathbf X^T$|$N\times N$|
|Daten $n$|Feature $\times$ Output|$\mathbf X^T\mathbf T$|$D\times K$|
|Daten $n$|Output $\times$ Output|$\mathbf R^T\mathbf R$|$K\times K$|
|Daten **und** Outputs|nichts (Skalar)|$\mathrm{Tr}{\mathbf R^T\mathbf R}$|$1\times1$|

### 10.2 Matrixidentitäten

|Identität|Bemerkung|
|---|---|
|$(\mathbf A\mathbf B)^T = \mathbf B^T\mathbf A^T$|Reihenfolge kehrt sich um — häufigster Fehler|
|$(\mathbf A\mathbf B\mathbf C)^T = \mathbf C^T\mathbf B^T\mathbf A^T$|gilt für beliebig viele Faktoren|
|$\mathrm{Tr}(\mathbf A\mathbf B) = \mathrm{Tr}(\mathbf B\mathbf A)$|zyklisch schieben|
|$\mathrm{Tr}(\mathbf A\mathbf B\mathbf C) = \mathrm{Tr}(\mathbf C\mathbf A\mathbf B)$|zyklisch, **nicht** beliebig permutieren|
|$\mathrm{Tr}(\mathbf A) = \mathrm{Tr}(\mathbf A^T)$|Kreuzterme zusammenfassen|
|$\mathbf a^T\mathbf A\mathbf b = \mathbf b^T\mathbf A^T\mathbf a$|Skalare darf man gratis transponieren|
|$\mathbf X^T\mathbf X = \sum_n \mathbf x_n\mathbf x_n^T$|Summe über Daten von äußeren Produkten|
|$\lVert\mathbf A\rVert_F^2 = \mathrm{Tr}(\mathbf A^T\mathbf A) = \mathrm{Tr}(\mathbf A\mathbf A^T)$|Frobenius-Norm|

### 10.3 Matrixableitungen

|Ableitung|Skalares Analogon|
|---|---|
|$\dfrac{\partial}{\partial\mathbf w}(\mathbf w^T\mathbf a) = \mathbf a$|$\frac{d}{dw}(aw) = a$|
|$\dfrac{\partial}{\partial\mathbf w}(\mathbf w^T\mathbf A\mathbf w) = (\mathbf A + \mathbf A^T)\mathbf w$|$\frac{d}{dw}(aw^2) = 2aw$|
|$\dfrac{\partial}{\partial\mathbf W}\mathrm{Tr}{\mathbf W^T\mathbf B} = \mathbf B$|$\frac{d}{dw}(bw) = b$|
|$\dfrac{\partial}{\partial\mathbf W}\mathrm{Tr}{\mathbf W^T\mathbf A\mathbf W} = (\mathbf A + \mathbf A^T)\mathbf W$|$\frac{d}{dw}(aw^2) = 2aw$|

Kostenloser Check: _Die Ableitung nach einem Parameter hat immer die Shape des Parameters._

### 10.4 Die Stufenleiter in einer Tabelle

|            | Stufe 0                     | Stufe 1                                           | Stufe 2                                           |
| ---------- | --------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Input      | $x_n$ (Skalar)              | $\mathbf x_n$ ($D\times1$)                        | $\mathbf x_n$ ($D\times1$)                        |
| Parameter  | $w$ (Skalar)                | $\mathbf w$ ($D\times1$)                          | $\mathbf W$ ($D\times K$)                         |
| Vorhersage | $wx_n$                      | $\mathbf X\mathbf w$ ($N\times1$)                 | $\mathbf X\mathbf W$ ($N\times K$)                |
| Residuum   | $r_n$                       | $\mathbf r$ ($N\times1$)                          | $\mathbf R$ ($N\times K$)                         |
| Loss       | $\tfrac12\sum_n r_n^2$      | $\tfrac12\mathbf r^T\mathbf r$                    | $\tfrac12\mathrm{Tr}{\mathbf R^T\mathbf R}$       |
| Gradient   | $\sum_n r_n x_n$            | $\mathbf X^T\mathbf r$                            | $\mathbf X^T\mathbf R$                            |
| Lösung     | $\mathrm{Cov}/\mathrm{Var}$ | $(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf t$ | $(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf T$ |

Die Zeile „Gradient" ist die Essenz: **Residuum, gewichtet mit dem Input, summiert über die Daten.** Von Stufe 0 bis zum Backpropagation-Algorithmus ändert sich daran nichts — nur die Anzahl der Indizes.

---

_Quellen: Bishop, Deep Learning: Foundations and Concepts, Kapitel 4 (Regression) und 5.1 (Diskriminanzfunktionen, Least Squares für Klassifikation). Matrixidentitäten in Appendix A._
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]