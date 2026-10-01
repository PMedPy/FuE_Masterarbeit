Tags: #Python #FuE Status: #extended

# 🔢 NumPy Quickinfo – Befehlsübersicht und Konzepte

> Diese Datei soll wachsen. Erweitere Tabellen bei Bedarf, ergänze eigene Spickzettel-Beispiele unten.

---

## 🧠 Das Wichtigste in 3 Sätzen

1. **ndarray = homogener Zahlenblock.** Ein `dtype` für alles, beschrieben durch `shape` (Anordnung) und Achsen (nummerierte Richtungen). Homogenität = Geschwindigkeit.
2. **Achsen-Denken:** Bei jeder Reduktion (`sum`, `mean`, …) ist die Frage „entlang welcher Achse?". `axis=k` bedeutet **Achse k verschwindet**.
3. **Vektorisierung statt Schleifen:** Operationen wirken elementweise auf ganze Arrays. `a * 2`, `a + b`, `np.exp(a)` – kein `for` nötig, kompiliert und schnell.

---

## 🛠️ Import & Grundsetup

```python
import numpy as np
```

Konvention: **immer `np`** als Alias.

---

## 🏗️ Arrays erstellen

|Aufgabe|Befehl|
|---|---|
|aus Liste|`np.array([1, 2, 3])`|
|aus verschachtelter Liste (2D)|`np.array([[1, 2], [3, 4]])`|
|Nullen|`np.zeros((3, 4))`|
|Einsen|`np.ones((2, 3))`|
|konstant gefüllt|`np.full((2, 2), 7)`|
|leer (uninitialisiert, schnell)|`np.empty((3, 3))`|
|Einheitsmatrix|`np.eye(3)`|
|Ganzzahl-Bereich|`np.arange(0, 10, 2)` → `[0,2,4,6,8]`|
|N Punkte auf Intervall|`np.linspace(0, 1, 5)` → 5 Werte 0…1|
|wie ein anderes Array|`np.zeros_like(x)`, `np.ones_like(x)`|
|Zufallszahlen (moderne API)|siehe Zufall-Abschnitt|

> ⚠️ **`arange` vs. `linspace`:** `arange` für ganzzahlige Schritte, `linspace` für eine feste **Anzahl** Punkte auf einem kontinuierlichen Intervall. Für Achsen/Plots/Gitter fast immer `linspace` – `arange` mit Fließkommaschritten hat Rundungstücken.

---

## 📐 Die drei beschreibenden Attribute

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])

x.shape     # (2, 3)  – Anordnung: 2 Zeilen, 3 Spalten
x.dtype     # int64   – Typ ALLER Elemente (homogen)
x.ndim      # 2       – Anzahl Achsen
x.size      # 6       – Gesamtzahl Elemente (2×3)
```

> 💡 Attribute, keine Methoden – kein `()`. Sie beschreiben das Array, ohne zu rechnen.

> 💡 **`(n,)` ≠ `(1,n)` ≠ `(n,1)`.** Ein 1D-Vektor, ein Zeilenvektor und ein Spaltenvektor sind technisch verschiedene Objekte mit unterschiedlichem Broadcasting-Verhalten. Das einsame Komma in `(5,)` markiert 1D.

---

## 🔢 dtype (Datentyp)

|dtype|Bedeutung|Nutzung|
|---|---|---|
|`int64`|Ganzzahl|Indizes, Zählungen|
|`float64`|Fließkomma, ~16 Stellen|**PyMC-Default** (exakte Inferenz)|
|`float32`|Fließkomma, ~7 Stellen|**PyTorch-Default** (halber Speicher, GPU-schnell)|
|`bool`|True/False|Masken|

```python
np.array([1, 2, 3], dtype=np.float32)   # explizit setzen
x.astype(np.float64)                     # umwandeln (neue Kopie)
```

> 💡 Eine Fließkommazahl „infiziert" das ganze Array: `np.array([1, 2, 3.0])` → alle `float64`. Wie der `object`-Fall in Pandas, nur strenger: NumPy erzwingt Homogenität.

> ⚠️ **dtype-Mismatch Pandas→PyTorch:** NumPy ist oft `float64`, PyTorch will `float32`. Bei Fehlermeldung: `.astype(np.float32)` oder `tensor.float()`.

---

## 🎯 Indexing & Slicing

```python
x = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])

x[0, 2]        # 12    – eine Zelle: [Zeile, Spalte]
x[1]           # ganze Zeile 1: [20,21,22,23]
x[:, 2]        # ganze Spalte 2: [12,22,32]
x[0:2, 1:3]    # Block: Zeilen 0-1, Spalten 1-2
x[-1]          # letzte Zeile
x[::2]         # jede zweite Zeile
```

Ein einzelner Doppelpunkt `:` heißt „alles auf dieser Achse". Slicing-Ende **exklusiv** (wie bei Listen, anders als `.loc` in Pandas).

### Fancy Indexing – mit Integer-Arrays

```python
idx = np.array([0, 2, 0])
x[idx]                 # holt Zeilen 0, 2, 0 (mit Wiederholung!)

mu = np.array([0.4, 0.55])
codes = np.array([0, 0, 1, 0, 1])
mu[codes]              # → [0.4, 0.4, 0.55, 0.4, 0.55]  ← DAS PyMC-mu[idx]-Muster!
```

> 💡 **Fancy Indexing ist das `mu[genotype_idx]` aus dem PyMC-Modell.** Ein Integer-Array greift die passenden Elemente heraus – dieselbe Bewegung wie `categories[codes]` bei Kategorien.

### Boolean-Masken

```python
maske = x > 20         # Boolean-Array gleicher Shape
x[maske]               # 1D-Array aller Werte > 20
x[x > 20] = 0          # bedingtes Setzen
```

---

## ⚠️ Views vs. Copies ⭐

Der wichtigste Fallstrick von NumPy – der alte Listen-Referenz-Bug, mit schärferen Zähnen.

```python
a = np.arange(10)
b = a[2:5]         # VIEW – teilt den Speicher mit a!
b[0] = 99          # ändert AUCH a[2]!
```

|Operation|Ergebnis|
|---|---|
|**Slicing** `a[2:5]`|**View** (geteilter Speicher)|
|**Fancy Indexing** `a[[0,2,4]]`|**Copy** (eigener Speicher)|
|**Boolean-Maske** `a[a>5]`|**Copy**|
|`.copy()`|erzwungene **Copy**|
|`.reshape()`|meist View|

```python
b = a[2:5].copy()   # sicher: explizite Kopie, a bleibt unberührt
```

> ⚠️ **Slicing gibt einen View, Fancy/Boolean gibt eine Copy.** Wenn du ein Slice änderst, änderst du das Original. Im Zweifel `.copy()`. Prüfen: `b.base is a` (View, wenn `True`).

---

## 🔄 Shape ändern

```python
x = np.arange(12)

x.reshape(3, 4)        # (12,) → (3,4)
x.reshape(3, -1)       # -1 = "berechne selbst" → (3,4)
x.reshape(-1)          # zurück zu 1D (Flatten)
x.ravel()              # 1D-View (wenn möglich)
x.flatten()            # 1D-Copy (immer)

a.T                    # Transponieren (Achsen tauschen)
a.transpose(1, 0, 2)   # Achsen gezielt umordnen (nD)

np.newaxis             # neue Achse einfügen:
v = np.array([1,2,3])  # (3,)
v[:, np.newaxis]       # (3,1) – Spaltenvektor
v[np.newaxis, :]       # (1,3) – Zeilenvektor
```

> 💡 `reshape` verlangt, dass die Gesamtzahl passt (`3*4 = 12`). `-1` lässt NumPy die fehlende Dimension ausrechnen.

---

## 📊 Reduktionen & das Achsen-Prinzip

**`axis=k` = Achse k verschwindet.** Die übrigen Achsen bleiben.

```python
x = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2,3)

x.sum()             # 21      – alles weg, Skalar
x.sum(axis=0)       # [5,7,9] – Achse 0 (Zeilen) weg → ein Wert pro SPALTE, shape (3,)
x.sum(axis=1)       # [6,15]  – Achse 1 (Spalten) weg → ein Wert pro ZEILE, shape (2,)
```

|Aufruf|reduziert|Ergebnis pro|(2,3)→|
|---|---|---|---|
|`x.sum(axis=0)`|Zeilen runter|Spalte|`(3,)`|
|`x.sum(axis=1)`|Spalten rüber|Zeile|`(2,)`|
|`x.sum()`|alles|—|`()`|

> ⚠️ **Denk „welche Achse verschwindet", nicht „welche behalte ich".** `axis=0` klingt nach „erste/Zeilen", eliminiert aber die Zeilen-Achse → Ergebnis pro Spalte.

Reduktionsfunktionen: `sum` `mean` `std` `var` `min` `max` `prod` `median` `argmin` `argmax` `any` `all` `cumsum`.

```python
x.mean(axis=0)              # Spaltenmittel
x.argmax(axis=1)            # Position des Max pro Zeile
np.median(x, axis=0)        # manche nur als np.funktion(x, axis=)
```

### `keepdims` – Achse als Länge-1 behalten

```python
x.mean(axis=1)                 # (2,)   – Achse weg
x.mean(axis=1, keepdims=True)  # (2,1)  – Achse bleibt, Länge 1
```

> 💡 **`keepdims=True`, wenn du das Ergebnis danach gegen das Original verrechnest** (Broadcasting). `x - x.mean(axis=1, keepdims=True)` zentriert jede Zeile; ohne keepdims (`(2,)` gegen `(2,3)`) knallt es.

---

## 📡 Broadcasting

NumPy „streckt" kleinere Arrays auf die Form größerer, ohne Kopien. Regeln (Achsen von **rechts** vergleichen):

1. Achsen gleich lang → passt
2. Eine Achse hat Länge 1 → wird gestreckt
3. Sonst → Fehler

```python
a = np.array([[1,2,3],[4,5,6]])   # (2,3)
a + 10                             # Skalar → auf alle: (2,3)
a + np.array([10,20,30])           # (3,) → auf jede Zeile: (2,3)
a + np.array([[10],[20]])          # (2,1) → auf jede Spalte: (2,3)
```

|Shape A|Shape B|Ergebnis|
|---|---|---|
|`(2,3)`|`()` Skalar|`(2,3)`|
|`(2,3)`|`(3,)`|`(2,3)`|
|`(2,3)`|`(2,1)`|`(2,3)`|
|`(2,3)`|`(2,)`|**Fehler**|

> 💡 Vektorisierung = Broadcasting statt Schleife. `np.exp(-x**2)` wirkt auf jedes Element gleichzeitig.

---

## ➕ Elementweise Operationen & ufuncs

```python
a + b, a - b, a * b, a / b, a ** 2     # elementweise (NICHT Matrixmultiplikation!)
np.exp(a), np.log(a), np.sqrt(a)       # elementweise Funktionen
np.sin(a), np.abs(a), np.round(a, 2)
np.maximum(a, b), np.minimum(a, b)     # elementweises Max/Min zweier Arrays
np.clip(a, 0, 1)                       # Werte auf [0,1] begrenzen
np.where(a > 0, a, 0)                  # bedingt: wo True → a, sonst 0 (wie ReLU)
```

> ⚠️ `a * b` ist **elementweise**, nicht Matrixmultiplikation. Matrix = `@` (siehe unten).

---

## 🧮 Lineare Algebra

### Matrixmultiplikation: `@` (Kontraktion, NICHT elementweise)

```python
a * b        # ELEMENTWEISE (Broadcasting-Welt)
a @ b        # MATRIXMULTIPLIKATION: Zeile × Spalte, aufsummiert
```

$(AB)_{ij} = \sum_k A_{ik}B_{kj}$ — das Skalarprodukt der i-ten Zeile von A mit der j-ten Spalte von B.

**Die Dimensions-Regel (inner/outer):**

$$A: (m, \underbrace{n}_{\text{innen}}) \quad @ \quad B: (\underbrace{n}_{\text{innen}}, p) \quad \Rightarrow \quad (m, p)$$

Die **inneren** Dimensionen müssen übereinstimmen und **verschwinden** (werden kontrahiert/summiert); die **äußeren** überleben.

```python
A = np.ones((3, 5)); B = np.ones((5, 2))
(A @ B).shape        # (3, 2) — die 5en berühren sich, verschwinden
A @ A                # FEHLER: (3,5)@(3,5), innere 5≠3
A.T @ A              # (5,3)@(3,5) → (5,5): 3er-Achse kontrahiert
A @ A.T              # (3,5)@(5,3) → (3,3): 5er-Achse kontrahiert
```

> 💡 **Transposition bringt die zu kontrahierende Achse nach innen.** Denk semantisch: „Welche Achse soll wegsummiert werden?" bestimmt, wo `.T` stehen muss. Bei `X.T @ X` (X ist `(n,d)`) wird die Daten-Achse `n` kontrahiert → `(d,d)` Feature-gegen-Feature.

> 💡 **`X.T @ X` = Gram-Matrix** (Skalarprodukte der Spalten). Auf **zentrierten** Daten ist $\frac1n X^TX$ die Kovarianzmatrix, auf **standardisierten** die Korrelationsmatrix (Einsen auf Diagonale).

### Werkzeuge

```python
A @ B                      # Matrixmultiplikation (bevorzugt)
A.T                        # Transponierte
np.linalg.solve(A, b)      # löst A x = b  ← STATT inv(A) @ b
np.linalg.lstsq(A, b, rcond=None)  # Kleinste-Quadrate direkt (am stabilsten)
np.linalg.inv(A)           # Inverse – nur wenn du sie als Objekt brauchst!
np.linalg.det(A)           # Determinante (0 → singulär)
np.linalg.cond(A)          # KONDITIONSZAHL
np.linalg.norm(v)          # Vektornorm
np.linalg.eig(A)           # Eigenwerte/-vektoren (allgemein)
np.linalg.eigh(A)          # "  für symmetrische Matrizen (schneller, stabil)
np.linalg.svd(A)           # Singulärwertzerlegung (hinter cond & lstsq)
```

### Normalengleichung: drei Wege, wachsende Stabilität

$$\hat\beta = (X^TX)^{-1}X^Ty \quad\text{(löst überbestimmtes } X\beta\approx y \text{ per Kleinste-Quadrate)}$$

```python
np.linalg.inv(X.T @ X) @ X.T @ y     # 1. naiv – zeigt die Formel, instabil
np.linalg.solve(X.T @ X, X.T @ y)     # 2. besser – kein explizites inv
np.linalg.lstsq(X, y, rcond=None)[0]  # 3. am besten – bildet X^T X gar nicht
```

> ⚠️ **Nie `inv(A) @ b` — immer `solve(A, b)`.** `solve` ist genauer (weniger Rundungsfehler) und ~2× schneller. `inv` nur, wenn du die Inverse selbst als Objekt weiterverwendest.

> 💡 **Warum `lstsq` am besten:** Bilden von `X.T @ X` **quadriert** die Konditionszahl ($\kappa(X^TX)=\kappa(X)^2$). `lstsq` umgeht das über QR/SVD und arbeitet mit $\kappa(X)$ statt $\kappa(X)^2$.

### Konditionierung – wie nah am Abgrund

$$\kappa(A) = \frac{\sigma_{max}}{\sigma_{min}} = \frac{\text{längste Halbachse}}{\text{kürzeste Halbachse der Ellipse}}$$

Eine Matrix bildet den Einheitskreis auf eine **Ellipse** ab. Die Singulärwerte sind die Halbachsenlängen. $\kappa$ = ihr Verhältnis.

|$\kappa$|Geometrie|Zustand|
|---|---|---|
|$=1$|Kreis (Orthogonalmatrix)|perfekt konditioniert|
|moderat (<100)|leichte Ellipse|gutartig|
|groß ($10^3$–$10^6$)|lange dünne Zigarre|schlecht konditioniert|
|$\infty$|Ellipse zur Linie kollabiert|**singulär** ($\sigma_{min}=0$)|

> 💡 **Faustregel:** $\kappa \approx 10^k$ → du verlierst ~$k$ Dezimalstellen Genauigkeit. Bei `float64` (~16 Stellen) ist ab $\kappa\approx10^{16}$ alles Rauschen – ohne Fehlermeldung (stiller Unsinn).

> 💡 **Warum Fehler explodieren:** Die Inverse **dehnt die kürzeste Halbachse um $1/\sigma_{min}$**. Ist $\sigma_{min}$ winzig, wird ein kleiner Fehler in $y$ um diesen riesigen Faktor verstärkt: $\frac{|\Delta\beta|}{|\beta|} \le \kappa \cdot \frac{|\Delta y|}{|y|}$. Singulär: $\sigma_{min}=0 \Rightarrow \kappa=\infty \Rightarrow$ keine Inversion.

### Zwei verschiedene Achsen NICHT verwechseln

|Frage|Achse|Begriffe|
|---|---|---|
|Gleichungen vs. Unbekannte?|Form $(m$ vs. $n)$|über-/bestimmt/unterbestimmt|
|Spalten unabhängig?|Rang|voll / **singulär** / schlecht konditioniert|

Regression ist meist **überbestimmt** ($n \gg d$) → Kleinste-Quadrate. Davon **unabhängig**: ob $X^TX$ singulär ist (kollineare Features).

### Die Reparatur-Familie (harter vs. weicher Fall)

|Situation|$\kappa$|Reparatur|
|---|---|---|
|**exakt** kollinear (One-Hot + Intercept)|$\infty$|`drop_first` (Spalte weg)|
|**fast** kollinear (korrelierte Proteine)|groß|Ridge: $(X^TX + \lambda I)^{-1}X^Ty$|

> 💡 `drop_first` und Ridge sind dieselbe Medizin für das harte/weiche Ende. Ridge hebt $\sigma_{min}$ von „fast 0" auf „$\geq\lambda$" → Ellipse wird runder. **Ridge ↔ Gauß-Prior** (kommt in der Bayes-Brücke): $\lambda$ = Prior-Stärke „Koeffizienten sind klein".

---

## 🎲 Zufallszahlen (moderne API)

```python
rng = np.random.default_rng(42)        # Generator mit Seed (reproduzierbar)

rng.random((2, 3))                     # gleichverteilt [0,1)
rng.normal(0, 1, size=100)             # Normalverteilung (mu, sigma, size)
rng.integers(0, 10, size=5)            # Ganzzahlen [0,10)
rng.choice(werte, size=n, replace=False)  # ohne Zurücklegen ziehen
rng.permutation(n)                     # Indizes 0..n-1 mischen
rng.shuffle(a)                         # Array in-place mischen
```

> 💡 **Fester Seed = reproduzierbar.** Für Masterarbeit unverzichtbar. Nutze `default_rng()`, nicht das alte `np.random.seed()` + `np.random.rand()`.

---

## 🔗 Kombinieren & Aufteilen

```python
np.concatenate([a, b], axis=0)     # aneinanderhängen entlang Achse
np.vstack([a, b])                  # vertikal stapeln (untereinander)
np.hstack([a, b])                  # horizontal (nebeneinander)
np.stack([a, b], axis=0)           # NEUE Achse erzeugen
np.split(a, 3, axis=0)             # in 3 Teile teilen
```

---
## Bedingungen elementweise: `np.where`

Ein `if` verlangt einen einzelnen Wahrheitswert. Bei einem Array liefert ein Vergleich aber ein ganzes Boolean-Array, deshalb scheitert der Versuch:

```python
x = np.array([1.0, 2.0, 3.0])
y = (1/x) if x > 0 else np.nan
# ValueError: The truth value of an array with more than one element is ambiguous
```

Das elementweise Pendant zum Ternary-Operator ist `np.where`:

```python
np.where(bedingung, wert_wenn_wahr, wert_wenn_falsch)
```

Alle drei Argumente werden gebroadcastet, das Ergebnis hat die gebroadcastete Shape. Skalare sind erlaubt:

```python
nenner = mu_max - D
S_stern = np.where(nenner > 0, D * K_s / nenner, np.nan)
```

|Aufruf|Ergebnis|
|---|---|
|`np.where(a > 0, a, 0)`|negative Werte auf 0 setzen (ReLU)|
|`np.where(m, A, B)`|elementweise zwischen zwei Arrays wählen|
|`np.where(a > 0, a, np.nan)`|ungültige Bereiche als `nan` markieren|
|`np.where(bedingung)`|**einargumentig**: gibt Tupel von Index-Arrays zurück, nicht Werte|

### `np.where` vs. Boolean-Indexing

Der entscheidende Unterschied — und eine typische Fehlerquelle:

```python
a[maske]                       # SELEKTION: Länge ändert sich, Positionen verschieben sich
np.where(maske, a, np.nan)     # ERSETZUNG: Länge bleibt, Zuordnung bleibt intakt
```

Boolean-Indexing wirft Elemente weg. Steht das Array in Positionsbeziehung zu einem anderen (`D[i]` gehört zu `S_stern[i]`), ist diese Beziehung danach zerstört — ohne Fehlermeldung. `np.where` erhält die Shape und damit die Zuordnung.

**Faustregel:** Solange die Positionen etwas bedeuten, wird ersetzt, nicht selektiert. Selektieren erst am Ende, wenn keine parallelen Arrays mehr im Spiel sind.

### `nan` als Plot-Werkzeug

Matplotlib zeichnet an `nan`-Stellen nichts — die Linie bricht dort ab. Statt gültige Bereiche mühsam herauszuschneiden, maskiert man ungültige mit `nan` und plottet das volle Array:

```python
X_stern = np.where(D_fine < D_krit, Y * (S_in - S_stern), np.nan)
ax.plot(D_fine, X_stern)      # bricht bei D_krit sauber ab
```

> [!warning] ⚡ MEINE SCHWACHSTELLE: `np.where` wertet beide Zweige komplett aus `np.where(nenner > 0, a / nenner, np.nan)` rechnet `a / nenner` für **alle** Elemente, auch für `nenner == 0`, und wirft erst danach die unerwünschten weg. Das Ergebnis stimmt, aber es gibt ein `RuntimeWarning: divide by zero`. Sauber wird es, indem man den Nenner vorher entschärft:
> 
> ```python
> sicher = np.where(nenner > 0, nenner, 1.0)
> S_stern = np.where(nenner > 0, D * K_s / sicher, np.nan)
> ```
> 
> Oder pragmatisch: `with np.errstate(divide="ignore", invalid="ignore"):` drumherum.

### Verwandte Funktionen

|Funktion|Zweck|
|---|---|
|`np.clip(a, lo, hi)`|auf Intervall begrenzen — kürzer als zwei verschachtelte `where`|
|`np.select([c1, c2], [v1, v2], default=)`|mehr als zwei Fälle|
|`np.nan_to_num(a)`|`nan`/`inf` durch Zahlen ersetzen|
|`np.isnan(a)`|`nan` testen — `a == np.nan` ist **immer** `False`|
|`np.nanmean`, `np.nansum`, `np.nanmax`|Reduktionen, die `nan` überspringen|

Wichtig für Reduktionen: `np.mean` auf einem Array mit `nan` liefert `nan`. Wer maskiert hat, muss die `nan`-Varianten benutzen.
---
## 🐼 Brücke zu Pandas & PyTorch

```python
# Pandas → NumPy (so SPÄT wie möglich, direkt vorm Modell)
X = df[features].to_numpy()        # DataFrame → ndarray (Namen/Index fallen weg)
y = df["ziel"].to_numpy()

# NumPy → Pandas (Ergebnisse zurück, nutzt Index-Alignment)
df["vorhersage"] = modell.predict(X)

# NumPy → PyTorch
import torch
t = torch.from_numpy(X.astype(np.float32))   # dtype beachten!
```

> 💡 **Workflow:** Pandas = Vorbereitungslabor (laden, bereinigen, kategorisieren), NumPy/Tensor = Reaktor. `.to_numpy()` erst, wenn die Daten sauber und rein numerisch sind.

---

## 🐍 Brücke zu Vanilla-Python (Listen, dicts, Skalare)

### Liste ↔ Array

```python
arr = np.array([1, 2, 3])          # Liste → Array
arr.tolist()                        # Array → echte Python-Liste (nicht np-Typen!)
np.array([[1,2],[3,4]])             # verschachtelte Liste → 2D-Array
```

> ⚠️ **Nie `np.append` in einer Schleife!** Jedes `np.append` kopiert das ganze Array neu → O(n²). Stattdessen: in einer **Python-Liste sammeln**, danach **einmal** `np.array(...)`.
> 
> ```python
> ergebnisse = []                    # Liste sammeln (schnell)
> for x in werte:
>     ergebnisse.append(f(x))
> ergebnisse = np.array(ergebnisse)  # EINMAL umwandeln
> ```

> 💡 **Wann Liste, wann Array?** Liste: heterogen, wächst dynamisch, kein Rechnen. Array: homogen, feste Größe, vektorisiertes Rechnen. Aufbauen in Liste, rechnen in Array.

### dict ↔ Array

dicts sind in NumPy kein nativer Typ – die Brücke geht über Werte-Listen oder Pandas:

```python
d = {"a": 1.0, "b": 2.0, "c": 3.0}
np.array(list(d.values()))          # nur die Werte → Array
keys = list(d.keys())               # Namen separat behalten (wie coords!)

# dict von Spalten → 2D über Pandas (sauberster Weg):
pd.DataFrame({"x": [1,2], "y": [3,4]}).to_numpy()
```

> 💡 Das `keys` separat halten + `values` als Array ist genau das **`coords`/`codes`-Muster** aus PyMC: Namen und Zahlen getrennt.

### Iterieren – meist ein Antipattern

```python
for x in arr:        # ⚠️ langsam, unpythonisch für Zahlen-Arrays
    ...
```

> ⚠️ **Über ein Array zu iterieren wie über eine Liste ist fast immer falsch.** Erst fragen: Geht das vektorisiert? `arr * 2`, `np.where(arr>0, ...)`, eine ufunc? Iteration ist die letzte Wahl (dann `np.nditer` oder gleich eine Liste).

### Skalar herausholen

```python
m = arr.max()        # ist ein numpy-Skalar (0-d Array), KEIN Python-float
m.item()             # → echter Python-float/int
float(arr.mean())    # ebenso: sauberer Python-Typ
int(arr.argmax())    # Position als echter int
```

> 💡 `.item()` braucht man oft für saubere `print`-Ausgaben, JSON-Serialisierung oder wenn eine Funktion einen echten Python-Typ erwartet (numpy-Skalare machen z.B. beim Speichern manchmal Ärger).

---

## 🎲 Bayes: MLE, MAP & Grid-Approximation

Drei Arten zu schätzen (zunehmende Reichhaltigkeit):

||nutzt Daten|nutzt Prior|Ergebnis|
|---|---|---|---|
|**MLE**|✓|✗|ein Punkt (Gipfel der Likelihood)|
|**MAP**|✓|✓|ein Punkt (Gipfel von Likelihood×Prior)|
|**Posterior**|✓|✓|ganze Verteilung mit Unsicherheit|

$$\hat\theta_{MLE} = \arg\max_\theta p(D\mid\theta) \qquad \hat\theta_{MAP} = \arg\max_\theta p(D\mid\theta),p(\theta)$$

> 💡 **Log-Trick:** Maximieren im log-Raum macht aus Produkten Summen (`log(a·b)=log a+log b`), ändert das Maximum nicht (log ist monoton). Numerisch stabil.

> 💡 **Ridge = MAP mit Gauß-Prior.** Der Strafterm $\lambda|\beta|^2$ ist der log eines Priors $\beta\sim\mathcal N(0,\tau^2)$, mit $\lambda=\sigma^2/\tau^2$. Kleine Prior-Varianz → großes λ → starke Regularisierung. Flacher Prior ($\tau^2\to\infty$) → λ→0 → **MLE = OLS**.

> 💡 **MLE = MAP bei großen Daten:** Die log-Likelihood ist eine Summe über $N$ Datenpunkte (wächst mit $N$), der log-Prior ist **ein** Term (konstant). Bei großem $N$ erdrückt die Likelihood den Prior → Posterior konvergiert zur schmalen Glocke um den MLE (Bernstein–von–Mises).

### Grid-Approximation (Posterior als reines Broadcasting)

Posterior $\propto$ Likelihood × Prior, an jedem Punkt eines Parameter-Gitters ausgewertet:

```python
daten = np.array([0.42, 0.55, 0.48, 0.61, 0.50])
sigma = 0.1
mu_grid = np.linspace(0, 1, 200)

# 1. Likelihood: (200,1)-(1,5) → (200,5) Gitter, dann Daten-Achse reduzieren
diff = mu_grid[:, None] - daten[None, :]      # (200, 5)
log_lik = (-0.5 * (diff/sigma)**2).sum(axis=1)  # (200,) Summe über Daten

# 2. Prior (Gauß um 0.5)
log_prior = -0.5 * ((mu_grid - 0.5)/0.3)**2

# 3. Posterior im log-Raum, stabilisieren, exponenzieren, normalisieren
log_post = log_lik + log_prior
log_post -= log_post.max()                     # numerische Stabilität VOR exp
post = np.exp(log_post); post /= post.sum()    # → Verteilung (Summe 1)

# 4. Die drei Schätzer ablesen
mu_mle  = mu_grid[np.argmax(log_lik)]          # Gipfel der Likelihood
mu_map  = mu_grid[np.argmax(log_post)]         # Gipfel des Posteriors
mu_mean = np.sum(mu_grid * post)               # Erwartungswert (Schwerpunkt)
```

> 💡 Das ist das Broadcasting-Gitter aus dem Reduktions-Abschnitt, gefüllt mit Wahrscheinlichkeit. Die **Breite** von `post` ist die Unsicherheit – das, was MLE/MAP verschweigen. Skaliert nicht über 1–2 Parameter hinaus (Gitter wächst exponentiell) → dafür gibt es MCMC/PyMC.

> ⚠️ **`log_post -= log_post.max()` vor `np.exp`** – sonst `exp(-große Zahl)=0` (Unterlauf). Ändert nach Normalisierung nichts am Ergebnis.

---

## ⚠️ Häufige Stolperfallen

|Problem|Ursache|Lösung|
|---|---|---|
|Original ändert sich unerwartet|Slice ist ein **View**|`.copy()` beim Slicen|
|`a * b` gibt falsches Ergebnis|elementweise statt Matrix|`a @ b` für Matrixprodukt|
|Broadcasting-Fehler `(2,3)` vs `(2,)`|Achse fehlt zum Ausrichten|`keepdims=True` oder `[:, np.newaxis]`|
|`axis` verwirrt|„behalten" statt „verschwinden" gedacht|`axis=k` = Achse k **weg**|
|`inv` schlägt fehl|singuläre Matrix (Kollinearität)|`solve` statt `inv`; Kollinearität prüfen|
|Rundungstücken bei `arange`|Fließkommaschritte|`linspace` mit Anzahl Punkten|
|dtype-Fehler in PyTorch|NumPy `float64` ≠ Torch `float32`|`.astype(np.float32)`|
|Integer-Division überrascht|`int`-Array bleibt `int`|einen Operanden `float` machen|
|`np.ones(3, 5)` → dtype-Fehler|Shape muss **ein Tupel** sein|`np.ones((3, 5))` – doppelte Klammern|
|`np.append` in Schleife lahm|kopiert jedes Mal alles (O(n²))|Liste sammeln, einmal `np.array`|
|`for x in arr` langsam|Iteration statt Vektorisierung|`arr*2`, `np.where`, ufunc|
|numpy-Skalar macht Ärger (JSON/print)|`arr.max()` ist 0-d Array, kein float|`.item()` oder `float(...)`|
|`exp` gibt 0 (Grid/Posterior)|Unterlauf bei `exp(-große Zahl)`|`log_post -= log_post.max()` vor `exp`|
|Plot-x-Achse stimmt nicht|gegen Positionen statt echte Werte geplottet|gegen `lambdas`/echte x-Werte plotten|

---

## ✅ Mini-Cheatsheet

```python
import numpy as np

# Erstellen
a = np.array([[1, 2, 3], [4, 5, 6]])
z = np.zeros((3, 4))
lin = np.linspace(0, 1, 100)

# Inspizieren
a.shape, a.dtype, a.ndim, a.size

# Indexing
a[0, 2]          # Zelle
a[:, 1]          # Spalte
a[a > 3]         # Boolean-Maske (Copy)
a[[0, 1]]        # Fancy (Copy)
a[0:1].copy()    # Slice + explizite Copy (Slice ist View!)

# Reduktionen (axis = welche Achse verschwindet)
a.sum(axis=0)                     # pro Spalte
a.mean(axis=1, keepdims=True)     # pro Zeile, Form (n,1) behalten

# Umformen
a.reshape(3, -1)
a.T
v[:, np.newaxis]                  # (n,) → (n,1)

# Rechnen
a @ b                             # Matrixprodukt
np.linalg.solve(X.T @ X, X.T @ y) # Normalengleichung (stabil)

# Zufall
rng = np.random.default_rng(42)
rng.normal(0, 1, size=(2, 3))

# Brücke
X = df[features].to_numpy()
```

# Referenzen