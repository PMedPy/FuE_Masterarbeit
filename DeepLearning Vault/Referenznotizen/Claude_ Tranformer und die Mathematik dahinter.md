
# Transformer: die Mathe, komprimiert

> [!abstract] Leitfrage für jede Formel **Welche Form hat das, und was bedeutet jede Achse?** Zeilen sind fast immer Tokens, Spalten sind Eigenschaften. Wenn du bei einer Größe die Form und die Bedeutung jeder Achse sagen kannst, hast du sie verstanden.


## 0 · Symbole und was sie bedeuten

|Symbol|Name|Bedeutung|Typisch (BERT-base / ESM-2 650M)|
|---|---|---|---|
|$n$|Sequenzlänge|Anzahl Tokens (Wörter, Aminosäuren)|beliebig, max. 512 / 1024|
|$d$|Modellbreite|Zahlen pro Token im Residual Stream|768 / 1280|
|$h$|Köpfe|parallele Attention-Einheiten|12 / 20|
|$d_k$|Kopfbreite|Zahlen pro Token _in einem Kopf_|64 / 64|
|$d_{ff}$|FFN-Breite|Zwischenbreite im MLP, meist $4d$|3072 / 5120|
|$V$|Vokabular|Anzahl verschiedener Tokens|30 522 / 33|
|$L$|Tiefe|Anzahl gestapelter Blöcke|12 / 33|
|$B$|Batch|Sequenzen gleichzeitig (nur im Code)||

> [!important] Die wichtigste Unterscheidung
> 
> ||**Gewichte** (Parameter)|**Aktivierungen** (Werte)|
> |---|---|---|
> |Beispiele|$W^Q, W^K, W^V, W^O, E$|$X, Q, K, V$, Scores, Ausgaben|
> |Herkunft|im Training gelernt|aus der Eingabe berechnet|
> |Pro Satz neu?|nein|ja|
> |Enthält $n$ in der Form?|**nie**|**immer** (als Zeilen)|
> 
> Bild: $W$ ist das **Protokoll**, die Aktivierung ist das **Messergebnis**. Das gleiche Protokoll gilt für 5 oder 500 Proben.

---

## 1 · Grundregeln

**Matrix = Tabelle.** $X \in \mathbb{R}^{n \times d}$: $n$ Zeilen (Tokens) × $d$ Spalten (Eigenschaften). Wie eine Messtabelle: Proben × Messgrößen.

**Matrixmultiplikation:** Die inneren Zahlen müssen gleich sein und verschwinden, die äußeren bleiben.

$$(a \times b),(b \times c) = (a \times c)$$

```latex
(a \times b)\,(b \times c) = (a \times c)
```

- Die innere Dimension verschwindet, weil über sie **aufsummiert** wird (Skalarprodukt).
- $X W$ mit $W \in \mathbb{R}^{b \times c}$ heißt: **jedes Token einzeln** von $b$ Eigenschaften in $c$ neue umrechnen. Jede neue Eigenschaft ist eine gewichtete Summe der alten.
- Die Zeilen (Tokens) überleben jede Multiplikation von rechts.

**Transponieren** $^\top$: Zeilen und Spalten tauschen. Aus $n \times d_k$ wird $d_k \times n$.

**Skalarprodukt** $a \cdot b = \sum_l a_l b_l$: groß, wenn beide in dieselbe Richtung zeigen. Lies es als **Übereinstimmung**.

---

## 2 · Embedding und Position

$$x_i = E[t_i] + p_i, \qquad E \in \mathbb{R}^{V \times d}, \quad X \in \mathbb{R}^{n \times d}$$

```latex
x_i = E[t_i] + p_i, \qquad E \in \mathbb{R}^{V \times d}, \quad X \in \mathbb{R}^{n \times d}
```

|Größe|Form|Bedeutung|
|---|---|---|
|$E$|$V \times d$|Nachschlagetabelle: eine Zeile pro möglichem Token. **Gewicht.**|
|$E[t_i]$|$d$|„Wer bin ich?" Zeile Nr. $t_i$, nur Nachschlagen, keine Rechnung|
|$p_i$|$d$|„Wo stehe ich?" Positionsstempel|
|$+$||Beides wird in denselben Vektor gemischt; es gibt keine reservierten Spalten|

**Warum Position?** Ohne sie ist Attention permutations-äquivariant: $\text{Attn}(PX) = P,\text{Attn}(X)$. Vertauschte Eingabe → vertauschte Ausgabe, sonst nichts. Der Satz wäre ein Wortsalat.

Sinus-Kodierung (Original 2017):

$$p_{i,2k} = \sin!\left(\frac{i}{10000^{2k/d}}\right), \qquad p_{i,2k+1} = \cos!\left(\frac{i}{10000^{2k/d}}\right)$$

```latex
p_{i,2k} = \sin\!\left(\frac{i}{10000^{2k/d}}\right), \qquad p_{i,2k+1} = \cos\!\left(\frac{i}{10000^{2k/d}}\right)
```

Lies das als eine Uhr mit $d/2$ Zeigern: kleines $k$ = schneller Zeiger (unterscheidet Nachbarn), großes $k$ = langsamer Zeiger (unterscheidet Anfang und Ende). Modern ist **RoPE** (u. a. ESM-2): Q und K werden je nach Position _gedreht_, sodass $q_i \cdot k_j$ nur vom Abstand $i-j$ abhängt.

---

## 3 · Attention mit einem Kopf

### Projektionen: drei Sichtweisen auf dasselbe Token

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V, \qquad W^{Q,K,V} \in \mathbb{R}^{d \times d_k}$$

```latex
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V, \qquad W^{Q,K,V} \in \mathbb{R}^{d \times d_k}
```

- **Q** („Query"): _Was suche ich?_
- **K** („Key"): _Wofür bin ich zuständig?_ Das Etikett, an dem andere mich finden.
- **V** („Value"): _Was gebe ich weiter, wenn man mich auswählt?_
- $X$ selbst ist _wer bin ich_.

### Die Formel

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```latex
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

### Dimensions-Durchlauf (Beispiel: $n=5$, $d=8$, $d_k=4$)

|Schritt|Rechnung|Form|Achsen bedeuten|Enthält $n$?|
|---|---|---|---|---|
|Eingabe|$X$|$5 \times 8$|Token × Eigenschaft|ja|
|Protokoll|$W^Q$|$8 \times 4$|alte Eig. × neue Eig.|**nein**|
|Queries|$(5{\times}8)(8{\times}4)$|$5 \times 4$|Token × Suchmerkmal|ja|
|Scores|$QK^\top = (5{\times}4)(4{\times}5)$|$5 \times 5$|**fragendes Token × gefragtes Token**|ja, zweimal|
|Gewichte|softmax zeilenweise|$5 \times 5$|Zeile $i$ = Aufmerksamkeitsbudget von $i$, summiert zu 1|ja|
|Ausgabe|$(5{\times}5)(5{\times}4)$|$5 \times 4$|Token × gemischter Inhalt|ja|

> [!tip] Warum verschwindet $d_k$ in $QK^\top$? Eintrag $(i,j) = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il},k_{jl}$. Über $d_k$ wird aufsummiert, übrig bleibt **eine Zahl pro Token-Paar**. Die Eigenschaftsachse ist weg, es bleibt eine reine Beziehungstabelle.

> [!note] $QK^\top$ ist keine Kovarianz Beides sind Tabellen aus Skalarprodukten, aber:
> 
> 1. Kovarianz vergleicht **Features** ($d \times d$), $QK^\top$ vergleicht **Tokens** ($n \times n$).
> 2. $QK^\top$ ist **nicht symmetrisch**, weil $W^Q \neq W^K$: „es" sucht „Tier", aber nicht umgekehrt. Aufmerksamkeit ist eine Einbahnstraße.
> 3. Nicht zentriert. Besserer Name: Ähnlichkeitsmatrix (verwandt mit der Gram-Matrix).

### Pro Token ausgeschrieben

$$\text{out}_i = \sum_{j=1}^{n} \alpha_{ij}, v_j, \qquad \alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{m=1}^{n} \exp(q_i \cdot k_m / \sqrt{d_k})}$$

```latex
\text{out}_i = \sum_{j=1}^{n} \alpha_{ij}\, v_j, \qquad \alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{m=1}^{n} \exp(q_i \cdot k_m / \sqrt{d_k})}
```

- $\sum_j \alpha_{ij} v_j$: **gewichteter Mittelwert** der Values. Attention ist ein _unscharfes Dictionary_: jeder Key passt ein bisschen.
- $\exp$: macht alles positiv und verstärkt Unterschiede (+1 Score = Faktor 2,7).
- Durch die Summe teilen: Anteil am Ganzen, also Prozente.
- softmax = weiches Maximum. Differenzierbar, deshalb lernbar.

### Warum $\sqrt{d_k}$?

Sind die Einträge von $q$ und $k$ unabhängig mit Mittelwert 0 und Varianz 1:

$$\text{Var}(q \cdot k) = \sum_{l=1}^{d_k} \text{Var}(q_l k_l) = d_k \quad\Rightarrow\quad \text{Std}(q\cdot k) = \sqrt{d_k}$$

```latex
\text{Var}(q \cdot k) = \sum_{l=1}^{d_k} \text{Var}(q_l k_l) = d_k \quad\Rightarrow\quad \text{Std}(q\cdot k) = \sqrt{d_k}
```

Lies es als: Eine Summe aus $d_k$ Zufallstermen streut mit $\sqrt{d_k}$. Bei $d_k = 64$ liegen die Scores bei ±8. Softmax wird dann fast hart und die Gradienten der Verlierer sterben. Teilen durch $\sqrt{d_k}$ = **Temperatur zurück auf 1**. Dieselbe Rolle spielt die Länge von $q$: eine lange Query ist entschieden, eine kurze unentschlossen.

---

## 4 · Multi-Head Attention

$$\text{head}_r = \text{Attention}(XW^Q_r,, XW^K_r,, XW^V_r), \qquad r = 1,\dots,h$$

```latex
\text{head}_r = \text{Attention}(XW^Q_r,\, XW^K_r,\, XW^V_r), \qquad r = 1,\dots,h
```

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h), W^O, \qquad W^O \in \mathbb{R}^{(h, d_k) \times d}$$

```latex
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O, \qquad W^O \in \mathbb{R}^{(h\, d_k) \times d}
```

|Größe|Form ($n=5, d=8, h=2, d_k=4$)|Bedeutung|
|---|---|---|
|$W^Q_r$|$8 \times 4$|Protokoll von Kopf $r$: jeder Kopf fragt nach etwas anderem|
|$\text{head}_r$|$5 \times 4$|Ergebnis von Kopf $r$|
|Concat|$5 \times 8$|Köpfe **nebeneinander**: Spalten 1–4 = Kopf 1, 5–8 = Kopf 2|
|$W^O$|$8 \times 8$|**mischt die Köpfe**: jede Ausgabespalte liest aus allen Köpfen|
|MHA|$5 \times 8$|wieder dieselbe Form wie $X$ → Residual möglich|

> [!important] Die einzige harte Bedingung Die Ausgabe muss $n \times d$ sein, damit $x + \text{MHA}(x)$ addierbar ist. $d_k = d/h$ ist nur eine **Konvention** (dann ist $h,d_k = d$, $W^O$ quadratisch, die Kosten sind unabhängig von $h$). Wählt man $d_k$ anders, muss $W^O$ eben $(h,d_k) \times d$ sein.

**Bild:** $h$ Ausschüsse tagen parallel (Grammatik, Bezugswörter, Nachbarschaft …). $W^O$ ist die Sitzung, in der die Protokolle zusammengeführt werden.

**Im Code** stehen alle $W^Q_r$ nebeneinander in einer $d \times d$-Matrix. Danach wird umgeformt: $(n, d) \to (h, n, d_k)$. Das ist dieselbe Rechnung, nur schneller.

---

## 5 · Kausale Maske (Decoder)

$$\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \qquad M_{ij} = \begin{cases} 0 & j \le i \ -\infty & j > i \end{cases}$$

```latex
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
```

- $M$ ist $n \times n$, genau wie die Scores. Zeile = wer fragt, Spalte = wer gefragt wird.
- $-\infty$, weil $e^{-\infty} = 0$. Eine 0 würde $e^0 = 1$ ergeben, also Gewicht.
- Oberes Dreieck gesperrt, Diagonale frei (jedes Token darf sich selbst sehen).
- Encoder (BERT, ESM-2): keine Maske, jedes Token sieht alle.

---

## 6 · Der Block: Sublayer, Residual, Norm, FFN

**Sublayer** = einer der zwei Teile eines Blocks: (1) MHA = _reden_, (2) FFN = _denken_. In Formeln ist „Sublayer" ein Platzhalter für beide.

### Residual

$$x \leftarrow x + \text{Sublayer}(x)$$

```latex
x \leftarrow x + \text{Sublayer}(x)
```

- Lies das $+$ als: **Behalten und ergänzen**, nicht ersetzen. Wie Randnotizen im Laborbuch.
- Setzt voraus, dass $\text{Sublayer}(x)$ dieselbe Form $n \times d$ hat.
- Beim Trainingsstart sind die Gewichte klein und zufällig, der Sublayer liefert also Kleinkram. Mit Residual gilt $x + \text{klein} \approx x$: Der Block startet als „fast nichts tun". Ohne Residual würde die Eingabe durch Zufall ersetzt.
    - Beispiel: $x = (2;,0;,1)$, Sublayer $= (0{,}03;,-0{,}02;,0{,}01)$ → mit Residual $(2{,}03;,-0{,}02;,1{,}01)$, ohne $(0{,}03;,-0{,}02;,0{,}01)$.
- Gradienten fließen über das $+$ ungedämpft zurück, deshalb lassen sich tiefe Netze trainieren.

### LayerNorm (pro Token)

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta, \qquad \mu, \sigma \text{ über die } d \text{ Einträge eines Tokens}$$

```latex
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta, \qquad \mu, \sigma \text{ über die } d \text{ Einträge eines Tokens}
```

Lies das als: jedes Token auf Standardmaß bringen, dann gelernt skalieren ($\gamma \in \mathbb{R}^d$) und verschieben ($\beta \in \mathbb{R}^d$). $\odot$ heißt Eintrag für Eintrag.

### FFN (pro Token)

$$\text{FFN}(x) = W_2,\sigma(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{d_{ff} \times d},; W_2 \in \mathbb{R}^{d \times d_{ff}}$$

```latex
\text{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{d_{ff} \times d},\; W_2 \in \mathbb{R}^{d \times d_{ff}}
```

Formen: $d \to 4d \to d$. Erst aufblähen, dann nichtlinear filtern ($\sigma$ = GELU), dann zurück. Die Tokens reden hier **nicht** miteinander.

### Zusammengesetzt (Pre-LN)

$$\begin{aligned} h &= x + \text{MHA}(\text{LN}(x)) \ y &= h + \text{FFN}(\text{LN}(h)) \end{aligned}$$

```latex
\begin{aligned} h &= x + \text{MHA}(\text{LN}(x)) \\ y &= h + \text{FFN}(\text{LN}(h)) \end{aligned}
```

Rein $n \times d$, raus $n \times d$ → beliebig stapelbar. Das Residual umschließt den **ganzen** Sublayer, inklusive $W^O$.

---

## 7 · Ausgabe und Loss

$$\text{logits} = \text{LN}(Y),W_U, \qquad W_U \in \mathbb{R}^{d \times V}, \qquad \text{logits} \in \mathbb{R}^{n \times V}$$

```latex
\text{logits} = \text{LN}(Y)\,W_U, \qquad W_U \in \mathbb{R}^{d \times V}, \qquad \text{logits} \in \mathbb{R}^{n \times V}
```

Achsen: Position × mögliches Token. Softmax über $V$ ergibt pro Position eine Wahrscheinlichkeitsverteilung über das Vokabular.

|Familie|Maske|Loss|
|---|---|---|
|Decoder (GPT)|kausal|$\mathcal{L} = -\sum_t \log p_\theta(x_t \mid x_{<t})$|
|Encoder (BERT, ESM-2)|keine|$\mathcal{L} = -\sum_{t \in \mathcal{M}} \log p_\theta(x_t \mid x_{\setminus \mathcal{M}})$|
|Encoder-Decoder (T5)|Decoder kausal + Cross-Attn|wie Decoder, bedingt auf Encoder|

- $\log$: macht aus einem Produkt von Wahrscheinlichkeiten eine Summe und bestraft selbstsichere Fehler hart ($\log 0{,}01 \approx -4{,}6$).
- $x_{<t}$ vs. $x_{\setminus\mathcal{M}}$: nur die Vergangenheit oder alles außer den Lücken. Das ist der ganze Unterschied.
- **Cross-Attention:** $Q$ aus dem Decoder ($n_{dec} \times d_k$), $K, V$ aus dem Encoder ($n_{enc} \times d_k$) → Scores $n_{dec} \times n_{enc}$. Rechteckig, weil zwei verschiedene Sequenzen.

---

## 8 · Parameter und Kosten

> [!question]- Parameter pro Block (erst nach Aufgabe 5 aufklappen) Mit $h,d_k = d$:
> 
> - $W^Q, W^K, W^V$: je $h$ Köpfe × $(d \times d_k)$ = $d^2$ → zusammen $3d^2$
> - $W^O$: $d \times d$ = $d^2$
> - FFN: $d \cdot 4d + 4d \cdot d = 8d^2$
> 
> $$\text{Parameter pro Block} \approx 12,d^2$$
> 
> ```latex
> \text{Parameter pro Block} \approx 12\,d^2
> ```
> 
> Unabhängig von $n$, denn Gewichte enthalten nie $n$. Beispiel GPT-2 small: $12 \cdot 12 \cdot 768^2 + 50,257 \cdot 768 \approx 85,\text{M} + 39,\text{M} \approx 124,\text{M}$.

**Rechenaufwand pro Schicht:**

$$\underbrace{\mathcal{O}(n^2 d)}_{\text{Scores und Mischen}} + \underbrace{\mathcal{O}(n, d^2)}_{\text{Projektionen und FFN}}$$

```latex
\underbrace{\mathcal{O}(n^2 d)}_{\text{Scores und Mischen}} + \underbrace{\mathcal{O}(n\, d^2)}_{\text{Projektionen und FFN}}
```

Doppelte Länge → Attention-Matrix × 4, FFN × 2. Beispiel $n$: 5 → 20 heißt $Q$ × 4, aber $QK^\top$ × 16.

---

## 9 · Dimensionsfluss auf einen Blick

```text
Token-IDs            (n,)          welche Tokens
  │ E[·] + p         E: (V, d)
X                    (n, d)        Token × Eigenschaft          ◄── Residual Stream
  │ LN
  │ ×W_r^Q,W_r^K,W_r^V  (d, d_k)   pro Kopf r = 1..h
Q_r, K_r, V_r        (n, d_k)      Token × Such-/Etikett-/Inhaltsmerkmal
  │ Q_r K_rᵀ / √d_k (+M)
S_r                  (n, n)        fragendes × gefragtes Token
  │ softmax (Zeile)
A_r                  (n, n)        Aufmerksamkeitsbudget pro Zeile = 1
  │ × V_r
head_r               (n, d_k)      Token × gemischter Inhalt
  │ Concat über r
                     (n, h·d_k)
  │ × W^O            (h·d_k, d)
MHA                  (n, d)   ──►  + X  (Residual)
  │ LN → FFN: (n,d)→(n,4d)→(n,d) ──►  + (Residual)
Y                    (n, d)        ×L Blöcke
  │ LN, × W_U        (d, V)
logits               (n, V)        Position × mögliches Token
```

---

## 10 · PyTorch: Mini-GPT

Getestet (PyTorch 2.x, CPU). Lernt eine Peptidsequenz auswendig und generiert sie weiter. In den Shape-Kommentaren steht $B$ für Batch.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# B = Batch, n = Tokens, d = Modellbreite, h = Köpfe, d_k = d/h, V = Vokabular


class MultiHeadAttention(nn.Module):
    def __init__(self, d, h, causal=True):
        super().__init__()
        assert d % h == 0, "Konvention d_k = d/h braucht einen Teiler"
        self.h, self.d_k, self.causal = h, d // h, causal
        # Alle Köpfe auf einmal: W^Q_1..W^Q_h nebeneinander = eine d x d Matrix.
        self.W_q = nn.Linear(d, d, bias=False)   # Gewicht: d x (h*d_k)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)   # Gewicht: (h*d_k) x d

    def forward(self, x):                        # x: (B, n, d)
        B, n, d = x.shape

        def split(t):                            # (B, n, d) -> (B, h, n, d_k)
            return t.view(B, n, self.h, self.d_k).transpose(1, 2)

        Q, K, V = split(self.W_q(x)), split(self.W_k(x)), split(self.W_v(x))
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, h, n, n)  wer-auf-wen
        if self.causal:
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
            S = S.masked_fill(mask, float("-inf"))           # Zukunft sperren
        A = S.softmax(dim=-1)                                # jede Zeile summiert zu 1
        out = A @ V                                          # (B, h, n, d_k)
        out = out.transpose(1, 2).reshape(B, n, d)           # Concat: (B, n, h*d_k)
        return self.W_o(out)                                 # (B, n, d)


class Block(nn.Module):
    def __init__(self, d, h, causal=True):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, causal)          # Sublayer 1: reden
        self.ffn = nn.Sequential(                            # Sublayer 2: denken
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):                        # (B, n, d)
        x = x + self.attn(self.ln1(x))           # Residual: dazuaddieren
        x = x + self.ffn(self.ln2(x))
        return x                                 # (B, n, d): gleiche Form -> stapelbar


class MiniGPT(nn.Module):
    def __init__(self, V, d=64, h=4, L=2, n_max=128):
        super().__init__()
        self.tok = nn.Embedding(V, d)            # E: V x d   ("wer bin ich")
        self.pos = nn.Embedding(n_max, d)        # P: n_max x d ("wo stehe ich")
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)  # W_U: d x V (Unembedding)

    def forward(self, idx, targets=None):        # idx: (B, n) Token-IDs
        B, n = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(n, device=idx.device))  # (B, n, d)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))         # (B, n, V): pro Position eine Verteilung
        loss = None
        if targets is not None:                  # nächstes Token vorhersagen
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, steps):
        for _ in range(steps):
            logits, _ = self(idx)
            p = logits[:, -1].softmax(-1)        # nur die letzte Position zählt
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    stoi = {c: i for i, c in enumerate(aa)}
    text = "GIGKFLHSAKKFGKAFVGEIMNS" * 40         # Magainin-2 als Wiederholung
    data = torch.tensor([stoi[c] for c in text])

    model = MiniGPT(V=len(aa))
    print("Parameter:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    n = 32
    for step in range(300):
        i = torch.randint(0, len(data) - n - 1, (16,))
        xb = torch.stack([data[j:j + n] for j in i])          # (16, 32)
        yb = torch.stack([data[j + 1:j + n + 1] for j in i])  # um 1 verschoben
        _, loss = model(xb, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")
    out = model.generate(torch.tensor([[stoi["G"]]]), 22)
    print("".join(aa[i] for i in out[0]))
```

**Zum Nachvollziehen:** Setz nach jeder Zeile in `forward` ein `print(x.shape)` und vergleiche mit Abschnitt 9. Der Loss startet bei ≈ 3,0 ≈ $\ln 20$: Das ist Raten bei 20 gleich wahrscheinlichen Aminosäuren.

---

## 11 · Bonus: Diffusion mit Transformer-Denoiser

> [!abstract] Einordnung **Diffusion** = Framework (Was wird gelernt? Rauschen vorhersagen). **Transformer** = Architektur des Denoisers $\varepsilon_\theta$. Die beiden Achsen sind unabhängig, hier wird beides kombiniert (Idee von DiT; bei Proteinen z. B. Diffusion im Embedding-Raum eines Protein-Sprachmodells).

### Vorwärtsprozess: Rauschen hinzufügen (fest, nichts gelernt)

$$q(x_t \mid x_0) = \mathcal{N}!\big(\sqrt{\bar\alpha_t},x_0,; (1-\bar\alpha_t),I\big) \quad\Leftrightarrow\quad x_t = \sqrt{\bar\alpha_t},x_0 + \sqrt{1-\bar\alpha_t},\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

```latex
q(x_t \mid x_0) = \mathcal{N}\!\big(\sqrt{\bar\alpha_t}\,x_0,\; (1-\bar\alpha_t)\,I\big) \quad\Leftrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
```

|Symbol|Bedeutung|
|---|---|
|$\beta_t$|Rauschmenge in Schritt $t$ (Rauschplan, z. B. linear $10^{-4} \to 0{,}02$)|
|$\alpha_t = 1-\beta_t$|Signalanteil, der Schritt $t$ überlebt|
|$\bar\alpha_t = \prod_{s \le t} \alpha_s$|Signalanteil nach $t$ Schritten; Produkt, weil sich Anteile multiplizieren|
|$\sqrt{\bar\alpha_t},x_0 + \sqrt{1-\bar\alpha_t},\varepsilon$|**Überblendung** Signal ↔ Rauschen; Varianz bleibt ≈ 1, weil $\bar\alpha + (1-\bar\alpha) = 1$|

### Training: Welches Rauschen steckt drin?

$$\mathcal{L} = \mathbb{E}_{x_0,,t,,\varepsilon}\Big[\big|\varepsilon - \varepsilon_\theta(x_t, t)\big|^2\Big]$$

```latex
\mathcal{L} = \mathbb{E}_{x_0,\,t,\,\varepsilon}\Big[\big\|\varepsilon - \varepsilon_\theta(x_t, t)\big\|^2\Big]
```

Lies das als: Nimm ein echtes Beispiel, würfle einen Zeitpunkt und ein Rauschen, verrausche. Der Denoiser soll das Rauschen zurückraten. $|\cdot|^2$ = quadratischer Fehler (Energie der Abweichung), $\mathbb{E}$ = Mittel über alle Würfe.

### Dimensionen: was sich gegenüber GPT ändert

|Mini-GPT|Transformer-Denoiser|
|---|---|---|
|Eingabe|Token-IDs $(n,)$ → Embedding $(n, d)$|kontinuierlich $x_t \in \mathbb{R}^{n \times d}$, z. B. PLM-Embedding pro Aminosäure|
|Zusatzeingabe|–|Zeitschritt $t$ → $c \in \mathbb{R}^{d}$ (Sinus-Embedding + MLP)|
|Maske|kausal|**keine**: alle Positionen werden gemeinsam entrauscht|
|Ausgabe|logits $(n, V)$|$\hat\varepsilon \in \mathbb{R}^{n \times d}$, **gleiche Form wie $x_t$**|
|Loss|Cross-Entropy|MSE|

### Konditionierung auf $t$: adaLN

$$\text{adaLN}(x, c) = \big(1 + \gamma(c)\big) \odot \frac{x - \mu}{\sigma} + \beta(c), \qquad \gamma(c), \beta(c) \in \mathbb{R}^{d}$$

```latex
\text{adaLN}(x, c) = \big(1 + \gamma(c)\big) \odot \frac{x - \mu}{\sigma} + \beta(c), \qquad \gamma(c), \beta(c) \in \mathbb{R}^{d}
```

Wie LayerNorm, nur werden $\gamma, \beta$ nicht fest gelernt, sondern **aus dem Zeitschritt berechnet**. Der Zeitschritt dreht also an den Reglern jedes Blocks: bei viel Rauschen grob, bei wenig Rauschen fein. DiT initialisiert zusätzlich ein Gate $a(c)$ mit 0, dann gilt $x + 0 \cdot \text{Sublayer} = x$. Der Block startet exakt als Identität, dasselbe Residual-Prinzip wie in Abschnitt 6.

### Sampling: Rauschen schrittweise entfernen (DDPM)

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}},\varepsilon_\theta(x_t, t)\right) + \sqrt{\beta_t},z, \qquad z \sim \mathcal{N}(0,I)$$

```latex
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right) + \sqrt{\beta_t}\,z, \qquad z \sim \mathcal{N}(0,I)
```

Lies das als: Ziehe das geschätzte Rauschen ab (passend skaliert), verstärke das übrige Signal ($1/\sqrt{\alpha_t}$) und würfle etwas frisches Rauschen dazu, damit Vielfalt entsteht. Start bei $x_T \sim \mathcal{N}(0, I)$, $T$ Schritte zurück.

### Code: Mini-DiT (getestet)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, d):                    # t: (B,) -> (B, d)
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class DiTBlock(nn.Module):
    """Transformer-Block, dessen LayerNorm vom Zeitschritt gesteuert wird (adaLN)."""
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)   # keine kausale Maske
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)           # aus c: je gamma, beta, gate für beide Sublayer
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)  # Start = Identität

    def forward(self, x, c):                     # x: (B, n, d), c: (B, d)
        g1, b1, a1, g2, b2, a2 = self.ada(c)[:, None].chunk(6, dim=-1)  # je (B, 1, d)
        y = self.ln1(x) * (1 + g1) + b1
        x = x + a1 * self.attn(y, y, y, need_weights=False)[0]
        y = self.ln2(x) * (1 + g2) + b2
        return x + a2 * self.ffn(y)              # (B, n, d)


class Denoiser(nn.Module):
    def __init__(self, d=64, h=4, L=3, n_max=64):
        super().__init__()
        self.d = d
        self.pos = nn.Parameter(torch.zeros(1, n_max, d))
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([DiTBlock(d, h) for _ in range(L)])
        self.out = nn.Linear(d, d)               # sagt eps voraus: gleiche Form wie x_t

    def forward(self, x_t, t):                   # x_t: (B, n, d), t: (B,)
        c = self.t_mlp(timestep_embedding(t, self.d))       # (B, d)
        x = x_t + self.pos[:, :x_t.size(1)]
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(x)                       # eps_hat: (B, n, d)


T = 200
beta = torch.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, 0)              # Signalanteil nach t Schritten


def train_step(model, opt, x0):
    B = x0.size(0)
    t = torch.randint(0, T, (B,))
    eps = torch.randn_like(x0)                   # (B, n, d)
    ab = alpha_bar[t][:, None, None]             # (B, 1, 1) -> broadcast
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps # Vorwärtsprozess in einem Schritt
    loss = F.mse_loss(model(x_t, t), eps)        # "welches Rauschen steckt drin?"
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def sample(model, shape):
    x = torch.randn(shape)                       # Start: reines Rauschen
    for t in reversed(range(T)):
        tt = torch.full((shape[0],), t)
        eps_hat = model(x, tt)
        x = (x - beta[t] / (1 - alpha_bar[t]).sqrt() * eps_hat) / alpha[t].sqrt()
        if t > 0:
            x = x + beta[t].sqrt() * torch.randn_like(x)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 16, 64
    def fake_x0(B):                              # glatte Spielzeug-"Embeddings"
        ph = torch.rand(B, 1, 1) * 6.28
        pos = torch.arange(n)[None, :, None] / n
        ch = torch.arange(d)[None, None, :] / d
        return torch.sin(6.28 * pos + ph + 3 * ch)
    model = Denoiser(d=d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for s in range(300):
        l = train_step(model, opt, fake_x0(32))
        if s % 100 == 0:
            print(f"step {s:3d}  loss {l:.3f}")
    print("Sample shape:", tuple(sample(model, (2, n, d)).shape))
```

### Die Stellschrauben, an denen man später tiefer gräbt

|Kern|Frage|Stichworte|
|---|---|---|
|**Rauschplan**|Wie schnell verschwindet das Signal?|linear, cosine, SNR $= \bar\alpha_t / (1-\bar\alpha_t)$|
|**Parametrisierung**|Was sagt das Netz voraus?|$\varepsilon$, $x_0$ oder $v$ (mathematisch ineinander umrechenbar, trainieren verschieden)|
|**Konditionierung**|Wie kommen $t$ und Zusatzinfos hinein?|adaLN, Cross-Attention (Text/Eigenschaften), Classifier-free Guidance|
|**Sampler**|Wie viele Schritte, wie viel Zufall?|DDPM (stochastisch), DDIM (deterministisch, weniger Schritte), ODE-Solver|
|**Raum**|Worin wird diffundiert?|Pixel/Koordinaten vs. latenter Raum / PLM-Embeddings; bei Sequenzen: wie kommt man vom Embedding zurück zur Sequenz (Decoder, Rundung)?|
|**Längen**|Wie wird eine variable Sequenzlänge behandelt?|Padding + Maske, Länge vorab sampeln|

---

## Quellen

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Xiong et al. (2020), [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- Su et al. (2021), [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Elhage et al. (2021), [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Ho, Jain & Abbeel (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Song, Meng & Ermon (2020), [DDIM](https://arxiv.org/abs/2010.02502)
- Peebles & Xie (2022), [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- Ho & Salimans (2022), [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Lin et al. (2023), [ESM-2 / ESMFold](https://www.science.org/doi/10.1126/science.ade2574)