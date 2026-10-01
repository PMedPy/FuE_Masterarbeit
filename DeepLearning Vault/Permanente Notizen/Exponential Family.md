07-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
S94
Jede Wahrscheinlichkeitsverteilung der Exponentialfamilie hat die Form
$$p(\mathbf{x}\mid \eta)=g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp\{\boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x})\}\, \mathrm{d}\mathbf{x} $$
Wobei:
- $\mathbf{\eta}$ ein Satz aus Parametern, oder auch "natürliche Parameter"
- $\mathbf{u(x)}$ eine suffiziente Statistik [[Sufficient Statistics]] 
- $\mathbf{g(\eta)}$ als die Normalisierungskonstante
Verteilungen der Exponentialfamilie sind Gauss, [[Multivariant Gaussian]], Bernoulli, Multinomial als Beispiele. 
Um auf Exponentialfamilie zu prüfen, wird die Verteilung ummodelliert, um die einzelnen Komponenten zu benennen: 
# Beispiele
### Bernoulliverteilung
Die Bernoulliverteilung zeigt die Wahrscheinlichkeiten eines Binären Zufallsraums aus, wo die Zufallsvariable $x \in {0,1}$ enthalten ist: Es gibt also nur zwei Ausgänge eines Zufallsexperiments von Bernoulliverteilten Größen. 
#### Zielform

Die Exponentialfamilie hat die allgemeine Form:

$$ p(x \mid \eta) = h(x), g(\eta), \exp{\eta^{\top} u(x)} $$

Ziel: Die Bernoulli-Verteilung in genau diese Struktur bringen und $h(x)$, $g(\eta)$, $u(x)$ sowie $\eta$ ablesen.

---

#### Schritt 1 – Ausgangspunkt

Bernoulli für $x \in {0,1}$ mit $\mu = p(x=1)$:

$$ p(x \mid \mu) = \mu^{x},(1-\mu)^{1-x} $$

_Kompaktschreibweise: $x=1$ liefert $\mu$, $x=0$ liefert $1-\mu$._

---

#### Schritt 2 – exp-ln-Trick

Ein Produkt von Potenzen wird über $\exp{\ln(\cdots)}$ in eine Summe im Exponenten verwandelt:

$$p(x \mid \mu) = \exp({ x \ln \mu + (1-x)\ln(1-\mu)}) $$

_Potenzen im Basisausdruck werden zu Faktoren im Exponenten — jetzt ist alles additiv._

---

#### Schritt 3 – Nach $x$ sortieren

Terme mit und ohne $x$ trennen:

$$ p(x \mid \mu) = \exp {\left( x \ln\frac{\mu}{1-\mu} + \ln(1-\mu)\right)} $$

_Der Koeffizient von $x$ ist die **Log-Odds** $\ln\frac{\mu}{1-\mu}$ — genau die Struktur, die den natürlichen Parameter liefert._

---

#### Schritt 4 – Natürlichen Parameter identifizieren

Der Koeffizient von $x$ ist der natürliche Parameter:

$$ \eta = \ln\left(\frac{\mu}{1-\mu}\right) $$

Umstellen nach $\mu$ (exponenzieren und auflösen):

$$ \mu = \frac{1}{1 + e^{-\eta}} = \sigma(\eta) $$

_Der Mittelwert $\mu$ ist die Sigmoid des natürlichen Parameters — der Ursprung der Sigmoid in der logistischen Regression._

---

#### Schritt 5 – Konstanten Term umschreiben

Der Term $\ln(1-\mu)$ soll in $\eta$ ausgedrückt werden. Mit $\mu = \sigma(\eta)$ und der Sigmoid-Symmetrie $1-\sigma(\eta)=\sigma(-\eta)$:

$$ 1 - \mu = \frac{1}{1 + e^{\eta}} = \sigma(-\eta) $$

Damit wird der konstante Faktor:

$$ \exp{\ln(1-\mu)} = 1-\mu = \sigma(-\eta) $$

---

#### Schritt 6 – Ablesen

Zusammengesetzt:

$$ p(x \mid \eta) = \underbrace{\sigma(-\eta)}_{g(\eta)} \cdot \exp{, \eta \cdot \underbrace{x}_{u(x)},} $$

Vor dem $\exp$ steht kein weiterer $x$-abhängiger Faktor, also $h(x)=1$.

|Komponente|Bernoulli|
|---|---|
|$u(x)$|$x$|
|$h(x)$|$1$|
|$g(\eta)$|$\sigma(-\eta)$|
|$\eta$|$\ln\frac{\mu}{1-\mu}$|

---

#### Kernaussagen

- Die Bernoulli-Verteilung ist Teil der Exponentialfamilie, weil sich ihre Dichte als $\exp{\text{linear in } x}$ mal Konstante schreiben lässt.
- Die **suffiziente Statistik** ist $u(x)=x$ — der Datensatz wird vollständig durch $\sum_n x_n$ zusammengefasst.
- $g(\eta)$ ist der **Normierungsfaktor**, der $\sum_x p(x\mid\eta)=1$ sichert.

---

#### Natürlicher Parameter ≠ Erwartungswert

- Natürlicher Parameter: $\eta = \ln\frac{\mu}{1-\mu}$ (Log-Odds)
- Erwartungswert (Mittelwertparameter): $\mathbb{E}[x] = \mu$
- Verbindung über die Sigmoid: $\mu = \sigma(\eta)$

Allgemeine Brücke über die Log-Partitionsfunktion $A(\eta) = -\ln g(\eta) = \ln(1+e^{\eta})$:

$$ \mathbb{E}[u(x)] = \frac{dA(\eta)}{d\eta} = \sigma(\eta) = \mu $$

_Der Erwartungswert ist die Ableitung von $A(\eta)$ nach $\eta$ — gilt für die gesamte Exponentialfamilie._

### Multinomial bzw Kategorial
Die Multinomialverteilung verallgemeinert die Binomialverteilung von zwei Ausgängen auf $K$ Ausgänge. Analog verallgemeinert die **kategoriale Verteilung** (Multinoulli) die Bernoulli-Verteilung auf $K$ Kategorien für einen einzelnen Versuch.

|Ausgänge|Ein Versuch|$n$ Versuche|
|---|---|---|
|2|Bernoulli|Binomial|
|$K$|Kategorial (Multinoulli)|Multinomial|

Parameter: $\boldsymbol{\mu} = (\mu_1,\dots,\mu_K)$ mit $\mu_k \ge 0$ und $\sum_{k=1}^{K}\mu_k = 1$.

---

#### Kategoriale Verteilung – ein Versuch

Ein einzelner Versuch mit Ergebnis in einer von $K$ Kategorien wird als **One-Hot-Vektor** $\mathbf{x} = (x_1,\dots,x_K)$ kodiert, wobei $x_k \in {0,1}$ und $\sum_k x_k = 1$ (genau ein Eintrag ist 1).

$$ p(\mathbf{x} \mid \boldsymbol{\mu}) = \prod_{k=1}^{K} \mu_k^{x_k} $$

_Nur der Faktor der tatsächlich eingetretenen Kategorie überlebt, alle anderen sind $\mu_k^0 = 1$. Direkte Verallgemeinerung von $\mu^x(1-\mu)^{1-x}$ der Bernoulli._

---

#### Multinomialverteilung – n Versuche

Bei $n$ unabhängigen Versuchen zählt $m_k$, wie oft Kategorie $k$ eingetreten ist, mit $\sum_k m_k = n$.

$$ p(m_1,\dots,m_K \mid \boldsymbol{\mu}, n) = \binom{n}{m_1,\dots,m_K}\prod_{k=1}^{K}\mu_k^{m_k} $$

Der **Multinomialkoeffizient** zählt die Anordnungsmöglichkeiten der Ergebnisse:

$$ \binom{n}{m_1,\dots,m_K} = \frac{n!}{m_1!,m_2!\cdots m_K!} $$

_Verallgemeinerung des Binomialkoeffizienten $\binom{n}{k}$. Bei $K=2$ reduziert sich alles exakt auf die Binomialverteilung._

---

#### Zusammenhang zur Bernoulli / Binomial

|Setze|Ergebnis|
|---|---|
|$K = 2$|Multinomial $\to$ Binomial|
|$K = 2$, $n = 1$|Kategorial $\to$ Bernoulli|
|$n = 1$|Multinomial $\to$ Kategorial|

_Die zwei Achsen sind unabhängig: $K$ steuert die Zahl der Kategorien, $n$ die Zahl der Versuche._

---

#### Kategoriale Verteilung als Exponentialfamilie

Zielform: $p(\mathbf{x}\mid\boldsymbol{\eta}) = h(\mathbf{x}),g(\boldsymbol{\eta})\exp{\boldsymbol{\eta}^\top \mathbf{u}(\mathbf{x})}$.

Der exp-ln-Trick auf $\prod_k \mu_k^{x_k}$:

$$ p(\mathbf{x}\mid\boldsymbol{\mu}) = \exp\Big{\textstyle\sum_{k=1}^{K} x_k \ln\mu_k\Big} $$

_Der Koeffizient von $x_k$ ist $\eta_k = \ln\mu_k$ — das Pendant zur Log-Odds der Bernoulli._

Wegen der Nebenbedingung $\sum_k \mu_k = 1$ sind nur $K-1$ der Parameter frei. Löst man die Redundanz auf (Kategorie $K$ als Referenz), erhält man die natürlichen Parameter als **Log-Verhältnisse**:

$$ \eta_k = \ln!\left(\frac{\mu_k}{\mu_K}\right), \qquad k = 1,\dots,K-1 $$

Die Rückrichtung ist die **Softmax-Funktion**:

$$ \mu_k = \frac{\exp(\eta_k)}{\sum_{j}\exp(\eta_j)} $$

_Softmax ist die mehrdimensionale Verallgemeinerung der Sigmoid — der Grund, warum Softmax in der Multiclass-Klassifikation auftaucht, ist exakt derselbe wie bei der Sigmoid in der logistischen Regression._

---

#### Suffiziente Statistik

Die suffiziente Statistik ist der Vektor der Kategorienzähler:

$$ \mathbf{u}(\mathbf{x}) = \mathbf{x}, \qquad \text{über } n \text{ Versuche: } \sum_{i=1}^{n}\mathbf{x}_i = (m_1,\dots,m_K) $$

_Der gesamte Datensatz wird vollständig durch die Zählvektoren $m_k$ zusammengefasst. Verallgemeinerung von $\sum_n x_n$ bei der Bernoulli._

---

#### MaxEnt-Charakterisierung

Die kategoriale Verteilung ist die **Maximum-Entropie-Verteilung** auf dem Träger ${1,\dots,K}$ bei gegebenen Kategorienwahrscheinlichkeiten (bzw. gegebenem $\mathbb{E}[\mathbf{x}]=\boldsymbol{\mu}$).

Kette (analog zu Bernoulli und Normal):

$$ \text{MaxEnt unter } \mathbb{E}[\mathbf{x}]=\boldsymbol{\mu} ;\Longrightarrow; \text{Exponentialfamilienform} ;\Longrightarrow; \mathbf{u}(\mathbf{x})=\mathbf{x} \text{ suffizient} $$

_Ohne Nebenbedingung ($K$ Kategorien, sonst nichts) wäre die MaxEnt-Verteilung die diskrete Gleichverteilung $\mu_k = 1/K$. Die Momentbedingung erzeugt die Exponentialfamilienform; die suffiziente Statistik ist deren Konsequenz, nicht Ursache._

---

#### Kernaussagen

- Multinomial verhält sich zu Kategorial wie Binomial zu Bernoulli — nur mit $K$ statt 2 Ausgängen.
- Der natürliche Parameter ist ein **Log-Verhältnis** $\ln(\mu_k/\mu_K)$; die Rückabbildung ist **Softmax**.
- Die Nebenbedingung $\sum_k\mu_k = 1$ macht einen Parameter redundant — nur $K-1$ sind frei.
- Suffiziente Statistik: die Kategorienzähler $m_k$.
- MaxEnt-Verteilung auf $K$ Kategorien unter Momentbedingung; die suffiziente Statistik folgt aus dem MaxEnt-Prinzip.
# MaxEnt bei der Exponentialfamilie
Unter der **Momentenbedinung** sind die Verteilung mit höchster Entropie die der Exponentialfamilie:

>Die MaxEnt-Verteilung unter Momentbedingungen $E[u_{k}(x)]=c_{k}​$ hat notwendigerweise die Form der Exponentialfamilie, und in dieser Form sind die $u_{k}(x)$ genau die suffizienten Statistiken.

Hier sollten jedoch zwei Konzepte nicht vertauscht werden: Die Datenkompression zur [[Sufficient Statistics]] der Stichprobe und auf der anderen Seite der Informationsgehalt der Verteilung. 
# Zusammenhang von Parametern und der Suffizienten Statistik.
Eine Statistik ist suffizient, wenn alle natürlichen Parameter aus $\mathbf{u(x)}$ abgeleitet werden können bzw die Likelihoodfunktion der Verteilung nur davon Abhängt. Alle Verteilungen aus der Exponentialfamilie besitzen diese Suffiziente Statistik. Das heißt, würde man die Likelihood jeder dieser Verteilungen nehmen und Anordnen, müsste sie die Fisher-Neymann Zerlegung erfüllen. Es gilt:
$p(x∣θ)=h(x)⋅k(T(x),θ)$
Wobei $T(x)$ der Datensatz ist und $\theta$ der natürliche Parameter. 
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]
Siehe Fisher-Neyman zerlegung. 