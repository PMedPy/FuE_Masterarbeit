28-08-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Der Conditional Variational Autoencoder

**Von der Bildklassifikation zur Bildgenerierung — Theorie, Herleitung und Implementierung auf MNIST**

---

## Abstract

Ein Bildklassifikator lernt $p(y \mid x)$ — die Wahrscheinlichkeit einer Klasse gegeben ein Bild. Die naheliegende Frage, ob sich dieses Modell umkehren lässt, um $p(x \mid y)$ zu erhalten, also gezielt Bilder einer Klasse zu erzeugen, ist zu verneinen: dem diskriminativen Modell fehlt der Bildprior $p(x)$ strukturell. Dieses Dokument entwickelt stattdessen den Conditional Variational Autoencoder (CVAE), ein generatives Modell, das diesen Prior implizit mitlernt. Es leitet die Loss-Funktion (das ELBO) vollständig her, diskutiert die Reparametrisierung als Voraussetzung für Gradientenoptimierung über stochastische Knoten, und dokumentiert eine vollständige Implementierung in PyTorch. Abschließend wird erklärt, warum die erzeugten Bilder unscharf sind — ein Effekt, der nicht aus mangelndem Training folgt, sondern aus der Struktur der Zielfunktion selbst.
 

---

## 1. Motivation

### 1.1 Die Ausgangsfrage

Angenommen, ein MNIST-Klassifikator ist trainiert. Er bildet ab:

$$\mathbb{R}^{784} \longrightarrow \mathbb{R}^{10}, \qquad x \longmapsto p(y \mid x)$$

Was passiert, wenn man einen One-Hot-Vektor für „7" als Eingabe nimmt und den Prozess rückwärts laufen lässt? Erhält man eine Art gemittelte 7?

Nein — und zwar aus drei unabhängigen Gründen.

**Die Abbildung ist nicht injektiv.** 784 Dimensionen werden auf 10 abgebildet. Unendlich viele Bilder ergeben denselben Ausgabevektor. Ein Inverses existiert nicht. Es existiert bestenfalls ein _Urbild_ — eine hochdimensionale Mannigfaltigkeit, in der die lesbaren Siebenen eine verschwindend kleine Teilmenge bilden.

**Die Operationen sind nicht umkehrbar.** ReLU hat alle negativen Werte auf 0 abgebildet; diese Information ist gelöscht, nicht versteckt. Und für die Gewichtsmatrizen gilt $W^\top \neq W^{-1}$.

**Der entscheidende Grund ist konzeptuell.** Ein Klassifikator lernt nicht, _wie eine 7 aussieht_. Er lernt, _was eine 7 von den anderen neun Klassen unterscheidet_. Er ist **diskriminativ**: er zieht Grenzen, er beschreibt keine Gebiete. Alles, was zwischen den Klassen nicht unterscheidet — Strichstärke, Neigung, Schnörkel — darf er verwerfen, und er tut es.

### 1.2 Was formal fehlt

Das Bayes-Theorem macht die Lücke sichtbar:

$$p(x \mid y) ;\propto; \underbrace{p(y \mid x)}_{\text{Klassifikator}} \cdot \underbrace{p(x)}_{\text{fehlt}}$$

Gelesen als Logik: _„Ein plausibles Bild einer 7 ist eines, das (a) vom Klassifikator als 7 erkannt wird **und** (b) überhaupt wie ein Bild aussieht."_ Der Faktor $p(x)$ ist der **Bildprior** — das Wissen darüber, wie Bilder aussehen. Der Klassifikator hat ihn nie gelernt, weil er ihn nie brauchte.

Optimiert man nur die erste Bedingung — etwa per Gradientenaufstieg im Inputraum, $x \leftarrow x + \eta \nabla_x z_7(x)$ — erhält man strukturiertes Rauschen mit 99,99 % Konfidenz, das für einen Menschen nach nichts aussieht. Das ist ein bekanntes und gut dokumentiertes Ergebnis ([Nguyen et al. 2015](https://arxiv.org/abs/1412.1897)); man baut sich im Grunde ein adversariales Beispiel aus dem Nichts. Mit Regularisierung (L2 auf die Pixel, Total Variation, gelegentliches Weichzeichnen) wird es besser ([Simonyan et al. 2014](https://arxiv.org/abs/1312.6034), ausführlich bei [Olah et al. 2017](https://distill.pub/2017/feature-visualization/)) — aber der Prior kommt dann vom Regularisierer, nicht vom Modell.

> **Merksatz.** Diskriminative Modelle ziehen Grenzen, generative Modelle beschreiben Gebiete. Eine Grenze rückwärts zu lesen sagt dir, wo das Gebiet _nicht_ ist — nicht, wo sein Zentrum liegt.

### 1.3 Der Ausweg

Ein Modell bauen, das $p(x)$ mitlernt. Der Variational Autoencoder tut das, indem er ein Bild durch einen Flaschenhals zwingt und aus der komprimierten Beschreibung wieder aufbaut. Die _class-conditional_ Variante lernt zusätzlich, nach Klasse zu unterscheiden — und kann am Ende gezielt eine 7 erzeugen.

---

## 2. Modellarchitektur und Grundbegriffe

### 2.1 Der Autoencoder-Gedanke

$$x ;\xrightarrow{\ \text{Encoder}\ }; z ;\xrightarrow{\ \text{Decoder}\ }; \hat{x}$$

Zwinge ein Netz, ein Bild durch einen engen Kanal zu quetschen und danach wieder aufzubauen. Gelingt das, muss im Kanal eine komprimierte Beschreibung stecken.

**Encoding heißt hier nicht Formatumwandlung.** Es ist nicht wie UTF-8 oder Base64 — dieselbe Information, andere Schreibweise, verlustfrei umkehrbar. Es ist **Kompression auf das Wesentliche**, und die ist notwendigerweise verlustbehaftet: 784 Zahlen passen nicht in 2.

Analogie: statt ein Gemälde Pixel für Pixel abzumalen, sagst du „Landschaft, Abenddämmerung, Berge links". Jemand anders kann daraus etwas Ähnliches malen, aber nie das Original. Genau das ist der Decoder.

Und der Zwang zur Kürze **ist** der Lernmechanismus. Weil nur wenige Zahlen zur Verfügung stehen, muss das Netz herausfinden, was an einer Ziffer wirklich zählt und was Zufall ist.

### 2.2 Das „V": warum Verteilungen statt Punkte

Ein gewöhnlicher Autoencoder bildet jedes Bild auf einen **Punkt** ab. Das genügt zum Rekonstruieren, ist aber für Generierung unbrauchbar: der Raum zwischen den Punkten ist Niemandsland. Zieht man dort einen Punkt und dekodiert ihn, kommt Müll heraus. Der Latentraum ist dann ein Adressbuch mit Löchern, keine Landkarte.

Der VAE gibt stattdessen für jedes Bild eine **Verteilung** aus — eine kleine Gaußwolke $\mathcal{N}(\mu, \sigma^2)$ im Latentraum. Aus der wird gezogen. Weil benachbarte Bilder überlappende Wolken haben, wird der Raum lückenlos gefüllt.

Genau diese Lückenlosigkeit ist die Voraussetzung dafür, dass man später blind ein $z$ würfeln und ein sinnvolles Bild bekommen kann.

Der Encoder hat deshalb **zwei Ausgabeköpfe**: einen für $\mu$, einen für $\log \sigma^2$.

> ### Kasten: Was `nn.Module` eigentlich tut
> 
> Die häufigste Verständnishürde. PyTorch trennt strikt zwischen zwei Dingen:
> 
> **`__init__` legt die Bauteile hin.** Hier fließt _nichts_. Du sagst nur: „ich brauche später einen Linear-Layer von 794 auf 512, den nenne ich `self.enc`." Wie ein Labor einräumen — es wird nichts pipettiert.
> 
> Warum `self.`? Weil dort die **Gewichte** stecken. `nn.Linear(794, 512)` legt beim Anlegen eine Matrix mit 794×512 Zufallszahlen an, und genau die sollen trainiert werden. Hängst du sie an `self`, findet PyTorch sie automatisch — deshalb funktioniert `model.parameters()` später, ohne dass du irgendetwas registrierst.
> 
> **Die Methoden beschreiben die Wege.** Erst hier fließen Daten. `encode`, `reparameterize`, `decode`, `forward` — vier Wege durch dieselben Bauteile.
> 
> **`forward` ist eine ganz normale Methode.** Ihr einziges Privileg ist eine Konvention: PyTorch definiert `__call__` so, dass `model(x, y)` intern `model.forward(x, y)` ausführt. Man schreibt immer die kurze Form, weil die lange Hooks überspringt.
> 
> `forward` ist also _der Hauptweg_ — der, der beim Training genommen wird. `encode` und `decode` sind Nebeneingänge für später, wenn man das Modell auseinandernimmt.
> 
> **Und warum reicht `nn.Sequential` nicht?** `nn.Sequential` ist selbst ein `nn.Module`, dessen `forward` ungefähr so aussieht:
> 
> ```python
> def forward(self, x):
>     for layer in self:
>         x = layer(x)
>     return x
> ```
> 
> Eine stumpfe Kette. Genau eine Sache kann sie nicht: sich verzweigen. Unser Modell teilt sich nach `self.enc` auf zwei Köpfe auf, würfelt dazwischen, und speist `y` an zwei Stellen ein. Das passt in keine Liste — deshalb die eigene Klasse.

### 2.3 Der Datenfluss

```
   x (B,784)        y (B,10)
        └──────┬───────┘
          cat  │ (B,794)
               ▼
           self.enc          794 → 512 → 256
               │
        ┌──────┴──────┐
        ▼             ▼
    self.fc_mu   self.fc_logvar        je (B,2)
        └──────┬──────┘
               ▼
      reparameterize          z = μ + σ·ε      ε ~ N(0,I)
               │ (B,2)
               ├──────── y (B,10)
          cat  ▼ (B,12)
           self.dec          12 → 256 → 512 → 784
               ▼
          logits (B,784)     roh, kein Sigmoid
```

### 2.4 Wo das „conditional" hinkommt

Der One-Hot-Vektor $y$ wird an **beide** Eingänge gehängt:

- Encoder: `cat([x, y])` → $q_\phi(z \mid x, y)$
- Decoder: `cat([z, y])` → $p_\theta(x \mid z, y)$

Das ist der ganze Unterschied zum gewöhnlichen VAE ([Sohn, Lee & Yan 2015](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)).

Die Konsequenz ist bemerkenswert und wird uns in Abschnitt 7 wiederbegegnen: weil der Decoder die Klasse **geschenkt** bekommt, wäre es Verschwendung, sie auch noch in $z$ zu kodieren — der KL-Term bestraft ja jedes gespeicherte Bit. Also lernt $z$, alles _außer_ der Identität zu speichern: Strichstärke, Neigung, Rundung.

> **$y$ trägt die Identität, $z$ trägt den Stil.** Diese Arbeitsteilung wurde nirgends hineinprogrammiert. Sie fällt aus der Zielfunktion heraus.

---

## 3. Datenimport und Aufbereitung

### 3.1 Der Datensatz

MNIST: 70.000 handgeschriebene Ziffern, 28×28 Pixel, Graustufen. 60.000 im Trainings-, 10.000 im Test-Split. Die Trennung stammt von den Autoren und ist nicht willkürlich — Trainings- und Testdaten kommen von unterschiedlichen Schreibergruppen (Angestellte des US Census Bureau vs. Highschool-Schüler). Das macht den Testsplit zu einem ehrlicheren Generalisierungstest als ein zufälliger Schnitt.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(666)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

raw_train = datasets.MNIST(root="./data", train=True,  download=True,
                           transform=transforms.ToTensor())
raw_test  = datasets.MNIST(root="./data", train=False, download=True,
                           transform=transforms.ToTensor())
```

> ### Kasten: `Dataset`/`DataLoader` vs. sklearn
> 
> Über sklearn (`fetch_openml("mnist_784")`) ginge das auch — man bekäme ein NumPy-Array `(70000, 784)`. Für Deep Learning ist das aber der schlechtere Weg.
> 
> sklearn setzt voraus, dass **alle Daten in den RAM passen**. Das Paradigma ist „Matrix rein, `fit()` drauf". Bei MNIST (55 MB) stimmt das noch; bei 100.000 hochauflösenden Mikroskopiebildern nicht mehr.
> 
> `torch.utils.data.Dataset` ist ein Vertrag über zwei Methoden: `__len__` (wie viele?) und `__getitem__(i)` (gib mir Nummer _i_). Wo die Daten liegen, ist egal. Der `DataLoader` ruft `__getitem__` **lazy** auf — genau dann, wenn ein Batch gebraucht wird — und kann das über mehrere Worker-Prozesse parallelisieren, während die GPU noch am vorigen Batch rechnet.
> 
> Die üblichen Importwege:
> 
> |Weg|Wann|
> |---|---|
> |`torchvision.datasets`|Standard-Benchmarks (MNIST, CIFAR, ImageNet)|
> |`ImageFolder`|eigene Bilder in Ordnern `klasse_a/`, `klasse_b/`|
> |eigene `Dataset`-Subklasse|alles andere — der Normalfall in der Forschung|
> |HuggingFace `datasets`|NLP und zunehmend alles, mit Memory-Mapping|
> |`TensorDataset(X, y)`|wenn es wirklich in den RAM passt|
> 
> **Indexkonvention:** `raw_train[0]` ist _Sample Nummer 0_, und liefert ein Tupel `(bild, label)`. Nicht „`[0]` ist Input und `[1]` ist Output" — das Tupel kommt aus `__getitem__`, nicht aus dem Index. Die Konvention (Input, Target) gilt in ganz PyTorch.

### 3.2 Drei Transformationen

**Skalierung auf $[0,1]$.** `transforms.ToTensor()` teilt die `uint8`-Werte durch 255. Das ist hier **zwingend**, nicht kosmetisch: wir modellieren jedes Pixel als Bernoulli-Variable, und der Decoder gibt eine Wahrscheinlichkeit aus. Hätte man wie üblich auf Mittelwert 0 standardisiert, wäre die Binary Cross-Entropy nicht definiert.

> Die Bernoulli-Annahme ist eine kleine Lüge: MNIST-Pixel sind graustufig, nicht binär. In der Praxis funktioniert BCE auf $[0,1]$-Werten trotzdem sehr gut und ist Standard seit dem Originalpaper. Sauberer wäre eine kontinuierliche Bernoulli ([Loaiza-Ganem & Cunningham 2019](https://arxiv.org/abs/1907.06845)).

**Flattening: $(1,28,28) \to 784$.** Die führende 1 ist die Kanaldimension (bei RGB stünde da 3). PyTorch nutzt durchgehend `(Channel, Height, Width)` — anders als matplotlib, das `(H, W)` bzw. `(H, W, C)` erwartet. Daher das `.squeeze()` beim Plotten.

**One-Hot-Encoding des Labels.** Der einzige Punkt, an dem es sich wirklich zu denken lohnt.

Das Label `7` als Zahl ins Netz zu geben wäre ein Fehler: das Netz würde eine **Ordnungsstruktur** unterstellen (7 liegt zwischen 6 und 8, ist „größer" als 3). Ziffern-Identität ist aber **nominal**, nicht ordinal. One-Hot gibt jeder Klasse ihre eigene, orthogonale Achse, alle im gleichen Abstand $\sqrt{2}$ zueinander.

> **Unterschied zum Klassifikator:** Bei Iris gibt man das Label als Integer an `nn.CrossEntropyLoss`, die intern One-Hot macht. Hier wird das Label ein **Input** ins Netz — deshalb muss man es selbst kodieren.

```python
NUM_CLASSES = 10

def prepare(x, y, num_classes=NUM_CLASSES):
    """x: (B,1,28,28) -> (B,784) float ; y: (B,) -> (B,10) float one-hot"""
    x = x.view(x.size(0), -1).float()
    y = F.one_hot(y, num_classes).float()
    return x, y
```

`x.view(B, -1)` heißt „behalte die Batch-Achse, walze den Rest platt"; die `-1` bedeutet „rechne dir selbst aus, wie viel übrig bleibt". Das `.float()` bei `one_hot` ist nötig, weil `F.one_hot` `long` liefert und `nn.Linear` `float` verlangt.

---

## 4. Das Modell in Code

```python
class CVAE(nn.Module):
    def __init__(self, x_dim=784, y_dim=10, h_dim=512, h2_dim=256, z_dim=2):
        super().__init__()
        self.z_dim, self.y_dim = z_dim, y_dim

        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h2_dim),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(h2_dim, z_dim)
        self.fc_logvar = nn.Linear(h2_dim, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + y_dim, h2_dim), nn.ReLU(),
            nn.Linear(h2_dim, h_dim),         nn.ReLU(),
            nn.Linear(h_dim, x_dim),          # KEIN Sigmoid
        )

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        return self.dec(torch.cat([z, y], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
```

### 4.1 Warum `logvar` statt `sigma`

$\sigma$ **muss** positiv sein, aber ein `nn.Linear` kann alles ausgeben, auch $-3$. Statt die Ausgabe künstlich einzusperren (ReLU? Softplus? beides schlecht konditioniert), lässt man sie frei in $\mathbb{R}$ laufen und interpretiert sie als Logarithmus der Varianz:

$$\sigma = \exp!\Big(\tfrac{1}{2}\log \sigma^2\Big)$$

Der Faktor $\tfrac12$ ist nur die Wurzel: $\sqrt{\sigma^2} = (\sigma^2)^{1/2}$, und im Log-Raum wird aus der Potenz eine Multiplikation. Gelesen als Logik: _lass das Netz in einem Raum arbeiten, in dem es sich nicht falsch bewegen kann, und übersetze am Ende._ Nebenbei ist der Log-Raum numerisch gutmütiger, weil sehr kleine Varianzen nicht auf 0 unterlaufen.

### 4.2 Der Reparametrisierungstrick

Ein handfestes Problem: der Encoder gibt eine Verteilung aus, wir müssen daraus **ziehen**, und Backpropagation kann nicht durch einen Zufallsgenerator hindurch. `torch.normal(mu, std)` hat keinen Gradienten bezüglich `mu` — die Kette reißt.

Die Umgehung:

$$z \sim \mathcal{N}(\mu, \sigma^2) \quad\Longleftrightarrow\quad z = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

Gelesen als Logik: **verschiebe den Zufall aus dem Rechenweg an den Rand.** $\varepsilon$ ist jetzt ein _Input_ wie das Bild — reiner Rauschen-Input ohne Parameter. Der Weg von $\mu$ nach $z$ ist eine Addition, der von $\sigma$ nach $z$ eine Multiplikation. Beides differenzierbar:

$$\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \varepsilon$$

Dass das mathematisch zulässig ist, folgt aus der Verteilungsgleichheit: eine affine Transformation einer Normalverteilung ist wieder normalverteilt, mit $\mathbb{E}[z] = \mu$ und $\text{Var}[z] = \sigma^2$. Der Erwartungswert im ELBO ändert sich also nicht — nur die Art, wie wir ihn schätzen.

> ### Kasten: `rand` vs. `randn` — der stumme Fehler
> 
> ```python
> eps = torch.rand_like(std)    # FALSCH — uniform in [0,1)
> eps = torch.randn_like(std)   # richtig — normal, μ=0, σ=1
> ```
> 
> Das `n` steht für _normal_. `rand` ist gleichverteilt: immer positiv, Mittelwert 0.5, Standardabweichung ≈ 0.29.
> 
> Mit `rand` liegt $z$ systematisch **über** $\mu$ und streut viel zu wenig. Die tatsächliche Ziehverteilung ist nicht $\mathcal{N}(\mu,\sigma^2)$ — die KL-Formel bestraft also die Abweichung von etwas, das gar nicht passiert. Beim Generieren zieht man später aus $\mathcal{N}(0,I)$, einer Verteilung, die der Decoder nie gesehen hat.
> 
> **Und nichts davon wirft einen Fehler.** Shapes stimmen, der Loss fällt, die Kurven sehen überzeugend aus. In einem dokumentierten Fall stieg die KL nach dem Fix von 2.47 auf 4.91 nats — der Kanal war vorher künstlich um die Hälfte eingeschnürt.
> 
> Konsequenz: bei probabilistischen Modellen sind Asserts über _statistische_ Eigenschaften Pflicht, nicht Kür.
> 
> ```python
> big = model.reparameterize(torch.zeros(50_000, 2), torch.zeros(50_000, 2))
> assert abs(big.mean()) < 0.02          # E[z] ≈ 0
> assert abs(big.std() - 1.0) < 0.02     # Std[z] ≈ 1
> 
> mu_g = torch.zeros(4, 2, requires_grad=True)
> model.reparameterize(mu_g, torch.zeros(4, 2)).sum().backward()
> assert mu_g.grad.abs().sum() > 0       # fliesst der Gradient?
> ```
> 
> Der letzte Test ist der eigentliche Punkt: `mu_g.grad` ist nur dann ungleich Null, wenn zwischen `mu` und `z` ein differenzierbarer Pfad liegt. Ohne den Trick wäre er `None`, und das Training würde still fehlschlagen.

> ### Kasten: kein NumPy im Modell
> 
> Drei unabhängige Gründe:
> 
> 1. **Skalar statt Tensor.** `rng.normal(0,1)` liefert _eine_ Zahl. Man braucht ein eigenes $\varepsilon$ pro Bild und pro Latentdimension, also Form `(B, z_dim)`. `randn_like` übernimmt Form, dtype und Device automatisch.
> 2. **Autograd bricht ab.** `np.exp` auf einem Tensor mit `requires_grad=True` wirft `RuntimeError: Can't call numpy() on Tensor that requires grad`. NumPy weiß nichts von Autograd; es müsste den Tensor in ein rohes Array verwandeln, und damit wäre die Kette gekappt. PyTorch verbietet das lieber, als es still passieren zu lassen.
> 3. **Device.** NumPy rechnet immer auf der CPU. Liegt der Tensor auf CUDA oder MPS, geht es ohnehin nicht.
> 
> Zu jeder NumPy-Funktion gibt es eine Torch-Entsprechung: `np.exp` → `torch.exp`, `rng.normal` → `torch.randn_like`.

### 4.3 Hyperparameter der Breite

Die Zahlen 512 und 256 sind **willkürlich**. Hart ist nur, dass benachbarte Schichten zusammenpassen: was aus `nn.Linear(a, b)` kommt, muss in `nn.Linear(b, c)`. Zwingend festgelegt sind nur die Ränder: 794 (784 + 10), `z_dim`, 784.

Zweierpotenzen sind Konvention ohne heutigen technischen Grund. Die Trichterform (schrittweise halbieren) ist eine Faustregel: die Kompression von 794 auf 2 passiert in Etappen, nicht in einem Sprung.

Zu schmal → Underfitting. Zu breit → langsamer, mehr Overfitting-Gefahr. Für MNIST-MLPs sind 256–1024 üblich. Der interessante Hyperparameter ist ohnehin nicht `h_dim`, sondern **`z_dim`** — er bestimmt, wie viel Information überhaupt durch den Flaschenhals passt.

---

## 5. Herleitung der Loss-Funktion

Dies ist der theoretische Kern.

### 5.1 Die generative Geschichte

Wir behaupten, MNIST-Bilder entstehen so:

1. Ziehe $z \sim \mathcal{N}(0, I)$ — _würfle einen Stil_
2. Nimm die Klasse $y$ — _entscheide dich für eine Ziffer_
3. Erzeuge $x \sim p_\theta(x \mid z, y)$ — _male sie_

$\theta$ sind die Decoder-Gewichte. Wählt man $\theta$ gut, ist das ein Bildgenerator. Bleibt die Frage, wie man $\theta$ wählt.

### 5.2 Das Problem: ein unlösbares Integral

Maximum Likelihood sagt: maximiere die Wahrscheinlichkeit der beobachteten Daten.

$$p_\theta(x \mid y) = \int p_\theta(x \mid z, y), p(z), \mathrm{d}z$$

Gelesen als Logik: _„Wie wahrscheinlich ist dieses Bild? Summiere über **alle** möglichen Stile $z$ und gewichte jeden mit seiner Prior-Wahrscheinlichkeit."_ Das $\int \dots p(z),\mathrm{d}z$ **ist** eine gewichtete Mittelung — dieselbe Struktur wie ein $\sum_i w_i f_i$, nur kontinuierlich.

Und genau das macht es unlösbar. Für ein gegebenes $x$ ist praktisch jedes zufällig aus dem Prior gezogene $z$ nutzlos: der Decoder malt irgendeine andere Ziffer. Monte-Carlo-Sampling bräuchte astronomisch viele Ziehungen, um die winzige Region zu treffen, die zu _diesem_ $x$ passt.

### 5.3 Die Lösung: einen Vorschlag raten

Wir führen einen zweiten Netzteil ein, den **Encoder** $q_\phi(z \mid x, y)$, der die Frage beantwortet: _„welche $z$ kämen für dieses Bild überhaupt in Frage?"_ Statt blind aus dem Prior zu ziehen, ziehen wir aus einem gezielten Vorschlag.

### 5.4 Die exakte Zerlegung

Wir starten bei der KL-Divergenz zwischen unserem Vorschlag und dem **wahren** Posterior:

$$D_{\mathrm{KL}}\big(q_\phi(z\mid x,y) ,|, p_\theta(z \mid x,y)\big) = \mathbb{E}_{q_\phi}\big[\log q_\phi(z\mid x,y) - \log p_\theta(z\mid x,y)\big]$$

Nun nach Bayes: $p_\theta(z \mid x, y) = \dfrac{p_\theta(x, z \mid y)}{p_\theta(x \mid y)}$, also

$$\log p_\theta(z\mid x,y) = \log p_\theta(x,z\mid y) - \log p_\theta(x\mid y)$$

Einsetzen:

$$D_{\mathrm{KL}} = \mathbb{E}_{q_\phi}\big[\log q_\phi - \log p_\theta(x,z\mid y)\big] + \log p_\theta(x\mid y)$$

Der Term $\log p_\theta(x\mid y)$ hängt nicht von $z$ ab und kommt deshalb unverändert aus dem Erwartungswert heraus. Umstellen:

$$\boxed{;\log p_\theta(x\mid y) ;=; \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x,z\mid y) - \log q_\phi(z\mid x,y)\big]}_{\textstyle \mathcal{L}(\theta,\phi)\ =\ \text{ELBO}} ;+; \underbrace{D_{\mathrm{KL}}\big(q_\phi ,|, p_\theta(z\mid x,y)\big)}_{\textstyle \ge, 0};}$$

**Wie man das liest.** Der rechte Term ist eine KL-Divergenz, also **nie negativ**. Er misst, wie sehr unser Encoder-Rat vom wahren Posterior abweicht. Wir kennen ihn nicht — aber weil er $\ge 0$ ist, folgt sofort:

$$\log p_\theta(x \mid y) ;\ge; \mathcal{L}$$

Deshalb: **Evidence Lower Bound**. Wir maximieren die Untergrenze statt der Evidenz.

Der schöne Nebeneffekt: Die linke Seite hängt nicht von $\phi$ ab. Schiebt man $\mathcal{L}$ über $\phi$ hoch, muss die Lücke also **schrumpfen** — der Encoder wird nebenbei ein besserer Approximator des echten Posteriors. Ein Optimierungsschritt, zwei Verbesserungen.

### 5.5 Das ELBO in seine zwei Terme zerlegen

Mit $p_\theta(x,z\mid y) = p_\theta(x \mid z,y), p(z)$:

$$\begin{aligned} \mathcal{L} &= \mathbb{E}_{q_\phi}\big[\log p_\theta(x\mid z,y) + \log p(z) - \log q_\phi(z\mid x,y)\big] \[4pt] &= \underbrace{\mathbb{E}_{q_\phi}\big[\log p_\theta(x\mid z,y)\big]}_{\text{Rekonstruktion}} ;-; \underbrace{D_{\mathrm{KL}}\big(q_\phi(z\mid x,y) ,|, p(z)\big)}_{\text{Regularisierung}} \end{aligned}$$

- **Term 1** fragt: _„Wenn ich den Code nehme, den der Encoder für dieses Bild vorschlägt — trifft der Decoder das Bild wieder?"_ Er will, dass jedes Bild einen eigenen, präzisen Platz bekommt.
- **Term 2** fragt: _„Wie weit weicht der Encoder von der Standardnormalverteilung ab?"_ Er zieht alle Codes Richtung $\mathcal{N}(0,I)$ und will am liebsten, dass der Encoder das Bild ignoriert.

**Die beiden ziehen gegeneinander, und dieser Konflikt ist das ganze Verfahren.** Term 1 allein gäbe einen gewöhnlichen Autoencoder mit löchrigem Latentraum. Term 2 stopft die Löcher. Erst dadurch wird der Raum **sampelbar**.

Optimierer minimieren, also:

$$\mathcal{L}_{\text{loss}} = -\mathcal{L} = \text{Rekonstruktionsfehler} + \text{KL}$$

### 5.6 Term 1 ist Binary Cross-Entropy

Mit der Bernoulli-Annahme pro Pixel:

$$p_\theta(x \mid z,y) = \prod_{i=1}^{784} \hat{x}_i^{,x_i}(1-\hat{x}_i)^{1-x_i}$$

Logarithmieren und negieren:

$$-\log p_\theta(x\mid z,y) = -\sum_{i=1}^{784} \Big[x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)\Big]$$

Das **ist** die BCE-Summe — kein Zufall, dieselbe Formel.

Der Log macht hier seine klassische Arbeit: **aus dem Produkt über 784 Pixel wird eine Summe.** Ohne ihn multipliziert man 784 Zahlen $< 1$ und landet sofort im numerischen Unterlauf.

Jeder Summand fragt: _„hier war Tinte ($x_i = 1$) — wie überzeugt warst du?"_ Volle Überzeugung kostet $-\log 1 = 0$, sicherer Irrtum $-\log 0 \to \infty$. Der Log ist die Bestrafung für selbstbewusste Fehler.

### 5.7 Term 2 hat eine geschlossene Form

Hier der Glücksfall: beide Verteilungen sind gaußsch, die KL ist analytisch lösbar — kein Sampling nötig. Herleitung für eine Dimension, mit $q = \mathcal{N}(\mu,\sigma^2)$ und $p = \mathcal{N}(0,1)$:

$$\log q(z) = -\tfrac12\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}, \qquad \log p(z) = -\tfrac12\log(2\pi) - \frac{z^2}{2}$$

Erwartungswerte unter $q$, wobei $\mathbb{E}_q[(z-\mu)^2] = \sigma^2$ und $\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$:

$$\mathbb{E}_q[\log q] = -\tfrac12\log(2\pi\sigma^2) - \tfrac12, \qquad \mathbb{E}_q[\log p] = -\tfrac12\log(2\pi) - \tfrac{1}{2}(\mu^2+\sigma^2)$$

Differenz bilden; die $\log(2\pi)$ heben sich weg:

$$D_{\mathrm{KL}} = -\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12\mu^2 + \tfrac12\sigma^2 = -\tfrac12\Big(1 + \log\sigma^2 - \mu^2 - \sigma^2\Big)$$

Bei diagonaler Kovarianz faktorisiert alles, also einfach über die Dimensionen summieren:

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\sigma^2) ,|, \mathcal{N}(0,I)\big) = -\frac{1}{2}\sum_{j=1}^{d}\Big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\Big)$$

(Anhang B in [Kingma & Welling 2013](https://arxiv.org/abs/1312.6114).)

**Gelesen als Logik** — jeder Summand stellt zwei Fragen an eine Latentdimension:

- $-\mu_j^2$ → _„liegst du nahe am Ursprung?"_ Quadratische Strafe, exakt wie ein L2-Regularisierer. Zieht alle Codes ins Zentrum.
- $\log\sigma_j^2 - \sigma_j^2$ → _„ist deine Unschärfe ungefähr 1?"_ Maximum exakt bei $\sigma = 1$; nach $\sigma^2$ ableiten gibt $\frac{1}{\sigma^2} - 1 = 0$. Zu scharf ist teuer, zu breit auch.
- Die $+1$ ist Kalibrierung, damit bei $\mu=0,\sigma=1$ glatt 0 herauskommt.

**Einheit beider Terme: nats** (natürlicher Logarithmus). Sie sind dadurch direkt addierbar, und die KL sagt buchstäblich, wie viele nats an Information der Encoder pro Bild durch den Flaschenhals schiebt.

### 5.8 Implementierung

```python
def vae_loss(logits, x, mu, logvar, beta=1.0):
    """Negatives ELBO. Rückgabe: (loss, recon, kl) — alle in nats pro Bild."""
    B = x.size(0)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / B
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return recon + beta * kl, recon, kl
```

**`reduction="sum"` statt `"mean"`**, weil über die 784 Pixel _summiert_ wird (Produkt der Bernoullis!), nicht gemittelt. Das `/B` macht daraus „pro Bild" — dann sind beide Terme in derselben Einheit und sauber addierbar.

> ### Kasten: `_with_logits` — warum der Decoder kein Sigmoid hat
> 
> Mathematisch ist `sigmoid` + `BCE` identisch zu `binary_cross_entropy_with_logits`. Numerisch nicht.
> 
> Die kombinierte Variante nutzt intern den Log-Sum-Exp-Trick. Rechnet man getrennt, wird $\hat x$ bei einem sicheren Decoder irgendwann auf exakt `0.0` oder `1.0` gerundet, und $\log(0) = -\infty$ macht den Loss zu `nan`. Der Bug erscheint typischerweise nach einigen hundert Schritten — genau dann, wenn das Modell anfängt, gut zu werden.
> 
> Konsequenz: der Decoder gibt **Logits** aus. Zum Anzeigen muss man `torch.sigmoid` von Hand nachholen.

**Validierung durch analytisch nachrechenbare Testfälle:**

```python
x_, lg_ = torch.rand(32, 784), torch.randn(32, 784)

# q = Prior exakt  ->  KL = 0
_, _, kl0 = vae_loss(lg_, x_, torch.zeros(32,2), torch.zeros(32,2))
assert abs(kl0.item()) < 1e-5

# mu = 1, sigma = 1  ->  KL = ½‖μ‖² = ½·2 = 1.0
_, _, kl1 = vae_loss(lg_, x_, torch.ones(32,2), torch.zeros(32,2))
assert abs(kl1.item() - 1.0) < 1e-4

# KL ist nie negativ
for _ in range(200):
    _, _, k = vae_loss(lg_, x_, torch.randn(32,2)*3, torch.randn(32,2)*3)
    assert k.item() >= -1e-5
```

Der zweite Fall: bei $\sigma = 1$ ist $\log\sigma^2 = 0$ und $\sigma^2 = 1$, beide heben sich gegen die $+1$ weg. Übrig bleibt $\tfrac12|\mu|^2$. Schöne Nebenerkenntnis: **die KL zwischen zwei gleich breiten Gaußkurven ist der halbe quadrierte Abstand ihrer Zentren** — dieselbe „Energie"-Struktur wie beim Least-Squares-Fehler.

Diese Asserts sind wichtiger als sonst: ein Tippfehler wie `mu.exp()` statt `mu.pow(2)` läuft fehlerfrei durch und trainiert Unsinn.

---

## 6. Training

### 6.1 Setsplitting

|Set|Größe|Wozu|Wie oft angefasst|
|---|---|---|---|
|Train|54.000|Gradienten, Gewichte ändern|jede Epoche|
|Validation|6.000|Hyperparameter, „läuft's?"|jede Epoche, ohne Gradient|
|Test|10.000|eine finale Zahl|genau **einmal**, ganz am Ende|

Sobald man wegen einer Zahl eine Entscheidung trifft (Lernrate ändern, früher stoppen, Architektur anpassen), hat man sich an dieses Set _angepasst_. Es ist dann kein unabhängiger Schätzer mehr — auch wenn man nie darauf trainiert hat. Diese schleichende Kontamination ist der häufigste Grund, warum publizierte Zahlen sich nicht reproduzieren lassen.

```python
BATCH_SIZE = 128

train_set, val_set = random_split(
    raw_train, [54_000, 6_000],
    generator=torch.Generator().manual_seed(666),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False)
test_loader  = DataLoader(raw_test,  batch_size=512, shuffle=False)
```

**`shuffle=True` nur beim Training.** Ohne Mischen wären aufeinanderfolgende Batches korreliert und die Gradienten systematisch verzerrt. Bei Val/Test ist die Reihenfolge egal, weil nichts gelernt wird.

**`drop_last=True`** wirft den letzten, unvollständigen Batch weg. Hier kosmetisch, bei BatchNorm-Modellen mit Restbatch der Größe 1 ein echter Crash-Grund.

**Eigener `Generator`** macht den Schnitt reproduzierbar, unabhängig davon, wie viel Zufall vorher verbraucht wurde.

> ### Kasten: Was Batches sind, und warum
> 
> `batch_size=128` heißt: 128 Bilder pro Portion. Bei 54.000 Bildern sind das 421 Batches pro Epoche. Eine **Epoche** ist ein vollständiger Durchlauf durch alle Trainingsdaten.
> 
> Der Unterschied zu einem kleinen Datensatz wie Iris ist strukturell:
> 
> ```python
> # Iris: full-batch, 300 Epochen = 300 Gewichtsupdates
> for epoch in range(300):
>     outputs = model(X_train)          # ALLE 120 Samples auf einmal
>     loss = criterion(outputs, y_train)
>     loss.backward(); optimizer.step()
> 
> # CVAE: mini-batch, 15 Epochen = 6.315 Gewichtsupdates
> for ep in range(15):
>     for xb, yb in train_loader:       # 421 Batches à 128
>         ...
>         loss.backward(); opt.step()
> ```
> 
> Zwei Gründe für Batches:
> 
> 1. **Speicher.** 54.000 Bilder gleichzeitig durchs Netz sprengt die GPU.
> 2. **Viele kleine Schritte schlagen wenige große.** Jeder einzelne Gradient ist ungenauer — er basiert nur auf 128 Bildern — aber genau dieses Rauschen hilft dem Optimierer, aus flachen lokalen Minima herauszurutschen. Daher _stochastic_ gradient descent.
> 
> Deshalb reicht bei Mini-Batches `lr=1e-3`, wo full-batch `lr=0.1` brauchte: bei 300 großen Schritten muss jeder weit sein, bei 6.315 kleinen wäre 0.1 sofort instabil.

### 6.2 β-Warm-up gegen Posterior Collapse

Am Anfang ist der Decoder nutzlos — er kann mit $z$ nichts anfangen. Der billigste Weg, den Loss zu senken, ist dann: KL auf 0 drücken, $z$ ignorieren, und aus $y$ allein die mittlere Ziffer malen. Das Modell landet in diesem lokalen Minimum und kommt nicht wieder heraus, weil der Decoder nie lernt, $z$ zu benutzen. Das heißt **Posterior Collapse**.

Gegenmittel: den KL-Term anfangs herunterwiegen und langsam einblenden.

$$\mathcal{L}_{\text{loss}} = \text{BCE} + \beta \cdot \text{KL}, \qquad \beta: 0 \to 1$$

([Bowman et al. 2016](https://arxiv.org/abs/1511.06349).) $\beta$ dauerhaft $> 1$ zu lassen ist ein eigenes Forschungsprogramm: [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) erzwingt damit stärker entwirrte Latentdimensionen — auf Kosten der Bildschärfe.

### 6.3 Die Trainingsschleife

```python
Z_DIM, EPOCHS, LR, WARMUP_EP = 2, 15, 1e-3, 5

model = CVAE(z_dim=Z_DIM).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=LR)


def run_epoch(loader, train, beta):
    model.train() if train else model.eval()
    tot = totr = totk = n = 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            x, y = prepare(xb.to(device), yb.to(device))
            logits, mu, logvar = model(x, y)
            loss, rec, kl = vae_loss(logits, x, mu, logvar, beta)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            B = x.size(0)
            tot += loss.item()*B; totr += rec.item()*B; totk += kl.item()*B; n += B

    return tot/n, totr/n, totk/n


history = {"train": [], "val": [], "recon": [], "kl": []}

for ep in range(1, EPOCHS + 1):
    beta = min(1.0, ep / WARMUP_EP)
    tr_loss, tr_rec, tr_kl = run_epoch(train_loader, True,  beta)
    va_loss, va_rec, va_kl = run_epoch(val_loader,   False, 1.0)

    for key, value in zip(history.keys(), [tr_loss, va_loss, va_rec, va_kl]):
        history[key].append(value)

    print(f"{ep:3d} {beta:5.2f} │ {tr_loss:8.2f} {tr_rec:8.2f} {tr_kl:7.3f} │ {va_loss:8.2f}")
```

### 6.4 Die einzelnen Bestandteile

**`.to(device)` — was „aufs Device schieben" heißt.** Ein Tensor liegt physisch in einem bestimmten Speicher: im RAM (CPU) oder im GPU-Speicher (CUDA bzw. MPS bei Apple Silicon). Rechnen kann man nur zwischen Tensoren am selben Ort. Das Modell ist per `.to(device)` schon dort; die Daten kommen frisch von der Platte und liegen zunächst auf der CPU. Vergisst man es: _„expected all tensors to be on the same device"_.

**`torch.set_grad_enabled(train)` — der Autograd-Graph.** Rechnet man mit einem Tensor, der `requires_grad=True` hat, notiert PyTorch nebenbei, _wie_ die Rechnung zustande kam — inklusive aller Zwischenergebnisse. Dieses Protokoll ist der **Autograd-Graph**; `loss.backward()` läuft es rückwärts ab und wendet an jedem Knoten die Kettenregel an.

Es kostet Speicher. Bei der Validierung wird es aufgebaut und ungenutzt weggeworfen — dort schaltet man es ab. `torch.no_grad()` ist dasselbe ohne Schalter; `@torch.no_grad()` als Dekorator gilt für eine ganze Funktion.

Ein zweiter, praktischer Grund: an einem Tensor mit Autograd-Graph verweigern `.cpu()` und `.numpy()` die Arbeit. Ohne `no_grad` müsste man überall `.detach()` einstreuen.

> **Faustregel: alles, was nicht trainiert, kommt in `no_grad`.**

**`with` ist reines Python.** Die Konstruktion für „mach etwas, und räum danach garantiert wieder auf" — bekannt von `with open(...) as f`. Hier ist das Aufräumen das Zurücksetzen eines globalen Schalters, auch wenn mittendrin eine Exception fliegt.

**`model.train()` / `model.eval()`.** Bei diesem Modell wirkungslos (kein Dropout, kein BatchNorm), aber sobald man es erweitert, ist das Fehlen ein sehr unangenehmer Bug.

**Die Reihenfolge `zero_grad` → `backward` → `step` ist zwingend.** PyTorch **addiert** Gradienten, statt sie zu überschreiben. Vergisst man `zero_grad`, summiert man die Gradienten aller bisherigen Batches auf und das Training explodiert.

**`.item()` beim Mitzählen.** `loss` ist ein Tensor mit dranhängendem Graphen. Summiert man Tensoren auf, hält man den ganzen Graphen im Speicher fest. `.item()` holt die nackte Python-Zahl heraus.

**`B = x.size(0)`** ist nicht „die Länge von x", sondern die **Größe der Achse 0**. Bei `x.shape == (128, 784)` ist `size(0) = 128`, also die Anzahl Bilder im Batch. Warum nicht `BATCH_SIZE` einsetzen? Weil der letzte Batch kleiner sein kann — beim `val_loader` ohne `drop_last` hat er bei 6.000/512 genau 368 Bilder. `x.size(0)` fragt nach, statt anzunehmen.

### 6.5 Ein typischer Verlauf

```
 Ep  beta │    train    recon      KL │      val
  1  0.20 │   165.31   163.90   7.060 │   145.34
  2  0.40 │   136.87   134.52   5.870 │   139.00
  3  0.60 │   134.15   131.02   5.221 │   136.08
  4  0.80 │   133.40   129.50   4.878 │   134.66
  5  1.00 │   133.20   128.55   4.649 │   133.89   ← β erreicht 1
  6  1.00 │   132.28   127.62   4.661 │   133.38   ← KL kehrt um
 ...
 15  1.00 │   128.50   123.59   4.913 │   130.60
```

**Der Wendepunkt bei Epoche 5/6 ist die interessanteste Stelle.** Bis dahin fällt die KL — das ist $\beta$, das den Druck erhöht. Ab Epoche 6 ist $\beta$ konstant, und die KL **steigt** wieder, langsam aber monoton.

Das ist das Gleichgewicht in Aktion: der Decoder ist inzwischen gut genug, dass sich zusätzliche Information _lohnt_. Er kauft KL, weil er dafür mehr recon spart, als sie kostet.

Ein alarmierender Verlauf wäre: KL geht auf 0 und bleibt dort, recon stagniert hoch.

---

## 7. Validierung

Bei einem Klassifikator gibt es eine ehrliche Zahl: Accuracy. Bei einem Generator nicht — „ist dieses erfundene Bild eine gute 7?" hat keine richtige Antwort. Deshalb drei Blickwinkel.

### 7.1 Wie man recon und KL liest

> ### Kasten: Der Gesamt-Loss ist eine Loss-Funktion. Die Einzelterme sind es nicht.
> 
> **Der Gesamt-Loss** ist eine ganz normale Zielgröße: kleiner = besser. Er ist eine Obergrenze für $-\log p(x\mid y)$, also für die Unwahrscheinlichkeit der Daten unter dem Modell.
> 
> Zwei Einschränkungen: Er kann **nie 0 erreichen** — die Daten haben eine Grundzufälligkeit, und die kostet Bits. Und er ist nur **innerhalb eines Setups** vergleichbar; anderes `z_dim` oder andere Datenskalierung, andere Zahl. Es gibt keinen absoluten Maßstab wie bei einer Accuracy.
> 
> **Die Einzelterme dagegen sind Diagnose, kein Score.** Jeder für sich ist trivial auf 0 zu bringen, indem man den anderen ruiniert:
> 
> - `recon = 0` → perfekte Rekonstruktion, erreichbar durch isolierte Punkte. Latentraum wird zum Adressbuch mit Löchern, Sampling produziert Müll.
> - `KL = 0` → der Encoder gibt für jedes Bild dieselbe Verteilung aus. Er hat aufgehört hinzuschauen. Posterior Collapse.
> 
> Man liest sie also als **Aufteilung**: wie viel des Loss-Budgets geht in Genauigkeit, wie viel in Ordnung.
> 
> **Ein Rechenbeispiel.** `KL = 4.91 nats`. Geteilt durch $\ln 2$ sind das **7.1 Bit** — die gesamte Information, die der Encoder pro Bild durch den Flaschenhals schiebt. Zum Vergleich: die Ziffernidentität allein wäre $\log_2 10 = 3.3$ Bit wert, die bekommt der Decoder aber über $y$ geschenkt. Was in diesen 7.1 Bit steckt, ist **alles außer der Identität**: Neigung, Strichstärke, Rundung. Der Stil einer handgeschriebenen Ziffer, quantifiziert.

**Häufige Verwechslung:** Der _Abstand_ zwischen Train- und Val-Loss misst Overfitting, nicht Modellgüte. Ein Modell, das gar nichts gelernt hat, hat Abstand null. Der beste Fit ist schlicht der **niedrigste Val-Loss**.

Zwei Fallstricke beim Vergleich der beiden Kurven: (a) Bis $\beta = 1$ erreicht ist, rechnen Train und Val mit unterschiedlichem $\beta$ und sind gar nicht vergleichbar. (b) Der Train-Loss ist ein Mittelwert über eine Epoche, _während_ sich die Gewichte verbessern — er ist systematisch pessimistisch. Ein wenig Abstand ist normal, auch ohne Overfitting.

### 7.2 Rekonstruktionen

```python
model.eval()
xb, yb = next(iter(val_loader))
x, y = prepare(xb[:10].to(device), yb[:10].to(device))

with torch.no_grad():
    logits, mu, logvar = model(x, y)
    recon = torch.sigmoid(logits).view(-1, 28, 28).cpu()

fig, axes = plt.subplots(2, 10, figsize=(13, 3))
for i in range(10):
    axes[0, i].imshow(xb[i].squeeze(), cmap="gray")
    axes[0, i].set_title(str(yb[i].item()), fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(recon[i], cmap="gray")
    axes[1, i].axis("off")
plt.tight_layout(); plt.show()
```

Die Kette `sigmoid → view → cpu`: Logits in Wahrscheinlichkeiten umrechnen, von 784 zurück in Bildform, von der GPU in den RAM (matplotlib kann mit einem MPS-Tensor nichts anfangen).

`xb[i].squeeze()` hat den `squeeze`, `recon[i]` nicht: `xb` kommt roh aus dem DataLoader und trägt die Kanalachse `(1,28,28)`; `recon` wurde per `.view(-1,28,28)` selbst geformt.

> **Zu Subplot-Gittern:** `plt.subplots(2, 10)` liefert ein NumPy-Array der Form `(2,10)`. Zugriff wie bei einer Matrix — `axes[zeile, spalte]`. Achtung: bei `subplots(1, 10)` ist es _flach_, dann `axes[i]`. Im Zweifel `print(axes.shape)`.

**Ergebnis:** Ziffern werden erkennbar rekonstruiert, aber unscharf. Neigung und Strichführung bleiben erhalten. Bei ungewöhnlichen Exemplaren (eine stark verschnörkelte 5) bricht es zusammen und das Modell fällt auf eine Durchschnittsform zurück — zwei Zahlen reichen nicht, um Ausreißer zu beschreiben.

Dass Form und Ausrichtung erhalten bleiben, ist genau das erwartete Verhalten: **es ist das Einzige, was $z$ überhaupt speichert.**

### 7.3 Der Latentraum

```python
mus, labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        x, y = prepare(xb.to(device), yb.to(device))
        mu, _ = model.encode(x, y)          # nur der Encoder, Nebeneingang
        mus.append(mu.cpu()); labels.append(yb)

mus    = torch.cat(mus).numpy()
labels = torch.cat(labels).numpy()

fig, ax = plt.subplots(figsize=(7, 6.5))
sc = ax.scatter(mus[:, 0], mus[:, 1], c=labels, cmap="tab10", s=4, alpha=.6)

th = np.linspace(0, 2*np.pi, 200)
for r, ls in [(1, "-"), (2, "--")]:         # 1σ- und 2σ-Kreis des Priors
    ax.plot(r*np.cos(th), r*np.sin(th), "k", ls=ls, lw=1, alpha=.5)

ax.set_aspect("equal"); plt.colorbar(sc, ticks=range(10)); plt.show()

print("E[μ] =", mus.mean(0).round(3))   # Ziel: [0, 0]
print("Std  =", mus.std(0).round(3))    # Ziel: [1, 1]
```

Hier wird zum ersten Mal `model.encode(...)` direkt aufgerufen statt `model(...)` — der Nebeneingang aus Abschnitt 2.2. Der Unterstrich bei `mu, _ =` wirft `logvar` weg; Konvention für „interessiert mich nicht".

`for r, ls in [(1, "-"), (2, "--")]` ist Tupel-Auspacken in einer Schleife: erst `r=1, ls="-"`, dann `r=2, ls="--"`. Zwei Namen links, zwei Elemente pro Tupel rechts. Die Kreise kommen aus der Parametrisierung $(\cos\theta,\sin\theta)$ und markieren, wo bei einer 2D-Standardnormalverteilung 39 % bzw. 86 % der Masse innerhalb liegen.

**Typisches Ergebnis:** `E[μ] = [0.025, -0.017]`, `Std = [1.039, 1.048]`. Die Wolke sitzt auf dem Prior — Voraussetzung fürs Sampling erfüllt.

> ### Kasten: Warum die Klassenfarben durchmischt sein _sollen_
> 
> Bei einem gewöhnlichen VAE wäre eine durchmischte Wolke ein Alarmsignal — dort sollen sich Klassencluster bilden, und dieser Plot wird gern als Beweis gezeigt, dass das Modell die Ziffern „verstanden" hat.
> 
> Beim CVAE ist die Durchmischung der Beweis, dass es funktioniert. Der Decoder bekommt $y$ ohnehin über den zweiten Eingang. Die Klasse zusätzlich in $z$ zu speichern, wäre Verschwendung — und der KL-Term bestraft jedes gespeicherte Bit.
> 
> **Die Farbmischung _ist_ die Arbeitsteilung, sichtbar gemacht.**
> 
> **Beobachtung zur Ziffer 1:** Einsen bilden oft einen schmalen Streifen statt einer Wolke. Grund: der Stilraum einer 1 ist niedrigdimensional — im Wesentlichen nur die Neigung. Eine 2 hat mehrere unabhängige Achsen (Schleife rund oder eckig, Schlaufe unten oder nicht, Breite). Der Stilraum der 1 _ist_ fast eine Linie.
> 
> Zwei Einschränkungen: (a) Der Prior gilt nur im **Aggregat** — die Gesamtwolke muss rund sein, einzelne Klassen dürfen Struktur haben. (b) **Welche Achse was bedeutet, ist Zufall.** Es gibt keinen Grund, warum ausgerechnet $z_2$ die Neigung tragen sollte; ein anderer Seed legt sie woanders hin. Der Latentraum hat keine kanonische Orientierung. Genau das versucht β-VAE zu erzwingen — auf Kosten der Bildqualität.

---

## 8. Generierung mit dem Decoder

### 8.1 Es läuft nichts rückwärts

Hier lohnt sich die Rückkehr zur Ausgangsfrage. Beim Klassifikator scheiterte die Umkehrung. Hier ist sie — fast enttäuschend — **schon eingebaut**.

`forward` besteht aus zwei Hälften:

$$\underbrace{x, y ;\to; z}_{\text{Encoder}} \qquad\qquad \underbrace{z, y ;\to; \text{Bild}}_{\text{Decoder}}$$

Der Decoder **ist** die Abbildung von einer kurzen Beschreibung auf ein Bild. Er kann das, weil er genau dafür trainiert wurde. Nichts muss invertiert werden, keine Gradienten rückwärts laufen, keine Gewichte transponiert.

Man muss nur **aufhören, den Encoder zu benutzen.**

Bisher kam $z$ immer vom Encoder — der hat ein echtes Bild angeschaut. Aber der Decoder fragt nicht nach, woher sein $z$ kommt. Er nimmt zwei Zahlen und malt.

**Und woher nimmt man diese Zahlen ohne Bild? Man würfelt sie.** Genau dafür war der KL-Term da: er hat den Latentraum so zusammengeschoben, dass er $\mathcal{N}(0,I)$ ähnelt. Deshalb ist ein zufällig gezogenes $z$ mit hoher Wahrscheinlichkeit ein Punkt, an dem der Decoder etwas Sinnvolles malt.

### 8.2 Implementierung

```python
@torch.no_grad()
def generate(model, digit, n=8, z=None):
    """Erzeugt n Bilder der Ziffer `digit`. z=None -> aus dem Prior würfeln."""
    model.eval()
    y = F.one_hot(torch.full((n,), digit, dtype=torch.long),
                  model.y_dim).float().to(device)
    if z is None:
        z = torch.randn(n, model.z_dim, device=device)
    return torch.sigmoid(model.decode(z, y)).view(n, 28, 28).cpu()
```

`torch.full((n,), 7)` erzeugt `tensor([7,7,7,...])` — `n` Kopien derselben Ziffer, die dann one-hot zu `(n, 10)` werden, `n`-mal dieselbe Zeile.

### 8.3 Vier Experimente

**(a) Freie Exemplare.** Für jede Ziffer zehn gewürfelte Stile:

```python
fig, axes = plt.subplots(10, 10, figsize=(10, 10.5))
for d in range(10):
    imgs = generate(model, d, n=10)
    for j in range(10):
        axes[d, j].imshow(imgs[j], cmap="gray"); axes[d, j].axis("off")
plt.tight_layout(); plt.show()
```

Keines dieser hundert Bilder existiert im Datensatz. Sie sind nicht abgerufen, nicht interpoliert — der Decoder hat sie aus zwei Zufallszahlen und einem One-Hot-Vektor gemalt.

**(b) Stil vs. Identität.** Dasselbe $z$, alle zehn Klassen:

```python
fig, axes = plt.subplots(6, 10, figsize=(10, 6.6))
for row in range(6):
    z_fix = torch.randn(1, model.z_dim, device=device)
    for d in range(10):
        axes[row, d].imshow(generate(model, d, n=1, z=z_fix)[0], cmap="gray")
        axes[row, d].axis("off")
plt.tight_layout(); plt.show()
```

Innerhalb einer Zeile ändert sich nur $y$ — und trotzdem teilen sich alle zehn Ziffern sichtbar eine Neigung und eine Strichstärke. **Das Modell hat einen Stilbegriff gelernt, den niemand hineinprogrammiert hat.** Er ist nur der Rest, der übrig blieb, nachdem der KL-Term alles Redundante herausgedrückt hatte.

**(c) Der Prototyp bei $z = 0$**, verglichen mit dem naiven Pixel-Mittelwert:

```python
fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
zeros = torch.zeros(1, model.z_dim, device=device)
for d in range(10):
    axes[0, d].imshow(generate(model, d, n=1, z=zeros)[0], cmap="gray")
    axes[0, d].axis("off")
    mask = raw_train.targets == d
    axes[1, d].imshow(raw_train.data[mask].float().mean(0)/255.0, cmap="gray")
    axes[1, d].axis("off")
plt.tight_layout(); plt.show()
```

**(d) Der Latentraum als Landkarte** (nur bei `z_dim = 2`):

```python
from torch.distributions import Normal

DIGIT, GRID = 7, 13
ppf = Normal(0., 1.).icdf(torch.linspace(0.02, 0.98, GRID))

canvas = np.zeros((GRID*28, GRID*28))
for i, zy in enumerate(reversed(ppf.tolist())):
    for j, zx in enumerate(ppf.tolist()):
        z = torch.tensor([[zx, zy]], dtype=torch.float32, device=device)
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = generate(model, DIGIT, n=1, z=z)[0].numpy()

plt.figure(figsize=(8.5, 8.5)); plt.imshow(canvas, cmap="gray"); plt.show()
```

**Warum Quantile (`icdf`) statt `linspace`?** Der Latentraum ist normalverteilt, nicht gleichverteilt. Ein `linspace(-3,3)` würde die dicht besetzte Mitte unterabtasten und in den Rändern lauter Punkte zeigen, die im Training nie vorkamen. Die Quantilfunktion sorgt dafür, dass jede Kachel dieselbe **Wahrscheinlichkeitsmasse** repräsentiert — man schaut sich den Raum so an, wie das Modell ihn erlebt hat.

Die Karte ist stetig: benachbarte Kacheln sehen ähnlich aus, nirgends kippt es abrupt. Das ist die Löcherfreiheit, die der KL-Term erzwungen hat — und der Grund, warum Sampling funktioniert.

---

## 9. Warum die Ergebnisse aussehen, wie sie aussehen

Die erzeugten Ziffern sind weich, wattig, an den Rändern ausgefranst. Das ist kein Trainingsfehler. Es folgt aus drei Ursachen, von denen zwei strukturell sind.

### 9.1 Der Mittelungsdruck des Rekonstruktionsterms

Der Rekonstruktionsterm ist ein **Erwartungswert** über $q(z \mid x, y)$. Und der optimale Punktschätzer unter einem mittleren Fehler ist immer ein **Mittelwert**.

Konkret: gegeben ein $z$ minimiert der Decoder die erwartete BCE, indem er für jedes Pixel die bedingte Wahrscheinlichkeit ausgibt:

$$\hat{x}_i^{,\star} = p(x_i = 1 \mid z, y) = \mathbb{E}[x_i \mid z, y]$$

Wo das Modell unsicher ist, ob ein Strich links oder rechts verläuft, ist der erwartungsminimierende Wert eben _beides halb_. Ein grauer, verwaschener Bereich ist nicht die Kapitulation des Modells — er ist seine **korrekte Antwort** auf die gestellte Frage.

### 9.2 Die faktorisierte Likelihood

Die Bernoulli-Annahme setzt voraus, dass die Pixel **gegeben $z$ unabhängig** sind:

$$p_\theta(x \mid z,y) = \prod_i p_\theta(x_i \mid z,y)$$

Der Decoder kann also keine Korrelationen zwischen Pixeln modellieren, die nicht schon in $z$ stehen. Er kann nicht sagen „entweder ist der ganze Strich links **oder** ganz rechts" — er entscheidet über jedes Pixel einzeln. Ein solches Modell _kann_ keine scharfen Alternativen darstellen; es kann nur hedgen.

Genau hier liegt der Unterschied zu GANs (die einen Diskriminator dazu zwingen, dass das Gesamtbild plausibel aussieht) und zu Diffusionsmodellen (die in vielen kleinen Schritten arbeiten und Korrelationen dabei aufbauen).

### 9.3 Der Flaschenhals

Bei `z_dim = 2` müssen zwei Zahlen ein 784-Pixel-Bild tragen. Alles, was nicht in 7 Bit passt, ist weg. Erhöht man auf `z_dim = 16`, werden die Bilder sichtbar schärfer — die KL steigt entsprechend, weil mehr Information durchgeschoben wird.

Das ist die einzige der drei Ursachen, die man durch Umkonfigurieren beheben kann. Die anderen beiden sind in der Zielfunktion verankert.

### 9.4 Geisterziffern: zwei Arten von Mittelwert

Der Vergleich aus Experiment (c) ist die Pointe des ganzen Projekts.

**Untere Zeile — Mittelwert im Pixelraum.** Der naive Durchschnitt aller Trainings-7en. Ein verwaschener Geist: Striche an verschiedenen Positionen löschen sich gegenseitig aus. Man sieht die _Überlagerung_ von tausend Siebenen, nicht eine Sieben.

**Obere Zeile — Mittelwert im Stilraum.** Der Decoder bei $z = 0$, dem Modus des Priors. Deutlich klarer und strichhafter, weil das Modell erst die Variation herausrechnet und dann **eine** Ziffer malt.

Das ist die Antwort auf die Ausgangsfrage aus Abschnitt 1. Ein „Mittelwert der 7" existiert — aber nur, wenn ein Modell vorher gelernt hat, **worüber** gemittelt werden soll. Der Klassifikator konnte das nicht liefern, weil er nie eine Repräsentation von „7-Sein" aufgebaut hat, sondern nur eine Grenze gegen alles Nicht-7.

Und selbst die obere Zeile ist noch geisterhaft — aus den Gründen 9.1 und 9.2. Ein VAE mittelt immer irgendwo. Er tut es nur an der richtigen Stelle.

### 9.5 Wohin von hier

|Modellklasse|Kernidee|Einstieg|
|---|---|---|
|**VQ-VAE**|diskreter Latentraum statt Gauß, kein Blur|[van den Oord et al. 2017](https://arxiv.org/abs/1711.00937)|
|**Normalizing Flows**|exakte Likelihood durch invertierbare Abbildungen|[Rezende & Mohamed 2015](https://arxiv.org/abs/1505.05770)|
|**Diffusion**|iteratives Entrauschen; heutiger State of the Art|[Ho et al. 2020](https://arxiv.org/abs/2006.11239)|
|**Conditional Diffusion**|Klassensteuerung ohne Klassifikator|[Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)|

Der rote Faden bleibt derselbe: **irgendwo muss $p(x)$ herkommen.** VAEs lernen ihn implizit über einen Flaschenhals, Flows explizit über eine invertierbare Abbildung, Diffusionsmodelle über seinen Gradienten (die _Score-Funktion_). Diskriminative Modelle lernen ihn gar nicht — und deshalb kann man sie nicht umkehren.

---

## 10. Anhang

### 10.1 Fehlerkatalog

Nach Tückengrad sortiert. Die ersten drei werfen keine Fehlermeldung.

|Fehler|Symptom|Ursache|
|---|---|---|
|`torch.rand_like` statt `randn_like`|Training läuft, Samples schlecht, KL zu niedrig|uniform statt normal|
|`mu.exp()` statt `mu.pow(2)` in der KL|läuft, trainiert Unsinn|Formel falsch übersetzt|
|Trainingszelle in Jupyter erneut ausführen|„Epoche 0" startet bei niedrigem Loss|trainiert auf alten Gewichten weiter|
|`opt.zero_grad()` vergessen|Loss explodiert|PyTorch addiert Gradienten|
|`torch.cuda.is_available` ohne `()`|`device` immer `cuda`, später Crash|Funktionsobjekt ist truthy|
|`plt.show` ohne `()`|Ausgabe zeigt `<function ...>`|dito|
|`dim=1` außerhalb der `cat`-Klammer|_„Sizes must match except in dimension 0"_|entlang der Batch-Achse konkateniert|
|`torch.cat(z, y, dim=1)`|TypeError|`cat` braucht eine **Liste**|
|NumPy im `nn.Module`|_„Can't call numpy() on Tensor that requires grad"_|Autograd bricht ab|
|`.to(device)` vergessen|_„expected all tensors on the same device"_|Daten auf CPU, Modell auf GPU|
|Sigmoid im Decoder **und** `_with_logits`|Loss stagniert, Bilder grau|doppelt gequetscht|
|`.cpu()` beim Plotten vergessen|matplotlib-Fehler|MPS/CUDA-Tensor|

**Der wichtigste Reflex:** Steht in der Ausgabe irgendwo `<function ...>`, wurde eine Funktion _erwähnt_ statt _aufgerufen_.

### 10.2 Glossar

|Begriff|Bedeutung|
|---|---|
|**Batch**|Portion von $B$ Samples, die gemeinsam durchs Netz geht|
|**Epoche**|ein vollständiger Durchlauf durch alle Trainingsdaten|
|**Autograd-Graph**|von PyTorch mitgeschriebenes Protokoll des Forward-Passes, Grundlage von `backward()`|
|**Logit**|Rohausgabe vor der Sigmoid/Softmax-Umrechnung, Wertebereich $\mathbb{R}$|
|**Latentraum**|Raum der komprimierten Codes $z$|
|**Prior** $p(z)$|angenommene Verteilung der Codes, hier $\mathcal{N}(0,I)$|
|**Posterior** $p(z\mid x)$|wahre Verteilung der Codes gegeben ein Bild — unbekannt|
|**Variational Posterior** $q_\phi(z\mid x)$|die Näherung, die der Encoder ausgibt|
|**ELBO**|Evidence Lower Bound; berechenbare Untergrenze für $\log p(x)$|
|**nat**|Informationseinheit bei natürlichem Log; $1\ \text{nat} = 1/\ln 2 \approx 1.44$ Bit|
|**Posterior Collapse**|$q_\phi$ ignoriert das Bild, KL $\to 0$|
|**Reparametrisierung**|$z = \mu + \sigma\varepsilon$, macht Sampling differenzierbar|

### 10.3 Nachbau-Checkliste

Für den Neuaufbau aus dem Kopf, in dieser Reihenfolge:

1. **Daten** — `datasets.MNIST` mit `ToTensor()`, Shapes und Wertebereich drucken
2. **`prepare`** — flatten + one-hot + float. Assert auf Shapes und `argmax`
3. **Modell** — `__init__` (Bauteile), `encode`, `reparameterize`, `decode`, `forward`. Assert auf Shapes, Stochastizität und **Gradientenfluss**
4. **Loss** — BCE-with-logits summiert / $B$, plus KL in geschlossener Form. Assert auf die analytischen Fälle KL$(\mu{=}0)=0$ und KL$(\mu{=}1)=1$
5. **Split** — `random_split` mit eigenem Generator, drei DataLoader
6. **Training** — `run_epoch` mit Schalter, β-Warm-up, erst 2 Epochen zum Test
7. **Validierung** — Kurven, Rekonstruktionen, Latentraum-Scatter mit `E[μ]`/`Std`
8. **Generierung** — `generate`, dann die vier Experimente

Wenn Schritt 3 und 4 mit ihren Asserts stehen, ist der Rest Handwerk.

### 10.4 Übungen zur Vertiefung

1. **`z_dim = 16`.** Schärfere Bilder, höhere KL. Bei `z_dim = 64`: steigt die KL proportional weiter? (Nein — warum?)
2. **Posterior Collapse provozieren.** `WARMUP_EP = 1`, `z_dim = 32`. KL beobachten.
3. **β fest auf 0.2 bzw. 5.0.** Der Trade-off heißt _rate–distortion_ ([Alemi et al. 2018](https://arxiv.org/abs/1711.00464)).
4. **`nn.Embedding(10, 16)` statt One-Hot.** Warum ist der Unterschied bei 10 Klassen klein, bei 10.000 entscheidend?
5. **Conditioning weglassen.** Jetzt _sollten_ sich Klassencluster bilden — der sauberste Beweis für die Stil/Identität-Trennung.
6. **CNN statt MLP.** `Conv2d` im Encoder, `ConvTranspose2d` im Decoder, $y$ als konstanter Extra-Kanal.
7. **$y$ interpolieren.** `0.5*onehot(3) + 0.5*onehot(8)` — was kommt raus, und was sagt das über die Geometrie des Conditioning-Inputs?
8. **Der Kreisschluss.** Einen Klassifikator trainieren, per Gradientenaufstieg eine „7" halluzinieren lassen, und das Ergebnis neben die CVAE-Samples stellen.

### 10.5 Literatur

- Kingma & Welling (2013), _Auto-Encoding Variational Bayes_ — [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Sohn, Lee & Yan (2015), _Learning Structured Output Representation using Deep Conditional Generative Models_ — [NeurIPS](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html)
- Doersch (2016), _Tutorial on Variational Autoencoders_ — [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)
- Kingma & Welling (2019), _An Introduction to Variational Autoencoders_ — [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)
- Bowman et al. (2016), _Generating Sentences from a Continuous Space_ — [arXiv:1511.06349](https://arxiv.org/abs/1511.06349)
- Higgins et al. (2017), _β-VAE_ — [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
- Alemi et al. (2018), _Fixing a Broken ELBO_ — [arXiv:1711.00464](https://arxiv.org/abs/1711.00464)
- Loaiza-Ganem & Cunningham (2019), _The Continuous Bernoulli_ — [arXiv:1907.06845](https://arxiv.org/abs/1907.06845)
- Nguyen, Yosinski & Clune (2015), _Deep Neural Networks are Easily Fooled_ — [arXiv:1412.1897](https://arxiv.org/abs/1412.1897)
- Simonyan, Vedaldi & Zisserman (2014), _Deep Inside Convolutional Networks_ — [arXiv:1312.6034](https://arxiv.org/abs/1312.6034)
- Olah, Mordvintsev & Schubert (2017), _Feature Visualization_ — [Distill](https://distill.pub/2017/feature-visualization/)

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]