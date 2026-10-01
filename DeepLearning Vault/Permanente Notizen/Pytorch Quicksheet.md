02-09-2026
Tags: #FuE #MachineLearning 
Status: #unextended



> Konvention: `x`, `y` sind Tensoren, `B` = Batchgröße, `D` = Feature-Dimension. Stand: PyTorch 2.x

---

## 1. Tensoren

### 1.1 Erzeugen

```python
import torch

torch.tensor([1., 2., 3.])          # aus Python-Daten (kopiert!)
torch.zeros(B, D); torch.ones(B, D)
torch.empty(B, D)                    # uninitialisiert, schnell
torch.full((B, D), 3.14)
torch.eye(D)                         # Einheitsmatrix
torch.arange(0, 10, 2)               # [0,2,4,6,8]  (Ende exklusiv)
torch.linspace(0, 1, steps=5)        # Ende inklusiv
torch.randn(B, D)                    # N(0,1)
torch.rand(B, D)                     # U[0,1)
torch.randint(0, 10, (B,))
torch.randperm(n)                    # zufällige Permutation

torch.zeros_like(x); torch.randn_like(x)   # übernimmt shape, dtype, device
```

`*_like` ist fast immer besser als `torch.zeros(x.shape)` — es erbt **dtype und device** und erspart dir Device-Mismatch-Fehler.

### 1.2 dtype, device, Umwandlung

```python
x.dtype, x.device, x.shape, x.ndim, x.numel()

x.float()          # -> float32
x.double()         # -> float64
x.long()           # -> int64 (Indizes, Labels)
x.bool()
x.to(torch.float32)
x.to("cuda"); x.cuda(); x.cpu()
x.to(device="cuda", dtype=torch.float16)

dev = "cuda" if torch.cuda.is_available() else "cpu"
```

Faustregeln:

- Labels für `CrossEntropyLoss` müssen `long` sein.
- Masken sind `bool`.
- `.to()` ist ein No-Op (gibt denselben Tensor zurück), wenn dtype/device schon stimmen.

### 1.3 NumPy-Interop

```python
t = torch.from_numpy(arr)   # teilt Speicher! Änderung wirkt in beide Richtungen
a = t.numpy()               # teilt Speicher, nur CPU + kein grad
a = t.detach().cpu().numpy()  # der sichere Standardweg
```

### 1.4 Shape-Operationen

```python
x.view(B, -1)        # nur auf contiguous Tensoren; teilt Speicher
x.reshape(B, -1)     # kopiert falls nötig; der robuste Default
x.flatten(start_dim=1)
x.squeeze(0)         # entfernt Dim der Größe 1
x.unsqueeze(1)       # fügt Dim der Größe 1 ein   -> (B,D) => (B,1,D)
x[:, None, :]        # dasselbe, kürzer
x.permute(0, 2, 1)   # beliebige Achsen-Umordnung
x.transpose(1, 2)    # genau zwei Achsen tauschen
x.T                  # nur 2D
x.expand(B, 5, D)    # Größe-1-Dims aufblasen, OHNE Kopie (read-only Sicht)
x.repeat(2, 3)       # echte Kopie, Kacheln
x.contiguous()       # Speicherlayout begradigen (nach permute nötig für view)

torch.cat([a, b], dim=0)     # entlang bestehender Achse
torch.stack([a, b], dim=0)   # NEUE Achse
torch.split(x, 4, dim=0); torch.chunk(x, 3, dim=0)
```

Mentales Modell: `view/permute/expand/transpose` ändern nur die **Sichtweise** (Strides) auf denselben Speicher; `reshape/repeat/contiguous/cat` können **kopieren**.

### 1.5 Indexing & Masken

```python
x[0]; x[:, 1]; x[..., -1]          # ... = "alle übrigen Achsen"
mask = x > 0
x[mask]                            # 1D-Ergebnis, Länge unbekannt
torch.where(mask, x, torch.zeros_like(x))   # shape-erhaltend
x.masked_fill(mask, float("-inf"))          # z.B. Attention-Masking

idx = torch.tensor([0, 2, 2])
x[idx]                             # fancy indexing entlang dim 0
torch.index_select(x, 0, idx)
torch.gather(x, dim=1, index=idx)  # elementweises Einsammeln
x.scatter_(dim=1, index=idx, src=v)  # Gegenstück: Verteilen
torch.take_along_dim(x, idx, dim=1)
```

### 1.6 Reduktionen

```python
x.sum(); x.sum(dim=1); x.sum(dim=1, keepdim=True)
x.mean(dim=0); x.std(dim=0, unbiased=True); x.var()
x.min(dim=1); x.max(dim=1)       # gibt (values, indices) zurück
x.argmin(); x.argmax(dim=1)
x.prod(); x.cumsum(dim=0); x.cumprod(dim=0)
x.norm(dim=1); torch.linalg.norm(x, ord=2, dim=1)
x.topk(k=5, dim=1)
x.any(); x.all(); mask.sum()      # bool -> Anzahl True
```

`keepdim=True` ist der Trick fürs Broadcasting: `x - x.mean(dim=1, keepdim=True)` funktioniert, ohne `keepdim` kracht es bei nicht-quadratischen Shapes.

### 1.7 Broadcasting

Regel, von rechts nach links gelesen: zwei Dimensionen passen zusammen, wenn sie **gleich** sind oder eine davon **1** ist. Fehlende Achsen links werden als 1 ergänzt.

```
(B, 1, D)  +  (   N, D)   ->  (B, N, D)
(B, D)     +  (   D,  )   ->  (B, D)
(B, D)     +  (B, 1  )    ->  (B, D)
```

Eine Größe-1-Achse heißt logisch: _"dieser Wert gilt für alle Einträge dieser Achse"_. Deshalb ist `x[:, None, :] - y[None, :, :]` die paarweise Differenz aller `x` gegen alle `y` — Shape `(B, N, D)`.

### 1.8 Mathematik & lineare Algebra

```python
a + b; a * b; a / b; a ** 2       # alle elementweise
a @ b                             # Matmul / Batch-Matmul
torch.matmul(a, b); torch.bmm(a, b)   # bmm: strikt (B,n,m)@(B,m,p)
torch.einsum("bij,bjk->bik", a, b)
torch.outer(u, v); torch.dot(u, v)

torch.exp(x); torch.log(x); torch.log1p(x); torch.expm1(x)
torch.sqrt(x); torch.abs(x); torch.sign(x)
torch.clamp(x, min=0, max=1)
torch.round(x); torch.floor(x)

torch.linalg.inv(A); torch.linalg.solve(A, b)      # solve statt inv @ b!
torch.linalg.cholesky(A)                            # A = L Lᵀ, A spd
torch.cholesky_solve(b, L)
torch.linalg.eigh(A); torch.linalg.svd(A)
torch.linalg.slogdet(A)                             # (sign, log|det|), stabil
torch.cdist(x, y, p=2)                              # paarweise Distanzen
```

Numerisch: nie `log(sum(exp(...)))` selbst schreiben, sondern `torch.logsumexp`. Nie `inv(A) @ b`, sondern `solve(A, b)`. Bei Kovarianzmatrizen mit Cholesky arbeiten — `slogdet` über `L` ist stabiler und billiger.

### 1.9 In-place

```python
x.add_(1); x.mul_(2); x.zero_(); x.clamp_(0)   # Unterstrich = in-place
```

Spart Speicher, kann aber Autograd zerstören ("a variable needed for gradient computation has been modified"). Im Zweifel: keine In-place-Ops auf Tensoren, die `requires_grad=True` haben oder in den Graphen eingehen.

### 1.10 Autograd

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()          # füllt x.grad
x.grad

with torch.no_grad():        # kein Graph -> Inferenz, Parameter-Updates
    ...
with torch.inference_mode(): # noch strikter/schneller als no_grad
    ...

x.detach()            # Tensor aus dem Graphen lösen
x.requires_grad_(True)
loss.backward(retain_graph=True)   # nur wenn du wirklich zweimal rückwärts musst

# Gradienten von Funktionen (z.B. für PINN-Residuen):
g, = torch.autograd.grad(y, x, create_graph=True)   # create_graph -> 2. Ableitung möglich
```

`backward()` akkumuliert Gradienten. Deshalb `optimizer.zero_grad()` in jedem Schritt.

### 1.11 Reproduzierbarkeit

```python
torch.manual_seed(0)
g = torch.Generator().manual_seed(0)
torch.randn(3, generator=g)
torch.use_deterministic_algorithms(True)
```

---

## 2. Neural Networks (`torch.nn`)

### 2.1 Eigenes Modul

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.1):
        super().__init__()                     # nie vergessen
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(1280, 512, 1).to(dev)
model(x)          # NIE model.forward(x) — sonst laufen Hooks nicht
```

Alles, was du als Attribut ein `nn.Module`, `nn.Parameter` oder in `nn.ModuleList`/`nn.ParameterList` legst, wird automatisch mitregistriert (taucht in `.parameters()` auf, wandert bei `.to(dev)` mit). Eine normale Python-Liste von Modulen wird **nicht** registriert.

```python
self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(4)])   # richtig
self.log_sigma = nn.Parameter(torch.zeros(1))                       # lernbarer Skalar
self.register_buffer("prior_prec", torch.tensor(1.0))               # nicht lernbar,
                                                                    # aber in state_dict & auf device
```

### 2.2 Wichtige Layer

|Zweck|Layer|
|---|---|
|Dense|`nn.Linear(in, out, bias=True)`|
|Embedding|`nn.Embedding(num, dim, padding_idx=0)`|
|Conv|`nn.Conv1d/2d(in_ch, out_ch, kernel_size, stride, padding, dilation)`|
|Pooling|`nn.MaxPool1d`, `nn.AvgPool1d`, `nn.AdaptiveAvgPool1d(1)`|
|Normalisierung|`nn.BatchNorm1d`, `nn.LayerNorm(d)`, `nn.GroupNorm`|
|Regularisierung|`nn.Dropout(p)`, `nn.Dropout1d`|
|Rekurrent|`nn.LSTM`, `nn.GRU` (`batch_first=True` setzen!)|
|Attention|`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`|
|Container|`nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`|

Aktivierungen: `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.ELU`, `nn.Tanh`, `nn.Sigmoid`, `nn.Softplus`.

`LayerNorm` normalisiert pro Sample über die Feature-Achse, `BatchNorm1d` pro Feature über den Batch. Bei kleinen oder stark variierenden Batches (typisch bei Proteinsequenzen) ist `LayerNorm` robuster.

### 2.3 Losses

```python
nn.MSELoss(reduction="mean")     # Regression
nn.L1Loss(); nn.HuberLoss(delta=1.0)     # robuster gegen Ausreißer
nn.SmoothL1Loss()
nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)  # erwartet LOGITS + long-Labels
nn.BCEWithLogitsLoss(pos_weight=pw)                 # erwartet LOGITS
nn.NLLLoss()                                        # erwartet log_softmax
nn.GaussianNLLLoss()                                # heteroskedastische Regression
nn.KLDivLoss(reduction="batchmean")                 # Input: log-probs
```

`*WithLogits` immer der Variante mit vorgeschaltetem `sigmoid`/`softmax` vorziehen — intern log-sum-exp-stabilisiert.

### 2.4 Initialisierung

```python
nn.init.xavier_uniform_(m.weight)      # für tanh/sigmoid
nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
nn.init.zeros_(m.bias)
nn.init.normal_(m.weight, std=0.02)

def init_fn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_fn)
```

### 2.5 Optimizer & Scheduler

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
torch.optim.Adam, SGD(momentum=0.9, nesterov=True), RMSprop

# Parametergruppen: z.B. kein weight decay auf Bias/Norm
groups = [
    {"params": decay_params,    "weight_decay": 1e-2},
    {"params": no_decay_params, "weight_decay": 0.0},
]
opt = torch.optim.AdamW(groups, lr=1e-3)

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        # ReduceLROnPlateau(opt, patience=5)  -> sched.step(val_loss)
        # OneCycleLR(opt, max_lr=1e-3, total_steps=...)  -> sched.step() pro BATCH
```

### 2.6 Trainingsschleife

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
    sched.step()

    model.eval()
    with torch.no_grad():
        val = sum(criterion(model(xb.to(dev)), yb.to(dev)).item() * len(xb)
                  for xb, yb in val_loader) / len(val_loader.dataset)
```

`model.train()` / `model.eval()` schalten **nur** Dropout und BatchNorm um — sie schalten Autograd nicht ab. Dafür brauchst du `no_grad()`. Beides zusammen ist der Standard für Validierung.

### 2.7 Daten

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

class SeqDataset(Dataset):
    def __init__(self, emb, y): self.emb, self.y = emb, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.emb[i], self.y[i]

loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True, collate_fn=None)
```

Für variable Sequenzlängen brauchst du eine eigene `collate_fn` mit `nn.utils.rnn.pad_sequence`.

### 2.8 Speichern & Laden

```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location=dev))

ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": e}
torch.save(ckpt, "ckpt.pt")
```

Immer `state_dict` speichern, nie das ganze Modellobjekt (pickle bricht bei Code-Änderungen).

### 2.9 Einfrieren, Nützliches

```python
for p in encoder.parameters():
    p.requires_grad = False              # z.B. ESM-2 als fixer Feature-Extractor

sum(p.numel() for p in model.parameters() if p.requires_grad)   # Parameterzahl
print(model)                                                     # Struktur
list(model.named_parameters())
```

---

## 3. Funktionen (`torch.nn.functional` & `torch.*`)

`nn.X` ist die Objekt-Variante mit Zustand/Parametern, `F.x` die zustandslose Funktion. Für Layer mit Gewichten nimmst du `nn.`, für reine Rechenoperationen `F.`.

```python
F.relu(x); F.gelu(x); F.silu(x); F.softplus(x)
F.softmax(x, dim=-1); F.log_softmax(x, dim=-1)
F.sigmoid(x)                     # meist besser: torch.sigmoid
F.normalize(x, p=2, dim=1)       # L2-Normalisierung pro Zeile
F.dropout(x, p=0.1, training=self.training)
F.linear(x, W, b)
F.mse_loss(pred, y); F.cross_entropy(logits, y); F.binary_cross_entropy_with_logits(z, y)
F.pad(x, (1, 1))
F.one_hot(idx, num_classes=10)
F.cosine_similarity(a, b, dim=-1)
F.scaled_dot_product_attention(q, k, v, is_causal=True)   # schnelle Attention
```

Numerisch stabile Helfer:

```python
torch.logsumexp(x, dim=-1)      # log Σ exp x, ohne Overflow
torch.log_softmax(x, dim=-1)
torch.nan_to_num(x, nan=0.0, posinf=1e4)
torch.isfinite(x); torch.isnan(x)
torch.clamp(x, min=1e-8)        # vor log/div
```

`logsumexp` liest sich als: _"weiche Maximum-Bildung im Log-Raum"_. Weil `log` Produkte in Summen verwandelt, ist es die Standardbrücke zwischen "Wahrscheinlichkeiten multiplizieren" und "Log-Likelihoods addieren" — deshalb taucht es überall auf, wo du über latente Zustände marginalisierst.

`einsum` als universelles Werkzeug:

```python
torch.einsum("bd,bd->b", a, b)        # Zeilenweises Skalarprodukt
torch.einsum("bij,bjk->bik", a, b)    # Batch-Matmul
torch.einsum("bhqd,bhkd->bhqk", q, k) # Attention-Scores
torch.einsum("bld->bd", x) / L        # Mittelung über Sequenzlänge
```

Lesart: jeder Buchstabe ist eine Achse. Buchstaben, die links vorkommen und rechts fehlen, werden **aufsummiert**. Alles andere wird elementweise ausgerichtet.

---

## 4. `torch.distributions`

Der Baukasten für alles Probabilistische: Likelihoods, Priors, Sampling mit Gradienten, KL-Divergenzen.

### 4.1 Grundmuster

```python
import torch.distributions as D

p = D.Normal(loc=mu, scale=sigma)     # sigma > 0 !
p.sample()                            # KEIN Gradient
p.rsample()                           # reparametrisiert -> Gradient fließt durch
p.log_prob(y)                         # elementweise Log-Dichte
p.mean, p.stddev, p.variance
p.entropy()
p.cdf(y); p.icdf(u)

p.batch_shape, p.event_shape          # unabhängige Verteilungen vs. Dimension EINES Ereignisses
```

Shape-Logik: `log_prob(y)` gibt einen Wert **pro Batch-Element**, nicht pro Event-Dimension. `Independent` verschiebt Achsen von `batch_shape` nach `event_shape` und summiert dadurch die Log-Dichten auf:

```python
p = D.Independent(D.Normal(mu, sigma), 1)   # (B, D) -> log_prob shape (B,)
```

### 4.2 Wichtige Verteilungen

```python
D.Normal(mu, sigma)
D.MultivariateNormal(mu, covariance_matrix=S)     # oder scale_tril=L (bevorzugt!)
D.LogNormal(mu, sigma)                            # positive Größen, z.B. kcat
D.StudentT(df, mu, sigma)                         # schwere Schwänze, robust
D.Laplace(mu, b)
D.Gamma(concentration, rate); D.InverseGamma(...)  # Priors auf Präzision/Varianz
D.HalfNormal(sigma); D.HalfCauchy(scale)           # Priors auf Skalenparameter
D.Beta(a, b); D.Dirichlet(alpha)
D.Bernoulli(logits=z); D.Categorical(logits=z)     # logits= statt probs= nutzen
D.Poisson(rate); D.NegativeBinomial(...)           # Counts (RNA-seq)
D.Uniform(a, b); D.Exponential(rate)
D.MixtureSameFamily(mix, comp)                     # Mischverteilungen
```

`scale_tril=L` statt `covariance_matrix=S`: du gibst den Cholesky-Faktor direkt, das ist stabiler und billiger, weil intern ohnehin zerlegt wird.

### 4.3 KL-Divergenz

```python
kl = D.kl_divergence(q, p).sum()      # analytisch, wo verfügbar (z.B. Normal||Normal)
```

Das ist der Regularisierungsterm im ELBO einer variationellen Näherung: du bezahlst dafür, dass dein Posterior `q` vom Prior `p` abweicht.

### 4.4 Transformierte Verteilungen

```python
base = D.Normal(0., 1.)
p = D.TransformedDistribution(base, [D.transforms.ExpTransform()])   # = LogNormal
D.transforms.AffineTransform(loc, scale)
D.transforms.SigmoidTransform(); D.transforms.SoftplusTransform()
```

`log_prob` korrigiert automatisch um den Log-Betrag der Jacobi-Determinante — das ist die Change-of-Variables-Formel und die Grundlage von Normalizing Flows.

### 4.5 Positivitäts-Constraints (praktisch wichtig)

Netze geben unbeschränkte Reals aus. Skalenparameter müssen positiv sein:

```python
sigma = F.softplus(raw) + 1e-6        # weich, stabil
sigma = torch.exp(log_sigma)          # üblich, aber empfindlich -> log_sigma clampen
log_sigma = log_sigma.clamp(-7, 7)
```

Vorteil, `log_sigma` als Parameter zu lernen: der Optimizer arbeitet auf einer unbeschränkten Skala, und multiplikative Änderungen der Varianz werden zu additiven Schritten.

### 4.6 Heteroskedastische Gaußsche NLL

Der Kopf sagt Mittelwert **und** Unsicherheit vorher. Die negative Log-Likelihood pro Datenpunkt:

$$ \mathcal{L} = \frac{1}{2}\log \sigma_i^2 ;+; \frac{(y_i - \mu_i)^2}{2\sigma_i^2} ;+; \text{const} $$

```latex
\mathcal{L} = \frac{1}{2}\log \sigma_i^2 \;+\; \frac{(y_i - \mu_i)^2}{2\sigma_i^2} \;+\; \text{const}
```

Als Logik gelesen: der zweite Term ist ein **präzisionsgewichteter** quadratischer Fehler — Punkte, denen das Modell hohe Unsicherheit zuschreibt, zählen weniger. Der erste Term ist die Strafe genau dafür; ohne ihn würde das Modell einfach `σ → ∞` setzen und den Fehlerterm wegdrücken. Die beiden Terme im Gleichgewicht ergeben eine kalibrierte Unsicherheit.

```python
mu, log_sigma = model(x).chunk(2, dim=-1)
log_sigma = log_sigma.clamp(-7, 7)
sigma = log_sigma.exp()

# Variante A: explizit
nll = (log_sigma + 0.5 * ((y - mu) / sigma) ** 2).mean()

# Variante B: über distributions (identisch bis auf Konstante)
nll = -D.Normal(mu, sigma).log_prob(y).mean()

# Variante C: eingebauter Loss (erwartet VARIANZ, nicht sigma)
nll = F.gaussian_nll_loss(mu, y, var=sigma ** 2)
```

### 4.7 Sampling mit Gradienten

```python
q = D.Normal(mu_q, sigma_q)
w = q.rsample()                     # differenzierbar: w = mu + sigma * eps
loss = -likelihood(w).mean() + D.kl_divergence(q, prior).sum()
```

`sample()` blockiert Gradienten, `rsample()` nutzt den Reparametrisierungstrick und lässt sie durch — für variationelle Inferenz brauchst du immer `rsample()`. Bei diskreten Verteilungen existiert `rsample` nicht; dort brauchst du REINFORCE (`log_prob`-Trick) oder Gumbel-Softmax (`D.RelaxedOneHotCategorical`).

---

## 5. Stolperfallen

|Symptom|Ursache|
|---|---|
|`Expected all tensors on same device`|Daten oder Buffer nicht `.to(dev)`|
|`Expected scalar type Long`|Labels sind float statt `long`|
|`view size is not compatible`|nach `permute` fehlt `.contiguous()`, oder nimm `reshape`|
|Loss wird `nan`|`log(0)`, Division durch σ≈0, zu hohe LR, fehlendes Grad-Clipping|
|Val-Loss unrealistisch gut|`model.eval()` vergessen (Dropout aktiv) oder Leakage in der Normalisierung|
|Gradienten wachsen über Epochen|`optimizer.zero_grad()` fehlt|
|Speicher läuft voll|`loss` statt `loss.item()` in einer Liste akkumuliert → Graph bleibt am Leben|
|`modified by an inplace operation`|In-place-Op auf einem Tensor im Autograd-Graph|

---

## 6. Referenzen

- Tensor-API: https://pytorch.org/docs/stable/tensors.html
- `torch.nn`: https://pytorch.org/docs/stable/nn.html
- `torch.nn.functional`: https://pytorch.org/docs/stable/nn.functional.html
- `torch.distributions`: https://pytorch.org/docs/stable/distributions.html
- `torch.linalg`: https://pytorch.org/docs/stable/linalg.html
- Broadcasting-Semantik: https://pytorch.org/docs/stable/notes/broadcasting.html
- Autograd-Mechanik: https://pytorch.org/docs/stable/notes/autograd.html
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]