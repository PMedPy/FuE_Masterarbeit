20-08-2026
Tags: #Python  #FuE 
Status: #unextended

# PyTorch – Das Grundgerüst

> [!note] Zweck dieses Dokuments Wiederkehrendes Skelett für PyTorch-Projekte, illustriert am Iris-Datensatz. Dient als Ausgangspunkt, um später fremden Code (Hugging Face Trainer, PyTorch Lightning, etc.) als Variation desselben Musters zu erkennen — nicht als fertiges Rezept für jedes Problem.

Das Skelett hat sechs wiederkehrende Stationen: Daten vorbereiten → Set-Splitting → Modellklasse → Trainings-Loop → Validierung → Inferenz. Diese Reihenfolge ist kein Zufall, sondern folgt direkt aus PyTorchs Autograd-Design (Gradienten werden explizit verwaltet, nicht implizit wie in manch anderem Framework).

---

## 1. Daten vorbereiten

Rohdaten (hier: Iris als CSV oder via `sklearn.datasets`) müssen in PyTorch-Tensoren überführt werden. Bei kleinen Datensätzen wie Iris reicht ein direkter Tensor; bei größeren kommt die `Dataset`/`DataLoader`-Abstraktion dazu (siehe Hinweis unten).

```python
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import load_iris

# Laden
iris = load_iris()
X = iris.data          # shape: (150, 4) — 4 Features
y = iris.target        # shape: (150,)   — 3 Klassen (0,1,2)

# In Tensoren konvertieren
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # long für Klassifikation (CrossEntropyLoss)
```

> [!important] PyTorch-Eigenheit: `dtype` ist kein Detail `float32` ist bei PyTorch der De-facto-Standard für Feature-Tensoren (nicht `float64` wie oft in NumPy) — Performance- und Speichergründe, besonders auf GPU. Labels für Klassifikation brauchen `torch.long` (int64), da `CrossEntropyLoss` das intern erwartet. Ein `dtype`-Mismatch ist einer der häufigsten ersten Fehler.

> [!note] Fallabhängig: Dataset/DataLoader-Abstraktion Bei 150 Iris-Zeilen kannst du den kompletten Tensor auf einmal durchs Netz schicken (ein Batch = gesamter Datensatz). Sobald Datenmengen wachsen (z.B. Proteinsequenzen im MB/GB-Bereich) oder du Shuffling/Mini-Batches brauchst, wird eine `Dataset`-Klasse mit `__len__` und `__getitem__` plus ein `DataLoader` nötig. Das ist keine Stilfrage, sondern eine Notwendigkeit, sobald Daten nicht mehr komplett in den Speicher passen oder Batch-Training gebraucht wird.

---

## 2. Set-Splitting

Aufteilen in Trainings-, Validierungs- (und ggf. Test-)Daten, meist vor dem Training, manchmal schon vor der Tensor-Konversion.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
)
```

> [!note] Fallabhängig: Stratifizierung und Split-Anzahl `stratify=y` sorgt dafür, dass Klassenverhältnisse in Train/Val gleich bleiben — bei balancierten Datensätzen wie Iris weniger kritisch, bei unbalancierten (z.B. seltene Enzymklassen) essentiell. Ob du zwei (Train/Val) oder drei Sets (Train/Val/Test) brauchst, hängt davon ab, ob du während der Entwicklung Hyperparameter tunst (dann Val zum Tunen, Test bleibt unberührt bis zum Schluss).

---

## 3. Modellklasse definieren

Das Herzstück: jede `nn.Module`-Subklasse braucht `__init__` (Layer definieren) und `forward` (wie Daten durchfließen).

```python
import torch.nn as nn

class IrisNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x  # Rohe Logits — kein Softmax hier!

model = IrisNet()
```

> [!important] PyTorch-Eigenheit: `super().__init__()` ist Pflicht Ohne diesen Aufruf registriert `nn.Module` die Layer nicht korrekt intern (Parameter-Tracking, `.to(device)`, `state_dict()` funktionieren dann nicht). Leicht zu vergessen, schwer zu debuggen, wenn man den Grund nicht kennt.

> [!important] PyTorch-Eigenheit: Logits statt Wahrscheinlichkeiten `forward` gibt bei Klassifikation meist rohe Logits zurück, kein Softmax. Der Grund: `nn.CrossEntropyLoss` wendet Softmax intern selbst an (numerisch stabiler als getrennt). Softmax nur explizit anwenden, wenn du tatsächlich Wahrscheinlichkeiten sehen willst (z.B. bei der Inferenz).

> [!note] Fallabhängig: Architekturkomplexität Iris braucht ein simples 2-Layer-MLP. Bei deinem FuE-Projekt wird die Modellklasse später ESM-2 (vortrainierter Transformer) als Backbone einbinden und einen KAN- sowie Bayesian Head daransetzen — strukturell bleibt das Schema (`__init__` + `forward`) gleich, nur die Layer-Komposition wird komplexer.

---

## 4. Trainings-Loop

Der Kern von PyTorchs expliziter Autograd-Philosophie: Forward → Loss → Backward → Optimizer-Step, in dieser Reihenfolge, jede Epoche.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100
train_losses = []

for epoch in range(n_epochs):
    model.train()                     # Trainings-Modus aktivieren
    optimizer.zero_grad()             # Gradienten zurücksetzen
    outputs = model(X_train)          # Forward-Pass
    loss = criterion(outputs, y_train)
    loss.backward()                   # Backward-Pass (Gradienten berechnen)
    optimizer.step()                  # Parameter-Update

    train_losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

> [!important] PyTorch-Eigenheit: `optimizer.zero_grad()` ist kein Boilerplate-Ritual Gradienten akkumulieren in PyTorch standardmäßig (werden bei jedem `.backward()`-Aufruf addiert, nicht überschrieben). Das ist absichtliches Design (nützlich für z.B. Gradient Accumulation bei großen Batches), bedeutet aber: vergisst du `zero_grad()`, summieren sich Gradienten über Epochen — ein klassischer stiller Bug, der das Training scheinbar zufällig divergieren lässt.

> [!important] PyTorch-Eigenheit: `loss.item()` statt `loss` `loss` ist ein Tensor mit angehängtem Computational Graph. Für reines Logging (z.B. `train_losses.append`) willst du `.item()` — den reinen Python-Skalar. Sonst hältst du unnötig den gesamten Graphen im Speicher fest, über viele Epochen ein Speicherleck.

> [!note] Fallabhängig: Batch- vs. Full-Batch-Training Der Loop oben ist Full-Batch (ganzer Trainingssatz pro Epoche) — bei Iris unproblematisch. Mit `DataLoader` würde eine innere Schleife über Batches iterieren (`for batch in dataloader:`). Für dein FuE-Projekt mit ESM-2-Embeddings wird Mini-Batch-Training die Norm sein, allein aus Speichergründen.

> [!note] Fallabhängig: Learning-Rate-Scheduling Bei Iris reicht ein konstantes `lr=0.01`. Bei komplexeren/längeren Trainingsläufen kommt oft ein Scheduler dazu (`torch.optim.lr_scheduler`), der die Lernrate über Epochen anpasst — nicht immer nötig, aber ein häufiger nächster Baustein.

---

## 5. Validierung

Gleiche Forward-Logik wie im Training, aber ohne Gradientenberechnung und im Evaluationsmodus.

```python
model.eval()                  # Evaluations-Modus aktivieren
with torch.no_grad():         # Keine Gradienten berechnen
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    val_preds = torch.argmax(val_outputs, dim=1)
    val_accuracy = (val_preds == y_val).float().mean()

print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {val_accuracy.item():.4f}")
```

> [!important] PyTorch-Eigenheit: `model.eval()` und `torch.no_grad()` sind unterschiedliche Dinge `model.eval()` schaltet Layer wie Dropout oder BatchNorm auf Inferenz-Verhalten um (die verhalten sich beim Training anders als bei der Vorhersage). `torch.no_grad()` deaktiviert separat die Gradientenberechnung, um Speicher/Rechenzeit zu sparen. Beide werden fast immer zusammen verwendet, sind aber technisch unabhängig — bei Iris' einfachem MLP ohne Dropout/BatchNorm macht `.eval()` hier wenig sichtbaren Unterschied, ist aber trotzdem guter Stil.

> [!note] Fallabhängig: Validierung während vs. nach dem Training Hier wird nach dem kompletten Training einmal validiert. Häufiger (und besser für Diagnostik, Stichwort Loss-Kurven) ist Validierung _innerhalb_ der Trainings-Loop, alle paar Epochen — dann siehst du Overfitting live entstehen, statt es erst am Ende festzustellen.

---

## 6. Inferenz / Vorhersage

Trainiertes Modell auf neue, ungesehene Daten anwenden.

```python
model.eval()
new_sample = torch.tensor([[5.1, 3.5, 1.4, 0.2]], dtype=torch.float32)

with torch.no_grad():
    logits = model(new_sample)
    probabilities = torch.softmax(logits, dim=1)   # jetzt explizit Softmax
    predicted_class = torch.argmax(probabilities, dim=1)

print(f"Vorhergesagte Klasse: {predicted_class.item()}, Wahrscheinlichkeiten: {probabilities}")
```

> [!note] Fallabhängig: Checkpointing vor der Inferenz Hier wird direkt im selben Skript vorhergesagt. In der Praxis liegt zwischen Training und Inferenz meist ein Speicher-/Ladeschritt (`torch.save(model.state_dict(), "model.pt")` und `model.load_state_dict(torch.load("model.pt"))`) — gerade bei langen Trainingsläufen (ESM-2-Fine-Tuning!) nicht optional, sondern notwendig, um nicht bei jedem Absturz von vorne anzufangen.

---

## Was in diesem Minimalbeispiel fehlt (aber bei größeren Projekten dazukommt)

- **Device-Handling** (`model.to(device)`, `X.to(device)`) — bei CPU-only wie Iris hier irrelevant, bei ESM-2 auf GPU zwingend
- **Dataset/DataLoader** — siehe Hinweis oben
- **Learning-Rate-Scheduling** — siehe Hinweis oben
- **Checkpointing** — siehe Hinweis oben
- **Metriken-Tracking über mehrere Epochen** für Diagnostikplots (Loss-Kurven, Pred-vs-True) — thematisch näher an Matplotlib-Modul 3C als an PyTorch selbst

---

## Quicksheet

| Konzept                 | Code                                             | Kurz-Erklärung                                                    |
| ----------------------- | ------------------------------------------------ | ----------------------------------------------------------------- |
| Tensor aus NumPy/Pandas | `torch.tensor(X, dtype=torch.float32)`           | Feature-Tensoren: `float32`. Klassifikations-Labels: `torch.long` |
| Modellklasse            | `class Net(nn.Module):` mit `super().__init__()` | Pflicht-Aufruf für internes Parameter-Tracking                    |
| Layer                   | `nn.Linear(in_dim, out_dim)`                     | Vollverbundene Schicht                                            |
| Aktivierung             | `nn.ReLU()`, `torch.relu(x)`                     | Nichtlinearität zwischen Layern                                   |
| Forward-Pass            | `outputs = model(X)`                             | Ruft intern `forward()` auf                                       |
| Loss (Klassifikation)   | `nn.CrossEntropyLoss()`                          | Erwartet rohe Logits, kein Softmax vorher                         |
| Optimizer               | `optim.Adam(model.parameters(), lr=0.01)`        | `model.parameters()` liefert alle trainierbaren Gewichte          |
| Gradienten zurücksetzen | `optimizer.zero_grad()`                          | Vor jedem Backward-Pass — Gradienten akkumulieren sonst           |
| Backward-Pass           | `loss.backward()`                                | Berechnet Gradienten via Autograd                                 |
| Parameter-Update        | `optimizer.step()`                               | Wendet Gradienten auf Gewichte an                                 |
| Trainingsmodus          | `model.train()`                                  | Aktiviert Dropout/BatchNorm-Trainingsverhalten                    |
| Evaluationsmodus        | `model.eval()`                                   | Deaktiviert Dropout/BatchNorm-Trainingsverhalten                  |
| Ohne Gradienten rechnen | `with torch.no_grad():`                          | Spart Speicher/Zeit bei Validierung/Inferenz                      |
| Skalar aus Tensor       | `loss.item()`                                    | Löst Tensor vom Computational Graph, reiner Python-Wert           |
| Klassenvorhersage       | `torch.argmax(logits, dim=1)`                    | Index der höchsten Logit/Wahrscheinlichkeit                       |
| Modell speichern        | `torch.save(model.state_dict(), "model.pt")`     | Nur Gewichte, nicht die Architektur                               |
| Modell laden            | `model.load_state_dict(torch.load("model.pt"))`  | Modellklasse muss vorher instanziiert sein                        |

### Schwachstellen _(persönlicher Abschnitt — hier eigene Stolperstellen ergänzen)_

# Referenzen
[[Pandas]]
[[NumPy]]
[[Klassen oder Classes]]