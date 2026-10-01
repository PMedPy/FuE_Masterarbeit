03-06-2026
Tags: #Python  #FuE 
Status: #unextended


 # 🔁 `enumerate()` – Index und Wert gleichzeitig in einer Schleife

## 🎯 Was macht es?

`enumerate()` ist eine eingebaute Python-Funktion, die dir in einer Schleife **gleichzeitig den Index (Zähler) und den Wert** eines Iterables liefert.

Statt zwei Variablen manuell zu verwalten, übernimmt Python das.

---

## 🧠 Das Grundprinzip

`enumerate(liste)` gibt bei jeder Iteration ein **Tupel** der Form `(index, wert)` zurück.

```python
früchte = ["Apfel", "Birne", "Kirsche"]

for paar in enumerate(früchte):
    print(paar)
# (0, 'Apfel')
# (1, 'Birne')
# (2, 'Kirsche')
```

Mit **Tupel-Unpacking** schreibt man das direkt in zwei Variablen:

```python
for index, frucht in enumerate(früchte):
    print(f"{index}: {frucht}")
# 0: Apfel
# 1: Birne
# 2: Kirsche
```

---

## 🎚️ Der `start=`-Parameter

Standardmäßig zählt `enumerate` ab **0**. Du kannst aber bei einem anderen Wert anfangen:

```python
for nummer, frucht in enumerate(früchte, start=1):
    print(f"{nummer}. {frucht}")
# 1. Apfel
# 2. Birne
# 3. Kirsche
```

Typische Anwendungen für `start`:

- `start=1` für menschenlesbare Nummerierung
- `start=2` für Excel-Zeilen (Zeile 1 ist meist der Header)

---

## 🆚 Vergleich zu anderen Mustern

### ❌ Manueller Zähler (umständlich)

```python
i = 0
for frucht in früchte:
    print(i, frucht)
    i += 1
```

### ❌ `range(len(...))` (unschön)

```python
for i in range(len(früchte)):
    print(i, früchte[i])
```

### ✅ `enumerate` (pythonisch)

```python
for i, frucht in enumerate(früchte):
    print(i, frucht)
```

---

## 🎁 Verschachteltes Tupel-Unpacking

Wenn jedes Element der Liste selbst ein Tupel ist, kannst du **direkt auspacken**:

```python
messdaten = [
    ("Mo", 21.5, 55),
    ("Di", 22.0, 58),
    ("Mi", 20.8, 62),
]

for index, (tag, temp, feuchte) in enumerate(messdaten, start=2):
    print(f"Zeile {index}: {tag} – {temp}°C – {feuchte}%")
# Zeile 2: Mo – 21.5°C – 55%
# Zeile 3: Di – 22.0°C – 58%
# Zeile 4: Mi – 20.8°C – 62%
```

→ Statt `eintrag[0]`, `eintrag[1]`, `eintrag[2]` hast du sprechende Namen.

---

## 💡 Typische Anwendungsfälle

### 1. Nummerierte Listenausgabe

```python
for n, aufgabe in enumerate(todo_liste, start=1):
    print(f"{n}. {aufgabe}")
```

### 2. In Excel/Tabellen schreiben (openpyxl)

```python
for zeilen_index, (tag, wert) in enumerate(daten, start=2):
    ws.cell(row=zeilen_index, column=1, value=tag)
    ws.cell(row=zeilen_index, column=2, value=wert)
```

### 3. Alle Positionen eines Wertes finden

```python
namen = ["Anna", "Bernd", "Clara", "Bernd"]

for i, name in enumerate(namen):
    if name == "Bernd":
        print(f"Position {i}")
# Position 1
# Position 3
```

> `list.index("Bernd")` würde nur **die erste** Position liefern – `enumerate` findet alle.

### 4. Sequenzen mit Position verarbeiten

```python
sequenzen = ["MKWVTFISLL", "QFEAQDR", "VTKAGYTL"]

for i, seq in enumerate(sequenzen):
    print(f"Sequenz #{i}: Länge {len(seq)}")
```

---

## 🎯 Die Faustregel

> **Immer wenn du in einer Schleife sowohl den Wert als auch den Index brauchst → `enumerate`.**

Nicht den Index manuell hochzählen. Nicht über `range(len(...))` gehen.

---

## ⚠️ Wann brauchst du `enumerate` NICHT?

Wenn du den Index **gar nicht** brauchst:

```python
# Schlecht – Index ungenutzt
for i, frucht in enumerate(früchte):
    print(frucht)

# Besser – direkt durchgehen
for frucht in früchte:
    print(frucht)
```

---

## 🔌 Bezug zum FuE-Projekt

`enumerate` wird dich beim Verarbeiten von Aminosäuresequenzen oder Trainingsdaten ständig begleiten:

```python
# Beispiel: Trainings-Validierungs-Split
for i, sequenz in enumerate(sequenzen):
    if i % 5 == 0:           # jede 5. ins Validierungsset
        validierung.append(sequenz)
    else:
        training.append(sequenz)
```

Auch in PyTorch DataLoader-Schleifen siehst du oft:

```python
for batch_idx, (eingaben, labels) in enumerate(dataloader):
    ...
```

→ Dasselbe Muster wie bei deinen Messdaten – nur größer.

# Referenzen
[[Datenverwaltung]]