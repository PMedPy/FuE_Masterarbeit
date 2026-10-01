2026-01-31
Tags: #Python  #FuE 
Status: #unextended

## 📋 `for`Schleife – Übersicht

---

### 🔹 Mit einer Liste

```python
früchte = ["Apfel", "Banane", "Kirsche"]

for frucht in früchte:
    print(frucht)          # Apfel, Banane, Kirsche
```

---

### 🔹 Mit Liste + Index (`enumerate`)

```python
for i, frucht in enumerate(früchte):
    print(f"{i}: {frucht}")  # 0: Apfel, 1: Banane, 2: Kirsche
```

---

### 🔹 Mit `range()`

```python
for i in range(5):          # 0, 1, 2, 3, 4
for i in range(2, 6):       # 2, 3, 4, 5
for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
```

---

### 🔹 Mit Dictionary

```python
sensor = {"name": "X1", "wert": 23.5, "einheit": "°C"}

# Nur Keys
for key in sensor:
    print(key)             # name, wert, einheit

# Keys + Values
for key, value in sensor.items():
    print(f"{key}: {value}")  # name: X1, wert: 23.5 ...
```

---

### 🔹 Mit Liste von Dictionaries

```python
sensoren = [
    {"name": "S1", "wert": 21.3},
    {"name": "S2", "wert": 23.7},
]

for sensor in sensoren:
    print(f"{sensor['name']}: {sensor['wert']}°C")
# S1: 21.3°C
# S2: 23.7°C
```

---

### 🔹 Akkumulator – Werte aufsammeln

```python
messungen = [21.3, 23.7, 22.1]

summe = 0                    # ← startet bei 0
for wert in messungen:
    summe = summe + wert     # ← wächst mit jedem Durchlauf

print(summe / len(messungen))  # Durchschnitt
```

---

### 🎯 Merksatz

> `wert` in `for wert in liste` **ist bereits der Wert** – nie nochmal mit `liste[wert]` drauf zugreifen!

---

# Referenzen