## 📋 Operatoren in Python

### 🔹 Vergleichsoperatoren
=-Zeichen sind "Zuweisungen" und nicht Gleichung von Variablen/Zahlen


```python
x = 10
y = 5

x == y    # Gleich?              → False  (Achtung: == nicht = !)
x != y    # Ungleich?            → True   ← dein != !
x > y     # Größer?              → True
x < y     # Kleiner?             → False
x >= y    # Größer oder gleich?  → True
x <= y    # Kleiner oder gleich? → False
```

---

### 🔹 Mathematische Operatoren

```python
x + y     # Addition        → 15
x - y     # Subtraktion     → 5
x * y     # Multiplikation  → 50
x / y     # Division        → 2.0
x // y    # Ganzzahldivision → 2  (ohne Nachkommastellen)
x % y     # Modulo          → 0  (Rest der Division)
x ** y    # Potenz          → 100000  (10 hoch 5)
x += Wert # Addiert jenen Wert zu variablen x (Erweiterte Zuweisung)
```

---

### 🔹 Logische Operatoren

```python
True and True    # Beide müssen True sein  → True
True and False   #                         → False
True or False    # Mindestens einer True   → True
False or False   #                         → False
not True         # Umkehrung               → False
```

Kombiniert mit Vergleichen:

```python
temperatur = 23
druck = 10

if temperatur > 20 and druck < 15:
    print("Alles im grünen Bereich!")
```

---

### ⚠️ Häufigster Anfängerfehler

```python
x = 5    # ← Zuweisung – speichert den Wert
x == 5   # ← Vergleich  – prüft ob x gleich 5 ist
```

---

### Weitere Operatoren:
#### Is-Operator
```python
# is-Operator als Alternative um IDentitäten zu vergleichen
>>> pi1 = 3.141592653589793

>>> pi2 = 3.141592653589793

>>> pi1 is pi2 # the same as id(pi1) == id(pi2)

False

>>> pi1 = 3.141592653589793

>>> pi2 = pi1

>>> pi1 is pi2

True
```
Ist nicht zu verwechseln mit Gleichheitsoperatoren zu vertauschen, weil diese auf 