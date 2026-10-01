20-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Topic
Ist ein Datentyp in dessen ein Objekt nur einmal vorkommen darf
```Python 
staedte = {"Hamburg," "Köln", "Frankfurt"}
```
- Wir wie Dicitonairy definiert, ohne Keys
- Sets können keine Listen als Elemente haben
- Sets sind veränderlich
Die hast du eigentlich alle schon drin – `intersection` (`&`), `union` (`|`), `isdisjoint`, `issubset`, `issuperset` stehen in deiner Liste. Vielleicht meinst du, dass sie nicht klar genug zusammenstehen? Hier die komplette, sauber gruppierte Version:

```python
s = {1, 2, 3}

# Hinzufügen / Entfernen
s.add(4)          # einzelnes Element
s.update([5, 6])  # mehrere Elemente
s.remove(2)       # entfernt, KeyError wenn fehlt
s.discard(99)     # entfernt, kein Fehler wenn fehlt
s.pop()           # entfernt & gibt beliebiges Element zurück
s.clear()         # leert das Set

a = {1, 2, 3}
b = {3, 4, 5}

# Union  (Vereinigung: alles aus beiden)
a | b   # a.union(b)                -> {1, 2, 3, 4, 5}
a.union(b, {6})                     # mehrere Iterables erlaubt

# Intersection  (Schnittmenge: was in beiden ist)
a & b   # a.intersection(b)         -> {3}
a.intersection(b, {3, 9})           # mehrere Iterables erlaubt

# Difference  (gerichtet: in a, aber nicht in b)
a - b   # a.difference(b)           -> {1, 2}
b - a   # b.difference(a)           -> {4, 5}
a.difference(b, {1})                # mehrere Iterables erlaubt

# Symmetric Difference  (in genau einem von beiden)
a ^ b   # a.symmetric_difference(b) -> {1, 2, 4, 5}

# In-place Varianten
a |= b  # update                          (Vereinigung)
a &= b  # intersection_update             (Schnittmenge)
a -= b  # difference_update               (Differenz)
a ^= b  # symmetric_difference_update     (symm. Differenz)

# Relationen / Tests
3 in a              # Mitgliedschaft
a.isdisjoint(b)     # True, wenn keine gemeinsamen Elemente
a.issubset(b)       # a <= b   (ist a ganz in b enthalten?)
a < b               # echte Teilmenge (a <= b und a != b)
a.issuperset(b)     # a >= b   (enthält a ganz b?)
a > b               # echte Obermenge

# Sonstiges
len(a)              # Anzahl Elemente
a.copy()            # flache Kopie
frozenset({1, 2})   # unveränderliches Set (hashbar)
```



### Wichtige Operationen
# Referenzen