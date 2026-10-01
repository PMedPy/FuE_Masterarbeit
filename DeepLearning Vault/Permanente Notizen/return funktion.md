2026-01-31
Tags: #Python  #FuE 
Status: #unextended

## `return` – was bedeutet das? 🔍

---

### 🔹 Generell – `return` beendet eine Funktion sofort

```python
def begruessung():
    print("Zeile 1")
    return              # ← Funktion endet hier!
    print("Zeile 2")    # ← wird NIE erreicht!

begruessung()  # → nur "Zeile 1"
```

`return` ist ein **Stoppschild** – alles danach wird ignoriert.

---

### 🔹 `return` mit Wert – kennst du schon!

```python
def durchschnitt(werte):
    summe = sum(werte)
    return summe / len(werte)  # ← gibt den Wert zurück

ergebnis = durchschnitt([10, 20, 30])  # ergebnis = 20.0
```

Hier gibt `return` einen **Wert zurück** an den Aufrufer.

---

### 🔹 `return` ohne Wert – der spezielle Fall

```python
def messung_hinzufuegen(self, sensor_name, wert):
    for sensor in self.sensoren:
        if sensor['name'] == sensor_name:
            sensor['messungen'].append(wert)
            return   # ← kein Wert, nur "ich bin fertig!"
    print(f"Sensor '{sensor_name}' nicht gefunden!")
```

Hier gibt `return` **nichts zurück** – es sagt nur: _"Aufgabe erledigt, sofort aufhören!"_

---

### 🧠 Warum ist das in der for-Schleife wichtig?

Stell dir vor du hast 3 Sensoren und suchst `"Temp-Sensor"`:

```
Durchlauf 1 → sensor = "Temp-Sensor" ✅ gefunden! → append → return → STOP
Durchlauf 2 → wird nie erreicht!
Durchlauf 3 → wird nie erreicht!
```

**Ohne `return`:**

```
Durchlauf 1 → sensor = "Temp-Sensor" ✅ gefunden! → append → weitersuchen...
Durchlauf 2 → sensor = "Druck-Sensor" ❌
Durchlauf 3 → sensor = "pH-Sensor" ❌
→ Schleife fertig → print("Sensor nicht gefunden!") ← FALSCH! 😱
```

Ohne `return` würde Python nach dem Fund **weiterlaufen** und am Ende trotzdem die Fehlermeldung ausgeben – obwohl der Sensor gefunden wurde!

---

### 🎯 Merksatz

|Situation|Nutze...|
|---|---|
|Funktion gibt Ergebnis zurück|`return wert`|
|Funktion ist fertig, kein Ergebnis|`return`|
|Fehlerfall früh abbrechen|`return`|

---

Alles klar? Dann weiter zur **Bonusaufgabe oder Lektion 12**! 😊



# Referenzen