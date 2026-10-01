2026-01-31
Tags: #Python 
Status: #unextended

---

## Lektion – Klassen (`class`) 🏗️

---

## 📖 Theorie – Was ist eine Klasse?

Erinnerst du dich an deinen Sensor-Dictionary aus Lektion 9?

```python
sensor = {"name": "Sensor 3000", "messwert": 23.5, "einheit": "°C"}
```

Das Problem: Ein Dictionary speichert nur **Daten** – aber keine **Funktionen** die damit arbeiten.

Eine Klasse kombiniert beides – **Daten + Funktionen** in einem Paket:

```python
class Sensor:
    def __init__(self, name, messwert, einheit):
        self.name = name
        self.messwert = messwert
        self.einheit = einheit

    def ausgabe(self):
        print(f"{self.name}: {self.messwert} {self.einheit}")

# Sensor "erschaffen" (Objekt erstellen)
sensor1 = Sensor("Sensor 3000", 23.5, "°C")
sensor1.ausgabe()  # → Sensor 3000: 23.5 °C
```

---

### 🔹 Die wichtigsten Teile

|Teil|Bedeutung|
|---|---|
|`class Sensor:`|Definiert die Klasse – wie ein Bauplan|
|`__init__`|Wird automatisch aufgerufen wenn ein Objekt erstellt wird|
|`self`|Verweist auf **das eigene Objekt** – immer erster Parameter!|
|`self.name`|Speichert einen Wert **im Objekt**|
|`def ausgabe(self)`|Eine Funktion der Klasse – nennt man **Methode**|

---

### 🔹 Klasse vs. Objekt

```python
# Klasse = Bauplan (existiert nur einmal)
class Sensor:
    ...

# Objekt = das fertige Produkt nach dem Bauplan (beliebig viele!)
sensor1 = Sensor("Sensor A", 23.5, "°C")
sensor2 = Sensor("Sensor B", 18.2, "°C")
sensor3 = Sensor("Sensor C", 30.1, "°C")
```

Stell es dir vor wie eine **Keksform** 🍪 – die Form ist die Klasse, jeder Keks ist ein Objekt!

---

## Warum nicht einfach Print?

## `__repr__` – warum nicht einfach `print()`? 🤔

---

### 🔹 Was passiert bei `print(reactor1)` ohne `__repr__`?

```python
# Ohne __repr__
print(reactor1)
# → <__main__.Reaktor object at 0x000002A4F3B1C5E0>
```

Python weiß nicht wie es dein Objekt als Text darstellen soll – also gibt es die **Speicheradresse** aus. Völlig nutzlos! 😄

---

### 🔹 `__repr__` = "wie soll ich aussehen wenn man mich ausgibt?"

```python
def __repr__(self):
    return f"{self.name} ist {self.status} und hat {self.temperatur}°C"
```

Du gibst Python eine **Anleitung** – und `print(reactor1)` wird automatisch schön:

```
Reaktor 1 ist eingeschaltet und hat 80°C erreicht
```

---

### 🔹 Warum nicht einfach eine normale Methode?

Du könntest auch schreiben:

```python
def info(self):
    print(f"{self.name} ist {self.status}...")

reactor1.info()   # funktioniert ✅
print(reactor1)   # → hässliche Speicheradresse ❌
```

`__repr__` ist Pythons **eingebauter Mechanismus** – er wird automatisch aufgerufen bei:

```python
print(reactor1)       # direkte Ausgabe
f"Info: {reactor1}"   # im f-String
str(reactor1)         # Umwandlung zu String
```

Eine normale `info()` Methode müsstest du immer manuell aufrufen – `__repr__` arbeitet im Hintergrund automatisch! 😊

---

# Referenzen