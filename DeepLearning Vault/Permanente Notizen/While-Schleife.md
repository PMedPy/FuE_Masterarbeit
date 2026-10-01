2026-01-31
Tags: #Python  #FuE 
Status: #unextended

---

## 📖 Theorie – Teil 2: `while`Schleife

Die `for`-Schleife läuft durch eine **bekannte Sammlung**. Die `while`-Schleife läuft **solange eine Bedingung wahr ist** – egal wie oft das ist.

```python
# for → "für jedes Element in der Liste"
for temp in messungen:
    print(temp)

# while → "solange diese Bedingung stimmt"
temperatur = 20
while temperatur < 25:
    print(f"Temperatur: {temperatur}°C – heize weiter!")
    temperatur = temperatur + 1

print("Zieltemperatur erreicht!")
```

---

### 🔧 Aufbau

```python
while bedingung:
    # Block wird wiederholt solange bedingung True ist
```

|Teil|Bedeutung|
|---|---|
|`while`|Startet die Schleife|
|`bedingung`|Wird vor jedem Durchlauf geprüft|
|`:`|...kennst du ja schon 😄|
|eingerückter Block|Wird wiederholt solange Bedingung `True`|

---

### ⚠️ Die Endlosschleife – größte Gefahr!

```python
# ❌ NICHT AUSFÜHREN – läuft ewig!
temperatur = 20
while temperatur < 25:
    print(temperatur)
    # temperatur wird nie verändert → Bedingung bleibt immer True!
```

Die Variable in der Bedingung **muss sich in der Schleife verändern** – sonst läuft sie ewig!

---

### 🔹 Wann `for`, wann `while`?

|Situation|Nimm...|
|---|---|
|Anzahl der Durchläufe bekannt|`for`|
|Anzahl der Durchläufe unbekannt|`while`|
|Warte bis eine Bedingung erfüllt ist|`while`|

Ein klassisches Beispiel – auf Benutzereingabe warten:

```python
eingabe = ""
while eingabe != "stop":
    eingabe = input("Befehl eingeben: ")

print("Programm beendet!")
```

Hier weiß Python vorher nicht wie oft der Nutzer etwas eingibt – perfekt für `while`! 😊

---

# Referenzen