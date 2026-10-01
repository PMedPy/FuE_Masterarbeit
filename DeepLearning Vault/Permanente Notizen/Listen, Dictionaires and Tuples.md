
### Listen

```python
liste = [10, 20, 30, "hallo"]
liste[0]              # Element lesen → 10
liste[-1]             # Letztes Element → "hallo"
liste[0] = 99         # Element ändern
liste.append(40)      # Element hinzufügen (hinten)
liste.remove(20)      # Element löschen (nach Wert)
liste.insert(2,"Geld")# Fügt "Geld" auf zweite Position ein, und verschiebt                              element auf Position 3
liste.extend()
liste.pop(x).         # Streicht Element mit index x. Wenn keiner angegeben ist,                         wird das letzte Element gestrichen
del liste[0]          # Element löschen (nach Index)
len(liste)            # Länge → 4
20 in liste           # Enthalten? → True/False
liste.sort()          # Sortieren
liste.count(x)        # Gibt anzahl an Elemente x aus
enumerate()           # Gibt den Element Index + Wert aus. 
```

### Dictionaires

```python
dict = {"name": "Max", "alter": 25}

dict["name"]          # Wert lesen → "Max"
dict["name"] = "Anna" # Wert ändern
dict["stadt"] = "Berlin" # Neuen Key hinzufügen
del dict["alter"]     # Key löschen
"name" in dict        # Key vorhanden? → True/False
dict.keys()           # Alle Keys
dict.values()         # Alle Values
len(dict)             # Anzahl Keys
dict.get(key[, default]) #Gibt Key zurück, wenn dieser nicht vorhanden ist, dann einen default wert.
dict.items()              # Alle Key-Value-Paare → [("name", "Max"), ...]
dict.update({"job": "Dev"}) # Mehrere Keys hinzufügen/überschreiben
dict.pop("name")          # Key löschen UND Wert zurückgeben
dict.pop("x", None)       # Wie oben, aber default statt Fehler
dict.popitem()            # Letztes Key-Value-Paar entfernen + zurückgeben
dict.setdefault("stadt", "Berlin") # Key holen; falls fehlt, anlegen mit default
dict.clear()              # Alle Einträge löschen → {}
dict.copy()               # Flache Kopie erstellen
dict | other              # Zwei Dicts zusammenführen (Python 3.9+) → neues dict
dict |= other             # other in dict mergen (in-place)
list(dict)                # Alle Keys als Liste → ["name", "alter"]
dict.fromkeys(["a","b"], 0) # Neues dict aus Keys mit gleichem Wert → {"a":0,"b":0}

for k, v in dict.items(): # Standard-Iteration über Key + Value
    print(k, v)

{k: v for k, v in dict.items() if v} # Dict Comprehension (filtern/transformieren)
```

Mit for-schleife:

### 🔹 Dictionary mit `.items()`

python

`for key, value in sensor.items(): print(key, "→", value)`

Hier passiert etwas Besonderes – du bekommst **zwei Variablen auf einmal**:

|Teil|Bedeutung|
|---|---|
|`key`|Bekommt den Namen (z.B. `"name"`, `"wert"`)|
|`value`|Bekommt den dazugehörigen Wert (z.B. `"Sensor 1"`, `23.5`)|
|`.items()`|Gibt das Dictionary als **Paare** zurück – ohne das geht es nicht!|

### Tuples

```python
punkt = (10, 20, 30)

punkt[0]              # Element lesen → 10
x, y, z = punkt       # Auspacken
len(punkt)            # Länge → 3
20 in punkt           # Enthalten? → True/False

❌ Ändern/Löschen → nicht möglich!
```