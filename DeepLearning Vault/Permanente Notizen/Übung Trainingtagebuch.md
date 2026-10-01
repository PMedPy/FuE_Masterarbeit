2026-01-31
Tags: #Python  #FuE 
Status: #unextended

---

# 🛠 Übungsaufgabe Tag 3: Trainingstagebuch 🏋️

Da du etwas Neues willst, das alle bisherigen Konzepte verbindet – hier ist deine Aufgabe.

## 📋 Szenario

Du baust ein **Trainingstagebuch**, das deine Workouts speichert und auswertet. Es nutzt:

- ✅ **Klassen** (Lektion 12)
- ✅ **Dictionaries & Listen** (Lektion 8–9)
- ✅ **Schleifen** (Lektion 10)
- ✅ **Funktionen** (Lektion 11)
- ✅ **`try`/`except`** (Lektion 13 – NEU)
- ✅ **Datei lesen & schreiben** (Lektion 14 – NEU)

---

## 🏗 Anforderungen

### Klasse `Trainingstagebuch`

**Attribute:**

- `eintraege` – Liste von Dictionaries, jedes mit:
    - `"datum"` (String, z.B. `"2026-04-29"`)
    - `"uebung"` (String, z.B. `"Bankdrücken"`)
    - `"gewicht"` (float, in kg)
    - `"wiederholungen"` (int)

**Methoden:**

1. **`eintrag_hinzufuegen(datum, uebung, gewicht, wdh)`** Fügt einen neuen Eintrag (als Dict) zur Liste hinzu.
    
2. **`speichern(dateiname)`** Schreibt alle Einträge in eine `.txt`-Datei. **Format pro Zeile:**
    
    ```
    2026-04-29;Bankdrücken;80.0;8
    2026-04-29;Kniebeuge;100.0;5
    ```
    
    (Werte mit Semikolon getrennt – das ist ein einfaches **CSV-ähnliches Format**.)
    
3. **`laden(dateiname)`** Liest eine solche Datei ein und füllt `self.eintraege` damit. Soll `FileNotFoundError` abfangen und freundlich melden, wenn die Datei nicht existiert.
    
4. **`bestleistung(uebung)`** Gibt das **maximale Gewicht** für eine bestimmte Übung zurück. Falls die Übung nicht existiert: freundliche Meldung, kein Crash.
    
5. **`uebersicht()`** Druckt alle Einträge schön formatiert aus.
    

---

## 🎯 Testcode (zum Ausprobieren am Ende)

```python
tb = Trainingstagebuch()
tb.eintrag_hinzufuegen("2026-04-27", "Bankdrücken", 75.0, 8)
tb.eintrag_hinzufuegen("2026-04-27", "Kniebeuge", 100.0, 5)
tb.eintrag_hinzufuegen("2026-04-29", "Bankdrücken", 80.0, 6)

tb.speichern("training.txt")

# Neues Tagebuch, lädt aus Datei
tb2 = Trainingstagebuch()
tb2.laden("training.txt")
tb2.uebersicht()

print(tb2.bestleistung("Bankdrücken"))    # → 80.0
print(tb2.bestleistung("Klimmzug"))        # → freundliche Meldung
tb2.laden("gibtsnicht.txt")                # → FileNotFoundError abfangen!
```

---

## 💡 Tipps

### Zum Speichern

```python
with open(dateiname, "w") as datei:
    for eintrag in self.eintraege:
        zeile = f"{eintrag['datum']};{eintrag['uebung']};{eintrag['gewicht']};{eintrag['wiederholungen']}\n"
        datei.write(zeile)
```

### Zum Laden

Pro Zeile musst du:

1. `strip()` anwenden
2. Mit `"split(";")` die Zeile in eine **Liste** aufteilen
3. Die einzelnen Werte in die richtigen Datentypen konvertieren (`float`, `int`)

```python
zeile = "2026-04-29;Bankdrücken;80.0;8"
teile = zeile.split(";")
# teile = ['2026-04-29', 'Bankdrücken', '80.0', '8']
```

### Zur Bestleistung

- Filtere mit einer Schleife alle Einträge, die zur gesuchten Übung passen
- Sammle deren Gewichte in einer Liste
- Wenn die Liste leer ist → freundliche Meldung
- Sonst → `max(gewichte)` zurückgeben

> 💡 **`max()`** ist eine eingebaute Python-Funktion: `max([5, 2, 9, 1])` → `9`

---

## 🆕 Neue Konzepte, die du brauchst

|Konzept|Erklärung|
|---|---|
|**`split(trenner)`**|Zerlegt einen String in eine Liste an jedem Trennzeichen. `"a;b;c".split(";")` → `["a", "b", "c"]`|
|**`max(liste)`**|Gibt das größte Element zurück|
|**`{eintrag['datum']}`**|In f-Strings Dict-Werte holen – `'`-Zeichen wechseln, weil außen `f"..."`|

---

## ⏱ Zeit: 30–40 Minuten

Das ist **größer** als die bisherigen Aufgaben – nimm dir Zeit. Bau Schritt für Schritt:

1. Erst die Klasse mit `__init__` und `eintrag_hinzufuegen`
2. Dann `uebersicht()` (zum Debuggen)
3. Dann `speichern()` testen
4. Dann `laden()` testen (mit der eben gespeicherten Datei)
5. Dann `bestleistung()`
6. Zum Schluss `try`/`except` für die Fehler

**Ohne Copilot.** Wenn du irgendwo hängst – frag gezielt. 💪

Schick mir den Code, wenn du fertig bist!

# Referenzen