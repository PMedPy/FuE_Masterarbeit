2026-01-31
Tags: #Python  #FuE 
Status: #unextended

# 📘 Lektion 14: Übersicht
---
## 📂 Datei öffnen & schließen

|Befehl|Zweck|
|---|---|
|`open(name, modus)`|Öffnet eine Datei, gibt ein Datei-Objekt zurück|
|`datei.close()`|Schließt die Datei manuell (selten nötig dank `with`)|
|`with open(...) as datei:`|Öffnet & schließt automatisch – **immer bevorzugen**|

### Modi

|Modus|Bedeutung|Datei existiert nicht|Datei existiert|
|---|---|---|---|
|`"r"`|read (lesen)|❌ `FileNotFoundError`|✅ liest|
|`"w"`|write (schreiben)|✅ wird erstellt|⚠️ wird **überschrieben**|
|`"a"`|append (anhängen)|✅ wird erstellt|✅ hängt ans Ende an|

---

## 📖 Aus Datei lesen

|Methode|Rückgabe|Wann nutzen?|
|---|---|---|
|`datei.read()`|gesamter Inhalt als **ein String**|kleine Dateien|
|`datei.readlines()`|**Liste** aller Zeilen (mit `\n`)|wenn alle Zeilen gleichzeitig gebraucht|
|`for zeile in datei:`|iteriert zeilenweise|**große Dateien**, speicherschonend ⭐|

---

## 📝 In Datei schreiben

|Methode|Was sie tut|
|---|---|
|`datei.write(string)`|schreibt einen String in die Datei|
|`datei.writelines(liste)`|schreibt mehrere Strings (kein automatisches `\n`!)|

⚠️ **`write()` braucht einen String** – Dicts/Zahlen vorher mit f-String formatieren:

```python
datei.write(f"{name};{wert}\n")
```

---

## 🧵 String-Methoden (in Lektion 14 wichtig)

|Methode|Effekt|Beispiel|
|---|---|---|
|`s.strip()`|entfernt Leerzeichen + `\n` an beiden Enden|`" hi\n".strip()` → `"hi"`|
|`s.split(trenner)`|zerlegt String in Liste|`"a;b;c".split(";")` → `["a","b","c"]`|
|`s.upper()` / `s.lower()`|Groß-/Kleinschreibung|`"abc".upper()` → `"ABC"`|
|`s.replace(alt, neu)`|ersetzt Teile|`"abc".replace("b","X")` → `"aXc"`|

---

## 🔢 Typumwandlung beim Einlesen

Aus Dateien kommen **immer Strings** – beim Lesen umwandeln:

|Funktion|Wandelt um in|
|---|---|
|`int(s)`|Ganzzahl|
|`float(s)`|Kommazahl|
|`str(x)`|String|

```python
zahl = float(zeile.strip())    # Standardablauf: strippen, dann konvertieren
```

---

## 🎨 f-Strings – Nützliches für Dateien

|Syntax|Effekt|
|---|---|
|`f"{wert}"`|Wert einsetzen|
|`f"{wert:.2f}"`|Float mit 2 Nachkommastellen|
|`f"{wert:>10}"`|rechtsbündig mit Mindestbreite 10|
|`f"{dict['key']}"`|Dict-Wert holen (innen `'`, außen `"`)|
|`\n`|Zeilenumbruch|
|`\t`|Tab|

---

## 🛡 Fehlerbehandlung mit Dateien

```python
try:
    with open("datei.txt", "r") as f:
        inhalt = f.read()
except FileNotFoundError:
    print("Datei nicht gefunden!")
```

`try`/`except` **um** den `with`-Block, nicht innen.

---

## 📋 Standard-Patterns

### Datei zeilenweise einlesen & verarbeiten

```python
with open("daten.txt", "r") as datei:
    for zeile in datei:
        sauber = zeile.strip()
        # ... verarbeiten
```

### CSV-ähnliche Datei mit Trennzeichen einlesen

```python
with open("daten.txt", "r") as datei:
    for zeile in datei:
        teile = zeile.strip().split(";")
        name = teile[0]
        wert = float(teile[1])
```

### Liste von Dicts in Datei speichern

```python
with open("daten.txt", "w") as datei:
    for eintrag in liste:
        zeile = f"{eintrag['name']};{eintrag['wert']}\n"
        datei.write(zeile)
```

### Datei mit Fehlerbehandlung lesen

```python
try:
    with open("daten.txt", "r") as datei:
        for zeile in datei:
            ...
except FileNotFoundError:
    print("Keine Datei vorhanden – starte leer.")
```

---

## ⚠️ Häufige Stolpersteine

|Problem|Ursache|Lösung|
|---|---|---|
|`\n` im String|Vergessen zu strippen|`zeile.strip()`|
|`int("5\n")` crasht _(manchmal)_|Whitespace im String|erst `strip()`, dann konvertieren|
|Datei plötzlich leer|`"w"` statt `"a"` benutzt|Modus prüfen|
|Alles in einer Zeile|Kein `\n` beim `write`|`f"...\n"`|
|`TypeError: write() argument must be str`|Dict/Zahl direkt geschrieben|Mit f-String in String umwandeln|
|`for i in liste:` und dann `liste[i]`|`i` ist schon das Element!|direkt `i` (oder `eintrag`) nutzen|

---

## 🔑 Goldene Regeln

1. **Immer `with`** statt manuelles `close()`
2. **Immer `strip()`** beim Einlesen
3. **`write()` will Strings** – f-Strings sind dein Freund
4. **`"w"` überschreibt** – wenn du anhängen willst, nimm `"a"`
5. **`FileNotFoundError`** mit `try`/`except` abfangen, wenn die Datei optional ist
6. **`for x in datei:`** ist speicherschonend – immer bevorzugen für große Dateien

---

# 📘 Lektion 14: Dateien lesen & schreiben (`.txt`)

---

## 🎯 Lernziel

Nach dieser Lektion kannst du:

- Textdateien **lesen** und **schreiben**
- Den Unterschied zwischen den Modi `r`, `w`, `a` erklären
- Den `with`-Block sauber einsetzen (und verstehen, _warum_)
- Daten zwischen Programmläufen **persistent** speichern

> ⚡ **Warum das wichtig ist:** Bisher waren alle deine Daten weg, sobald das Programm endete. Mit Dateien kann sich dein Programm Sachen merken – Messdaten speichern, Berichte ablegen, Konfigurationen laden.

---

## 📖 1. Eine Datei lesen – das Grundprinzip

```python
datei = open("notizen.txt", "r")    # öffnen
inhalt = datei.read()                # gesamten Inhalt als String
datei.close()                        # schließen!
print(inhalt)
```

Drei Schritte: **öffnen → lesen → schließen**.

Das `"r"` steht für **read** – Lesemodus.

> ⚠️ **`close()` nicht vergessen!** Sonst bleibt die Datei "im Speicher hängen", andere Programme können nicht zugreifen, und im schlimmsten Fall gehen Daten verloren. Gleich kommt eine viel elegantere Lösung.

---

## 🔐 2. `with` – der saubere Weg

In der Praxis schreibt man fast immer:

```python
with open("notizen.txt", "r") as datei:
    inhalt = datei.read()
    print(inhalt)

# Hier außerhalb des with-Blocks ist die Datei automatisch geschlossen!
```

**Was macht `with`?**

- Öffnet die Datei
- Übergibt sie an die Variable `datei` (das `as datei`)
- Sorgt dafür, dass sie **automatisch geschlossen wird**, sobald der Block endet – **selbst wenn ein Fehler auftritt!**

**Faustregel:** **Immer `with` benutzen.** Kein `close()` mehr nötig, fehlersicher, lesbar.

---

## 🎚 3. Die drei wichtigsten Modi

|Modus|Bedeutung|Was passiert|
|---|---|---|
|`"r"`|**read** (lesen)|Datei muss existieren, sonst `FileNotFoundError`|
|`"w"`|**write** (schreiben)|⚠️ **Überschreibt** existierende Datei komplett!|
|`"a"`|**append** (anhängen)|Hängt **ans Ende** an, behält alten Inhalt|

> ⚠️ **`"w"` ist gefährlich:** Es löscht den alten Inhalt **sofort** beim Öffnen – noch bevor du etwas geschrieben hast. Wenn du nur ergänzen willst, nimm `"a"`.

---

## 📝 4. In eine Datei schreiben

```python
with open("notizen.txt", "w") as datei:
    datei.write("Erste Zeile\n")
    datei.write("Zweite Zeile\n")
```

Wichtig: **`\n` bedeutet Zeilenumbruch.** Ohne `\n` landet alles in einer einzigen Zeile.

```python
datei.write("Hallo")
datei.write("Welt")
# Datei enthält: "HalloWelt"   ← keine Trennung!

datei.write("Hallo\n")
datei.write("Welt\n")
# Datei enthält:
# Hallo
# Welt
```

---

## 📚 5. Drei Wege zu lesen

Angenommen, `notizen.txt` enthält:

```
Apfel
Banane
Kirsche
```

### Variante A: `read()` – alles als ein String

```python
with open("notizen.txt", "r") as datei:
    inhalt = datei.read()
print(inhalt)
# → 'Apfel\nBanane\nKirsche\n'
```

Gut für kleine Dateien.

### Variante B: `readlines()` – Liste von Zeilen

```python
with open("notizen.txt", "r") as datei:
    zeilen = datei.readlines()
print(zeilen)
# → ['Apfel\n', 'Banane\n', 'Kirsche\n']
```

⚠️ Die `\n` sind **noch dran** am Ende jeder Zeile!

### Variante C: Zeile für Zeile mit `for` – **die beste für große Dateien**

```python
with open("notizen.txt", "r") as datei:
    for zeile in datei:
        print(zeile)
```

Speicherschonend – Python lädt jeweils nur eine Zeile.

---

## ✂️ 6. `\n` loswerden mit `.strip()`

Da `\n` oft im Weg ist:

```python
with open("notizen.txt", "r") as datei:
    for zeile in datei:
        sauber = zeile.strip()    # entfernt Leerzeichen + \n vorne und hinten
        print(sauber)
```

`strip()` ist eine **String-Methode**, die Leerzeichen, Tabs und Zeilenumbrüche an beiden Enden entfernt.

```python
"  Hallo \n".strip()    # → "Hallo"
```

---

## 🧪 7. Daten verarbeiten – realistisches Beispiel

Angenommen `messwerte.txt`:

```
12.5
8.3
15.7
9.1
```

Lade sie als Zahlen und berechne den Mittelwert:

```python
zahlen = []

with open("messwerte.txt", "r") as datei:
    for zeile in datei:
        zahl = float(zeile.strip())    # strip + Umwandlung
        zahlen.append(zahl)

mittelwert = sum(zahlen) / len(zahlen)
print(f"Mittelwert: {mittelwert:.2f}")
```

> 💡 **`{wert:.2f}`** in einem f-String → Zahl mit 2 Nachkommastellen. Sehr nützlich!

---

## 🛡 8. Fehler abfangen – Brücke zu Lektion 13

Was, wenn die Datei nicht existiert?

```python
try:
    with open("messwerte.txt", "r") as datei:
        inhalt = datei.read()
except FileNotFoundError:
    print("Die Datei wurde nicht gefunden.")
```

Sehr typisches Muster: `try`/`except` **um den `with`-Block** herum, nicht innen.

---

## 📂 9. Dateipfade – wo liegt die Datei eigentlich?

Wenn du `open("notizen.txt")` schreibst, sucht Python im **aktuellen Arbeitsverzeichnis** – meistens dort, wo dein Python-Skript liegt.

**Absoluter Pfad** (komplette Adresse):

```python
open("C:/Users/Adrian/Documents/notizen.txt")
```

**Relativer Pfad** (von wo das Skript läuft):

```python
open("daten/notizen.txt")    # Unterordner "daten"
open("../notizen.txt")       # ein Verzeichnis nach oben
```

> 💡 In Lektion 19 lernst du `pathlib` – das macht Pfade plattformunabhängig (Windows vs. Linux).

---

## ⚠️ 10. Stolpersteine

|Problem|Ursache|Lösung|
|---|---|---|
|`FileNotFoundError`|Datei nicht da / falscher Pfad|`try`/`except` oder Pfad prüfen|
|Datei wird unerwartet **leer**|`"w"` statt `"a"` benutzt|Modus prüfen|
|Alles in einer Zeile|Kein `\n` beim Schreiben|`datei.write("...\n")`|
|`\n` taucht im String auf|Vergessen zu strippen|`.strip()` aufrufen|
|`int(zeile)` crasht|`\n` noch dran|erst `strip()`, dann `int()`|

---

# Referenzen
[[Übung Trainingtagebuch]]