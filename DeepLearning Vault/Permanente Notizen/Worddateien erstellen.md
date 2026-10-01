13-05-2026
Tags: #Python  #FuE 
Status: #unextended

# 📘 Lektion 16 – Word-Dokumente automatisiert erstellen mit `python-docx`

## 🎯 Kernidee

Mit `python-docx` lassen sich Word-Dokumente **vollautomatisch** aus Python heraus erzeugen, befüllen und formatieren – ohne Word zu öffnen. Das ist die Grundlage für automatisierte Berichte aus Messdaten, Analyse-Ergebnissen oder ML-Vorhersagen.

---

## 📦 Installation

```bash
pip install python-docx
```

⚠️ **Wichtig:** Paketname ≠ Importname!

|Befehl|Schreibweise|
|---|---|
|`pip install`|`python-docx` (mit Bindestrich)|
|`import`|`docx` (ohne `python-`)|

```python
from docx import Document
```

---

## 🏗️ Die Hierarchie eines Word-Dokuments

Das mentale Modell für alles, was folgt:

```
Document
   ├── Paragraph
   │     ├── Run  ← kleinste Einheit mit einheitlicher Formatierung
   │     ├── Run
   │     └── Run
   ├── Paragraph
   └── Table
         └── Row → Cell → Paragraph → Run
```

**Faustregel für Runs:**

> Anzahl Formatwechsel + 1 = Anzahl Runs

Beispiel: _„Das Enzym **Hexokinase** ist wichtig."_ → **3 Runs** (normal / fett / normal)

---

## 🔧 Grund-Workflow

Jedes Skript folgt diesem Muster:

```python
from docx import Document

doc = Document()              # 1. Erstellen
doc.add_heading("Titel", 1)   # 2. Befüllen
doc.add_paragraph("Text")
doc.save("datei.docx")        # 3. Speichern
```

---

## 📝 Überschriften

```python
doc.add_heading("Hauptüberschrift", level=1)
doc.add_heading("Unterüberschrift", level=2)
doc.add_heading("Sub-Sub", level=3)
```

- `level=0` → Titel
- `level=1` → Hauptüberschrift
- `level=2`–`9` → tiefer verschachtelte Überschriften
- ⚠️ **Default ist `level=1`** – wenn nicht angegeben, immer Heading 1!

---

## 📄 Absätze und Runs

### Einfacher Absatz

```python
doc.add_paragraph("Ein normaler Text.")
```

### Absatz mit gemischter Formatierung

```python
p = doc.add_paragraph("Mein Name ist ")   # Variable speichern!
p.add_run("Paul").bold = True
p.add_run(" und ich lerne ")
p.add_run("Python").italic = True
p.add_run(".")
```

### Run-Eigenschaften

```python
run = p.add_run("Text")
run.bold = True
run.italic = True
run.underline = True
```

### Schriftgröße, -art, Farbe

```python
from docx.shared import Pt, RGBColor

run.font.size = Pt(14)
run.font.name = "Arial"
run.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)   # Rot
```

---

## 📊 Tabellen

### Struktur

```
Table → Row → Cell → Paragraph → Run
```

### Tabelle erstellen

```python
tabelle = doc.add_table(rows=3, cols=4)
tabelle.style = "Light Grid Accent 1"   # optional
```

### Auf Zellen zugreifen (0-basiert!)

```python
tabelle.cell(0, 0).text = "Kopfzelle oben links"
tabelle.cell(1, 2).text = "Zeile 1, Spalte 2"
```

### Dynamisch befüllen (**das wichtige Muster!**)

```python
daten = [
    {"name": "Hexokinase", "kcat": 450},
    {"name": "Glucokinase", "kcat": 65},
]

tabelle = doc.add_table(rows=1, cols=2)
tabelle.rows[0].cells[0].text = "Enzym"
tabelle.rows[0].cells[1].text = "kcat"

for eintrag in daten:
    zeile = tabelle.add_row().cells
    zeile[0].text = eintrag["name"]
    zeile[1].text = str(eintrag["kcat"])    # ⚠️ Pflicht-Cast!
```

⚠️ **`cell.text` braucht einen String** → bei Zahlen immer `str(...)` verwenden!

### Formatierter Zellinhalt

Wenn die Zelle fett/kursiv werden soll, nicht `cell.text =` benutzen, sondern:

```python
zelle = tabelle.cell(1, 0)
absatz = zelle.paragraphs[0]
run = absatz.add_run("Hexokinase")
run.bold = True
```

---

## 🖼️ Bilder einfügen

```python
from docx.shared import Cm, Inches

doc.add_picture("plot.png")                    # Originalgröße
doc.add_picture("plot.png", width=Cm(12))      # 12 cm breit
doc.add_picture("plot.png", width=Inches(4))   # 4 Zoll breit
```

⚠️ **Reine Zahlen** (`width=12`) funktionieren nicht – Einheit ist Pflicht.

Typischer Workflow mit matplotlib:

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("plot.png")

doc.add_picture("plot.png", width=Cm(15))
```

---

## 🎨 Weitere nützliche Elemente

### Listen

```python
doc.add_paragraph("Erstens", style="List Number")
doc.add_paragraph("Zweitens", style="List Number")

doc.add_paragraph("Apfel", style="List Bullet")
doc.add_paragraph("Birne", style="List Bullet")
```

### Ausrichtung

```python
from docx.enum.text import WD_ALIGN_PARAGRAPH

p = doc.add_paragraph("Zentriert")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
# Weitere: LEFT, RIGHT, JUSTIFY
```

### Seitenumbruch

```python
doc.add_page_break()
```

### Mehrere Absätze in einer Zelle

```python
zelle = tabelle.cell(0, 0)
zelle.text = "Erste Zeile"
zelle.add_paragraph("Zweite Zeile")
```

---

## ⚠️ Häufige Stolperfallen

|Problem|Ursache|Lösung|
|---|---|---|
|`ModuleNotFoundError: docx`|Falscher Import|`from docx import Document` (ohne `python-`)|
|Alle Überschriften gleich groß|`level` weggelassen|Explizit `level=2` etc. setzen|
|`TypeError: requires str`|Zahl in `cell.text`|Mit `str(...)` casten|
|`add_run()` schlägt fehl|Paragraph nicht in Variable|`p = doc.add_paragraph(...)`|
|`width=12` wirkt nicht|Einheit fehlt|`width=Cm(12)` oder `Inches(...)`|

---

## 🧠 Wiederkehrendes Muster: Daten → Dokument

Das **eigentliche Power-Pattern** dieser Lektion:

```python
for eintrag in datenliste:
    doc.add_heading(eintrag["titel"], level=2)
    doc.add_paragraph(eintrag["beschreibung"])

    tabelle = doc.add_table(rows=1, cols=2)
    # ... Tabelle aus Daten füllen ...

    doc.add_picture(eintrag["plot"], width=Cm(12))
    doc.add_page_break()
```

→ **Schleife über strukturierte Daten + Dokument-API = automatischer Bericht.** Dasselbe Schema funktioniert später auch für PDFs (Lektion 18), HTML-Reports oder LaTeX.

---

## 🔌 Bezug zum FuE-Projekt

|Word-Konzept|Entspricht später in PyTorch / Data Science|
|---|---|
|`Document → Paragraph → Run`|`nn.Module → nn.Sequential → nn.Linear` (Verschachtelung)|
|Liste von Dictionaries → Tabelle|DataFrame → Tensor / DataLoader-Batches|
|Schleife über Daten|Schleife über Enzym-Vorhersagen|
|matplotlib-Plot in `add_picture`|Loss-Kurven, Vorhersage-vs-Messwert-Plots|

---

## ✅ Mini-Cheatsheet (zum Abtippen)

```python
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Überschriften
doc.add_heading("Titel", level=1)
doc.add_heading("Unter", level=2)

# Absatz mit Runs
p = doc.add_paragraph("Normal ")
p.add_run("fett").bold = True
p.add_run(" und ")
p.add_run("kursiv").italic = True

# Tabelle
t = doc.add_table(rows=1, cols=2)
t.style = "Light Grid Accent 1"
t.rows[0].cells[0].text = "Spalte A"
t.rows[0].cells[1].text = "Spalte B"
for x, y in [(1, 2), (3, 4)]:
    z = t.add_row().cells
    z[0].text = str(x)
    z[1].text = str(y)

# Bild
doc.add_picture("plot.png", width=Cm(12))

# Liste & Umbruch
doc.add_paragraph("Punkt 1", style="List Number")
doc.add_page_break()

doc.save("bericht.docx")
```

# Referenzen