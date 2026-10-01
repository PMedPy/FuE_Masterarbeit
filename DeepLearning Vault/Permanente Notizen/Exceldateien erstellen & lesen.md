03-06-2026 Tags: #Python #FuE Status: #extended

# 📘 Lektion 17 – Excel-Dateien mit `openpyxl` und `pandas`

## 🎯 Kernidee

Excel-Dateien sind im Labor und in der Datenanalyse das Standard-Format. Python kann sie **vollautomatisch** erstellen, lesen, verändern und auswerten – mit zwei verschiedenen Werkzeugen, die unterschiedliche Stärken haben.

---

## 🛠️ Zwei Welten

|Werkzeug|Philosophie|Zweck|
|---|---|---|
|**`openpyxl`**|"Excel-Datei manipulieren"|Bericht-Excel mit Formatierung, Farben, Diagrammen|
|**`pandas`**|"Daten verarbeiten"|Einlesen, Rechnen, Filtern, Transformieren|

**Faustregel:**

- Bericht ist das Ziel → `openpyxl`
- Analyse ist das Ziel → `pandas`
- Kombinieren möglich (oft sinnvoll!)

---

## 📦 Installation & Import

```bash
pip install pandas openpyxl
```

```python
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
```

> ⚠️ Pandas benutzt openpyxl intern – aber wenn du dessen Funktionen direkt brauchst, musst du es **selbst importieren**.

---

# 🔧 Teil A: `openpyxl`

## 🏗️ Die Hierarchie – das Fundament

```
Workbook (wb)            (die ganze .xlsx-Datei)
   └── Worksheet (ws)    (ein Tabellenblatt/Sheet/Reiter)
         └── Cell        (eine einzelne Zelle, z.B. A1)
```

|Ebene|Beispielobjekt|Wie ansprechen|Was kann es?|
|---|---|---|---|
|Workbook|`wb`|`Workbook()` oder `load_workbook(...)`|Sheets enthalten, speichern|
|Worksheet|`ws = wb.active` oder `wb["Name"]`|per Name oder Position|Zellen enthalten, Spaltenbreiten, Title|
|Zeile|`ws[1]`|per Integer-Index|**Tupel** aus Cell-Objekten|
|Spalte|`ws["A"]`|per Buchstabe|**Tupel** aus Cell-Objekten|
|Zelle|`ws["A1"]` oder `ws.cell(row=1, column=1)`|per Adresse oder Index|`.value`, `.font`, `.fill`, `.alignment`|

**Beispiel zum Begreifen:**

```python
from openpyxl import Workbook

wb = Workbook()              # Workbook erstellen
ws = wb.active               # erstes Sheet holen
ws["A1"] = "Tag"             # → die Zelle A1 bekommt einen Wert

print(type(wb))              # <class 'openpyxl.workbook.workbook.Workbook'>
print(type(ws))              # <class 'openpyxl.worksheet.worksheet.Worksheet'>
print(type(ws["A1"]))        # <class 'openpyxl.cell.cell.Cell'>
print(type(ws[1]))           # <class 'tuple'>  ← Zeile = Tupel von Cells!
```

> ⚡ **MEINE SCHWACHSTELLE: Hierarchie-Verwechslung.** Ich habe öfter `ws` (Sheet) und `ws[1]` (Zeile) gleich behandelt. Aber: Ein Sheet hat `.column_dimensions`, eine Zeile nicht. Eine Zelle hat `.font`, eine Zeile nicht. **Immer fragen: auf welcher Ebene bin ich gerade?**

---

## 🆕 Workbook erstellen und speichern

|Aufgabe|Befehl|
|---|---|
|Neues Workbook|`wb = Workbook()`|
|Aktives Sheet holen|`ws = wb.active`|
|Sheet umbenennen|`ws.title = "Messdaten"`|
|Speichern|`wb.save("datei.xlsx")`|

**Beispiel – Minimale Excel-Datei:**

```python
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws.title = "Messdaten"
ws["A1"] = "Tag"
ws["B1"] = "Wert"
ws.cell(row=2, column=1, value="Montag")
ws.cell(row=2, column=2, value=21.5)
wb.save("messung.xlsx")
```

> ⚠️ openpyxl zählt Zeilen und Spalten **ab 1**, nicht ab 0 wie Listen!

---

## ✍️ Zellen ansprechen – zwei Schreibweisen

|Schreibweise|Wann sinnvoll?|
|---|---|
|`ws["A1"]`|Konkrete, feste Zellen (lesbar, kurz)|
|`ws.cell(row=1, column=1)`|In Schleifen, wenn Position eine Variable ist|

**Beispiel – Statische Adresse vs. Schleife:**

```python
# Statisch (Header):
ws["A1"] = "Tag"
ws["B1"] = "Wert"

# In Schleife (Daten):
messdaten = [("Mo", 21.5), ("Di", 22.0), ("Mi", 20.8)]
for i, (tag, wert) in enumerate(messdaten, start=2):
    ws.cell(row=i, column=1, value=tag)
    ws.cell(row=i, column=2, value=wert)
```

> ⚡ **MEINE SCHWACHSTELLE: Adresse als String in Schleifen.** Versuch nicht `ws[f"A{i}"]` – das funktioniert zwar, ist aber unsauber. In Schleifen **immer `ws.cell(...)`**.

---

## 📖 Lesen statt Schreiben

|Aufgabe|Befehl|
|---|---|
|Workbook laden|`wb = load_workbook("datei.xlsx")`|
|Sheet per Name|`ws = wb["Messdaten"]`|
|Sheet per Position|`ws = wb.worksheets[0]`|
|**Zellwert** lesen|`ws["B2"].value` ← `.value` nicht vergessen!|
|Alle Sheet-Namen|`wb.sheetnames`|

**Beispiel – Datei einlesen und auswerten:**

```python
from openpyxl import load_workbook

wb = load_workbook("messung.xlsx")
ws = wb.active

# Header lesen
print(ws["A1"].value)      # → "Tag"

# Alle Datenzeilen ab Zeile 2
for tag, wert in ws.iter_rows(min_row=2, values_only=True):
    print(f"{tag}: {wert}")
```

> ⚡ **MEINE SCHWACHSTELLE: Cell-Objekt vs. Wert.** `ws["A1"]` ist das **Objekt** (mit Metadaten wie Schriftart), `ws["A1"].value` ist der **Inhalt**. Wenn du in einem f-String schreibst `f"{ws['A1']}"` kriegst du den Müll `<Cell 'Sheet'.A1>`, nicht den Wert!

---

## 🔁 Über Zellen iterieren

|Aufgabe|Befehl|
|---|---|
|Eine Zeile|`for zelle in ws[1]: ...`|
|Eine Spalte|`for zelle in ws["A"]: ...`|
|Mehrere Zeilen (mit Werten)|`for row in ws.iter_rows(min_row=2, values_only=True): ...`|
|Mehrere Zeilen (mit Cell-Objekten)|`for row in ws.iter_rows(min_row=2): ...`|
|Über alle Sheets|`for ws in wb.worksheets: ...`|

**Beispiel – Werte und Cell-Objekte:**

```python
# Nur Werte (Tupel aus Zahlen/Strings)
for tag, wert in ws.iter_rows(min_row=2, values_only=True):
    print(f"{tag}: {wert}")
# Output: ('Mo', 21.5), ('Di', 22.0), ...

# Cell-Objekte (kannst Formatierung lesen/setzen)
for row in ws.iter_rows(min_row=2):
    for cell in row:
        if isinstance(cell.value, (int, float)) and cell.value > 22:
            cell.fill = PatternFill("solid", fgColor="FFFF00")
```

---

## 🎨 Formatierung – Font, Fill, Alignment

|Eigenschaft|Was sie tut|
|---|---|
|`zelle.font = Font(...)`|Schriftart, Größe, Farbe, fett, kursiv|
|`zelle.fill = PatternFill(...)`|Hintergrundfarbe der Zelle|
|`zelle.alignment = Alignment(...)`|Ausrichtung|

**Wichtige Parameter:**

```python
Font(bold=True, italic=False, size=14, color="FFFFFF", name="Calibri")
PatternFill("solid", fgColor="4472C4")
Alignment(horizontal="center", vertical="center", wrap_text=True)
```

> 💡 Farben sind Hex-Codes **ohne `#`**: `"FF0000"` = rot, `"FFFFFF"` = weiß, `"000000"` = schwarz.

**Beispiel – Header durchformatieren:**

```python
header_font = Font(bold=True, size=12, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="4472C4")
header_align = Alignment(horizontal="center")

# Style-Objekte EINMAL definieren, MEHRFACH zuweisen:
for zelle in ws[1]:                          # alle Zellen in Zeile 1
    zelle.font = header_font
    zelle.fill = header_fill
    zelle.alignment = header_align
```

> ⚡ **MEINE SCHWACHSTELLE: `font` mit `fill` verwechselt.** Copy-Paste-Fehler waren bei mir häufig: `zelle.font = header_fill`. Schau immer hin, **welche Eigenschaft** du gerade änderst. Tipp: Style-Objekte sprechend benennen (`header_font`, `header_fill`).

---

## 📐 Spaltenbreiten und Zeilenhöhen

|Aufgabe|Befehl|
|---|---|
|Spaltenbreite setzen|`ws.column_dimensions["A"].width = 20`|
|Zeilenhöhe setzen|`ws.row_dimensions[1].height = 25`|

> ⚡ **MEINE SCHWACHSTELLE (zweimal passiert!):** Niemals `ws.column_dimensions["A"] = 20` schreiben – das überschreibt das ganze Dimension-Objekt mit einer Zahl, und das Skript crasht beim Speichern mit `AttributeError: 'int' object has no attribute 'reindex'`. Immer **`.width`** dranhängen!

**Beispiel – Mehrere Spalten auf einmal:**

```python
# Mit Schleife (skaliert besser):
for spalte in "ABCDE":
    ws.column_dimensions[spalte].width = 15

# Einzeln (wenn unterschiedliche Breiten):
ws.column_dimensions["A"].width = 12
ws.column_dimensions["B"].width = 20
ws.column_dimensions["C"].width = 18
```

---

## ➕ Sheets verwalten

|Aufgabe|Befehl|
|---|---|
|Neues Sheet anhängen|`wb.create_sheet("Statistik")`|
|Neues Sheet an Position N|`wb.create_sheet("Übersicht", 0)`|
|Sheet löschen|`del wb["Statistik"]`|
|Alle Sheet-Namen|`wb.sheetnames`|

**Beispiel – Mehrere Sheets mit unterschiedlichen Inhalten:**

```python
wb = Workbook()
wb.active.title = "Statistik"           # erstes Sheet umbenennen

ws_rohdaten = wb.create_sheet("Rohdaten")
ws_warnungen = wb.create_sheet("Warnungen", 0)  # ganz vorne

print(wb.sheetnames)
# → ['Warnungen', 'Statistik', 'Rohdaten']
```

---

## 🧮 Formeln einfügen

|Aufgabe|Befehl|
|---|---|
|Formel als String|`ws["C2"] = "=A2*B2"`|
|Summe einer Spalte|`ws["B10"] = "=SUM(B2:B9)"`|

> ⚠️ Die Formel wird **als Text** geschrieben. Excel berechnet sie erst beim Öffnen der Datei.

---
# 🔗 Teil B: Kombinierter Workflow

Der echte Praxis-Workflow: **Pandas für die Datenverarbeitung, openpyxl für die optische Politur.**

```python
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

# === 1. Pandas: Daten bauen ===
df_lab = pd.concat([
    pd.Series([0.77, 0.86, 0.95], name="Absorption"),
    pd.Series([7.21, 7.18, 7.25, 7.22], name="pH"),
], axis=1)

statistik = pd.DataFrame({
    "Mittelwert":         df_lab.mean(),
    "Standardabweichung": df_lab.std(),
    "Anzahl der Werte":   df_lab.count(),
})

# === 2. Pandas: in Excel exportieren ===
with pd.ExcelWriter("bericht.xlsx") as writer:
    df_lab.to_excel(writer, sheet_name="Rohdaten", index=False)
    statistik.to_excel(writer, sheet_name="Statistik", index=True)

# === 3. openpyxl: Optik nachträglich polieren ===
header_font = Font(bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="1265FF")

wb = load_workbook("bericht.xlsx")
for ws in wb.worksheets:
    for zelle in ws[1]:                     # Header-Zeile
        zelle.font = header_font
        zelle.fill = header_fill
        zelle.alignment = Alignment(horizontal="center")
    for spalte in "ABCDE":
        ws.column_dimensions[spalte].width = 18

wb.save("bericht.xlsx")
```

---

# 🚨 Meine 7 wichtigsten Schwachstellen – kompakt

|#|Fehler|Folge|Korrekt|
|---|---|---|---|
|1|`ws.column_dimensions["A"] = 20`|Crash beim Speichern|`ws.column_dimensions["A"].width = 20`|
|2|`ws["A1"]` in f-String|"Cell 'Sheet'.A1" als Text|`ws["A1"].value`|
|3|`for ws in (ws1[1], ws2[1])`|`ws` ist Zeile, kein Sheet|`for ws in (ws1, ws2):`|
|4|`Index=False` (groß)|`TypeError: unexpected keyword`|`index=False`|
|5|`df["X"] > 5 and df["Y"] < 3`|TypeError|`(df["X"] > 5) & (df["Y"] < 3)`|
|6|`df["Reaktor"]` nach `groupby`|KeyError|`df.reset_index()` oder Index nutzen|
|7|`df_filt["x"] = ...` (nach Filter)|SettingWithCopyWarning|`df_filt = df[mask].copy()`|

---

# ⚖️ Vergleich: openpyxl vs. pandas

|Aufgabe|openpyxl|pandas|
|---|---|---|
|Datei erstellen|`Workbook()`|`df.to_excel(...)`|
|Datei einlesen|`load_workbook(...)`|`pd.read_excel(...)`|
|Einzelne Zelle ändern|✅ einfach|❌ umständlich|
|Mittelwert einer Spalte|`for`-Loop nötig|`df["x"].mean()`|
|Zellfarbe setzen|✅|❌|
|Daten filtern|`for`-Loop|`df[df["x"] > 10]`|
|Mehrere Sheets schreiben|✅|✅ (mit `ExcelWriter`)|

---

# 🔌 Bezug zum FuE-Projekt

|Excel-Konzept|Entspricht in PyTorch / Data Science|
|---|---|
|`DataFrame` mit Spalten|Input-Features für ML-Modelle|
|`df["spalte"].mean()`|Statistische Vorbereitung (Normalisierung)|
|`pd.read_excel(...)`|Standard-Weg, Trainingsdaten einzulesen|
|Vektor-Operationen|Direktes Vorbild für `tensor + tensor`, `tensor * 2`|
|`df.loc[bedingung]`|Filter für Trainings-/Validierungs-Split|
|`df.values`|Brücke zu NumPy/PyTorch: `torch.tensor(df.values)`|

---

# 🎓 Take-aways

- **openpyxl** denkt in **Zellen** – wie ein Excel-Mensch
- **pandas** denkt in **Spalten** – wie ein Datenanalyst
- Beide kombinierst du in echten Projekten oft im selben Skript
- Pandas-Vektor-Operationen sind das **Vorbild** für alles, was du später in NumPy und PyTorch siehst
- Bei Fehlern: **Auf welcher Hierarchie-Ebene bin ich gerade?**

# Referenzen

[[Worddateien erstellen]]