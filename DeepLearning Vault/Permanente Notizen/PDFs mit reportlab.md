
03-06-2026 Tags: #Python #FuE #PDF Status: #unextended

# 📘 Lektion 18 – PDF-Dateien mit `reportlab`

## 🎯 Kernidee

PDFs sind das **finale Berichts-Format** in der Wissenschaft. Sie sehen für alle Leser exakt gleich aus – Schriftarten, Positionen, Bilder sind eingebettet. Python kann sie **vollautomatisch** erzeugen, was perfekt für reproduzierbare Laborberichte und ML-Auswertungen ist.

---

## 🌍 Drei Welten – wie unterscheidet sich PDF?

|Format|Denkmodell|Layout-Entscheidung|
|---|---|---|
|**Word (.docx)**|Fließtext|Leser passt an|
|**Excel (.xlsx)**|Zellengitter|Leser passt an|
|**PDF (.pdf)**|Fixierte Seiten|Autor entscheidet|

> 💡 **Konsequenz:** PDFs sind _präsentations-final_. Was du baust, sieht jeder Empfänger exakt gleich.

---

## 📦 Installation & Import

```bash
pip install reportlab
```

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
```

> ⚠️ ReportLab importiert man **selektiv**, nicht als ganzes Modul. Konvention in Python: `from xy import abc`, nicht `import xy` und dann `xy.abc.def`.

---

# 🛠️ Zwei Ansätze: Canvas vs. Flowable

|Ansatz|Denken|Wann?|
|---|---|---|
|**Canvas** (low-level)|"Maler" – jedes Element an Pixel-Position zeichnen|Pixelgenaue Kontrolle, Hintergründe, Wasserzeichen|
|**Flowable / Platypus** (high-level)|"Story" – Elemente in Liste, Bibliothek macht Layout|Berichte, Tabellen, alles mit automatischem Umbruch|

**Wir nehmen Flowable.** Es ist konzeptuell ähnlich zu `python-docx`.

---

# 🏗️ Das mentale Modell von Platypus

```
SimpleDocTemplate           ← Rahmen (Seitengröße, Ränder, Dateiname)
        ↑
   doc.build(story)
        ↑
    story = [Flowable, Flowable, ...]   ← Liste von Elementen
        ↑
    ParagraphStyle / TableStyle         ← Aussehen
```

**Vergleich zu python-docx:**

|python-docx|reportlab Platypus|
|---|---|
|`Document()`|`SimpleDocTemplate(...)`|
|`doc.add_paragraph(...)`|`story.append(Paragraph(...))`|
|`doc.save(...)`|`doc.build(story)`|
|Run-Eigenschaften|`ParagraphStyle`|

> 💡 Hauptunterschied: Du baust **erst die ganze Liste**, dann übergibst du sie. Bei python-docx fügst du Schritt für Schritt ans Dokument an.

---

## 🆕 Minimales Beispiel – die kleinste PDF

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

doc = SimpleDocTemplate("test.pdf", pagesize=A4)
styles = getSampleStyleSheet()

story = []
story.append(Paragraph("Mein erster PDF-Bericht", styles["Heading1"]))
story.append(Paragraph("Das ist ein Absatz.", styles["Normal"]))

doc.build(story)
```

Vier Schritte – immer dieselben:

1. **Dokument** anlegen
2. **Stile** holen
3. **Story-Liste** befüllen
4. **`build`** aufrufen

---

# 📐 Seitengröße, Ränder, Einheiten

|Aufgabe|Befehl|
|---|---|
|Seitengröße A4|`pagesize=A4`|
|Seitengröße Letter|`pagesize=letter` (aus `pagesizes`)|
|Eigene Größe|`pagesize=(595, 842)` (Punkte)|
|Ränder setzen|`leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm`|

```python
from reportlab.lib.units import cm

doc = SimpleDocTemplate(
    "bericht.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2*cm,
)
```

> ⚠️ **Einheit ist standardmäßig Punkt** (1 Inch = 72 Punkt = 2.54 cm). Mit `cm` und `mm` kannst du in Zentimetern bzw. Millimetern arbeiten. Beispiel: `2*cm` = 56.7 Punkt.

> 📍 **Koordinatensystem:** Punkt (0, 0) ist **unten links**, nicht oben links! Wichtig nur beim Canvas-Ansatz – Flowables behandeln das automatisch.

---

# 📝 Flowables – die Story-Elemente

|Flowable|Was es ist|
|---|---|
|`Paragraph`|Ein Textabsatz mit Stil|
|`Spacer`|Vertikaler Abstand|
|`Table`|Eine Tabelle|
|`Image`|Ein Bild|
|`PageBreak`|Erzwungener Seitenumbruch|
|`HRFlowable`|Horizontale Linie|

```python
from reportlab.platypus import Paragraph, Spacer, Table, PageBreak, Image, HRFlowable

story = []
story.append(Paragraph("Titel", styles["Heading1"]))
story.append(Spacer(1, 12))                    # 12 Punkt vertikaler Abstand
story.append(Paragraph("Ein Absatz mit Text.", styles["Normal"]))
story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
story.append(PageBreak())                       # neue Seite
story.append(Paragraph("Zweite Seite", styles["Heading1"]))
```

> 💡 **Reihenfolge zählt!** Flowables werden in der Reihenfolge platziert, in der sie in der Liste stehen.

---

# 🔠 Text und Stile

## Stile aus Standard-Set holen und anpassen

```python
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()
# Verfügbar: "Normal", "Heading1", "Heading2", "Heading3",
#            "Title", "Bullet", "Definition", "Code", "BodyText", ...

# Anpassen:
styles["Normal"].fontSize = 11
styles["Heading1"].textColor = colors.darkblue
```

## Eigene Stile definieren

```python
from reportlab.lib.styles import ParagraphStyle

mein_titel = ParagraphStyle(
    "MeinTitel",                       # interner Name (Pflicht)
    fontName="Helvetica-Bold",
    fontSize=18,
    textColor=colors.darkblue,
    alignment=1,                       # 0=links, 1=zentriert, 2=rechts, 4=Blocksatz
    spaceBefore=12,
    spaceAfter=18,
    leading=22,                        # Zeilenabstand
)

story.append(Paragraph("Mein Titel", mein_titel))
```

## Verfügbare Standard-Schriftarten

|Schriftart|Varianten|
|---|---|
|`Helvetica`|`Helvetica-Bold`, `Helvetica-Oblique`, `Helvetica-BoldOblique`|
|`Times-Roman`|`Times-Bold`, `Times-Italic`, `Times-BoldItalic`|
|`Courier`|`Courier-Bold`, `Courier-Oblique`, `Courier-BoldOblique`|
|`Symbol`|(für mathematische Symbole)|
|`ZapfDingbats`|(für Symbole/Pfeile)|

> ⚠️ **In ReportLab gibt es kein `bold=True`!** Fett ist eine **eigene Schriftart**: `fontName="Helvetica-Bold"`. Das ist anders als python-docx oder openpyxl!

## Inline-Formatierung im Text (HTML-ähnlich)

Im Text eines `Paragraph` kannst du **HTML-Tags** zum Formatieren benutzen:

```python
text = """
Das ist <b>fett</b>, das ist <i>kursiv</i>,
und das ist <font color="red">rot</font>.
<br/>Eine neue Zeile innerhalb des Absatzes.
"""
story.append(Paragraph(text, styles["Normal"]))
```

Erlaubte Tags: `<b>`, `<i>`, `<u>`, `<font color="..." size="...">`, `<br/>`, `<sub>`, `<sup>`, `<para>`.

---

# 📊 Tabellen

Tabellen sind ein eigener Flowable mit **separater Stil-Logik** (TableStyle, nicht ParagraphStyle).

## Tabelle erstellen

```python
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

daten = [
    ["Tag", "Temperatur", "pH"],       # Header-Zeile
    ["Mo",  21.5,         7.21],
    ["Di",  22.0,         7.18],
    ["Mi",  20.8,         7.25],
]

tabelle = Table(daten)
story.append(tabelle)
```

## Tabelle formatieren

```python
tabelle = Table(daten)
tabelle.setStyle(TableStyle([
    # Header-Zeile
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
    ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN",      (0, 0), (-1, 0), "CENTER"),

    # Alle Zellen
    ("FONTSIZE",   (0, 0), (-1, -1), 10),
    ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
    
    # Datenzeilen
    ("ALIGN",      (1, 1), (-1, -1), "RIGHT"),    # Zahlen rechts
    ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),   # erste Spalte fett
]))
```

> 💡 **Koordinaten in TableStyle:** `(spalte, zeile)`, **0-basiert**.
> 
> - `(0, 0)` = oben links
> - `(-1, 0)` = letzte Spalte, erste Zeile (ganze Header-Zeile = `(0,0)` bis `(-1,0)`)
> - `(-1, -1)` = letzte Zelle (alle Zellen = `(0,0)` bis `(-1,-1)`)

## Tabellen-Style-Befehle (häufig genutzt)

|Befehl|Wirkung|
|---|---|
|`"BACKGROUND"`|Hintergrundfarbe|
|`"TEXTCOLOR"`|Textfarbe|
|`"FONTNAME"`|Schriftart|
|`"FONTSIZE"`|Schriftgröße|
|`"ALIGN"`|horizontale Ausrichtung: `"LEFT"`, `"CENTER"`, `"RIGHT"`|
|`"VALIGN"`|vertikale Ausrichtung: `"TOP"`, `"MIDDLE"`, `"BOTTOM"`|
|`"GRID"`|Alle Linien (Format: Dicke, Farbe)|
|`"BOX"`|Nur Außenrand|
|`"LINEBELOW"`|Linie unter angegebenen Zellen|
|`"BOTTOMPADDING"`|Innenabstand unten|

## Spaltenbreiten

```python
tabelle = Table(daten, colWidths=[3*cm, 4*cm, 4*cm])
```

---

# 🖼️ Bilder einbetten

```python
from reportlab.platypus import Image
from reportlab.lib.units import cm

story.append(Image("plot.png", width=12*cm, height=8*cm))
```

> 💡 Wenn du `matplotlib`-Plots einbinden willst (Lektion 25): erst `plt.savefig("plot.png")`, dann mit `Image(...)` einbetten. Diesen Workflow wirst du im FuE-Projekt ständig nutzen.

---

# 🎨 Farben

|Variante|Beispiel|
|---|---|
|Eingebaute Namen|`colors.red`, `colors.darkblue`, `colors.lightgrey`|
|Hex-Code|`colors.HexColor("#4472C4")`|
|RGB (0–1)|`colors.Color(0.27, 0.45, 0.77)`|

```python
from reportlab.lib import colors
mein_blau = colors.HexColor("#1265FF")
```

---

# 📋 Vollständiges Beispiel – Laborbericht

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

# 1. Dokument
doc = SimpleDocTemplate(
    "laborbericht.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

# 2. Stile
styles = getSampleStyleSheet()
mein_titel = ParagraphStyle(
    "MeinTitel",
    parent=styles["Heading1"],
    fontSize=20, textColor=colors.HexColor("#1265FF"),
    alignment=1, spaceAfter=20,
)

# 3. Story bauen
story = []
story.append(Paragraph("Laborbericht: Enzymkinetik", mein_titel))

story.append(Paragraph("Datum: 26.06.2026", styles["Normal"]))
story.append(Spacer(1, 20))

story.append(Paragraph("Messdaten", styles["Heading2"]))
daten = [
    ["Tag", "Temperatur", "pH"],
    ["Mo", "21.5", "7.21"],
    ["Di", "22.0", "7.18"],
    ["Mi", "20.8", "7.25"],
]
tabelle = Table(daten, colWidths=[3*cm, 4*cm, 4*cm])
tabelle.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1265FF")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(tabelle)

story.append(PageBreak())
story.append(Paragraph("Auswertung", styles["Heading2"]))
story.append(Paragraph("Die Messreihe zeigt <b>stabile</b> Bedingungen.", styles["Normal"]))

# 4. Bauen
doc.build(story)
```

---

# ⚠️ Häufige Stolperfallen

|Problem|Ursache|Lösung|
|---|---|---|
|`bold=True` funktioniert nicht|Gibt's nicht in ReportLab|`fontName="Helvetica-Bold"`|
|Tabelle wird nicht formatiert|`Table(daten)` allein hat keinen Style|`tabelle.setStyle(TableStyle([...]))`|
|Zahlen werden zu lang formatiert|Floats werden mit allen Stellen geschrieben|Vorher `f"{wert:.2f}"` formatieren|
|Tabelle bricht in der Mitte um|Default-Verhalten|`KeepTogether([tabelle])` aus `platypus`|
|Bilder zu groß / zu klein|Default-Pixelgröße|`width` und `height` mit `cm`/`mm` angeben|
|Datei lässt sich nicht überschreiben|PDF in Reader noch offen|Reader schließen, dann neu bauen|
|HTML-Tags erscheinen als Text|Falscher Stil|Im `Paragraph` werden `<b>`, `<i>` usw. interpretiert|

---

# 🔌 Bezug zum FuE-Projekt

|PDF-Konzept|Im FuE-Projekt|
|---|---|
|`SimpleDocTemplate`|Berichts-Template für jede Analyse|
|`story.append(Paragraph(...))`|Pro Enzym automatisch Abschnitt einfügen|
|`Table`|kcat/Km-Tabellen mit Vorhersagewerten|
|`Image` (matplotlib-Plot)|Loss-Kurven, Vorhersage-vs-Messwert|
|`PageBreak`|Eine Seite pro Enzym|

> 💡 Der typische FuE-Workflow später:
> 
> 1. PyTorch-Modell macht Vorhersagen → DataFrame
> 2. matplotlib zeichnet Plots → PNG
> 3. reportlab packt alles in PDF-Bericht → fertig

---

# ✅ Mini-Cheatsheet

```python
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

doc = SimpleDocTemplate("bericht.pdf", pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm)
styles = getSampleStyleSheet()

story = []
story.append(Paragraph("Titel", styles["Heading1"]))
story.append(Spacer(1, 12))
story.append(Paragraph("Ein Absatz mit <b>Hervorhebung</b>.", styles["Normal"]))

daten = [["Spalte A", "Spalte B"], [1, 2], [3, 4]]
t = Table(daten)
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
]))
story.append(t)

doc.build(story)
```

---

# 🎓 Take-aways

- PDF ist **autor-gesteuertes Layout** – jeder Leser sieht es gleich
- Platypus-Flowables ähneln python-docx, aber alles wird in **eine Liste** gepackt
- Stile sind **explizit**: `ParagraphStyle`, `TableStyle` – kein implizites "bold"
- Schriftart-Variante (`-Bold`, `-Italic`) macht die Formatierung, nicht Properties
- Bei Stolpersteinen: erst die **Tabellen-Koordinaten** prüfen (Format `(spalte, zeile)`!)

# Referenzen

[[Worddateien erstellen]] [[Exceldateien erstellen & lesen]]