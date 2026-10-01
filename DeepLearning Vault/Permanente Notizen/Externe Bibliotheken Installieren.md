Tags: #Python  #FuE 
Status: #unextended


# 📘 Lektion 15: Externe Bibliotheken mit `pip` – Quickinfo

> Pakete installieren und nutzen, virtuelle Umgebungen verwalten

---

## 📦 Was sind externe Bibliotheken?

Pakete, die andere Entwickler:innen geschrieben haben, gehostet auf **PyPI** (Python Package Index). Beispiele:

- **NumPy / Pandas / Matplotlib** – Data Science
- **python-docx / openpyxl / reportlab** – Office-Dateien
- **requests** – Webseiten abrufen
- **PyTorch / scikit-learn** – Machine Learning
- **rich** – schöne Konsolen-Ausgaben

---

## 🛠 `pip` – Pythons Paket-Manager

`pip` läuft im **Terminal**, nicht im Python-Skript.

### Wichtige Befehle

| Befehl                            | Wirkung                              |
| --------------------------------- | ------------------------------------ |
| `pip install paketname`           | Paket installieren                   |
| `pip install paketname==2.1.0`    | bestimmte Version                    |
| `pip install --upgrade paketname` | aktualisieren                        |
| `pip uninstall paketname`         | entfernen                            |
| `pip list`                        | alle installierten Pakete            |
| `pip show paketname`              | Details zu einem Paket               |
| `pip freeze > requirements.txt`   | aktuelle Pakete in Datei exportieren |
| `pip install -r requirements.txt` | aus Datei alles installieren         |

---

## 🐍 Pakete im Code nutzen

### Drei Import-Varianten

```python
import requests                    # ganzes Paket
antwort = requests.get("...")

from datetime import datetime      # nur einzelne Klasse/Funktion
jetzt = datetime.now()

import numpy as np                 # Kürzel (Konvention!)
zahlen = np.array([1, 2, 3])
```

### Bekannte Konventionen

|Paket|Übliches Kürzel|
|---|---|
|`numpy`|`np`|
|`pandas`|`pd`|
|`matplotlib.pyplot`|`plt`|
|`seaborn`|`sns`|

---

## 📂 Virtuelle Umgebungen (`venv`)

Isolierter Python-Container pro Projekt → keine Konflikte zwischen Paketversionen.

### Anlegen & nutzen (Mac/Linux)

```bash
# In Projektordner gehen
cd ~/Documents/MeinProjekt

# venv anlegen
python3 -m venv .venv

# Aktivieren
source .venv/bin/activate

# Erkennen am Prompt: (.venv) erscheint vorne

# Pakete installieren (landen NUR im venv)
pip install rich

# Deaktivieren
deactivate
```

### Konventionen

- Ordnername **`.venv`** (mit Punkt) ist Standard
- Eine venv pro Projekt
- venv **nie** in iCloud/Dropbox/OneDrive – führt zu kaputten Installationen
- `.venv` in `.gitignore` aufnehmen (wird nicht versioniert)

---

## 🔄 Conda – Alternative zu `venv`

Bevorzugt für Data Science / Machine Learning, da auch nicht-Python-Abhängigkeiten gemanaged werden (CUDA, MKL, etc.).

### Wichtige Befehle

```bash
conda create -n projektname python=3.12   # neue Umgebung
conda activate projektname                # aktivieren
conda deactivate                           # deaktivieren
conda install paketname                    # Paket installieren
conda env list                             # alle Umgebungen anzeigen
conda config --set auto_activate_base false   # base nicht auto-aktivieren
```

### venv vs. Conda

||venv|Conda|
|---|---|---|
|Schlank|✅|❌ (größer)|
|Für reine Python-Projekte|✅|✅|
|Für ML/DS mit CUDA|⚠️ pip|✅ besser|
|Bringt Python mit|❌ braucht systemweit|✅ je Umgebung|

---

## 📌 Best Practices

1. **Pro Projekt eine eigene Umgebung** (venv oder conda)
2. **Nie global installieren**, wenn möglich
3. **`requirements.txt`** pflegen – machst Projekte teilbar
4. **Code lokal halten**, nicht in iCloud
5. In VS Code: **`Cmd + Shift + P → Python: Select Interpreter`** zur richtigen Umgebung
6. Am Prompt **immer prüfen**: welches venv ist aktiv? (`which python`)

---

## ⚠️ Häufige Stolpersteine

|Problem|Ursache|Lösung|
|---|---|---|
|`command not found: pip`|venv nicht aktiv|`source .venv/bin/activate`|
|`ModuleNotFoundError` im Skript|falsches Python|Interpreter prüfen|
|Pakete "verschwinden" zwischen Sitzungen|venv nicht aktiviert|aktivieren!|
|pip kaputt nach Erstellung|iCloud/Sync hat venv beschädigt|venv lokal anlegen|
|Conflicting versions|Pakete wurden global installiert|venv nutzen|

---

## 🎯 Bezug zum FuE-Projekt (PyTorch)

Späterer FuE-Workflow für Deep Learning auf Aminosäure-Sequenzen:

```bash
# Eigene Conda-Umgebung
conda create -n fue python=3.12
conda activate fue

# PyTorch (mit GPU/CPU-Support)
conda install pytorch torchvision -c pytorch

# Bioinformatik-Pakete
pip install biopython transformers
```

→ Conda zentral, weil PyTorch + CUDA-Treiber konsistent zusammenspielen müssen.

---

#python #lernkurs #pip #venv #lektion15

---

---

# 🛠 Übungsaufgabe Lektion 15: Geburtstagstabelle mit `rich`

## 📋 Aufgabe

Schreib ein Skript `geburtstage_anzeige.py`, das deine `geburtstage.txt` aus Lektion 14 lädt und als **bunte Tabelle** in der Konsole ausgibt.

### Erwartete Ausgabe

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃            🎂 Geburtstagsliste              ┃
┣━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Name             ┃ Geburtsdatum             ┃
┣━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Anna             ┃ 15.03.1995               ┃
┃ Ben              ┃ 22.07.1988               ┃
┃ Clara            ┃ 03.11.2001               ┃
┗━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

(Die genaue Optik bestimmt `rich` selbst – sieht meist noch schöner aus.)

---

## 🔧 Anforderungen

1. Verwende das Paket **`rich`** (im venv installieren mit `pip install rich`)
2. **Lade** die Geburtstage aus `geburtstage.txt` (Format: `Name;Datum`)
3. **Erstelle** eine `Table` mit Titel und zwei Spalten
4. **Fülle** die Tabelle mit den geladenen Daten
5. **Fang ab**, falls die Datei nicht existiert → freundliche Meldung

---

## 💡 Tipps & Code-Bausteine

### `rich` importieren und Tabelle bauen

```python
from rich import print
from rich.table import Table

tabelle = Table(title="🎂 Geburtstagsliste")
tabelle.add_column("Name", style="cyan")
tabelle.add_column("Geburtsdatum", style="magenta", justify="center")

tabelle.add_row("Anna", "15.03.1995")

print(tabelle)    # ← rich.print, nicht das normale print!
```

### Style-Optionen für Spalten

|Argument|Beispiel|Wirkung|
|---|---|---|
|`style`|`style="cyan"`|Textfarbe|
|`justify`|`justify="center"`|Ausrichtung (left/center/right)|
|`header_style`|`header_style="bold green"`|Stil der Überschrift|

### Datei einlesen (kennst du schon)

```python
with open("geburtstage.txt", "r") as datei:
    for zeile in datei:
        daten = zeile.strip().split(";")
        # daten[0] = Name, daten[1] = Datum
```

---

## ⏱ Zeit: ~15 Minuten

Schreib das Skript selbst, führ es aus mit `python geburtstage_anzeige.py` (im aktivierten venv!) und schick mir das Ergebnis.

> 💡 **Bezug zum FuE-Projekt:** `rich` wirst du später beim PyTorch-Training einsetzen, um Loss-Werte und Metriken pro Epoche schön anzuzeigen. Die Logik _"Daten einlesen → in Tabellenform anzeigen"_ ist auch genau das, was du später beim Auswerten von Modellvorhersagen brauchst (Aminosäuresequenz → vorhergesagte Parameter → tabellarische Übersicht). 🧬

Viel Erfolg! 💪

---

# Referenzen