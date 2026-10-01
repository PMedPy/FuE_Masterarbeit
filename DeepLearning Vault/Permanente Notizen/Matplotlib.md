20-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Quickinfo: Matplotlib

## Was ist Matplotlib und wozu?

**Matplotlib** ist die Standardbibliothek zum Erstellen von Diagrammen in Python. Das gängige Modul ist `matplotlib.pyplot`, fast immer importiert als `plt`:

```python
import matplotlib.pyplot as plt
```

Damit zeichnest du Linien-, Streu-, Balkendiagramme, 3D-Plots und mehr – und exportierst sie als PNG, SVG, PDF.

## Das zentrale Konzept: Figure vs. Axes

Das musst du auseinanderhalten, sonst verwirrt dich die API dauerhaft:

|Begriff|Was es ist|
|---|---|
|**Figure**|Die gesamte Leinwand / das Bild als Ganzes|
|**Axes**|EIN Koordinatensystem _innerhalb_ der Figure (das Plot-Feld)|
|**Axis**|Eine einzelne Achse (x oder y) – NICHT mit "Axes" verwechseln!|

Eine Figure kann mehrere Axes enthalten (= Subplots). "Axes" klingt wie "Achse", meint aber das **ganze Plotfeld**.

## Der empfohlene Stil: explizit (object-oriented)

Es gibt zwei Schreibweisen. Nutze die **explizite** – sie skaliert sauber auf mehrere Plots und ist Profi-Standard:

```python
fig, ax = plt.subplots()      # Tupel-Unpacking: (Figure, Axes)
ax.plot(x, y)                 # auf dem Axes zeichnen
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
```

`plt.subplots()` gibt ein **Tupel** `(Figure, Axes)` zurück – das packst du in `fig, ax` aus.

> [!warning] ⚡ MEINE SCHWACHSTELLE: wechselnde Form von `axes` Die Rückgabe von `plt.subplots()` ändert ihre Form je nach Anzahl Plots:
> 
> - `plt.subplots()` → **ein** Axes-Objekt → `ax.plot(...)`
> - `plt.subplots(3, 1)` → **1D-Array** → `axes[0]`, `axes[1]`, `axes[2]`
> - `plt.subplots(2, 2)` → **2D-Array** → `axes[0, 1]` usw. Wer bei mehreren Plots `ax.plot` statt `axes[i].plot` schreibt, bekommt einen Fehler. Immer fragen: "Ist `axes` hier ein Array oder ein einzelnes Objekt?"

## Mehrere Plots in EINER Figure (untereinander)

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, x, color="crimson")
axes[0].set_ylabel("x(t)")

axes[1].plot(t, y, color="seagreen")
axes[1].set_ylabel("y(t)")

axes[2].plot(t, z, color="royalblue")
axes[2].set_ylabel("z(t)")
axes[2].set_xlabel("Zeit t")

fig.suptitle("Titel über allem")
fig.tight_layout()
plt.show()
```

- `subplots(3, 1)` = 3 Zeilen, 1 Spalte → untereinander.
- `sharex=True` = gemeinsame x-Achse (x-Label nur einmal unten).
- `fig.tight_layout()` = verhindert überlappende Beschriftungen.

## Mehrere separate Figures

Wenn du mehrere **eigenständige** Figures erzeugst, zeigt **ein** `plt.show()` am Ende ALLE offenen Fenster:

```python
fig1, ax1 = plt.subplots()
ax1.plot(t, x)

fig2, ax2 = plt.subplots()
ax2.plot(t, y)

plt.show()   # öffnet BEIDE Fenster
```

Unterschied merken: `subplots(3,1)` = drei Plots in _einem_ Bild; mehrere `plt.subplots()`-Aufrufe = mehrere _getrennte_ Bilder.

## Cheatsheet: Grundgerüst

|Befehl|Wirkung|
|---|---|
|`fig, ax = plt.subplots()`|Eine Figure + ein Axes|
|`plt.subplots(z, s)`|Gitter aus z Zeilen, s Spalten|
|`figsize=(b, h)`|Größe in Zoll (Breite, Höhe)|
|`plt.show()`|Alle offenen Figures anzeigen|
|`fig.tight_layout()`|Layout entzerren|
|`fig.suptitle("...")`|Gesamttitel über der Figure|

## Cheatsheet: Zeichnen

|Befehl|Wirkung|
|---|---|
|`ax.plot(x, y)`|Linie (verbindet Punkte in Reihenfolge)|
|`ax.scatter(x, y)`|Streudiagramm (einzelne Punkte)|
|`ax.bar(x, höhe)`|Balkendiagramm|
|`ax.plot(x, y, linewidth=0.5)`|Liniendicke (dünn bei vielen Punkten!)|
|`ax.plot(x, y, color="crimson")`|Farbe|
|`ax.plot(x, y, "--")`|Linienstil (gestrichelt)|
|`ax.plot(x, y, label="...")`|Beschriftung für Legende|

## Cheatsheet: Beschriften & Achsen

| Befehl                   | Wirkung                                       |
| ------------------------ | --------------------------------------------- |
| `ax.set_xlabel("...")`   | x-Achsen-Beschriftung                         |
| `ax.set_ylabel("...")`   | y-Achsen-Beschriftung                         |
| `ax.set_title("...")`    | Titel dieses einen Axes                       |
| `ax.legend()`            | Legende anzeigen (braucht `label=`)           |
| `ax.set_aspect("equal")` | Beide Achsen gleich skaliert (nicht verzerrt) |
| `ax.set_xlim(a, b)`      | x-Bereich festlegen                           |
| `ax.grid(True)`          | Gitternetz                                    |
| `ax.axis("off")`         | Achsen/Rahmen ausblenden (gut für Kunst!)     |

## Cheatsheet: Speichern (für PNG-Export)

| Befehl                    | Wirkung                                         |
| ------------------------- | ----------------------------------------------- |
| `fig.savefig("bild.png")` | Als PNG speichern                               |
| `dpi=300`                 | Auflösung (300 = Druckqualität)                 |
| `transparent=True`        | Transparenter Hintergrund                       |
| `bbox_inches="tight"`     | Ränder eng beschneiden                          |
| `pad_inches=0`            | Rand um den beschnittenen Bereich (0 = randlos) |

Beispiel für sauberen Export:

```python
fig.savefig("attraktor.png", dpi=300, transparent=True,
            bbox_inches="tight", pad_inches=0)
```

> [!warning] ⚡ MEINE SCHWACHSTELLE: Reihenfolge savefig vor show **Erst `savefig()`, dann `plt.show()`.** In manchen Umgebungen leert `plt.show()` die Figure, und ein danach aufgerufenes `savefig()` speichert ein leeres Bild.
> 
> ```python
> fig.savefig("bild.png", dpi=300, transparent=True, bbox_inches="tight")
> plt.show()   # NACH dem Speichern
> ```

### 3D-Export: zusätzliche Transparenz-Falle

Bei 3D-Plots reicht `transparent=True` **nicht** – die grauen "Wände" (panes) des Koordinatensystems bleiben sonst als Kasten sichtbar. Lösung: Achsen komplett ausblenden, dann erst speichern.

```python
ax.set_axis_off()    # entfernt Achsen UND graue Wände -> echte Transparenz
fig.savefig("roessler.png", dpi=300, transparent=True,
            bbox_inches="tight", pad_inches=0)
plt.show()
```

Faustregel: **`set_axis_off()` ist bei 3D Voraussetzung für ein wirklich freigestelltes PNG.**

### Speicherort

`savefig("bild.png")` speichert **relativ zum Arbeitsverzeichnis** (dort, wo das Skript/Notebook liegt). Für einen festen Ort einen Pfad angeben:

```python
fig.savefig(r"C:\Dev\Collage\roessler.png", dpi=300, transparent=True)
```

## 3D-Plots

3D-Plotting ist eingebaut, aber das Axes muss explizit als 3D deklariert werden. Statt `plt.subplots()` erzeugt man die Figure und hängt ein 3D-Axes an:

```python
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")

ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5, color="#1400f2")
plt.show()
```

`ax.plot(x, y, z)` ist die direkte Erweiterung des 2D-Falls – einfach drei Koordinaten-Arrays statt zwei.

### Blickwinkel steuern: `view_init`

```python
ax.view_init(elev=30, azim=45)
```

|Parameter|Bedeutung|Werte|
|---|---|---|
|`elev`|Höhenwinkel der Kamera|`0` = von der Seite, `90` = senkrecht von oben|
|`azim`|horizontale Drehung|0–360 Grad, dreht das Objekt|

Tipp: Im interaktiven Fenster mit der Maus drehen, gewünschte Winkel ablesen, dann fest eintragen.

### Mehrere 3D-Plots nebeneinander

`plt.subplots()` eignet sich schlecht für 3D – nutze `add_subplot` mit Positionscode **(Zeilen, Spalten, Position)**:

```python
fig = plt.figure(figsize=(15, 5))
ax_a = fig.add_subplot(1, 3, 1, projection="3d")   # Gitter 1x3, Feld 1
ax_b = fig.add_subplot(1, 3, 2, projection="3d")   # Feld 2
ax_c = fig.add_subplot(1, 3, 3, projection="3d")   # Feld 3

for ax in (ax_a, ax_b, ax_c):
    ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5)

ax_a.view_init(elev=90, azim=0)    # von oben
ax_b.view_init(elev=30, azim=45)   # schräg
ax_c.view_init(elev=0,  azim=0)    # von der Seite
```

Positionen werden zeilenweise von links oben gezählt. Drei nebeneinander = `1, 3, n`; drei untereinander = `3, 1, n`.

### 3D-Achsen ausblenden (für Kunst)

```python
ax.set_axis_off()      # Achsen, Ticks, Gitter UND die 3D-"Wände" weg
```

Bei 3D entfernt `set_axis_off()` zusätzlich die grauen Hintergrundwände (panes) und Gitterlinien – übrig bleibt nur die Bahnkurve.

> [!warning] ⚡ MEINE SCHWACHSTELLE: t_span und t_eval müssen zusammenpassen Bei längerer Simulation IMMER beide Zahlen mitziehen:
> 
> ```python
> t_span = (0, 500)                   # Endzeit
> t_eval = np.linspace(0, 500, 50000) # gleiche Endzeit + Punktzahl hoch
> ```
> 
> Nur `t_span` erhöhen → Ausgabepunkte passen nicht zum Integrationsfenster. Endzeit hoch, aber Punktzahl gleich → Linie wird gröber (weniger Punkte pro Umlauf).

## Gestaltung & Ästhetik (für Kunst/Collage)

Wenn das Bild nicht Diagramm, sondern **Grafik** werden soll, willst du Achsen, Ränder und Hintergrund kontrollieren. Die wichtigsten Hebel:

### Alles Störende ausblenden

|Befehl|Wirkung|
|---|---|
|`ax.axis("off")`|Achsen, Ticks, Rahmen komplett weg|
|`ax.set_xticks([])`|Nur die x-Ticks entfernen (Achse bleibt)|
|`fig.patch.set_alpha(0)`|Figure-Hintergrund transparent|
|`ax.set_facecolor("none")`|Axes-Hintergrund transparent|

Für ein reines Kunstobjekt (nur die Linie, kein Drumherum):

```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(sol.y[0], sol.y[1], linewidth=0.6, color="#1400f2")
ax.set_aspect("equal")
ax.axis("off")                     # alles weg außer der Linie
fig.savefig("kunst.png", dpi=300, transparent=True, bbox_inches="tight")
```

### Farbe & Linie gestalten

|Parameter|Beispiel|Wirkung|
|---|---|---|
|`color`|`"#1400f2"` oder `"crimson"`|Linienfarbe (Hex oder Name)|
|`linewidth`|`0.4` – `2`|Dicke; bei vielen Punkten dünn wählen|
|`alpha`|`0.6`|Transparenz der Linie (0=unsichtbar, 1=voll)|
|`linestyle`|`"--"`, `":"`, `"-."`|Strichmuster|
|`solid_capstyle`|`"round"`|Runde Linienenden – weicher Look|

### Farbverlauf entlang der Trajektorie

Ein einziger Farbwert ist statisch. Für einen **Verlauf entlang der Bahn** (z. B. Zeit als Farbe) nutzt man `LineCollection` oder – einfacher – `scatter` mit `c=`:

```python
sc = ax.scatter(sol.y[0], sol.y[1], c=sol.t, cmap="plasma", s=0.5)
```

- `c=sol.t` → Farbe codiert die Zeit.
- `cmap="plasma"` → Farbpalette (s. u.).
- `s=0.5` → Punktgröße (klein = feine Bahn).

### Schöne Colormaps (`cmap=`)

|Name|Charakter|
|---|---|
|`"viridis"`|Standard, gut lesbar, blau→gelb|
|`"plasma"` / `"magma"` / `"inferno"`|Warm, kräftig, gut für dunklen Hintergrund|
|`"twilight"`|Zyklisch – Anfang = Ende, ideal für geschlossene Bahnen|
|`"coolwarm"`|Blau→Rot, divergierend|

### Hintergrund dunkel (für leuchtende Linien)

```python
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#0a0a0a")     # fast schwarz
ax.set_facecolor("#0a0a0a")
ax.plot(sol.y[0], sol.y[1], linewidth=0.5, color="#24ff6d")
ax.axis("off")
```

> Tipp für deine Collage: PNG immer mit `transparent=True` exportieren – dann kannst du die Figur in Photoshop frei über jeden Hintergrund legen. `bbox_inches="tight"` schneidet leere Ränder weg, damit kein unsichtbarer Rahmen stört.

### Stil-Presets

matplotlib bringt fertige Stile mit, die Schrift, Farben und Gitter auf einen Schlag ändern:

```python
plt.style.use("dark_background")   # dunkler Look
# weitere: "ggplot", "seaborn-v0_8", "bmh"
print(plt.style.available)         # alle anzeigen
```

## Bezug zum FuE-Projekt

Matplotlib begleitet dich im ML-Alltag dauerhaft: **Loss-Kurven** über Epochen, **Vorhersage vs. Realität** als Scatter, **Diagnostik** von Trainingsläufen. Genau dasselbe `fig, ax`-Muster, nur mit anderen Daten. Wer die Figure/Axes-Logik einmal verinnerlicht hat, plottet später Trainingsverläufe ohne nachzuschlagen.

# Referenzen