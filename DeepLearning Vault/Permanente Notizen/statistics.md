07-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Quickinfo: Das `statistics`-Modul

Das `statistics`-Modul gehört zur **Standardbibliothek** – keine Installation nötig, kein `pip`. Es deckt grundlegende deskriptive Statistik ab (Mittelwert, Median, Streuung) und ist gedacht für reine Python-Zahlenlisten. Für große Datenmengen oder Vektoroperationen greift man später zu NumPy/Pandas – `statistics` ist der saubere, lesbare Einstieg.

## Aufruf

```python
import statistics

werte = [4.1, 4.3, 4.0, 4.2, 4.4]
mw = statistics.mean(werte)
```

Häufige Kurzform:

```python
import statistics as stats
mw = stats.mean(werte)
```

Oder gezielt einzelne Funktionen importieren:

```python
from statistics import mean, stdev, median
mw = mean(werte)
```

## Methodenkonventionen

- Alle Funktionen erwarten ein **iterierbares Objekt** (Liste, Tuple, Generator) mit Zahlen.
- Eingabe muss **mindestens 1 Element** haben; Streuungsfunktionen wie `stdev`/`variance` brauchen **mindestens 2**.
- Funktionen sind **eigenständige Funktionen**, keine Methoden eines Objekts: also `statistics.mean(liste)`, nicht `liste.mean()`.
- Bei ungültiger Eingabe (leere Liste, zu wenige Werte) wird `statistics.StatisticsError` geworfen.
- Rückgabetyp richtet sich nach Eingabe: `int`-Listen können je nach Funktion `int`, `float` oder `Fraction` zurückgeben. `mean([1, 2, 3])` ergibt z. B. `2` als Typ, der den Mittelwert exakt darstellt.

## Wichtigste Funktionen

### Lagemaße (Mittelwert & Zentrum)

|Funktion|Beschreibung|
|---|---|
|`mean(data)`|Arithmetisches Mittel (Durchschnitt)|
|`fmean(data)`|Wie `mean`, aber immer `float` und deutlich schneller – Standardwahl für reine Float-Daten|
|`median(data)`|Median; bei gerader Anzahl der Durchschnitt der beiden mittleren Werte|
|`median_low(data)`|Bei gerader Anzahl der **kleinere** der beiden Mittelwerte (immer ein echter Datenwert)|
|`median_high(data)`|Bei gerader Anzahl der **größere** der beiden Mittelwerte|
|`mode(data)`|Häufigster Wert; bei mehreren gleich häufigen den ersten gefundenen|
|`multimode(data)`|Liste **aller** häufigsten Werte (auch bei Gleichstand)|

### Streuungsmaße (Variabilität)

|Funktion|Beschreibung|
|---|---|
|`stdev(data)`|**Stichproben**-Standardabweichung (Teiler n−1)|
|`pstdev(data)`|**Populations**-Standardabweichung (Teiler n)|
|`variance(data)`|Stichproben-Varianz (n−1)|
|`pvariance(data)`|Populations-Varianz (n)|

Faustregel: Hast du eine **Stichprobe** aus einer größeren Grundgesamtheit (z. B. einige Messreihen aus vielen möglichen), nimm `stdev`/`variance`. Sind die Daten die **komplette Grundgesamtheit**, nimm `pstdev`/`pvariance`. Im Laboralltag mit Messreihen ist meist `stdev` die richtige Wahl.

### Weitere Mittelwerte & Verteilung

|Funktion|Beschreibung|
|---|---|
|`geometric_mean(data)`|Geometrisches Mittel (Wachstumsraten, Verhältnisse)|
|`harmonic_mean(data)`|Harmonisches Mittel (Durchschnitt von Raten/Geschwindigkeiten)|
|`quantiles(data, n=4)`|Teilt Daten in `n` gleich große Bereiche; `n=4` gibt die Quartile zurück|

### Korrelation & Regression (ab Python 3.10)

|Funktion|Beschreibung|
|---|---|
|`correlation(x, y)`|Pearson-Korrelationskoeffizient zweier gleich langer Datenreihen|
|`covariance(x, y)`|Kovarianz zweier Datenreihen|
|`linear_regression(x, y)`|Lineare Regression; gibt `slope` (Steigung) und `intercept` (Achsenabschnitt) zurück|

> **Versionshinweis:** `correlation`, `covariance` und `linear_regression` gibt es erst ab Python 3.10. Deine Lernumgebung läuft auf **3.9.7** – dort sind diese Funktionen noch nicht verfügbar. Sie stehen dir aber später in der `fue`-Umgebung (3.11) zur Verfügung.

## Anwendungsbeispiel

```python
import statistics

ph_werte = [7.01, 6.98, 7.04, 7.00, 6.97]

print(f"Mittelwert:    {statistics.mean(ph_werte):.3f}")
print(f"Median:        {statistics.median(ph_werte):.3f}")
print(f"Std.-Abw.:     {statistics.stdev(ph_werte):.4f}")
print(f"Varianz:       {statistics.variance(ph_werte):.5f}")
```

Mit Fehlerbehandlung – passend zu deinem `datei_lesen`-Workflow, wo Spalten auch leer sein können:

```python
import statistics

def auswerten(werte):
    try:
        return {
            "mittelwert": statistics.mean(werte),
            "stdev": statistics.stdev(werte),
        }
    except statistics.StatisticsError as fehler:
        print(f"Statistik nicht berechenbar: {fehler}")
        return None
```

## Tipps zum `statistics`-Modul

- **`fmean` statt `mean` bei Float-Messdaten.** `fmean` ist schneller und gibt garantiert `float` zurück – ideal für deine pH-/Photometer-/Thermometer-Reihen. `mean` brauchst du nur, wenn du exakte Brüche/`Decimal` erhalten willst.
- **`stdev` vs. `pstdev` nicht verwechseln.** Der Teiler unterscheidet sich (n−1 vs. n). Bei Messreihen als Stichprobe ist `stdev` der Standard. Eine falsche Wahl verfälscht die Streuung systematisch, besonders bei wenigen Werten.
- **Mindestlänge prüfen.** `stdev`/`variance` werfen bei nur einem Wert einen `StatisticsError`. Wenn deine eingelesene Spalte nach dem Filtern leer oder einelementig sein kann, fang den Fehler mit `try/except` ab, statt das Programm abstürzen zu lassen.
- **Genau eine Datenquelle pro Funktion.** Die Funktionen sind eigenständig (`statistics.mean(liste)`), nicht an ein Objekt gebunden. Das ist ein anderes Muster als z. B. bei deiner `Messreihe`-Klasse, wo `self.mittelwert()` eine Methode ist – nicht durcheinanderbringen.
- **Generatoren werden konsumiert.** Übergibst du einen Generator statt einer Liste, ist er nach einem Funktionsaufruf „leer". Für mehrere Statistiken über dieselben Daten erst in eine Liste umwandeln: `werte = list(generator)`.
- **Versionsabhängigkeit beachten.** Korrelation/Regression erst ab 3.10. Plane das ein, wenn du Code zwischen `.venv` (3.9) und `fue` (3.11) hin- und herbewegst.
- **Grenze des Moduls kennen.** `statistics` ist für überschaubare Listen gedacht. Sobald du mit großen Arrays, Matrizen oder Performance arbeitest – also spätestens im PyTorch-/FuE-Kontext – übernehmen NumPy (`np.mean`, `np.std`) und Pandas. Konzeptionell ist es derselbe Gedanke: `statistics` baut dir das mentale Modell auf, das du dort wiederfindest.

# Referenzen