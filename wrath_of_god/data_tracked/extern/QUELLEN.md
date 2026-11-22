# Quellen der externen AMP-Datenbanken

Diese Datei dokumentiert, **was genau** aus den vier Datenbanken geholt wurde und
**wann**. Grund: die Datenbanken werden laufend aktualisiert. Ohne Version und
Abrufdatum ist später nicht mehr feststellbar, auf welchem Stand eine Auswertung
beruht — und genau das fragt ein Gutachter.

Die Rohdateien liegen **unverändert** unter `data/cache/DBs/<db>/` (untracked).
Nicht umbenennen, nicht in Excel öffnen und neu speichern: eine von Excel neu
geschriebene xlsx ist nicht mehr dieselbe Datei, und Excel macht aus Sequenzen
gelegentlich Datumsangaben.

Die daraus erzeugten normalisierten CSVs liegen in `data_tracked/extern/` und
gehen mit ins Git.

Hash berechnen (macOS):

```bash
shasum -a 256 data/cache/DBs/<db>/<datei>
```

Zeilen zählen:

```bash
wc -l data/cache/DBs/<db>/<datei>
```

---

## DBAASP

Zugang über die REST-API, kein Dateidownload.

- Endpunkt: `<URL, z.B. https://dbaasp.org/peptides>`
- Abrufdatum: `<JJJJ-MM-TT>`
- `totalCount` zum Abrufzeitpunkt: `<n>`
- Seitengröße (`limit`): `<n>`
- Anzahl geholter Seiten: `<n>`
- Eindeutige Peptid-IDs im Cache: `<n>`
- Cache: `data/cache/DBs/dbaasp/<ordner oder dateimuster>`
- Erzeugte CSV: `data_tracked/extern/<datei>.csv`, `<n>` Zeilen
- Version laut Website: `<z.B. DBAASP v3.x>`
- Bemerkungen: `<z.B. Detailabruf pro Peptid nötig ja/nein; Abbrüche; Auffälligkeiten>`

## DRAMP

- Version laut Website: `<z.B. DRAMP 4.0>`
- Downloadseite: https://dramp.cpu-bioinfor.org/downloads/
- Direkter Dateilink: `<URL>`
- Abrufdatum: `<JJJJ-MM-TT>`
- Datei: `data/cache/DBs/dramp/<datei>`
- Einträge: `<n>`
- SHA256: `<hash>`
- Erzeugte CSV: `data_tracked/extern/<datei>.csv`, `<n>` Zeilen
- Bemerkungen: `<welches Sub-Dataset: General / Activity / Specific; MIC-Felder als Freitext?>`

## dbAMP

- Version laut Website: `<z.B. dbAMP 3.0>`
- Downloadseite: https://awi.cuhk.edu.cn/dbAMP/
- Direkter Dateilink: `<URL>`
- Abrufdatum: `<JJJJ-MM-TT>`
- Datei: `data/cache/DBs/dbamp/<datei>`
- Einträge: `<n>`
- SHA256: `<hash>`
- Erzeugte CSV: `data_tracked/extern/<datei>.csv`, `<n>` Zeilen
- Bemerkungen: `<...>`

## APD3

- Version laut Website: `<z.B. APD3, Stand JJJJ-MM>`
- Seite: https://aps.unmc.edu/AP
- Direkter Dateilink: `<URL>`
- Abrufdatum: `<JJJJ-MM-TT>`
- Datei: `data/cache/DBs/apd3/<datei>`
- Einträge: `<n>`
- SHA256: `<hash>`
- Erzeugte CSV: `data_tracked/extern/<datei>.csv`, `<n>` Zeilen
- Bemerkungen: `<APD3 liefert überwiegend Sequenzen, quantitative MICs nur sporadisch>`

---

## Zusammengeführter Datensatz

- Erzeugt am: `<JJJJ-MM-TT>`
- Skript: `<pfad>`
- Eingänge: `<die vier CSVs mit Zeilenzahlen>`
- Dedupliziert nach: `<Schlüssel, z.B. (sequence, species, medium, modifikationen)>`
- Zeilen nach Deduplizierung: `<n>`
- Eindeutige Sequenzen: `<n>`
- Ausgabe: `data_tracked/extern/<datei>.csv`
- Bemerkungen: `<Einheitenumrechnung, Behandlung zensierter Werte, Kontaminationsflags>`
