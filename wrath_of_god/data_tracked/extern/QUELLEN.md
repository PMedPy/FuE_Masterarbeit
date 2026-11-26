# Quellen der externen AMP-Datenbanken

Diese Datei dokumentiert, **was genau** aus den externen Quellen geholt wurde und
**wann**. Grund: die Datenbanken werden laufend aktualisiert. Ohne Version und
Abrufdatum ist später nicht mehr feststellbar, auf welchem Stand eine Auswertung
beruht — und genau das fragt ein Gutachter.

Die Rohdateien liegen **unverändert** unter `data/cache/DBs/<db>/` (untracked).
Nicht umbenennen, nicht in Excel öffnen und neu speichern: eine von Excel neu
geschriebene xlsx ist nicht mehr dieselbe Datei, und Excel macht aus Sequenzen
gelegentlich Datumsangaben.

Die daraus erzeugten normalisierten CSVs liegen in `data_tracked/extern/`.
Ob sie in ein **oeffentliches** Repo mitgehen duerfen, ist pro Datenbank
unterschiedlich — siehe Abschnitt D in
`notizen/veroeffentlichung_checkliste.md`. Im jetzigen Arbeitsrepo gehen sie
mit ins Git.

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

Zugang ueber die REST-API, kein Dateidownload.

- Endpunkt: `https://dbaasp.org/peptides?limit=&offset=`
- Abrufdatum: 2026-11-24
- `totalCount` zum Abrufzeitpunkt: 25 542
- Seitengroesse (`limit`): 200
- Anzahl geholter Seiten: 128
- Detailabrufe nach Spezies- und Kandidatenfilter: 4 560 (von 25 542)
- Eindeutige Peptid-IDs im Cache: 4 560
- Cache: `data/cache/DBs/dbaasp/liste/`, `.../liste_<spezies>/`, `.../detail/`
- Erzeugte CSV: `data_tracked/extern/dbaasp_messungen.csv`, 40 725 Zeilen,
  4 195 Peptid-IDs, 3 799 eindeutige Sequenzen
- Version laut Website: keine laufende Nummer angegeben; Publikationsstand ist
  DBAASP v3 (Pirtskhalava et al. 2021, NAR 49, D288-D297)
- Lizenz: widerspruechlich. "Data access policy" nennt die Daten public domain
  und frei verteilbar, die Klausel "Non-Distribution of Data" verbietet jede
  Weitergabe. Siehe Abschnitt D der Veroeffentlichungs-Checkliste.
- Bemerkungen: Detailabruf pro Peptid noetig, die Aktivitaetsmessungen stehen
  nicht in der Listenantwort. Das Feld `activity` ist **immer** ug/ml und
  kollabiert die Zensierung: ">200" erscheint als `200.0`. Deshalb wird die
  Zensur aus dem Rohfeld `concentration` geparst und nur die Zahl aus
  `activity` genommen. Molmasse fuer 87,6 % der Zeilen (Rest ohne
  bestimmbares MW).

## DRAMP

- Version laut Website: **DRAMP 5.0** (Seite zuletzt aktualisiert 2026-10-20)
- Downloadseite: https://dramp.cpu-bioinfor.org/downloads/
  (dort unter den General-Datensaetzen; kein stabiler Direktlink hinterlegt,
  der Download laeuft ueber die Seite)
- Abrufdatum: 2026-11-24
- Datei: `data/cache/DBs/dramp/general_amps.xlsx`
- Eintraege: 12 784 Zeilen, 28 Spalten
- SHA256: `1f376e89596a01e0a186258ada2d2c29bcde2e520abfe2969a618b824958f25e`
- Erzeugte CSVs:
  - `data_tracked/extern/dramp_novelty.csv`, 11 485 eindeutige Sequenzen
  - `data_tracked/extern/dramp_vergleich.csv`, 2 808 Messungen ueber
    843 eindeutige Sequenzen (857 Peptid-IDs)
- Lizenz: CC BY 4.0 fuer die general- und clinical-Daten. Zu zitieren:
  Ma T. et al. (2026) DRAMP 5.0 ..., Nucleic Acids Research, gkag989.
  Fuer die Patent-AMPs verlangt die Seite eine Autorisierung — die sind hier
  **nicht** enthalten und duerfen auch nicht nachtraeglich dazugelegt werden.
- Bemerkungen: nur das General-Subset. `Antibacterial.xlsx` ist eine Teilmenge
  davon und wurde nicht geladen. Die MIC-Angaben stehen als Freitext im Feld
  `target_organism`, mit gemischten Einheiten und Schreibweisen des
  My-Zeichens; der Parser verwirft, was er nicht eindeutig lesen kann.

## dbAMP

**Nicht verwendet.** Fallengelassen, weil DBAASP und DRAMP den synthetischen
Bestand bereits abdecken und der Mehrertrag den Parsing-Aufwand nicht
rechtfertigt. Der leere Ordner `data/cache/DBs/dbamp/` kann weg.

## APD

- Version laut Website: **APD6**, Stand 2026-01-01 (6 309 Peptide gesamt:
  3 379 natuerlich, 2 290 synthetisch, 373 vorhergesagt)
- Seite: https://aps.unmc.edu/downloads
- Direkter Dateilink:
  https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta
- Abrufdatum: 2026-11-24
- Datei: `data/cache/DBs/apd/naturalAMPs_APD2024a.fasta`
- Eintraege: 3 306 Peptide in 3 307 FASTA-Headern, siehe Bemerkung unten
- SHA256: `b2d30bbbe57a383220d5fa68fda5a73728d2345f88585a78aa5f9e9546cd3d66`
- Erzeugte CSV: `data_tracked/extern/apd_novelty.csv`, 3 306 Zeilen,
  Spalten `apd_id, sequence, sequence_length`
- Lizenz: keine eingeraeumt. Seitenfuss "All Rights Reserved". Um Zitation
  wird gebeten: Wang G., Schmidt C., Li X., Wang Z. (2026) APD6 ...,
  Nucleic Acids Research 54, D363-D374.
- Bemerkungen: keine quantitativen MICs, deshalb nur Novelty-Referenz. Nur das
  Natural-Subset; synthetische und vorhergesagte AMPs gibt es nicht als Datei.
  Keine nicht-kanonischen Zeichen, keine doppelten Sequenzen. Laenge Median 29,
  min 2, max 183; sechs Eintraege liegen unter 5 Resten und bleiben drin, sie
  erzeugen bei `-c 0.8` ohnehin keine Treffer.

**Pseudo-Header in der Quelldatei.** Die Datei beginnt mit der Zeile
`>Your search led to 3306 peptides`. Das passiert nicht beim Herunterladen,
sondern steckt so in der Datei auf dem Server: APD erzeugt die FASTA offenbar
ueber dieselbe Suchfunktion wie die Weboberflaeche, und deren Trefferzeile wird
als erster Header mitgeschrieben. Daher 3 307 Header fuer 3 306 Peptide.
`extern/apd.py` filtert auf das Kennungsmuster `AP\d{4,6}` und meldet, wie
viele Header verworfen wurden -- erwartet ist genau 1. Wird daraus mehr, hat
sich das Format der Quelle geaendert.

**Zur Datei im Cache.** Gespeichert ueber den Browser, der die Endung ".txt"
anhaengt; die Endung wurde entfernt, damit der Name dem auf dem Server
entspricht. Der Inhalt blieb unangetastet, reine LF-Zeilenenden, keine
CRLF-Wandlung. Eine frueher von Hand aus der Browseransicht kopierte Fassung
(`natualAMPs_APD2024a.fasta`, ohne "r") war inhaltlich identisch -- gleiche
3 307 Header, gleiche 3 306 eindeutigen Sequenzen, gleiche Laengenstatistik --
und unterschied sich nur im fehlenden Zeilenumbruch am Dateiende. Sie ist
ersetzt; der Hash oben gehoert zum Download.

---

## UniProt (Negativquelle, kein AMP-Datensatz)

Keine AMP-Datenbank, sondern die Quelle fuer den Spezifitaets-Boden: Fragmente
aus Proteinen, die sicher keine AMPs sind.

- Seite: https://www.uniprot.org
- Query (Suchfeld, woertlich):

  ```
  (reviewed:true) AND (cc_scl_term:SL-0086) AND (length:[150 TO 2000])
  NOT (cc_scl_term:SL-0243)
  NOT (keyword:antimicrobial) NOT (keyword:antibiotic) NOT (keyword:fungicide)
  ```

  SL-0086 ist Cytoplasm, SL-0243 ist Secreted. Zytoplasmatisch statt nur
  "ohne AMP-Keyword", weil sekretierte Proteine membranexponiert und haeufig
  kationisch sind -- dort stecken die AMP-artigen Abschnitte.
- Abrufdatum: 2026-11-25
- Trefferzahl laut UniProt: 149 816
- Datei: `data/cache/DBs/uniprot/uniprotkb_reviewed_true_AND_cc_scl_term_2026_11_25.fasta.gz`
  (FASTA canonical, gepackt belassen wie vom Server geliefert;
  `extern.apd.lese_fasta` liest .gz direkt)
- Eintraege in der Datei: 149 816 -- stimmt mit der Trefferzahl ueberein, der
  Download ist also vollstaendig
- SHA256: `8cdbeb006b728c3be62ee29d0517c06d1f39b79c71e25b34bb4ad260ea9bf4b7`
- Proteinlaenge: Median 353, min 150, max 2000
- Erzeugte Datei: `splits/uniprot_lower.fasta` (+ .csv), ein Fragment je
  Protein, cysteinfrei, kanonisch, Laenge 5-50, Rand 25 je Ende, fester Seed
- Lizenz: CC BY 4.0 (UniProt-Konsortium). Weiterverbreitung mit
  Namensnennung erlaubt.
- Bemerkungen: die Proteine werden vom Modul zufaellig mit festem Seed
  gezogen, nicht in Dateireihenfolge -- UniProt liefert nach Accession, und
  die korreliert mit Organismus und Einreichungsjahrzehnt.

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
