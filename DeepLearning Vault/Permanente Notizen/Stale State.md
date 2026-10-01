Tags: #Python  #FuE 
Status: #unextended

# Topic

Stale State ist ein Bug-Typ, der leise ohne Fehlermeldung existiert. Er beruht darauf, dass das System veraltete Daten nicht löscht.
Beispiel:

```python
tb = Trainingstagebuch()
tb.eintrag_hinzufuegen("2026-04-27", "Bankdrücken", 75.0, 8)
tb.bestleistung("Bankdrücken")        # speichert: bestwert = 75.0

# Eine Woche später, neuer Eintrag:
tb.eintrag_hinzufuegen("2026-05-04", "Bankdrücken", 90.0, 6)

tb.uebersicht()
```

**Was wird angezeigt?**

```
Datum: 2026-04-27 | Übung: Bankdrücken | Gewicht: 75.0 | ...
Datum: 2026-05-04 | Übung: Bankdrücken | Gewicht: 90.0 | ...
Die Bestleistung bei Bankdrücken lag bei 75.0 kg     ← FALSCH!
```
Lösung: Es sollen alle aus den Rohdaten abgeleitete Werte dort berechnet werden, wo sie gebraucht werden und nicht separat in listen etc gespeichert werden.
# Referenzen