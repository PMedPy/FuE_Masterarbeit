2026-01-31
Tags: #Python  #FuE 
Status: #unextended


# 📘 `try` / `except` – Quickinfo

> Fehlerbehandlung in Python: Programm crasht nicht, sondern fängt Fehler kontrolliert ab.

---

## 🔥 Was ist eine Exception?

Ein **Fehler zur Laufzeit** (z. B. fehlender Dict-Key, Division durch 0). Wird der Fehler nicht abgefangen, **bricht das Programm sofort ab**.

---

## 🪤 Grundgerüst

```python
try:
    # riskanter Code
    ...
except FehlerTyp:
    # läuft nur, wenn dieser Fehler auftritt
    ...
```

**Ablauf:**

1. `try`-Block wird ausgeführt
2. Tritt ein Fehler auf → Rest von `try` wird **übersprungen**, springt in passenden `except`
3. Tritt **kein** Fehler auf → `except` wird ignoriert
4. Code **nach** dem `try`/`except` läuft normal weiter

---

## 🏷 Wichtige Fehlertypen

|Exception|Tritt auf bei|Beispiel|
|---|---|---|
|`KeyError`|Dict-Key fehlt|`d["xy"]`|
|`IndexError`|Listen-Index zu groß|`liste[999]`|
|`ValueError`|Wert in falscher Form|`int("abc")`|
|`TypeError`|Falscher Datentyp|`"3" + 5`|
|`ZeroDivisionError`|Division durch 0|`10 / 0`|
|`FileNotFoundError`|Datei nicht da|`open("x.txt")`|
|`AttributeError`|Methode/Attribut fehlt|`liste.foo()`|

> 💡 Fehlertyp unbekannt? Fehler provozieren, letzte Zeile der Meldung lesen, Namen übernehmen.

---

## 🎛 Volle Form (selten komplett nötig)

```python
try:
    ...
except FehlerTyp:
    # bei Fehler
    ...
else:
    # nur wenn KEIN Fehler
    ...
finally:
    # IMMER (z. B. zum Aufräumen)
    ...
```

---

## 🎯 Mehrere Fehler abfangen

**Unterschiedlich behandeln:**

```python
except ValueError:
    ...
except ZeroDivisionError:
    ...
```

**Gleich behandeln:**

```python
except (ValueError, ZeroDivisionError):
    ...
```

**Fehlermeldung nutzen:**

```python
except KeyError as e:
    print(f"Fehler: {e}")
```

---

## ⚠️ Anti-Pattern: Fehler verschlucken

```python
try:
    ...
except:        # ❌ fängt ALLES, auch unerwartete Fehler
    pass       # ❌ und macht nichts → Bugs werden unsichtbar
```

**Regel:** Immer **spezifisch** abfangen, nur was du **erwartest**.

---

## 🐍 EAFP vs. LBYL – zwei Philosophien

|Stil|Slogan|Beispiel|
|---|---|---|
|**EAFP** (pythonic)|_"Versuch's, fang Fehler ab"_|`try: d[k] except KeyError:`|
|**LBYL**|_"Schau erst nach"_|`if k in d:`|

**Wann was?**

- Vorher leicht prüfbar → **LBYL**
- Prüfung teuer / mehrere Fehler möglich / nicht prüfbar (`int("abc")`) → **EAFP**

---

## 💡 Faustregeln

- `try`-Block so **klein wie möglich** halten – nur die Zeile, die wirklich fehlschlagen kann
- Spezifischer Fehlertyp statt blankem `except:`
- Nur abfangen, **wo der Fehler realistisch auftreten kann** (Invarianten beachten!)
- Fehlerbehandlung sollte **nicht** Bugs verstecken – manchmal ist Crashen ehrlicher

---

## 📌 Mini-Beispiel (komplett)

```python
def teile(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Division durch 0!")
        return None
    except TypeError:
        print("Beide Werte müssen Zahlen sein.")
        return None
```

---

# Referenzen