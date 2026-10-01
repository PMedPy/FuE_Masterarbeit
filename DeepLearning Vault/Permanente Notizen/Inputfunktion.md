

20-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Topic
Die Inputfunktion dient zur Eingabe, und gibt immer ein String aus. 

```python
# ---------------------------------------------------------------
# input() Cheat Sheet
# ---------------------------------------------------------------

# Grundform: input() gibt IMMER einen str zurück
name = input("Wie heißt du? ")
print(name, type(name))        # -> str

# ---------------------------------------------------------------
# Typumwandlung (Casting)
# ---------------------------------------------------------------
alter = int(input("Alter: "))          # "25"   -> 25
preis = float(input("Preis: "))        # "3.14" -> 3.14
text  = str(input("Name: "))           # str ist redundant, aber legal

# Vorsicht: passt der String nicht, gibt's ValueError
# int("3.5")  -> ValueError
# int("abc")  -> ValueError

# ---------------------------------------------------------------
# Mehrere Werte in einer Zeile  -> split()
# ---------------------------------------------------------------
# Eingabe: "3 5 7"
a, b, c = input("Drei Zahlen: ").split()       # split() trennt an Whitespace
# a, b, c sind noch strings!

# Eingabe: "3 5 7"  -> direkt zu ints
zahlen = list(map(int, input().split()))       # [3, 5, 7]

# Eingabe: "1,2,3"  -> eigener Trenner
werte = input().split(",")                     # ['1', '2', '3']

# Mehrere mit Casting + Entpacken
x, y = map(int, input("x y: ").split())        # "4 9" -> x=4, y=9

# ---------------------------------------------------------------
# eval()  -> wertet den String als Python-Ausdruck aus
# ---------------------------------------------------------------
# Eingabe: "3 + 4 * 2"
ergebnis = eval(input("Ausdruck: "))           # -> 11 (int!)

# Eingabe: "[1, 2, 3]"
liste = eval(input())                          # -> echte Liste [1, 2, 3]

# Eingabe: "2, 3, 4"
t = eval(input())                              # -> Tupel (2, 3, 4)

# ⚠️ WARNUNG: eval() führt BELIEBIGEN Code aus!
# Bei "__import__('os').system('rm -rf /')" wird das ausgeführt.
# Niemals eval() auf nicht vertrauenswürdige Eingaben anwenden.

# ---------------------------------------------------------------
# Sicherere Alternative: ast.literal_eval()
# ---------------------------------------------------------------
import ast
# Wertet nur Literale aus (Zahlen, Strings, Listen, Dicts, Tupel)
# -> KEIN Code wird ausgeführt, kein Sicherheitsrisiko
liste = ast.literal_eval(input())              # "[1, 2, 3]" -> [1, 2, 3]
d     = ast.literal_eval(input())              # "{'a': 1}"  -> {'a': 1}
# ast.literal_eval("os.system('...')")  -> ValueError (sicher!)

# ---------------------------------------------------------------
# Häufige Muster
# ---------------------------------------------------------------

# Whitespace entfernen
name = input().strip()                         # "  Paul " -> "Paul"

# Mit Default, falls leer
wert = input("Name [Gast]: ") or "Gast"        # leere Eingabe -> "Gast"

# Eingabe validieren (Schleife bis korrekt)
while True:
    try:
        n = int(input("Zahl: "))
        break
    except ValueError:
        print("Keine gültige Zahl, nochmal.")

# Mehrere Zeilen einlesen (bis leere Zeile)
zeilen = []
while (line := input()) != "":                 # Walross-Operator :=
    zeilen.append(line)

# Liste fester Länge einlesen
n = int(input("Anzahl: "))
items = [input(f"Element {i+1}: ") for i in range(n)]

# ---------------------------------------------------------------
# Übersicht: input + Casting
# ---------------------------------------------------------------
# int(input())                  -> eine Ganzzahl
# float(input())                -> eine Kommazahl
# input().split()               -> Liste von Strings
# list(map(int, input().split()))   -> Liste von Ints
# eval(input())                 -> Python-Ausdruck (UNSICHER)
# ast.literal_eval(input())     -> Literal (SICHER)
```

Wichtigster Punkt fürs Hinterkopf-Behalten: `input()` liefert **immer** einen String, alles andere ist deine Umwandlung. Und `eval()` ist bequem, aber bei fremden Eingaben ein echtes Sicherheitsloch – `ast.literal_eval()` ist die saubere Variante, wenn du strukturierte Daten (Listen, Dicts) einlesen willst.

# Referenzen