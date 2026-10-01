2026-01-31
Tags: #Python  #FuE 
Status: #unextended


# Funktionen
### Insistance
mit 
der funktion insistance, lässt sich der wahrheitgehalt über die Aussage des Datentyps einer Variable ausgeben:
```Python
>>> s =

"Ich bin ein String"

>>> isinstance(s, str)

True
```
Oder, ob die Variable ein Integer oder Float ist:
```Python
>>> x = 4

>>> isinstance(x, int) or isinstance(x, float)

True

>>> x = 4.8

>>> isinstance(x, int) or isinstance(x, float)

True
```
Oder etwas bequemer:
```Python
>>> x = 4.8

>>> isinstance(x, (int, float))

True

>>> x = (89, 123, 898)

>>> isinstance(x, (list, tuple))

True

>>> isinstance(x, (int, float))

False
```
### Type
Ansonsten gibt es noch die Type-Funktion, mit den sich direkt der Datentyp ausgeben lässt:
```Python
x=14.5
type(x)

 float
```

# Referenzen
[[Funktionen]]