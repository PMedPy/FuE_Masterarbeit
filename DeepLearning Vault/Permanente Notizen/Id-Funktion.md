2026-01-31
Tags: #Python  #FuE 
Status: #unextended


# Topic

Gibt die Identität eines Objektes zurück
```Python
>>> x = 42

>>> id(x)

9786176

>>> y = x

>>> id(y)

9786176

# Schauen, ob die beiden Variablen auf das selbe Objekt referenzieren
>>> x = 42

>>> y = x

>>> id(x) == id(y)

True
```
# Referenzen