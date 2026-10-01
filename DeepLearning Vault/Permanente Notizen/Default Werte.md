29-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Topic
Bestimmte Parameter von [[Funktionen Py]], die gleich bleiben, gesetz man ändert sie durch eine Parametereingabe nicht.

```python
def hallo(name="Namenloser"): 
... print("Hallo " + name + "!") ...  
hallo("Peter") 
>>>Hallo Peter! 
hallo() 
>>> Hallo Namenloser!
```

Um einen bestimmten wert eines Parameters einer Funktion festzulegen (ihn also nicht auf Default zu lassen) werden [[Schlüsselwortparameter]] genutzt:
```python
def umfang(laenge=2, breite=1):
	return 2 * (laenge + breite)	

umfang(breite=1.5) 7.0 
 
#Im folgenden Beispiel nutzen wir Schlüsselwortparameter, obwohl es nicht notwendig wäre. Es dient aber dazu, die Lesbarkeit eines Programms deutlich zu erhöhen:  

umfang(laenge=5, breite=3) 16  >>> umfang(5, 3) #so hätten wir auch statt des vorigen Aufrufs ω↗ schreiben können 16 

umfang(breite=3, laenge=5) #oder so 16
```
# Referenzen