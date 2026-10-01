25-06-2026
Tags: #Python  #FuE 
Status: #unextended

# Topic
Hier gibt es die Möglichkeit zu formatieren, mitttels sep und end.
- Um mit Print einen Value in die zweite Reihe auszugeben, nutzt man \n, wobei dies der default sep wert ist
- Kann auch genutzt werden, um die Ausgabe in eine Datei zu geben.
```python
 fh = open("daten.txt", "w") 
  print("42 ist die Antwort, aber was ist die Frage?", file=fh) fh.close()
# Referenzen