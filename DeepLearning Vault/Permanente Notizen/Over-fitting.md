2026-01-31
Tags: #MachineLearning #FuE 
Status: #unextended


# Allgemein
Unter Over-Fitting versteht man das Phänomen, wenn die Anzahl der Datenpunkte geringer ist als die der [[Weight]] und damit diese Weights die Datenpunkte überrepresentieren (man denke an das Polynom 9 Grades, das durch 4 Punkte geht, alle genaustens trifft aber eine Unförmige Schwankung zwischen den Punkten zulässt.)

Effekt: Der RMS (Root-Mean-Square wird 0 bei den [[Training Set]] , wenn M >= Anzahl der Daten aus dem Set. Bei den Test-Set wird der Fehler hingegen riesig!)

Verbesserung: 
Entweder Vergrößerung des Trainingssets, oder [[Regularization]]

### Heuristik aus der Statistik
Die Anzahl an Datenpunkte sollte nicht weniger sein, als ein 5-10 fache der Menge an lernbaren Parameter des Modells.

# Referenzen