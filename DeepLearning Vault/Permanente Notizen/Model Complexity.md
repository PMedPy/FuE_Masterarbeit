Tags: #FuE #MachineLearning 
Status: #unextended

## Allgemein
Model Complexity steigt, wenn ein Modell Funktionen flexibler darstellen kann.
- Ein einfacheres Modell ist steifer bei der Abbildung der Trainingsdaten
- Ein komplexes Modell kann eine größere Vielfalt von Mustern representieren
## Zusammenhang zur Datengröße
- Je größer das Trainingset ist, desto weniger ist das Modell Anfällig für [[Over-fitting]]. Höhere Datenmengen erlauben höhere **Model Complexity**
- Bei höherer Komplexität und geringer Datenmengen, kann [[Over-fitting]] entstehen
## Bias-Varianz-Tradeoff
Der Hauptgegenstand bei der Diskussion über die Modellkomplexität ist der Tradeoff zwischen [[Bias]] und [[Varianz]]. Bei Erhöhung der Komplexität (siehe Polynombeispiel mit Polynomordnung $M$) wird jeder Datenpunkt mehr gewichtet. Die [[Weight]]s werden größer, der Fehler mit Testset steigt mit dem Overfitting an. Reduktion von Komplexität führt zu Erhöhung des [[Bias]] und zur Reduktion der [[Varianz]] (Empfindlichkeit gegenüber Daten außerhalb des Trainingsdatensatzes). Der Bias ist hoch bei einfachen Modellen (Der Fehler der mittleren Vorhersage des Modells) Der gesamte erwartete Fehler setzt sich insgesamt zusammen aus:
$$\text{Erwarteter Fehler} = \text{Bias}^2 + \text{Varianz} + \text{irreduzibles Rauschen}$$
Rauschen: Ein Fehler, der aus der Datengenerierung selbst stammt.

# Referenzen
[@bishopDeepLearningFoundations2024]
[@kleinEinfuehrungPython32021]