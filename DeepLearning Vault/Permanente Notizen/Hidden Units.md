08-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Hidden Units sind Werte, die nicht im Trainingset vorkommen (Nur Input und Output-Daten)
Hidden Units sind sowohl in und Outputs von mehrschichtigen Modellen, es sind aufbereitete Daten, deren Komplexität reduziert sind, um sie für die nächste Schicht des Modells aufzubereiten.

Von Claude:
Hidden Units sind die **internen Variablen eines mehrschichtigen Modells**. Ihre Werte sind nicht in den Trainingsdaten vorgegeben — der Datensatz enthält nur Inputs x\mathbf{x} x und Targets y\mathbf{y} y. Hidden Units entstehen während der Vorwärtsrechnung und werden indirekt über das Training (Anpassung der Gewichte) gelernt.

Jede Hidden Unit ist gleichzeitig **Output ihrer eigenen Schicht** und **Input der nachfolgenden Schicht**. Sie repräsentiert ein vom Modell gelerntes Zwischenmerkmal: eine Transformation der Eingabe, die für die weitere Verarbeitung nützlicher ist als die Rohdaten. Diese Transformation kann die Repräsentation komprimieren, expandieren oder umstrukturieren — je nach Architektur und Aufgabe.

# Referenzen
[@bishopDeepLearningFoundations2024]