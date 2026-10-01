2026-01-31
Tags: #FuE #MachineLearning 
Status: #unextended


# Topic
Bei diesem Ansatz der **[[Regularization]]** werden die Gewichte nach Null gezogen, ohne sie Null werden zu lassen

Weight Decay schrumpft nicht um einen festen Betrag, sondern um einen festen Bruchteil.

Claude:
Bei jedem Gradientenschritt passiert (vereinfacht) folgendes: Das Gewicht wird erstens in Richtung besserer Daten-Fit bewegt (das ist der normale Gradient der Loss), und zweitens wird es zusätzlich ein kleines bisschen _zur Null gezogen_. **Der Zug zur Null ist proportional zum  [[Weight]] selbst.**

Ein Gewicht von 10 wird also stärker zurückgezogen als ein Gewicht von 0.1. Das ist der entscheidende Punkt — deshalb heißt es auch _decay_, Zerfall, analog zum radioaktiven Zerfall: die Zerfallsrate ist proportional zur Menge. Konkret, bei jedem Schritt wird jedes Gewicht mit einem Faktor knapp unter 1 multipliziert (sagen wir 0.9999), bevor der normale Gradient addiert wird.

Daraus folgt eine wichtige Konsequenz: Ein Gewicht wird nicht auf Null gedrückt, wenn die Daten dagegen sprechen. Es stellt sich ein Gleichgewicht ein — der Datenfit-Gradient zieht das Gewicht in eine bestimmte Richtung, Weight Decay zieht es Richtung Null, und am Ende landet es dort, wo beide Kräfte sich ausgleichen. Wichtige Features bekommen weiterhin hohe Gewichte, nur eben nicht unnötig hohe. Unwichtige Features, bei denen der Datenfit kaum zieht, werden effektiv nahe Null gedrückt.


**Von mir:** 
Kleinere Weights heißen höheren **Bias** aber geringere **Varianz**
# Referenzen