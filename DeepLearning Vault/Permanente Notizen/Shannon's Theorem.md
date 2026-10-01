01-07-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Die Entropie nicht nur als physikalisches Maß der Unordnung (Siehe [[Entropy]]) , sondern als Maß der Überraschung eines Ereignisses zu sehen, das postuliert das Shannon Theorem. 

# Definition
Die Shannonentropie ist Definiert, als die Summe von der Wahrscheinlichkeitsverteilung $p(x)$ und dessen negativen logarithmus. 
$$-\sum_{x} \log(p(x))p(x)dx$$

Informationstechnisch bedeutet das, ein **diskretes** Ereignis einen höheren Informationsgehalt hat wenn es unwahrscheinlich ist, im Vergleich zu einem Ereignis $x_2$ , was wahrscheinlicher ist und durch seine höhere Frequenz wenig Information liefert (Wir wissen schneller um Ereignis $x_2$, als um $x_1$). Die Informationsdichte hängt also von $p(x)$ ab.
 
 Zwei Informationen über $x$ und $y$ sind [[I.I.D]] : $h(x,y) = h(x)+h(y)$, damit muss $h$ der Logarithmus von $p(x)$ sein.
# Der Aufbau
Man beachte die Form der Gleichung: Man Gewichtet den Informationsgehalt $h(x)$ mit jedem der einzelnen Wahrscheinlichkeit. Wir bekommen also einen Punktschätzer bzw die Entropie ist ein Punktschätzer und sagt: Der mittlere Informationsgehalt einer Verteilung in [[Bits]] (Binary Digits). Bits ist hier die Einheit, da man konventionstechnisch den $\log_{2}$ für den Informationsgehalt verwendet. Da man ein und die selbe Information mit viel oder wenig Aufwand (viel oder wenig Bits) einen Empfänger senden kann, ist die Entropie daher die unterste Schranke für die kleinstmögliche Bitlänge. 

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]