20-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Topic
Im Umfeld von Parametisierungen oder generellen Umrechnen von Wahrscheinlichkeitsgrößen (z.B $x=g(y)$) ändern sich deren Wahrscheinlichkeitsdichten. Eine Dichtetransformation ist nötig:
$$
p_y(y) = p_x(g(y)) \cdot \left| \frac{dg}{dy} \right|
$$
Ohne die Ableitung der Größe nach der neuen Variablen, würde sich die Wahrscheinlichkeitsdichte nicht zu 1 Summieren: Man unterschätzt/überschätzt systematisch die Unsicherheit

Die Korrekte Transformation der Dichten einer zu einer anderen Größe ist in mehreren Fällen wichtig:
- **Verzerrte Punktschätzungen:** Wie im Bishop-Beispiel – der MAP-Schätzer im log-Raum ist nicht der log des MAP-Schätzers im Originalraum. Wer das ignoriert, gibt falsche optimale Werte an.
- **Falsche Unsicherheitsangaben:** Konfidenzintervalle oder Credible Intervals werden falsch breit, wenn die Jacobi-Verzerrung ignoriert wird. Bei skewen Transformationen wie $\log$ kann das dramatisch sein.
- **Generative Modelle trainieren nicht:** Bei Normalizing Flows ohne Jacobi-Term divergiert die Likelihood, und Maximum Likelihood wird sinnlos.
- **Sampling-Verfahren produzieren falsche Verteilungen:** Inverse Transform Sampling oder Reparametrisierungstricks bei VAEs funktionieren nur, weil die Transformationsregel mathematisch korrekt angewandt wird.
# Referenzen
[@bishopDeepLearningFoundations2024]