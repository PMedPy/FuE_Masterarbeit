2026-05-08
Tags: 
Status: #Referenz
# Freie Energie nach Karl Friston's Free Energy und Komplexitätsreduktion via Niklas Luhmann

## Meine Frage an Claude:
Was mir gerade in den Sinn schwebt: Hat Luhmann nicht auch was von Komplexitätsreduktion gesprochen um komplexität aufzubauen. Klar in seiner Systemtheorie. Gibt es da überlapp mit DeepLearning?
## Luhmanns Grundgedanke (sehr verdichtet)

Luhmanns Ausgangsbeobachtung: Die Welt (die "Umwelt") ist hochgradig komplex — mehr Möglichkeiten, als irgendein System je verarbeiten könnte. Ein System (z. B. eine Person, eine Organisation, das Rechtssystem) kann mit dieser Umweltkomplexität nur umgehen, indem es **selektiert**: Es reduziert Komplexität, indem es nur bestimmte Unterscheidungen trifft und alles andere ignoriert.

Das Paradoxe und für Luhmann Zentrale: Diese Reduktion ist **die Voraussetzung dafür, eigene, interne Komplexität aufzubauen**. Ein Gericht kann nicht "die ganze Welt" verhandeln — aber gerade weil es nur nach dem binären Code recht/unrecht operiert, kann es ein hochdifferenziertes inneres Begriffssystem, eine Dogmatik, eine Verfahrensordnung entwickeln. Reduktion außen → Aufbau innen.

Schlagwortartig: **"Nur Komplexität kann Komplexität reduzieren."** Und: Reduktion ist nicht Verlust, sondern Bedingung von Form.

## Wo Deep Learning das tatsächlich spiegelt

Hier wird es interessant, weil mehrere Strukturmerkmale passen:

**1. Die Bottleneck-Idee.** Autoencoder sind das vielleicht direkteste Beispiel. Du zwingst einen hochdimensionalen Input (z. B. ein Bild mit 256×256×3 = ~200 000 Dimensionen) durch eine schmale Engstelle, etwa 64 Dimensionen. Das Netz **muss reduzieren**. Und genau in dieser erzwungenen Reduktion entsteht eine **interne, strukturierte Repräsentation** — der Latent Space —, in dem semantisch sinnvolle Operationen plötzlich möglich werden (Interpolation zwischen Gesichtern, Stiltransfer, etc.). Das ist sehr luhmannisch: Reduktion erzeugt operative Anschlussfähigkeit.

**2. Hierarchische Feature-Extraktion.** In einem CNN lernt die erste Schicht Kanten, die zweite Texturen, die dritte Teile, die vierte Objekte. Jede Schicht **selektiert** aus dem Input ihrer Vorgängerschicht das, was für die Aufgabe relevant ist, und verwirft den Rest. Diese kaskadierende Reduktion baut eine zunehmend abstrakte interne Komplexität auf — Repräsentationen, die der rohe Pixelraum nicht hat.

**3. Der Information Bottleneck als Theorie.** Das ist der direkteste Kandidat für eine formale Brücke. Naftali Tishby hat vorgeschlagen, Deep Learning als Optimierungsproblem zu verstehen, bei dem Schichten **maximal informativ über das Label, minimal informativ über den Input** werden sollen:

$$\min_{p(t \mid x)} ; I(X; T) - \beta , I(T; Y)$$

```latex
\min_{p(t \mid x)} \; I(X; T) - \beta \, I(T; Y)
```

In Worten: Komprimiere $X$ zu einer Repräsentation $T$ (Reduktion), die so viel wie möglich über $Y$ behält (relevante Komplexität). Die Theorie ist umstritten, aber sie formalisiert genau die Intuition, dass **gezielte Reduktion die Bedingung für leistungsfähige Repräsentation** ist.

**4. Generalisierung als Folge von Reduktion.** Ein Modell, das den Trainingsdatensatz auswendig lernt, hat keine Komplexität reduziert — es hat sie nur gespeichert. Generalisierung entsteht erst, wenn das Modell **strukturelle Selektion** vornimmt: bestimmte Muster als relevant betrachtet, andere als Rauschen. Regularisierung, Dropout, frühes Stoppen — das sind alles Mechanismen, die Reduktion erzwingen, damit innere Struktur (= generalisierbare Repräsentation) entsteht. Hier hast du übrigens auch eine Verbindung zum Bias-Varianz-Tradeoff von eben: Komplexität "richtig" reduzieren heißt, den Sweet Spot finden.

**5. Selbstreferenz und Autopoiesis.** Vorsichtigere Analogie, aber: Luhmanns Systeme operieren selbstreferentiell — sie produzieren ihre eigenen Elemente. Bei rekurrenten Netzen, bei autoregressiven Sprachmodellen, bei selbstüberwachtem Lernen (das eigene Trainingssignale erzeugt) findet man strukturelle Echos davon. Das ist aber eher suggestiv als rigoros.

## Wo die Analogie endet

Damit das nicht zu glatt klingt, vier ehrliche Einschränkungen:

**Erstens**, Luhmann spricht über **Sinn** als Medium der Komplexitätsreduktion — und Sinn ist bei ihm an Bewusstsein und Kommunikation gebunden. Ein neuronales Netz reduziert Information, aber ob es "Sinn" konstituiert, ist eine ganz andere Frage. Man sollte den Begriff nicht überdehnen.

**Zweitens**, Luhmanns Systeme sind **operativ geschlossen** — sie verarbeiten nur ihre eigenen Operationen, nicht "die Umwelt direkt". Ein Netz im Training ist das gerade _nicht_: es ist offen, datengetrieben, von außen geformt durch den Gradienten. Erst zur Inferenzzeit könnte man von einer Art Geschlossenheit sprechen.

**Drittens**, "Komplexität" meint bei Luhmann etwas Soziales und Semantisches; bei Shannon, Kolmogorov und in der ML-Theorie etwas Informationstheoretisches. Die Begriffe sind verwandt, aber nicht deckungsgleich. Eine Brücke müsste über Information-Bottleneck-artige Formalisierungen laufen — und auch da bleibt der semantische Kern (was _bedeutet_ die reduzierte Information) unterspezifiziert.

**Viertens**, Luhmann selbst hätte den Vergleich vermutlich relativiert. Er war skeptisch gegenüber kybernetischen Engführungen seiner Theorie, gerade weil er Systeme nicht als Informationsverarbeiter, sondern als Kommunikations- bzw. Sinnbildungssysteme verstand.

## Wo der Vergleich produktiv ist

Trotz dieser Einschränkungen finde ich den Bezug heuristisch wertvoll, gerade für jemanden, der aus den Lebenswissenschaften kommt und sich gerade in ML einarbeitet. Er liefert eine **konzeptuelle Begründung** dafür, warum Reduktion (durch Bottlenecks, Regularisierung, Schichtarchitektur) nicht ein notwendiges Übel ist, sondern strukturell _notwendig_ dafür, dass ein Modell überhaupt etwas Brauchbares "weiß". Das wird in der ML-Literatur oft eher technisch motiviert ("sonst Overfitting"), aber Luhmann gibt einem ein Vokabular, um den positiven Aspekt zu sehen: **Reduktion ist konstitutiv für interne Ordnung**, nicht bloß ein Filter gegen Rauschen.

Wenn dich das vertieft interessiert: Es gibt einen kleinen, aber existierenden Diskurs an der Schnittstelle Systemtheorie/Kybernetik/ML — Stichworte wären Heinz von Foerster, Gregory Bateson (Vorläufer), und in jüngerer Zeit Arbeiten zu "Predictive Processing" und Karl Friston's Free-Energy-Prinzip, das tatsächlich versucht, Wahrnehmung und Lernen als Komplexitätsreduktion zu formalisieren. Friston ist näher dran an Luhmann, als man auf den ersten Blick denkt.

Eine ehrliche Schlussbemerkung: Solche Brücken zwischen Geistes- und Naturwissenschaften sind oft inspirierend, aber sie ersetzen nicht die saubere ML-Mathematik. Sie können dir helfen, ein intuitives Gespür dafür zu entwickeln, _warum_ bestimmte Architekturen funktionieren — aber wenn du einen Autoencoder baust, brauchst du am Ende doch den Reconstruction Loss und nicht Luhmann. Beides hat seinen Platz.


# Referenzen
