Tags: #FuE #MachineLearning #Statistik 
Status: #unextended

# Momente im Allgemeinen

Der Begriff der Momente kommt aus der Mechanik. Das Integral der Massendichte (Also Masse/Volumeneinheit) bekommt man die Gesamtmasse

- $\int \rho(x) , dx$ = Gesamtmasse

Generell gilt für Momente:

- $\mathcal{M_{n}}=\int x^{n}\rho(x)dx$  (Allgemeine Momentengleichung)

Moment kommt von Momentum (Bewegungskraft, Moment). Die Allgemeinemomentengleichung ist physikalisch gesehen ein Integral gegen eine Monomene Testfunktion.

In der Physik bekannte Momente sind:
 - $\int x \cdot \rho(x) , dx$ = statisches Moment um den Ursprung → bestimmt den **Schwerpunkt**
- $\int x^2 \cdot \rho(x) , dx$ = **Trägheitsmoment**

> Die Reihe aller Momente ist eine **Kodierung** einer Verteilung. Jedes Moment ist nur eine Projektion dieser Verteilung. 
> Beispiel: Das nullte Moment einer Scheibe ist ihre Masse. Diese Masse kann auch eine Kugel haben, ergo Kodiert dieses einzelne Moment die Verteilung nicht. 

# Statistisches Moment
In der Statistik spricht man von Momenten $k$-ter Ordnung
$$\text{$k$-tes Moment} = \mathbb{E}[x^k] = \int x^k , p(x) , dx$$
Wobei:
- **Erstes Moment (first-order moment):** $\mathbb{E}[x^1] = \mathbb{E}[x]$ → das ist der **Mittelwert*.
- **Zweites Moment (second-order moment):** $\mathbb{E}[x^2]$ → hängt mit der Varianz zusammen, aber ist _nicht_ die Varianz selbst (siehe Unten).
Das $0$-te Moment der Gaussverteilung muss, wegen Normierungsbedingungen immer 1 betragen wegen $\int_{-\infty}^{\infty} p_{Gauss}(x)dx=1$. Gilt im Generellen für alle Wahrscheinlichkeitsverteilungen
# Zentrale und rohe Momente
Die obige Definition vom zweiten Moment zeigt, das nicht direkt die Varianz gemeint ist, sondern eine Assoziation davon. Die Momente, die noch nicht auf eine Größe reduziert sind, werden rohe Momente genannt. Im Gegensatz stehen dazu die zentralen Momente:
$$\text{$k$-tes zentrales Moment} = \mathbb{E}[(x - \mu)^k]$$
Wenn von 2. Moment gesprochen wird, ist meist die Varianz, also das zentrale Moment gemeint. 
# Referenzen
[@bishopDeepLearningFoundations2024]
[[Claude _ Likelihood Schätzer und Momente]]