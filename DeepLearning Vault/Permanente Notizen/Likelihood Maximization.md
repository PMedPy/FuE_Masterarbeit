Tags: #FuE #MachineLearning #Statistik
Status: #unextended

# Allgemein: Maximum Likelihood Estimation
Um ein Modell an Daten anzupassen, muss in einer Weise eine Optimierung stattfinden z.B die Erniedrigung der [[Error Function]]. Eine solche Optimierung in Bezug auf die [[Likelihood]]-Funktion wird "Maximum Likelihood Estimation" oder MLE genannt. Die Idee:

> *Unter welchen Parametern $\theta$ werden die Daten $\mathcal{D}$ am Wahrscheinlichsten vom Modell repräsentiert*

Formal sieht das so aus: 
$$\theta_{ML} = \arg\max_\theta ; p(\mathcal{D} \mid \theta) = \arg\max_\theta ; \sum_{n=1}^{N} \ln p(x_n \mid \theta)$$ 
### Warnung: Unterschätzer der Varianz
Die maximierte Likelihood-Funktion unterschätzt systematisch die Varianz nach:
$$\mathbb{E}[\sigma^2_{ML}] = \frac{N-1}{N}  \sigma^2_{\text{wahr}}$$
Der Erwartungswert $\mathbb{E}$ (Varianz ermittelt über mehrere Datensätze) der Varianz ist kleiner als die wahre Varianz. Das ist der selbe Grund, warum in der Statistik
# Beispiel: Gauß-Likelihood
Bishop führt das am Beispiel der eindimensionalen Gauß-Verteilung durch. Die Dichte ist:

$$p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Gegeben i.i.d. Daten $\mathcal{D} = {x_1, \ldots, x_N}$ ist die Likelihood:

$$p(\mathcal{D} \mid \mu, \sigma^2) = \prod_{n=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp!\left(-\frac{(x_n-\mu)^2}{2\sigma^2}\right)$$

Logarithmieren und vereinfachen:

$$\ln p(\mathcal{D} \mid \mu, \sigma^2) = -\frac{1}{2\sigma^2} \sum_{n=1}^{N} (x_n - \mu)^2 - \frac{N}{2} \ln(2\pi\sigma^2)$$

### Maximierung nach $\mu$

Ableitung nach $\mu$ gleich null setzen:

$$\frac{\partial}{\partial \mu} \ln p = \frac{1}{\sigma^2} \sum_{n=1}^{N} (x_n - \mu) \stackrel{!}{=} 0$$

Auflösen liefert:

$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n$$

Der Maximum-Likelihood-Schätzer für den Mittelwert ist also schlicht der **empirische Mittelwert** der Daten. Dieses Ergebnis ist intuitiv – aber bemerkenswert ist, dass es als direkte Konsequenz des MLE-Prinzips herausfällt, nicht als Ad-hoc-Definition.

### Maximierung nach $\sigma^2$

Analog für die Varianz:

$$\sigma^2_{ML} = \frac{1}{N} \sum_{n=1}^{N} (x_n - \mu_{ML})^2$$

Auch das ist die empirische Varianz – mit einer wichtigen Eigenheit, siehe nächster Abschnitt.
# Erwartungswert vs Likelihood-Schätzer

Es konnte gezeigt werden, dass der Likelihood-Schätzer für die Gaussfunktion
$$\mu_{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n$$
ist. 
Es handelt sich dabei nicht um den Mittelwert von Likelihood. Der Schätzer bedeutet:

> Mit den gegebenen Daten $x_{1},x_{2,\dots}x_{n}$ repräsentiert der Mittelwert $\mu_{ML}$ die Daten am wahrscheinlichsten. 

Wenn nun aus weiteren Schätzer aus Daten der selben Verteilung (in dem Fall Gaussverteilung) der Durchschnitt berechnet wird, bekommt man den Erwartungswert für den wahren Mittelwert:
$$\mathbb{E}[\mu_{ML}] = \mu$$

# Referenzen
[@bishopDeepLearningFoundations2024]