29-08-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# ODEs und SDES
- Analogie zwischen Vektorfeld und Fluss: Das Vektorfeld hält gegebene Vektoren am Ort $x$ bereit. Der Flow ist dann für alle Punkte im Raum ein Gitter, dass sich durch das Vektorfeld verformt. Das heißt der Flow $\psi_{t}(x_{0})$, also ein Punkt von der Fluss-Karte, gibt eine Trajektorie $X$ an. 

# Propability Paths
- Ein Wahrscheinlichkeitspfad ist der Weg zwischen $p_{init}$ und $p_{data}$ 
- Beim Training: $z$ ist ein Datenpunkt, gezogen aus $p_{data}$. Über den Training soll der Punkt $x$ = $z$ haben. Während des Trainings gilt also $p(x \mid z)$. Für die Inferenz muss marginalisiert werden zu $p(z)$ 
- Man lernt mit dem bedingten Wahrscheinlichkeit $p_{t}(x \mid z)$ und möchte für die Generierung die marginale $p_{t}(x)$. Begründung: $p_{t}(x \mid z)$ zeigt immer auf einen Punkt, der durch die Daten gegeben ist, wir möchten aber ein neuen Punkt generieren, brauchen also das marginale $p_{t}(x)$ um neuen Content zu generieren. 
- Noise Scheduler: Zwei konitnuierliche monotone funktionen mit $\alpha_{0}= \beta_{0}$= 1 und $\alpha_{1} = \beta_{0} = 1$. Der Wahrscheinlichkeitspfad ist dann: $p_{t}(·|z) = \mathbf{N} (\alpha,t,z, \beta,  t ,I_{d})$
- two continuously differentiable, monotonic functions with α0 = β1 = 0 and α1 = β0 = 1. We then define the conditional probability path
#### Die Divergenz - Eine Mathematische Funktion
Die Divergenz ist Definiert als:
$$\mathrm{div}(v_t)(x) = \sum_{i=1}^{d} \frac{\partial}{\partial x_i} v_t^i(x)$$
Die Divergenz ist die lokale Volumenänderungsrate. Sie gibt an:
- im Vektorfeld an stelle x, wie die Steigung komponentenweise ist, ob die Funktion gestauch oder gestreckt wird
- $div>0$ → das Volumenelement bläht sich auf. Es fließt mehr heraus als hinein. Man sagt: **Quelle**.
- $div<0$ → es schrumpft. **Senke**.
- $div=0$ = das Volumen bleibt erhalten (das Kästchen darf sich verformen und verdrehen, nur die Größe bleibt). Das nennt man **inkompressibel**.
![[Pasted image 20260831213246.png]]

#### Kontinuitätsgleichung
Claude: Die Wahrscheinlichkeitsmasse an der Stelle xx x ändert sich genau um das, was netto abfließt.* Das Minus ist die Buchführung — was hinausströmt (positive Divergenz), fehlt hier. Nichts entsteht, nichts verschwindet, alles wird nur transportiert. Das ist die zentrale Bedingung dafür, dass ein Vektorfeld utu_t ut​ tatsächlich den Pfad ptp_t pt​ erzeugt.
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]