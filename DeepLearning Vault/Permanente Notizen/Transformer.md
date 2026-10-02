24-09-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemeines 

Transformer ist eine Art von [[Architectur]] und ist daher das "Wie": Aufbau, wie das Neuronale Netz rechnet. 
- Vergleich Diffusionsmodell: ist ein generatives [[ML Framework]] und sagt "Was" berechnet werden soll

Ein Transformer ist also eine Architektur, aufgebaut aus $N$ gleichen Blöcken
# Stichpunkte
- Im Grunde ist Transformer pro Block Attention (zuhören, wissen über gesamten Input) und einmal MLP (Token arbeitet alleine)
- Jedes Token wird zu einer Identität (Vektor $E$) und einer Position $p_{i}$ assoziiert. Bei $n$ tokens entsteht also eine  $d * n$  Matrix. 
- Aus jedem Tokenvektor entstehen 3 neue: Query (Zuständigkeit), Key (Fundort) und Value (Wert)
- Attention: Vergleicht alle Querys mit Keys. Je mehr übereinstimmung, umso mehr Value wird zurückgerufen: Also ein gewichteter Mittelwert
- Tokenbegrenzung bei Transformern: Weil Attention-Matrix mit steigender Tokenzahl quadratisch wächst
- **Multihead Attention**: Da pro Token nur eine Mischung entstehen kann, brauch es $h$ Köpfe.

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]