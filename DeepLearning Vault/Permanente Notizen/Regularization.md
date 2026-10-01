2026-01-31
Tags: #MachineLearning #FuE 
Status: #unextended


# Topic
Kern-Idee in einem Satz: Regularization ist alles, was ein Modell daran hindert, sich zu sehr an die Trainingsdaten zu klammern — also Overfitting reduziert.

**L(θ)=L(θ)+λ⋅R(θ)**

wobei R(θ) ein Strafterm ist, der "zu komplexe" oder "zu große" Weights bestraft, und λ ein Hyperparameter, der die Stärke steuert. Der Bias-Variance-Tradeoff wird dabei bewusst verschoben. Es wird Bias akzeptiert, damit die Varianz auf neuen Daten drastisch sinkt.
$$\widetilde{E}(\mathbf{w}) = \frac{1}{2}\sum_{n=1}^N \{y(x_n,\mathbf{w}) - t_n\}^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$$
Der Term $\frac{\lambda}{2}\|\mathbf{w}\|^2$ konkurriert in der Minimierung mit der eigentlichen Fehlerfunktion ${E}(\mathbf{w})$
Jetzt muss bei der minimierung nicht nur der Fehler der Datenpunkte kleiner, sondern auch die Weights kleiner werden, damit die Funktion minimiert wird. 
# Referenzen

