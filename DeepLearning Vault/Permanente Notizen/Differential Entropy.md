25-05-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Allgemein
Wir können die Definition der Entropie auf stetige Verteilungen anwenden. 
Für diskrete Verteilung gilt:
$$H[x] = -\sum_x p(x)\log_{2}p(x)$$ (Also die Summe über **ALLE** $x$ )

Durch die Anwendung des  "*mean value theorem*", wobei in definierten Kästen $\Delta$ ein wert für $x_{i}$ existieren muss. Dieser Wert muss in Spanne $i\Delta \leqslant x_{i} \leqslant (i+1)\Delta$ liegen. Es gilt dann 
$$\int_{i\Delta}^{(i+1)\Delta}p(x)dx = p(x_{i})\Delta$$
In Worten: Der Wert von $p(x)$ mit $x_{i}$ wird durch das Integral der diskreten Spanne gebildet. Die Wahrscheinlichkeit $x_{i}$  zu beobachten ist gegeben durch in  $p(x_{i})\Delta$. Einsetzen in die bekannte diskrete Gleichung der Entropie ergibt:
$$H_\Delta = \underbrace{-\sum_i p(x_i)\,\Delta\,\ln p(x_i)}_{\to\,-\int p(x)\ln p(x)\,dx}\;-\;\ln\Delta$$
und über den $Lim_{\Delta \to 0}$  und Vernachlässigung des Wertes $\ln\Delta$  gilt:
$$H(x)= -\int p(x) \ln p(x)dx$$
Man sieht: die diskrete und die differentielle Entropie unterscheiden sich um den Summanden $-\ln\Delta$. 
Diese additive Konstante kann vernachlässigt werden, da man im Deep Learning Kontext an Entropie-Unterschiede zwischen Verteilungen interessiert ist. 


# Weiterführung


# Referenzen
[@bishopDeepLearningFoundations2024]