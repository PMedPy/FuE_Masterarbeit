29-06-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Topic
Die Möglichkeit, tiefe Kopien von nicht-flachen Listen anzufertigen
```python 
from copy import deepcopy
person1 = ["Swen", ["Seestraße", "Konstanz"]]
person2 = deepcopy(person1)  id(person1[1]), id(person2[1]) 

>>>140450035848064, 140450035904576
```

# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]