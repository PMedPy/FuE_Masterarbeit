# ShuffleScorer — Konzept

Entwurfspapier zur Diskussion. Stand 19.11.2026.

## 1. Das Problem

Eine niedrige Perplexität kann zwei verschiedene Ursachen haben, und beide sehen im Score gleich aus:

1. Die Sequenz ist **natürlich angeordnet** — sie folgt Mustern, die das Sprachmodell aus echten Proteinen kennt.
2. Die Sequenz ist **niederkomplex** — sie benutzt kaum verschiedene Aminosäuren, und schon deshalb ist jede Position leicht zu raten.

Im Lauf Torres gegen Aguirre hat sich das als relevant erwiesen: 13 % von Torres' gefilterten Sequenzen liegen bei einer Shannon-Entropie um 1,0 bit, benutzen also effektiv zwei Aminosäuren. Ihre Pseudoperplexität fällt bis auf 1,10. Die Rangkorrelation zwischen Entropie und Pseudoperplexität beträgt in Torres' Top-30 +0,81 — es sind dieselben Sequenzen.

Sein Perplexitätsvorsprung ist damit zu einem erheblichen Teil kein Natürlichkeitsvorteil, sondern ein Komplexitätsdefizit.

Die Entropie danebenzustellen hilft nur begrenzt, weil beide Größen miteinander verwoben sind. Ein Quotient aus Perplexität und dem Kompositionsboden $2^H$ löst das Problem **nicht**: Dieser Quotient fällt monoton mit der Entropie (von 4,06 bei $H \le 0{,}5$ auf 1,39 bei $H > 3{,}5$) und ist damit nur eine umgerechnete Entropie, keine zweite Achse.

## 2. Die Idee

Statt die Zusammensetzung zu *modellieren*, halten wir sie **exakt konstant** und variieren nur die Reihenfolge.

Zu jeder Sequenz wird eine Permutation ihrer eigenen Residuen gebildet und mit demselben Modell bewertet. Die gemischte Sequenz hat dieselbe Länge, dieselben Buchstaben in derselben Häufigkeit und damit dieselbe Entropie. Der einzige Unterschied ist die Anordnung.

Was zwischen den beiden Bewertungen an Perplexität liegt, kann deshalb nur aus der Reihenfolge stammen.

## 3. Die Kennzahl

Sei $L(s)$ die mittlere negative Log-Likelihood pro Residuum, die das Modell der Sequenz $s$ zuweist. Die Perplexität ist $\mathrm{PPL}(s) = e^{L(s)}$.

Für $n$ Permutationen $\sigma_1 \dots \sigma_n$ einer Sequenz $s$:

$$\Delta L(s) = \frac{1}{n}\sum_{k=1}^{n} L(\sigma_k(s)) \;-\; L(s)$$

$$R(s) = e^{\Delta L(s)} = \frac{\mathrm{PPL}_{\text{gemischt}}}{\mathrm{PPL}_{\text{original}}}$$

Gelesen als Logik: $\Delta L$ ist der Zugewinn an Überraschung pro Residuum, wenn man die Reihenfolge zerstört. $R$ ist derselbe Zugewinn als Faktor auf die effektive Anzahl der Wahlmöglichkeiten.

Berichtet werden sollten beide: $\Delta L$ in nats pro Residuum ist additiv und besser vergleichbar, $R$ ist anschaulicher.

## 4. Warum die Entropie konstant bleiben muss

Das ist die tragende Annahme des ganzen Verfahrens, und sie steht und fällt mit einem Detail: **Es muss eine Permutation sein, kein Neuziehen.**

Würde man stattdessen aus der Häufigkeitsverteilung neu würfeln, schwankte die Zusammensetzung von Ziehung zu Ziehung, und mit ihr die Entropie. Der Unterschied zwischen Zähler und Nenner wäre dann eine Mischung aus Reihenfolge und Zusammensetzung — also genau die Vermengung, die wir auflösen wollen.

Bei einer Permutation ist die Multimenge der Buchstaben invariant. Entropie, Länge und jede andere reihenfolgeblinde Kenngröße sind in Zähler und Nenner identisch, per Konstruktion und nicht nur im Mittel.

Daraus folgt eine nützliche Selbstkontrolle: Bei einem Homopolymer wie `PPPPPPPPPPPPPP` sind alle Permutationen identisch mit dem Original. $R$ muss dort exakt 1 ergeben. Tut es das in der Implementierung nicht, stimmt etwas nicht.

## 5. Was hohe und niedrige Werte bedeuten

| Wert | Bedeutung |
|---|---|
| $R \gg 1$ | Die echte Reihenfolge ist deutlich vorhersagbarer als eine zufällige Anordnung derselben Buchstaben. Die Sequenz trägt Ordnungsinformation, die das Modell erkennt. |
| $R \approx 1$ | Die Reihenfolge trägt nichts bei. Die gesamte Vorhersagbarkeit kommt aus der Zusammensetzung. Erwartungswert für Wiederholungskonstrukte wie `KKLKKLLKK…`. |
| $R < 1$ | Die echte Reihenfolge ist **schwerer** vorhersagbar als eine zufällige. Ungewöhnlich; deutet auf eine Anordnung hin, die den Mustern des Modells zuwiderläuft. |

Der Nullpunkt der Skala ist also 1, nicht 0. Abweichung von 1 ist das Signal.

Für die Fragestellung des Projekts: Erwartet wird, dass die niederkomplexen Sequenzen in Torres' gefiltertem Satz bei $R \approx 1$ landen, während kompositionell reichere Sequenzen deutlich darüber liegen. Damit ließe sich die Aussage "Torres' bessere Perplexität ist Komplexitätsdefizit, kein Natürlichkeitsvorteil" direkt belegen, statt sie aus zwei Verteilungen zu erschließen.

Ein biologischer Nebeneffekt, der die Kennzahl für AMPs besonders interessant macht: Das Mischen zerstört die periodische Abfolge hydrophober und kationischer Residuen, die eine amphipathische Helix ausmacht. $R$ misst damit indirekt mit, wie stark das Modell seine Sicherheit aus genau dieser Periodizität bezieht.

## 6. Implementierungslogik

### Reihenfolge der Schritte

1. **Vorbehandlung** anwenden (Methionin, siehe Abschnitt 7) — auf **alle** Sätze gleich.
2. $L(s)$ für die behandelte Originalsequenz bestimmen.
3. $n$ Permutationen bilden, jede einzeln bewerten, $L$ je Permutation festhalten.
4. Die $L$-Werte der Permutationen **arithmetisch mitteln**.
5. Differenz zum Original bilden, exponenzieren.

### Gemittelt wird im Logarithmus

Perplexität ist ein Exponential. Wegen der Jensen-Ungleichung ist der Mittelwert mehrerer Perplexitäten nicht die Perplexität des mittleren Verlusts. Also die negativen Log-Likelihoods mitteln und erst am Schluss exponenzieren — dasselbe Argument wie beim Batching.

### Zahl der Permutationen

Eine einzelne Permutation ist eine Zufallsziehung aus sehr vielen möglichen Anordnungen; der Wert schwankt entsprechend. Zehn sind ein vernünftiger Ausgangswert. Ob das reicht, lässt sich empirisch prüfen: auf einer Stichprobe von einigen hundert Sequenzen $n = 5$, $10$ und $20$ rechnen und sehen, ab wann sich der Median nicht mehr bewegt.

### Determinismus

Der Zufallsgenerator muss **pro Sequenz** aus einem festen Startwert und der Sequenz selbst abgeleitet werden, nicht aus einem global fortlaufenden Zustand. Sonst hängt das Ergebnis davon ab, in welcher Reihenfolge und in welchen Batches die Sequenzen verarbeitet wurden — und ein Wiederholungslauf liefert andere Zahlen als der erste.

### Cache

Der Fingerabdruck muss alles enthalten, was das Ergebnis verändert: Modellname, Zahl der Permutationen, Startwert des Zufallsgenerators und die Methionin-Entscheidung. Was nur die Ausführung betrifft — Batchgröße, Gerät — gehört nicht hinein.

### Kosten

Der Aufwand ist $n$-mal ein normaler Scoring-Durchlauf. Auf 50.000 Sequenzen mit $n = 10$ sind das grob ein bis zwei Stunden auf der A30.

**Praktischer Vorschlag:** Die interessante Aussage betrifft die *gefilterten* Sätze, also 156 und 173 Sequenzen. Darauf kostet der Score mit zehn Permutationen Sekunden statt Stunden. Für die Gesamtsätze ist er nur nötig, wenn die Verteilung über alle 50.000 selbst Gegenstand der Arbeit sein soll.

## 7. Das Methionin-Problem

69 % der Aguirre-Sequenzen und 81 % der Torres-Sequenzen beginnen mit Methionin. Das ist ein Artefakt der Dekodierung über den ESM-2-Kopf, kein Merkmal der Peptide: Ohne Position 1 liegt der Methioninanteil in beiden Sätzen bei 0,8 bis 0,9 %, also unter dem Trainingssatz.

ProGen2 erwartet M an Position 1 sehr stark. Beim Mischen wandert das M an eine beliebige Stelle, und die Perplexität steigt — aber nicht, weil Ordnungsinformation verloren ginge, sondern wegen einer einzigen Positionskonvention. Der Zähler wird dadurch aufgebläht, und zwar für beide Sätze **unterschiedlich stark**, weil die M-Anteile verschieden sind. Der Vergleich wäre verzerrt.

Zwei Auswege:

**Ausweg 1 — Position 1 festhalten.** Nur die Residuen ab Position 2 permutieren. Einfach, aber unsauber: Die Sequenzen ohne führendes M werden dann anders behandelt als die mit, und die Anteile unterscheiden sich zwischen den Sätzen.

**Ausweg 2 — führendes Methionin vor dem Scoren abschneiden**, bei allen Sequenzen aller Sätze. Empfohlen. Das Artefakt verschwindet aus Zähler und Nenner gleichermaßen, alle Sequenzen werden gleich behandelt, und die Peptide verlieren nichts Biologisches.

**Wichtige Folge von Ausweg 2:** Der Nenner muss dann ebenfalls auf der gekürzten Sequenz berechnet werden. Die vorhandenen Perplexitäten im Cache sind auf den ungekürzten Sequenzen entstanden und dürfen hier **nicht** wiederverwendet werden. Zähler und Nenner müssen auf demselben Eingabetext beruhen, sonst misst der Quotient die Kürzung mit.

## 8. Grenzen

- $R$ misst Ordnungsvorhersagbarkeit **nach Maßgabe eines Modells**, nicht biologische Funktion. ProGen2 und ESM-2 werden verschiedene Werte liefern.
- Die Kennzahl sagt nichts über Wirksamkeit und löst insbesondere nicht das Zirkularitätsproblem bei APEX.
- Sehr kurze Sequenzen haben wenige unterscheidbare Permutationen; bei Längen unter etwa acht Residuen wird die Schätzung unzuverlässig.
- Was das Modell als "Ordnung" erkennt, ist das, was in seinen Trainingsdaten häufig war. Eine funktionale, aber seltene Anordnung erscheint als ungeordnet.

## 9. Offene Entscheidungen

1. **Brauchen wir die Kennzahl überhaupt?** Die bereits vorhandenen Größen — Rangkorrelation zwischen Entropie und Perplexität je Satz (Torres +0,49, Aguirre +0,22) und der Anteil unter $H = 1{,}5$ — tragen die Aussage möglicherweise schon.
2. Nur auf den gefilterten Sätzen oder auf allen 50.000?
3. Welches Modell — ProGen2, ESM-2, oder beide?
4. Methionin: Ausweg 1 oder 2?
5. Zahl der Permutationen.
