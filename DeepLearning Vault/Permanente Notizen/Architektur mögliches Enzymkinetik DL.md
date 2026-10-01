21-07-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Architektur-Design: Enzymkinetik-Vorhersage aus Sequenz

Konzeptioneller Entwurf für die Vorhersage kinetischer Parameter (k_cat, K_m) aus Aminosäuresequenzen. Dieses Dokument beschreibt die Bausteine, ihre Aufgaben und die empfohlene Aufbaureihenfolge — ohne Code.

## Abkürzungen (beim ersten Auftreten)

- **DNN** — Deep Neural Network (tiefes neuronales Netz)
- **ESM-2** — Evolutionary Scale Modeling 2 (Protein-Sprachmodell von Meta AI)
- **MLP** — Multi-Layer Perceptron (mehrschichtiges vollverbundenes Netz)
- **KAN** — Kolmogorov-Arnold Network (Netz mit lernbaren Funktionen auf den Kanten)
- **PINN** — Physics-Informed Neural Network (physik-informiertes neuronales Netz)
- **k_cat** — Wechselzahl (katalytische Konstante, Umsätze pro Zeit)
- **K_m** — Michaelis-Konstante (Substratkonzentration bei halbmaximaler Rate)
- **ΔG‡** — freie Aktivierungsenthalpie (Gibbs-Energie des Übergangszustands)
- **ΔG_bind** — freie Bindungsenthalpie (Substratbindung)
- **μ / σ** — Mittelwert / Standardabweichung (hier: Vorhersage und ihre Unsicherheit)
- **MSE** — Mean Squared Error (mittlerer quadratischer Fehler)

---
## 1. Gesamtidee

Rohe Aminosäuresequenzen sind ein hochdimensionaler, "verschränkter" Raum, in dem die Zielgröße (k_cat) nicht direkt ablesbar ist. Die Strategie:

1. Ein **großes, vortrainiertes Modell (ESM-2)** entwirrt die Sequenz in eine reiche numerische Repräsentation (Embeddings). Dieses Modell bleibt **eingefroren** — es wird nicht mittrainiert.
2. Ein **kleiner, flacher, trainierbarer Kopf** liest aus dieser Repräsentation die Zielgröße ab.

Der Kern: Die "Tiefe" (das biophysikalische Vorwissen) steckt im eingefrorenen Backbone. Der eigene Kopf bleibt bewusst flach, weil der Enzym-Datensatz klein und verrauscht ist — der eigentliche Gegner ist Overfitting, nicht mangelnde Kapazität.
![[architektur.svg|697]]

---

## 2. Die feste Basis (immer vorhanden)

### 2.1 Eingabe — Aminosäuresequenz

Ein Protein als "Satz" in einem Alphabet von ~20 Buchstaben (den Aminosäuren), Länge L.

### 2.2 ESM-2 — Backbone (eingefroren)

Ein auf Millionen Sequenzen vortrainiertes Protein-Sprachmodell. Es liefert pro Aminosäure einen hochdimensionalen Vektor. Bei der gebräuchlichen 650-Millionen-Parameter-Variante ist dieser **1280-dimensional**. Ergebnis für ein Protein: eine Matrix der Form **L × 1280**.

> _Eingefroren_ heißt: keine Gewichts-Updates. Die Embeddings können daher einmal vorab berechnet und gespeichert werden; trainiert wird nur der Kopf.

### 2.3 Pooling — von der Matrix zum Vektor

Für **eine** Vorhersage pro Protein muss die L × 1280-Matrix zu **einem** Vektor verdichtet werden. Übliche Wahl: **Mittelung über alle Positionen** (mean pooling) → ein 1280-dim Vektor pro Protein.

### 2.4 MLP-Kopf — flach und trainierbar

Zwei versteckte Schichten, absteigende Breite:

```
1280 → 256 → 64
```

Jede versteckte Schicht besteht aus: linearer Abbildung + Aktivierungsfunktion + Dropout (zufälliges Ausschalten von Einheiten während des Trainings, als Regularisierung). Die Kompression 1280 → 256 zwingt das Netz, die für k_cat relevanten Richtungen herauszuziehen.

**Zielgröße:** nicht k_cat direkt, sondern **log k_cat**. k_cat spannt viele Größenordnungen; im Logarithmus wird die Verteilung symmetrischer und der MSE-Verlust sinnvoll. (Dieselbe Logik wie bei freien Energien: Der Logarithmus verwandelt multiplikative in additive Struktur.)

**Schichtzahl-Faustregel:** Mit **einer** versteckten Schicht als Referenz beginnen, auf **zwei** gehen, und aufhören, sobald die Validierungsleistung nicht mehr steigt. Vier versteckte Schichten sind bei dieser Datenmenge fast sicher Overfitting.

---

## 3. Die drei Aufsätze (was, wofür, wie kombinierbar)

Diese lösen **verschiedene** Probleme und sind **stapelbar**, nicht konkurrierend.

### 3.1 Bayes'sche letzte Schicht → für _Unsicherheit_

**Problem:** Ein normaler Kopf gibt nur eine Punktvorhersage ("k_cat = 12,3") und verrät nicht, wie sicher er ist. Bei exotischen, trainingsfernen Sequenzen rät das Netz — ein Punktwert verbirgt das.

**Lösung (Last-Layer Bayesian):** Nicht das ganze Netz probabilistisch machen, sondern nur die **letzte lineare Schicht** (64 → 1). Ist alles davor fest, wird die letzte Schicht zu einer **linearen Regression auf 64 Basisfunktionen** — die 64 Aktivierungen sind die Spalten der Design-Matrix. Mit einem Gauß-Prior hat diese Regression eine geschlossene Lösung und liefert eine Verteilung über die Vorhersage.

**Ergebnis:** pro Vorhersage nicht nur μ, sondern auch σ. Ob diese Unsicherheiten _ehrlich_ sind, prüft man über Kalibrierung (z. B. Scaffold-Split: strukturell getrennte Trainings-/Testdaten).

**Reife:** hoch. Bestes Aufwand-Nutzen-Verhältnis.

### 3.2 Mechanistischer Kopf (PINN-Idee) → für _physikalische Konsistenz_

**Problem:** Der Standard-Kopf ist eine Black Box; er nutzt vorhandenes Wissen über Enzymkinetik nicht und könnte physikalisch unsinnige Werte liefern.

**Lösung:** Statt k_cat direkt vorherzusagen, sagt der Kopf eine **latente physikalische Größe** voraus — ΔG‡ (freie Aktivierungsenthalpie) — und ein **fester, nicht gelernter** Decoder rechnet daraus über die **Eyring-Gleichung** die Rate:

```
Kopf → ΔG‡ (latent)  →  [Eyring: k_cat = (k_B · T / h) · exp(−ΔG‡ / R·T)]  →  k_cat
                          ↑ feste Physik, keine Parameter
```

Vorteile: Die Vorhersage ist **per Konstruktion** physikalisch plausibel (die Eyring-Form ist eingebaut), und ΔG‡ ist **interpretierbar** — prüfbar gegen gemessene Werte. Der eigentliche PINN-Aspekt kommt hinzu, wenn Temperatur-/pH-Abhängigkeit als Nebenbedingung in den Verlust geschrieben wird (Strafterm bei Verletzung der Arrhenius-Beziehung über die Temperatur).

**Reife:** ambitioniert. Zahlt sich vor allem aus, wenn Daten **über verschiedene Bedingungen** (Temperatur, pH) vorliegen, an denen die Physik-Nebenbedingungen greifen. Bei ausschließlich Standardbedingungen ist es eher ein physikalisch geformter Ausgabe-Layer als ein echtes PINN.

### 3.3 KAN → für _interpretierbare gelernte Funktionsform_

**Wo es reinpasst:** **nicht** als Backbone (bei 1280-dim, verrauscht spielt der KAN-Vorteil nicht), sondern als **kleiner Decoder innerhalb des mechanistischen Kopfes**. Es lernt die Beziehung zwischen wenigen interpretierbaren Deskriptoren und der latenten Größe ΔG‡.

**Reiz:** In einem KAN sind die Kanten explizite eindimensionale Funktionen — man kann sie **ablesen** ("Abhängigkeit von Deskriptor X ist sigmoidal, von Y linear"). Das ist der Symbolic-Regression-Gedanke: eine Funktionsform _entdecken_, statt sie anzunehmen.

**Reife:** explorativ, optional. Risiko: kostet Zeit, und am Ende sagt ein einfaches MLP womöglich genauso gut vorher.

---

## 4. Zuordnung auf einen Blick

|Komponente|Löst|Ort im Netz|Reife|
|---|---|---|---|
|Flacher MLP-Kopf|Grundvorhersage|1280 → 256 → 64 → 1|Fundament|
|Bayes letzte Schicht|Unsicherheit (μ **und** σ)|ersetzt die 64 → 1-Schicht|reif|
|Mechanistisch (PINN)|physikalische Plausibilität + interpretierbares ΔG‡|fester Eyring-Decoder am Ausgang|ambitioniert|
|KAN|ablesbare Funktionsform|_innerhalb_ des mech. Kopfes|explorativ|

---

## 5. Vollausbaustufe (alle Optionen aktiv)

```
ESM-2 (eingefroren)
  → Pooling
    → MLP-Kopf → latente Deskriptoren
      → [KAN, optional] → ΔG‡                (interpretierbare Funktionsform)
        → [Eyring-Decoder, feste Physik] → Rate   (physikalisch plausibel)
          → [Bayes'sche letzte Schicht] → μ ± σ    (mit Unsicherheit)
```

Ohne den optionalen mechanistischen Zweig geht der MLP-Kopf direkt in die Bayes'sche letzte Schicht.

---

## 6. Empfohlene Aufbaureihenfolge

Der wichtigste Rat: **nicht** mit der Vollausbaustufe beginnen. Jede Stufe muss die vorige messbar **schlagen**, sonst wird sie verworfen.

1. **Baseline** — schlichter MLP-Kopf, Punktvorhersage, sauberer Scaffold-Split. Der Referenzwert. Ohne ihn ist nicht erkennbar, ob die komplexeren Teile überhaupt etwas bringen.
2. **Bayes'sche letzte Schicht** — billig, reif, liefert sofort Unsicherheit. Bestes Aufwand-Nutzen-Verhältnis.
3. **Mechanistischer Kopf** — nur wenn die Baseline steht **und** Daten über verschiedene Bedingungen vorliegen.
4. **KAN** — ganz zuletzt, als exploratives Extra.

Diese Disziplin ist der wirksamste Schutz vor einem überkomplexen Modell, das am Ende schlechter generalisiert als der schlichte Kopf — die klassische Falle bei ehrgeizigen Architekturen dieser Art.
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]