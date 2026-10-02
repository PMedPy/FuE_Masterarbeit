01-10-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Shape-Landschaft:

---
#### [[FuE_Shapelandschaft]]

---

# Fragenprotokoll

--- 

### [[FuE_Fragenprotokoll]]

---
# Bauplan:
## 0 Config

Konfigobjekt

## 1 Daten
FASTA, CSV, nur AMPs, Längenfilter

##  2 ESM-Wrapper
[[Pseudocode ESM Wrapper]]

## 3 PeptideDataset und DataLoader
- Batch + Paddingmaske

## 4 NoiseSchedule + DiffusionProcess

## 5 DenoiseTransformer
- Vorhersage

##  6 Trainer (+ EMA, Checkpoints, Logging)
- Run-Ordner als Ausgabe
- Ausgabe von Daten

## 7 Sampler - generierte Sequenzen

## 8 Evaluator + Plots
- Metriken figures 

# 9 (Stretch) externe Filter
- Apex, 
- Progen2  (Natürliche Sequenzen)
- Paarweise Ähnlichkeit

