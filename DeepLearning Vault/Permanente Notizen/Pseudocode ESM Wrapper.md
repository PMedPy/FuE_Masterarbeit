02-10-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Pseudocode

Erstellen einer ESMWrapper Klasse:
1. Importe: torch, fair-ESM, nn
2. __init__  beschließt, was bei einer Instanz aufgebaut werden soll. 
	1. Wir wollen also bei jeder Instanz wissen: Was ist die maximale Länge des AMPs, alphabet wird aufgerufen, 
	2. ist für decoder und encoder gleich. 
	3. Hier wird auch für das Decoden eine Liste mit alphabet 
3. __encode__ : normales ESM-2 mit 8 millionen Parametern. 
	1. Liste aus strings als Input, kein Tensor. @torchnograd, da gewichte hier eingefroren. (reine Operation, kein Trainierbarer Teil). Ausgabe Shape tensor (B, L+2, D). assert mit Länge darf nicht größer als L_Max
	2. Abschneiden von BOS und EOS und padding auf  L_Max (Vermute hier ein Stack aus Nullen hinten dran zu ballern)
4. __decode__ : ESM-2 Head. 
	1. Definition des ESM-HEads. 
	2. reversed time embedding (gegenoperation.) 
	3. Abschneiden von Padding-schwanz (0-Positionen über einen if-algorithmus abschneiden? vielleicht gibt es in Pytorch eine elegantere Variante)
	4. dann input (B, L,D) in den Head und eine tensor mit (B,L,AS) mit AS = 20 herausbekommen. 
	5. Argmaxing über dim =2, um mit AS-Tabelle zu übersetzen


### Encoder.
Baut aus einem Batch von Peptidsequenzen ein ESM-2 Embedding. BOS und EOS werden abgeschnitten, sodass Ausgabe mit Shape (B,L_batch, D) ausgegeben wird. 

#### Args:
- device (string, wird im Geräte unabhängigen Code definiert.)
- X als Inputliste mit Sequenzen, z.B ["ACDEFGHIKLMN", "KLVFFAED"]
