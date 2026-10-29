# Cluster-Anleitung — THM MNI GPU-Cluster

Für das Projekt FuE_Masterarbeit. Reiner Terminal-Workflow, kein VS Code
Remote-SSH, keine Container.

Grundlage: die THM-Doku unter https://slurm-cluster.mni.thm.de/slurm_sbatch/
Ansprechpartner bei Problemen: Eric Hartmann, eric.hartmann@mni.thm.de,
0641/309-2460, Raum A12 2.15

---

## 0. Das Grundprinzip

Es gibt zwei Sorten Rechner:

| | Name | wofür |
|---|---|---|
| **Login-Knoten** | `slurm-cluster.mni.thm.de` | anmelden, Dateien verwalten, Jobs abschicken |
| **Rechenknoten** | z.B. `bioinf-euryale.mni.thm.de` | rechnen — nur über SLURM erreichbar |

**Auf dem Login-Knoten wird nicht gerechnet.** Er hat keine GPU, und alle
Nutzer teilen ihn sich. Alles, was länger als ein paar Sekunden dauert, geht
über `sbatch` an einen Rechenknoten.

Daraus folgt eine Sache, die beim ersten Mal verwirrt: auf dem Login-Knoten
meldet `torch.cuda.is_available()` immer `False`. Das ist normal und kein
Fehler — dort ist schlicht keine GPU.

**Merkregel für jeden Befehl:** steht links im Prompt `pemd95@slurm-cluster`,
bist du auf dem Cluster. Steht dort `paulmedebach@...`, bist du auf dem Mac.
`rsync` läuft IMMER lokal, alles andere auf dem Cluster.

---

## 1. Verbinden

```bash
ssh pemd95@slurm-cluster.mni.thm.de
```

Beim allerersten Mal fragt SSH nach dem Fingerabdruck des Rechners — mit `yes`
bestätigen (ausgeschrieben, nicht `y`).

### Einmalig einrichten: Schlüssel statt Passwort

Spart bei jedem `ssh` und `rsync` das Passwort.

```bash
# lokal
ssh-keygen -t ed25519 -C "paul@macbook"      # 3x Enter
ssh-copy-id pemd95@slurm-cluster.mni.thm.de  # einmal noch das Passwort
```

Dazu lokal in `~/.ssh/config` eintragen:

```
Host thm
    HostName slurm-cluster.mni.thm.de
    User pemd95
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

Danach genügt `ssh thm`, und in `rsync` schreibst du `thm:` statt der langen
Adresse. `ControlMaster` hält die Verbindung zehn Minuten offen, sodass
mehrere Befehle hintereinander keine neue Sitzung aufbauen.

---

## 2. Dateien hin und zurück

Immer **lokal** ausführen.

### Code hochschieben (ohne die großen Caches)

```bash
rsync -avP \
  --exclude 'wrath_of_god/data/' \
  --exclude 'wrath_of_god/tmp/' \
  --exclude 'wrath_of_god/runs/' \
  ~/Documents/Projekte/FuE_Masterarbeit/ \
  thm:~/FuE_Masterarbeit/
```

Der **Schrägstrich am Ende des Quellpfads** ist wichtig — ohne ihn landet der
Ordner im Ordner (`~/FuE_Masterarbeit/FuE_Masterarbeit/`).

### Nur die Skripte nachschieben (der häufigste Fall)

```bash
rsync -avP \
  ~/Documents/Projekte/FuE_Masterarbeit/wrath_of_god/scripts/ \
  thm:~/FuE_Masterarbeit/wrath_of_god/scripts/
```

### Embeddings hochschieben (einmalig, 1,2 GB für 8M)

```bash
rsync -avP --mkpath \
  ~/Documents/Projekte/FuE_Masterarbeit/wrath_of_god/data/cache/*8M* \
  thm:~/FuE_Masterarbeit/wrath_of_god/data/cache/
```

`--mkpath` legt fehlende Zwischenverzeichnisse an. Ohne das bricht rsync mit
`mkdir ... failed: No such file or directory` ab, weil es nur die letzte Ebene
selbst erzeugt.

### Ergebnisse abholen

```bash
rsync -avP \
  thm:~/FuE_Masterarbeit/wrath_of_god/runs/ \
  ~/Documents/Projekte/FuE_Masterarbeit/wrath_of_god/runs/
```

`-P` zeigt den Fortschritt und macht nach einem Abbruch beim selben Befehl da
weiter, wo es aufgehört hat.

---

## 3. Umgebung auf dem Cluster

Es gibt dort **kein Modulsystem** (`module` existiert nicht) und kein
vorinstalliertes conda. Miniforge liegt deshalb im Home.

### Einmalig eingerichtet

```bash
cd ~
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash
exec bash

cd ~/FuE_Masterarbeit
mamba env create -f environment.yml
conda activate fue-amp
```

`mamba` statt `conda` für das Erzeugen — bei unserer Paketliste der
Unterschied zwischen zwei und fünfzehn Minuten.

### ESM-Gewichte vorladen

Muss **einmal pro Modellvariante** passieren, auf dem Login-Knoten:

```bash
export TORCH_HOME=~/FuE_Masterarbeit/.torch
mkdir -p $TORCH_HOME
python -c "import esm; esm.pretrained.esm2_t6_8M_UR50D(); print('ESM ok')"
```

Grund: `ESMWrapper` lädt die Gewichte beim ersten Aufruf aus dem Netz. Hat der
Rechenknoten keinen Internetzugang, bricht der Job sonst mitten im Lauf ab —
nach dem Laden der 1,2 GB Embeddings, also an der ärgerlichsten Stelle.

Für die 650M-Variante entsprechend `esm2_t33_650M_UR50D()`.

### Nach jedem Neuanmelden

```bash
conda activate fue-amp
```

---

## 4. Einen Lauf starten

```bash
cd ~/FuE_Masterarbeit
mkdir -p slurm_logs
JOB=$(sbatch --parsable wrath_of_god/scripts/train.slurm)
echo "Job-ID: $JOB"
```

`--parsable` gibt nur die nackte Job-Nummer aus statt "Submitted batch job
3212", sodass sie direkt in die Variable passt.

**Aus der Repo-Wurzel abschicken**, denn die `--output`-Pfade im Skript sind
relativ zum Abschickverzeichnis, und `slurm_logs/` muss vorher existieren.

### Hyperparameter ändern

Im Kopf von `wrath_of_god/scripts/train.slurm`:

```bash
RUN_NAME="cluster01"
EPOCHS=50
ESM="8M"
```

Alternativ ohne das Skript anzufassen, direkt als Flags von `06_train.py`:
`--loss-reduction`, `--loss-weight`, `--batch-size`, `--lr`, `--debug N`.

---

## 5. Überwachen

```bash
squeue -u $USER                          # eigene Jobs
tail -f slurm_logs/fue-train_3212.out    # Live-Ausgabe, Strg+C beendet nur die Anzeige
cat slurm_logs/fue-train_3212.err        # Fehlerausgabe
scancel 3212                             # Job abbrechen
sacct -j 3212 --format=JobID,State,ExitCode,Elapsed   # Nachschau, auch wenn der Job weg ist
```

Status in `squeue`:

| ST | Bedeutung |
|---|---|
| `PD` | wartet (Grund steht in der letzten Spalte) |
| `R` | läuft |
| (verschwunden) | fertig oder abgebrochen — `sacct` fragen |

Häufige PD-Gründe: `Resources` (Knoten belegt), `Priority` (andere vor dir),
`ReqNodeNotAvail` (Knoten down — mit `sinfo -N -l` prüfen).

### Knoten und GPUs abfragen

```bash
sinfo -o "%P %G %N %m %c %t"      # Partitionen, GPUs, RAM, Kerne, Status
sinfo -o "%N %G" | grep euryale   # GPUs eines bestimmten Knotens
```

`bioinf-euryale.mni.thm.de` hat `gpu:V100:3` — drei Tesla V100 mit je 32 GB,
32 Kerne, 478 GB RAM.

---

## 6. Die wichtigsten SBATCH-Parameter

Aus der THM-Doku, auf unser Projekt bezogen:

| Parameter | Bedeutung |
|---|---|
| `--job-name` | Name im `squeue` und in den Logdateinamen (`%x`) |
| `-w NODE` | auf welchem Knoten der Job laufen soll |
| `--gpus 1` | Anzahl GPUs. Mehr als eine braucht es fast nie — vorher mit dem Admin sprechen |
| `--gres=gpu:TYP:N` | GPUs eines bestimmten Typs, z.B. `gpu:V100:1` |
| `--cpus-per-task` | 1 bis 20. Die THM bittet ausdrücklich, mit möglichst wenigen zu testen |
| `--mem` | RAM, Angabe in MB oder mit Suffix (`32G`) |
| `--time` | Maximallaufzeit im Schema `Tage-Stunden:Minuten:Sekunden`, z.B. `0-12:00:00` |
| `--output` / `--error` | Logdateien. `%x` = Jobname, `%j` = Job-ID |
| `--mail-user` / `--mail-type` | Benachrichtigung bei `BEGIN,END,FAIL,ALL` |
| `--container-image` | nur falls doch mal ein Container nötig ist (Pyxis) |

---

## 7. Unser `train.slurm`, Zeile für Zeile

```bash
#SBATCH -w bioinf-euryale.mni.thm.de    # fester Knoten
#SBATCH --gpus 1                        # eine V100
#SBATCH --cpus-per-task 4               # DataLoader nutzt keine Worker, mehr bringt nichts
#SBATCH --mem 32G                       # Embeddings liegen komplett im RAM (1,2 GB)
#SBATCH --time 0-12:00:00
```

Im Rumpf drei Dinge, die nicht offensichtlich sind:

**`source "$(conda info --base)/etc/profile.d/conda.sh"`**
Ein SLURM-Job läuft in einer nicht-interaktiven Shell, die `~/.bashrc` nicht
einliest. Ohne diese Zeile kennt der Job `conda activate` nicht.

**`export PYTHONPATH="$PROJECT/wrath_of_god/src"`**
Macht das Paket `aguirre` auffindbar, ohne `pip install -e .`. Code-Änderungen
wirken dadurch sofort, ohne dass die Umgebung angefasst werden muss.

**`python scripts/06_train.py ...` ohne `srun`**
Ein `srun` INNERHALB des sbatch-Skripts scheitert auf diesem Cluster mit
`Task launch ... failed: Communication connection failure`. Die THM-Doku ruft
Python im Beispielskript auch direkt auf.

---

## 8. Was ein Lauf erzeugt

```
runs/train_<name>_<datum>/
    config.yaml                    exakte Config dieses Laufs
    metrics/metrics.csv            Loss pro Epoche (train, token, val)
    metrics/loss_per_t.csv         Loss je t-Rasterpunkt, eine Zeile pro Epoche
    checkpoints/checkpoint_last.pt wird überschrieben, Absturzversicherung
    checkpoints/checkpoint_best.pt bester Val-Loss
    final.pt                       Artefakt zum Sampeln (ohne Optimizer-State)
    plots/lernkurven.png
    plots/t_heatmap.png
    report.pdf
```

Die CSVs werden **nach jeder Epoche** geschrieben — ein abgebrochener Lauf
behält also seine Zeilen.

---

## 9. Fallstricke, die wir schon getroffen haben

| Symptom | Ursache | Abhilfe |
|---|---|---|
| `rsync: change_dir ... failed` | rsync auf dem Cluster gestartet | lokal ausführen |
| `mkdir ... failed: No such file or directory` | Zwischenverzeichnis fehlt | `--mkpath` oder vorher `ssh thm 'mkdir -p ...'` |
| `conda: command not found` im Job | nicht-interaktive Shell | `source .../conda.sh` vor `conda activate` |
| `Task launch failed: Communication connection failure` | `srun` im sbatch-Skript | `srun` weglassen |
| `cuda: False` auf dem Login-Knoten | dort ist keine GPU | normal, im Job prüfen |
| `module: command not found` | es gibt kein Modulsystem | Miniforge im Home |
| Job sofort `FAILED`, 10 s Laufzeit | meist Pfad oder Umgebung | `cat slurm_logs/*_<ID>.err` |

Noch offen: auf dem Login-Knoten meldet sich `torch 2.14.1` ohne CUDA, auf dem
Rechenknoten `2.6.0+cu124` mit CUDA. Es liegen offenbar zwei Installationen im
Spiel. Für die Ausarbeitung gilt die Version aus dem **Job-Log**, nicht die vom
Login-Knoten.

---

## 10. Spickzettel

```bash
# lokal: Code hoch
rsync -avP ~/Documents/Projekte/FuE_Masterarbeit/wrath_of_god/scripts/ \
           thm:~/FuE_Masterarbeit/wrath_of_god/scripts/

# Cluster: Lauf starten
ssh thm
conda activate fue-amp
cd ~/FuE_Masterarbeit
JOB=$(sbatch --parsable wrath_of_god/scripts/train.slurm); echo $JOB
tail -f slurm_logs/fue-train_${JOB}.out

# lokal: Ergebnisse holen
rsync -avP thm:~/FuE_Masterarbeit/wrath_of_god/runs/ \
           ~/Documents/Projekte/FuE_Masterarbeit/wrath_of_god/runs/
```
