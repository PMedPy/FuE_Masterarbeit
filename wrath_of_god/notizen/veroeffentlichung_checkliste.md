# Checkliste: personenbezogene Daten vor einer Veröffentlichung

Abzuarbeiten, **bevor** dieses Projekt in ein öffentliches GitHub-Repo wandert.

**Diese Datei nennt bewusst keine Adressen und keine Benutzernamen im Klartext**,
sondern nur Datei und Zeile. Sie wird selbst versioniert — eine Checkliste, die
die zu entfernenden Daten aufzählt, wäre sonst genau das Leck, das sie
verhindern soll. Die Prüfbefehle in Abschnitt G finden die Stellen von selbst.

Stand 2026-11-24. Aktuelles Remote ist `thm:FuE_Masterarbeit`, also ein
SSH-Remote auf dem Uni-Cluster — **nicht öffentlich**. Es besteht kein
Zeitdruck, aber die Punkte müssen vor dem ersten Push nach außen erledigt sein
(Begründung in Abschnitt E).

---

## A — Muss raus: Daten Dritter

Das sind die Punkte, die nicht in deinem Ermessen liegen. Deine eigenen Daten
preiszugeben ist deine Entscheidung; die von anderen nicht.

- [ ] **`wrath_of_god/notizen/cluster_anleitung.md`, Zeile 7** — Name und
      Mailadresse einer Person aus dem Rechenzentrum, als Ansprechpartner
      notiert. Diese Person wurde nicht gefragt. Ersetzen durch einen neutralen
      Hinweis, etwa "Ansprechpartner beim MNI-Rechenzentrum, Kontakt siehe
      Intranet".

- [ ] **`Beispiel Notebooks/` (6 Dateien, versioniert)** — fremdes Kursmaterial.
      Zwei Probleme auf einmal: das Urheberrecht liegt beim Kursautor, und die
      gespeicherten Zellausgaben enthalten deinen Benutzernamen
      (`01_pytorch_workflow.ipynb`, Zeile 1324). Gehört nicht in ein eigenes
      öffentliches Repo — im neuen Repo einfach nicht mitnehmen.

## B — Sollte raus: eigene private Daten

- [ ] **`wrath_of_god/pyproject.toml`, Zeile 41** — private Mailadresse im
      `authors`-Feld. Entweder durch die Uni-Adresse ersetzen oder das Feld
      weglassen. Begründung: eine Uni-Adresse ist eine Rollenadresse, die nach
      dem Studium wegfällt; die private begleitet dich Jahrzehnte.

- [ ] **`wrath_of_god/scripts/train.slurm`, Zeile 8** und
      **`scripts/train_test_pred_n.slurm`, Zeile 8** — Uni-Mailadresse als
      `#SBATCH --mail-user`. Vertretbar, sie stehenzulassen. Sauberer: auf
      `$SLURM_MAIL` oder eine Umgebungsvariable umstellen, dann ist das Skript
      auch für David direkt brauchbar.

- [ ] **`wrath_of_god/notizen/cluster_anleitung.md`, Zeilen 38 und 51** —
      Cluster-Benutzername in den `ssh`-Beispielen. Durch `<benutzer>` ersetzen;
      die Anleitung wird dadurch sogar allgemeiner.

- [ ] **`wrath_of_god/nb/data_test.ipynb` und `nb/trainer_test.ipynb`** —
      Home-Pfade in gespeicherten Zellausgaben, also Benutzername und
      Verzeichnisstruktur. Lösung ist nicht Suchen und Ersetzen, sondern
      `nbstripout` (Abschnitt F, Schritt 3): dann landen Ausgaben künftig gar
      nicht mehr im Commit.

- [ ] **`Ausarbeitung/abbildungen/TorresVsAguirre/lauf_metadaten.json`** —
      enthält den Rechnernamen aus `platform.node()`. Entweder das Feld
      entfernen oder die Datei nicht übernehmen. Für künftige Läufe
      überlegen, ob der Hostname im Manifest überhaupt gebraucht wird —
      die SLURM-Job-ID und der Git-Commit identifizieren den Lauf bereits.

## C — Darf bleiben, oder ist eine Entscheidung

- Dein **Name** auf der Titelseite der Ausarbeitung
  (`Ausarbeitung/vorspann/titelseite.tex`, Zeile 17) und als `\author`
  (`Ausarbeitung/Test/test.tex`, Zeile 12). Das ist der Zweck einer Thesis, kein
  Leck.

- [ ] **Entscheidung: gehört `Ausarbeitung/` überhaupt mit?** 49 versionierte
      Dateien LaTeX und Abbildungen. Für ein Code-Repo, das als Arbeitsprobe
      taugen soll, sind zwei getrennte Repos oft sauberer — Code öffentlich,
      Thesis-Text privat oder erst nach der Benotung. Muss jetzt nicht
      entschieden werden, aber vor dem Anlegen.

## D — Nicht personenbezogen, aber vorher zu klären: Datenlizenzen

Gehört in dieselbe Runde, weil es dasselbe Risiko ist: etwas
veröffentlichen, das man nicht veröffentlichen darf.

**Kein Rechtsrat.** Das ist eine Recherche der Seitentexte, Stand 24.11.2026.
Bei DBAASP und APD ist eine kurze Mail an die Betreiber der sauberste Weg,
falls die Daten wirklich mit ins öffentliche Repo sollen.

### Die drei genutzten Datenbanken

- [ ] **DRAMP 5.0** — unproblematisch. Die Daten stehen unter CC BY 4.0,
      Weiterverbreitung ist also mit Namensnennung erlaubt. Zu zitieren:
      Ma T. et al. (2026) DRAMP 5.0: an update integrating sequence and
      structure in a novel FASTS format for AI-driven design of antimicrobial
      peptides. Nucleic Acids Research, gkag989.
      Zwei Punkte zum Mitschreiben: (a) Wir haben **5.0** geladen, nicht 4.0 —
      die Versionsnummer in QUELLEN.md muss stimmen. (b) Die Freigabe gilt für
      die general- und clinical-Daten; für die Patent-AMPs verlangt die Seite
      eine Autorisierung. Wir nutzen nur `general_amps.xlsx`, sind also nicht
      betroffen — das gehört aber als Satz in QUELLEN.md, damit später niemand
      die Patent-Tabellen nachträglich dazulegt.

- [ ] **DBAASP** — widersprüchlich, deshalb vorsichtig behandeln.
      Die Terms and Conditions enthalten zwei Aussagen, die sich nicht
      vertragen. Der Abschnitt "Data access policy" bezeichnet die Daten als
      public domain, frei verteilbar und kopierbar, und verlangt nur eine
      öffentliche Nennung von DBAASP. Die Klausel "Non-Distribution of Data"
      verlangt dagegen, die Kontrolle über die Daten zu behalten, sie nicht zu
      verkaufen und auch nicht sonst an irgendwen weiterzugeben.
      Diese zweite Klausel ist wörtlich aus einer NIH-Data-Use-Agreement-
      Vorlage übernommen, wo sie an Position 3 steht und mit "except those
      authorized under this DUA" weitergeht — der Zusatz fehlt hier. Sie passt
      nicht zu einer offenen Datenbank und ist offenbar Boilerplate.
      Konsequenz für uns: `dbaasp_messungen.csv` nicht ins öffentliche Repo.
      Stattdessen `src/aguirre/extern/dbaasp.py` und QUELLEN.md versionieren —
      damit ist der Datensatz reproduzierbar, ohne ihn weiterzuverbreiten.

- [ ] **APD6** — am restriktivsten. Keine Lizenz, keine
      Nutzungsbedingungen, im Seitenfuß "© 2003-present ... All Rights
      Reserved". Ohne eingeräumte Erlaubnis bleibt es beim gesetzlichen
      Standard; dazu kommt in Deutschland das Datenbankherstellerrecht
      (§§ 87a ff. UrhG) auf die Sammlung als solche — unabhängig davon, dass
      einzelne Sequenzen keine eigene Schutzhöhe haben.
      Die Seite bittet um Zitation: Wang G., Schmidt C., Li X., Wang Z. (2026)
      APD6: The antimicrobial peptide database is expanded to promote research
      and development by deploying an unprecedented information pipeline.
      Nucleic Acids Research 54, D363-D374.
      Konsequenz: die FASTA und `apd_novelty.csv` nicht ins öffentliche Repo.

- [x] **Torres-Datensatz** (`torres_train.fasta`,
      `torres_cellbiomaterial.csv`, `torres_train_sequences.csv`) —
      **MIT, mit einer Einschränkung.** Die README von
      github.com/programmablebio/amp-diffusion nennt MIT; eine eigene
      LICENSE-Datei liegt nicht im Repo, die Angabe steht also nur im
      Fließtext. Die Trainings- und Ergebnisdaten liegen dort mit drin (`note/eval/train_eval.csv` = die 19 670 Trainingssequenzen,
      `results/cell_biomaterial/ampdiffusion.csv` = unser
      `torres_cellbiomaterial.csv`). Weiterverbreitung mit Lizenz- und
      Copyright-Hinweis also erlaubt.
      Ein Vorbehalt: MIT ist Torres' Lizenz auf seine Zusammenstellung. Die
      zugrundeliegenden Datenbankeinträge haben weiterhin ihre eigenen
      Bedingungen (siehe DBAASP und APD oben) — eine Lizenz kann niemand
      weitergeben, die er selbst nicht hatte. Für unseren Fall praktisch
      unproblematisch, weil wir die Torres-Dateien ohnehin nur als
      Trainings- und Vergleichsdaten benutzen und nicht als Datenbankauszug
      ausgeben.

### Was das für die Repo-Struktur heißt

- [ ] Wenn DBAASP und APD nicht weiterverbreitet werden dürfen, kann
      `data_tracked/extern/` nicht als versionierter Ordner bleiben. Dann:
      `data_tracked/extern/*` in die `.gitignore`, nur `QUELLEN.md` bleibt
      versioniert, und die Beschaffung läuft über die Skripte in
      `src/aguirre/extern/`. Für David heißt das: er zieht die Skripte aus dem
      Repo und baut die CSVs einmal selbst — bei DRAMP und APD je ein
      Download, bei DBAASP ein Lauf von etwa 70 Minuten.
      **Entscheidung offen.** Betrifft nur das öffentliche Repo, nicht das
      jetzige Arbeitsrepo.

- [x] **`Torres_Model/` und `Paper/`** sind bereits in der `.gitignore`
      (Zeilen 103 und 108). Gut so — die PDFs der Paper dürfen ohnehin nicht
      weiterverbreitet werden.

- [x] **`runs/`** ist ignoriert (Zeile 34). Damit landen Laufmetadaten mit
      Hostnamen nicht im Repo — bis auf die eine nach `Ausarbeitung/`
      kopierte Datei aus Abschnitt B.

---

## E — Warum ein frisches Repo und kein Umzug der History

Das eigentliche Problem ist nicht die aktuelle Fassung einer Datei, sondern die
**Commit-History**. Wer eine Adresse in einem späteren Commit löscht, hat sie
nicht entfernt: die alte Version bleibt abrufbar, und jeder Klon trägt sie
weiter. Rausbekommen geht nur über ein History-Rewrite (`git filter-repo`), und
das ändert alle Commit-Hashes — alle bestehenden Klone werden unbrauchbar, auch
Davids.

Für ein neues öffentliches Repo gibt es deshalb den einfacheren Weg: **`git
init` auf einem bereinigten Stand**, ein erster Commit, fertig. Kein Rewrite,
keine Altlasten, kein Risiko, eine Stelle übersehen zu haben, die nur noch in
der History steht.

Was dabei verloren geht, ist die Commit-History. Für ein Repo, das als
Arbeitsprobe und als Grundlage für die Veröffentlichung dient, ist das kaum ein
Verlust — und das bestehende Repo auf dem THM-Cluster bleibt als vollständiges
Arbeits- und Historienarchiv erhalten.

## F — Ablauf, wenn es soweit ist

1. Entscheiden, was mitkommt: Code (`src/`, `scripts/`, `nb/`), `notizen/`,
   `data_tracked/` ja/nein, `Ausarbeitung/` ja/nein. `Beispiel Notebooks/`
   nicht.
2. Neuen Ordner anlegen und den gewünschten Stand hineinkopieren — **ohne
   `.git`**. Damit ist die History von vornherein weg und kein Rewrite nötig.
3. `nbstripout` installieren und als Filter einrichten, dann die Notebooks
   einmal durchlaufen lassen, damit die bestehenden Ausgaben verschwinden.
4. Abschnitte A und B abarbeiten.
5. Prüfbefehle aus Abschnitt G laufen lassen, bis sie nichts mehr finden.
6. `git init`, erster Commit, Repo auf GitHub anlegen, pushen.
7. `README.md` und `LICENSE` ergänzen — ohne Lizenzdatei ist öffentlicher Code
   formal „alle Rechte vorbehalten", also für andere nicht benutzbar.
8. Das Cluster-Repo unverändert als Archiv behalten.

Zum Namen: das aktuelle Repo heißt `FuE_Masterarbeit`, was zwei Dinge
zusammenzieht, die verschieden sind. Das Python-Paket heißt `aguirre` und die
Modellvarianten Aguirre Small und Big — ein Repo-Name in dieser Richtung wäre
kohärenter. Deine Entscheidung.

## G — Prüfbefehle

Alle aus dem Projektwurzelverzeichnis, alle nur auf **versionierte** Dateien.
Sie sind so geschrieben, dass sie keine personenbezogenen Daten enthalten —
Benutzername und Rechnername holt die Shell selbst.

Mailadressen jeder Art:

```bash
git ls-files -z | xargs -0 grep -nIE "[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}" \
  | grep -viE "noreply|example\."
```

Eigener Benutzername (findet Home-Pfade und `ssh`-Beispiele):

```bash
git ls-files -z | xargs -0 grep -nIi "$(whoami)"
```

Rechnername:

```bash
git ls-files -z | xargs -0 grep -nIi "$(hostname -s)"
```

Absolute Pfade auf Nutzerverzeichnisse, unabhängig vom Namen:

```bash
git ls-files -z | xargs -0 grep -nIE "/(Users|home)/[a-z0-9._-]+" | head -40
```

Notebooks mit nicht geleerten Ausgaben:

```bash
git ls-files "*.ipynb" -z | xargs -0 grep -lE '"output_type"'
```

Erst wenn alle fünf leer ausgehen (bis auf gewollte Treffer wie den Namen auf
der Titelseite), ist der Stand veröffentlichungsreif.
