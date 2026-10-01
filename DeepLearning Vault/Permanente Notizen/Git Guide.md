31-07-2026
Tags: #FuE #MachineLearning 
Status: #unextended

# Quickinfo: Git & GitHub Basics

> Grundworkflow für Versionskontrolle des Python-Lernkurs-Repos (und späterer Projekte wie FuE/PyTorch)

---

## Grundbegriffe

|Begriff|Bedeutung|
|---|---|
|**Repository (Repo)**|Ordner, dessen Änderungshistorie Git mitverfolgt|
|**Commit**|Schnappschuss des Codes zu einem Zeitpunkt, mit Beschreibung|
|**Remote**|Entfernte Kopie des Repos (bei dir: GitHub)|
|**Push**|Lokale Commits zum Remote hochladen|
|**Pull**|Änderungen vom Remote herunterladen|
|**Staging Area**|Zwischenschritt: "was soll in den _nächsten_ Commit rein?"|

**Merksatz Staging vs. Commit:** `git add` lädt die Schrotflinte → `git commit` schießt sie ab. `add` = auswählen was rein soll, `commit` = Schnappschuss festmachen.

---

## Der Standard-Workflow (nach Code-Änderungen)

```bash
cd ~/Documents/Python_Lernkurs      # in den Projektordner wechseln
git status                          # Übersicht: was hat sich geändert?
git add -A                          # ALLES stagen (auch Löschungen/Verschiebungen)
git status                          # Kontrolle vor dem Commit
git commit -m "Kurze inhaltliche Beschreibung"
git push                            # Hochladen zu GitHub
```

**Wichtig zu `git status`-Ausgaben:**

- `deleted` → Datei fehlt am alten Ort (oft nur verschoben/umbenannt)
- `renamed` (nach `git add -A`) → Git hat Verschiebung/Umbenennung selbst erkannt
- `new file` → komplett neu, noch nie getrackt
- `Untracked files` → Git kennt diese Datei/Ordner noch gar nicht

**Commit-Message:** beschreibt _was_ sich inhaltlich geändert hat, nicht _wie_ (nicht "add ausgeführt", sondern z.B. "Modul 3 Pandas abgeschlossen").

---

## Neues Repo für ein anderes Projekt anlegen (z.B. Master-FuE)

Einmalig, falls Ordner noch kein Git-Repo ist (GitHub-Repo vorher leer anlegen):

```bash
cd ~/pfad/zum/projekt
git init
git remote add origin git@github.com:PMedPy/NAME-DES-REPOS.git
```

Danach normaler Workflow (add → commit → push) wie oben.

Der SSH-Key gilt für **alle** eigenen Repos auf GitHub — keine erneute SSH-Einrichtung nötig.

---

## SSH-Setup (nur bei Problemen/neuem Key nötig)

Neuen Key erzeugen:

```bash
ssh-keygen -t ed25519 -C "email@beispiel.com" -f ~/.ssh/id_ed25519_new
```

Public Key anzeigen (→ bei GitHub unter Settings → SSH and GPG keys eintragen):

```bash
cat ~/.ssh/id_ed25519_new.pub
```

Mac anweisen, den neuen Key zu nutzen — `~/.ssh/config`:

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_new
  IdentitiesOnly yes
```

Verbindung testen:

```bash
ssh -T git@github.com
```

Erfolg = `Hi <username>! You've successfully authenticated...`

---

## 🧠 Schwachstellen (persönliche Stolperpunkte)

- [ ] Verwechslung: `git add` ≠ `git commit` — add wählt aus, commit macht fest
- [ ] SSH-Passphrase ist **nirgends einsehbar** (auch nicht in der Key-Datei) — bei Vergabe sofort in Passwort-Manager notieren
- [ ] Terminal zeigt bei Passwort-/Passphrase-Eingabe **keine Zeichen an** — wirkt wie Einfrieren, ist aber normal
- [ ] `git add -A` (nicht nur `git add .`) nutzen, wenn auch Löschungen im ganzen Repo erfasst werden sollen

---

## Offene Punkte für später

- `.gitignore` einrichten (besonders vor FuE/PyTorch-Projekt: Checkpoints, `__pycache__/`, große Rohdaten ausschließen)
- Branches (parallele Experimente, ohne funktionierenden Code zu gefährden) — noch nicht behandelt
# Weiterführung

# Referenzen
[@bishopDeepLearningFoundations2024]