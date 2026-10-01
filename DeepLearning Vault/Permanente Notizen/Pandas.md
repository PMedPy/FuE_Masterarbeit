16-06-2026 Tags: #Python #FuE Status: #extended

# 📊 Pandas Quickinfo – Befehlsübersicht und Konzepte

> Diese Datei soll wachsen. Erweitere Tabellen bei Bedarf, ergänze eigene Spickzettel-Beispiele unten.

---

## 🧠 Das Wichtigste in 3 Sätzen

1. **DataFrame = Tabelle**, **Series = Spalte**. Beide sind „Listen mit Superkräften" (Index-Labels, Datentyp, vektorisierte Operationen, eingebaute Statistik).
2. **Vektorisiertes Denken**: Operationen wirken auf die **ganze Spalte gleichzeitig** – keine Schleife nötig, schneller und lesbarer.
3. **Boolesche Masken** (`df["x"] > 5`) sind die zentrale Filtermethode. Sie liefern eine `True/False`-Series, die du als „Schablone" auf den DataFrame legst.

---

## 🧭 Der Workflow: DataFrame → Modell

Der rote Faden über das ganze Modul: **erst verstehen, dann modellfertig machen.** Die Grenze zwischen beiden Hälften ist die wichtigste Struktur im Ablauf.

**Verstehen (explorativ):**

1. **Laden** → Index-Check (`index_col=`) → **dtypes prüfen** (`df.dtypes`): ist eine Zahlenspalte fälschlich `object`? → `pd.to_numeric(errors="coerce")`
2. **NaN messen** (`df.isna().mean().sort_values(ascending=False)`) → _dann_ Strategie: `dropna(subset=)` / `fillna` / lassen
3. **Kategorisieren** (`astype("category")`) — früh, schaltet `groupby`/`observed=False` frei
4. **Explorieren**: `groupby`, `agg`, plotten
5. **Gruppenstruktur erkennen & erhalten**: Sind Beobachtungen genestet (→ Hierarchie)? Struktur sauber kodiert halten

**Modellfertig machen (ganz am Ende):** 6. _(situativ)_ **wide → long** (`melt`) fürs hierarchische Modell / Auswertung 7. **Split** (auf Gruppenebene!) → **Encoding** (`get_dummies`) → **Standardisieren** (mit **Trainings**-Statistiken)

> 💡 **Kategorisieren = für die Analyse (früh). Encoding = für das Modell (spät).** Dazwischen liegt die ganze explorative Arbeit – Dummies sind für Menschen unlesbar.

> ⚠️ **Hierarchie ≠ Pooling.** Hierarchie ist eine _Eigenschaft der Daten_ (ablesbar: steckt X in Y?). Partial Pooling ist eine _Modellentscheidung_ über den Informationsaustausch zwischen Gruppen – gehört ins Modell, nicht in die Datenaufbereitung. (Und beides ist unabhängig von Kausalität.) Im Pandas-Schritt nur: Struktur sichtbar halten, Entscheidung offen lassen.

---

## 🛠️ Import & Grundsetup

```python
import pandas as pd
```

Konvention: **immer `pd`** als Alias. So sehen es alle Tutorials und Bücher.

---

## 🧭 Der Index – das Herzstück von Pandas

Der **Index** ist keine Deko, sondern eine **eigene Achse mit Identität**. Jede Series und jeder DataFrame hat einen. Er ist der Schlüssel, über den Pandas Daten ausrichtet, verbindet und adressiert.

- **Series** hat eine Achse: den Zeilenindex (`s.index`).
- **DataFrame** hat zwei: Zeilenindex (`df.index`, Achse 0) und Spaltenindex (`df.columns`, Achse 1).
- Der Index **reist mit den Zeilen mit**: Nach Sortieren, Filtern oder `dropna` klebt jedes Label an seiner Zeile – er ist dann nicht mehr `0,1,2,…`, sondern lückenhaft und ungeordnet.

|Aufgabe|Befehl|
|---|---|
|Spalte zum Index machen|`df = df.set_index("MouseID")`|
|Index zurück in Spalte|`df = df.reset_index()`|
|Index-Labels ansehen|`df.index`|
|Label an Position 4 nachschlagen|`df.index[4]`|
|Index umbenennen|`df.index.name = "ID"`|

> 💡 **Mental model:** DataFrame = ein `dict` von Series, die sich **einen gemeinsamen Index teilen**. Deshalb ist `df["spalte"]` eine Series, und deshalb hat jede Spalte ihren **eigenen dtype** (anders als ein NumPy-Array mit einem dtype für alles).

> ⚠️ **Erst zusammenfügen, dann Index umsetzen.** Wenn du `set_index` zu früh auf nur einem Teil machst, zerstörst du die gemeinsame Ausrichtungsbasis, und der nächste `join` läuft ins Leere → alles NaN. Regel: solange alle Teile denselben (Range-)Index haben zusammenfügen, dann den Index umsetzen.

---

## 🏗️ DataFrame & Series erstellen

|Aufgabe|Befehl|
|---|---|
|Series aus Liste|`pd.Series([1, 2, 3])`|
|Series mit Index-Labels|`pd.Series([21, 22], index=["Mo", "Di"])`|
|Series mit Name|`pd.Series([1, 2], name="Temp")`|
|DataFrame aus Dictionary|`pd.DataFrame({"A": [1, 2], "B": [3, 4]})`|
|DataFrame aus Liste-von-Listen|`pd.DataFrame([[1, 3], [2, 4]], columns=["A", "B"])`|
|Leerer DataFrame|`pd.DataFrame()`|

---

## 📂 Excel- und CSV-Dateien

|Aufgabe|Befehl|
|---|---|
|Excel einlesen|`pd.read_excel("datei.xlsx")`|
|Bestimmtes Sheet|`pd.read_excel("datei.xlsx", sheet_name="Daten")`|
|Alle Sheets als Dict|`pd.read_excel("datei.xlsx", sheet_name=None)`|
|Zeilen am Anfang überspringen|`pd.read_excel("datei.xlsx", skiprows=4)`|
|CSV einlesen|`pd.read_csv("datei.csv")`|
|CSV mit Trennzeichen Tab|`pd.read_csv("datei.csv", sep="\t")`|
|Excel speichern (ohne Index-Spalte)|`df.to_excel("ergebnis.xlsx", index=False)`|
|CSV speichern|`df.to_csv("ergebnis.csv", index=False)`|
|Mehrere Sheets schreiben|`pd.ExcelWriter(...)` (siehe Beispiel unten)|

**Mehrere Sheets schreiben:**

```python
with pd.ExcelWriter("bericht.xlsx") as writer:
    statistik_df.to_excel(writer, sheet_name="Statistik", index=False)
    rohdaten_df.to_excel(writer, sheet_name="Rohdaten", index=False)
```

> ⚠️ **`index=False`** verhindert, dass die 0/1/2-Indexspalte mitgeschrieben wird. Fast immer gewollt.

---

## 🔍 DataFrame inspizieren

|Aufgabe|Befehl|
|---|---|
|Erste 5 Zeilen|`df.head()`|
|Letzte 5 Zeilen|`df.tail()`|
|Erste N Zeilen|`df.head(10)`|
|Anzahl Zeilen × Spalten|`df.shape`|
|Spaltennamen|`df.columns`|
|Index-Labels|`df.index`|
|Datentypen pro Spalte|`df.dtypes`|
|**dtypes zählen**|`df.dtypes.value_counts()`|
|Zusammenfassung|`df.info()`|
|**Komplettstatistik**|`df.describe()`|
|Anzahl unique Werte|`df["spalte"].nunique()`|
|Wert-Häufigkeiten|`df["spalte"].value_counts()`|

> 💡 `df.describe()` ist Gold wert beim ersten Blick auf neue Daten: Mittelwert, Std., Min, Max, Quartile – auf einen Streich.

> 💡 **Series-Shape ist `(n,)`** – ein Tupel mit _einem_ Eintrag. Das Komma zeigt „eindimensional", eine Series hat nur eine Achse. Der DataFrame dagegen: `(zeilen, spalten)`.

---

## 🎯 Spalten und Zeilen auswählen: `loc` vs. `iloc`

Es gibt **zwei Adressierungssysteme**, weil man eine Zelle auf zwei Arten benennen kann: über ihr **Label** oder über ihre **Position**.

|`loc`|`iloc`||
|---|---|---|
|adressiert über|**Label** (l wie _label_)|**Position** (i wie _integer_)|
|Slice-Ende|**inklusive**|**exklusive** (wie bei Listen)|
|überlebt Sortieren|✅ folgt dem Namen|❌ folgt der Stelle|
|Reihenfolge|`df.loc[zeile, spalte]`|`df.iloc[zeile, spalte]`|

```python
df.loc["309_7", "DYRK1A_N"]   # Zeile mit Label "309_7", Spalte "DYRK1A_N"
df.iloc[6, 0]                 # siebte Zeile (ab 0!), erste Spalte

df.iloc[0:5]                  # FÜNF Zeilen (0,1,2,3,4) – Ende exklusiv
df.loc["309_1":"309_5"]       # FÜNF Zeilen inkl. "309_5" – Ende INKLUSIVE

df.loc["309_7", :]            # ganze Zeile (: = alle Spalten)
df.loc[:, "ITSN1_N"]          # ganze Spalte (: = alle Zeilen)
```

**Warum ist die Slice-Grenze bei `loc` inklusive?** Bei Labels wie `"309_7"` gibt es kein „eins davor", das Pandas ausschließen könnte – Labels haben keine Arithmetik. Also nimmt Pandas das Endlabel mit.

> ⚠️ **Eckige Klammern, immer.** `series.iloc[6]` ✅ – `series.iloc(6)` ❌ ruft `iloc` als _Funktion_ auf und liefert einen kryptischen `No axis named 6`-Fehler. `loc`/`iloc` sind Indexer-Objekte, in die man mit `[...]` hineingreift (wie `liste[0]`), keine Funktionen.

> 💡 **Regel:** `loc` ist der Default. `iloc` nur, wenn du wirklich Position meinst („die erste Zeile, egal welche").

### Der Kern-Effekt: Label folgt der Zeile, Position folgt der Stelle

Solange der Index brav `0,1,2,…` bzw. in Originalreihenfolge ist, zeigen `loc` und `iloc` auf **dieselbe** Zelle. **Nach dem Sortieren fallen sie auseinander:**

```python
df_sortiert = df.sort_values("DYRK1A_N", ascending=False)

df_sortiert["DYRK1A_N"].iloc[6]        # → siebtgrößter Wert (Position 6 der sortierten Reihe)
df_sortiert["DYRK1A_N"].loc["309_7"]   # → derselbe Wert wie VOR dem Sortieren
                                        #   Maus 309_7 hat sich nicht verändert, nur ihre Position
```

> 💡 `sort_values` schiebt **NaN standardmäßig ans Ende** (auf- wie absteigend). Steuerbar mit `na_position="first"`.

> 💡 **String-Sortierung ≠ numerische Sortierung.** Ist die MouseID ein String, sortiert Pandas zeichenweise: `309_10` kommt **vor** `309_2`, weil `"1" < "2"`. Kein Bug.

---

## 🔗 Stille Ausrichtung (Alignment) – die NaN-Falle

Operationen zwischen zwei Series/DataFrames richten sich **am Index aus, nicht an der Position**. Pandas bildet die **Vereinigung der Labels**, rechnet passende Labels zusammen und setzt alles Nicht-Passende auf `NaN` – **ohne Fehler, ohne Warnung**.

```python
a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["z", "y", "x"])
a + b        # x→31, y→22, z→13   (labelweise! NICHT [11,22,33])

# Die Falle:
beg = df["DYRK1A_N"].iloc[:10]    # erste 10 Mäuse
end = df["ITSN1_N"].iloc[-10:]    # letzte 10 Mäuse
beg + end                          # fast alles NaN – keine Maus in BEIDEN Series

# Der Beweis, dass Überlappung addiert wird:
df["DYRK1A_N"].iloc[5:15] + df["ITSN1_N"].iloc[5:15]   # hier überlappen alle Labels → echte Summen
```

> ⚠️ Das ist **kein `IndexError`**! Ein `IndexError` ist ein echter Fehler (z.B. `liste[999]`). Stille Ausrichtung wirft _keinen_ Fehler – genau das macht sie tückisch. Nenn es **Alignment**, nicht Error.

> 💡 Ausrichtung ist auch der Grund, warum `X.join(y)` über den Index läuft: Haben beide denselben Index, stehen die Zeilen korrekt nebeneinander. Hätte `y` einen anderen Index, entstünden lautlos NaN-Blöcke.

---

## 📐 Filter mit booleschen Masken

|Aufgabe|Befehl|
|---|---|
|Vergleich → Maske|`df["Temp"] > 22`|
|Maske anwenden|`df[df["Temp"] > 22]`|
|UND-Verknüpfung|`df[(df["A"] > 0) & (df["B"] < 10)]`|
|ODER-Verknüpfung|`df[(df["A"] == 0) \| (df["A"] == 1)]`|
|NICHT-Verknüpfung|`df[~(df["A"] > 0)]`|
|Wertebereich|`df[df["Temp"].between(20, 25)]`|
|Element in Liste|`df[df["Tag"].isin(["Mo", "Di"])]`|
|Strings, die enthalten|`df[df["Name"].str.contains("pH")]`|

> ⚠️ **Wichtig:** Bei kombinierten Bedingungen **Klammern** setzen und `&`/`|` statt `and`/`or` verwenden!

### Was eine Maske _ist_ – Minibeispiel

```python
df = pd.DataFrame({
    "Maus":    ["A","A","A","B","B","B","C","C","C"],
    "Protein": [0.20, 0.55, 0.80, 0.35, 0.62, 0.41, 0.75, 0.15, 0.90],
    "Gruppe":  ["ctrl","ctrl","ts","ctrl","ts","ts","ctrl","ts","ctrl"]
}, index=["m1","m2","m3","m4","m5","m6","m7","m8","m9"])

maske = df["Protein"] > 0.5
# m1 False, m2 True, m3 True, m4 False, m5 True,
# m6 False, m7 True, m8 False, m9 True   → dtype: bool
```

Drei Eigenschaften: die Maske ist eine **Series** (`dtype: bool`), sie hat **denselben Index** wie `df`, und **dieselbe Länge**. Entstanden durch elementweisen Vergleich (Vektorisierung: kein Loop im Code, neun Vergleiche im Ergebnis).

```python
df[maske]        # → m2, m3, m5, m7, m9   (5 Zeilen)
```

**Mental model: Lochschablone.** Wo `True`, fällt die Zeile durch und wird mitgenommen; wo `False`, bleibt sie zurück.

### Was `[...]` je nach Inhalt bedeutet

|In der Klammer|Interpretiert als|Ergebnis|
|---|---|---|
|String `df["Protein"]`|Spaltenname|Series (eine Spalte)|
|Liste `df[["Maus","Gruppe"]]`|Spaltennamen|DataFrame (mehrere Spalten)|
|**Boolean-Series** `df[maske]`|**Zeilenfilter**|DataFrame (Teilmenge Zeilen)|
|Slice `df[0:3]`|Zeilenpositionen|DataFrame (Zeilen)|

Immer „herausgreifen", aber **der Typ des Schlüssels entscheidet, ob Spalten oder Zeilen**.

> 💡 **Filtern läuft über den Index, nicht über die Position.** Eine umsortierte Maske (`maske.sort_index(ascending=False)`) liefert dasselbe Ergebnis – Pandas gleicht über Labels ab. Eine Maske mit **fremdem** Index dagegen scheitert. Sie funktioniert normalerweise, weil sie _aus_ dem DataFrame entsteht.

### Die Maske als eigenständiges Objekt

```python
maske.sum()     # 5     → wie viele Zeilen erfüllen die Bedingung
maske.mean()    # 0.556 → welcher Anteil (5/9)
~maske          # Negation: die anderen 4
```

### Warum `&` statt `and`, und warum Klammern

```python
m1 = df["Protein"] > 0.5      # F T T F T F T F T
m2 = df["Gruppe"] == "ctrl"   # T T F T F F T F T
df[m1 & m2]                   # → m2, m7, m9
```

- **`and` scheitert**, weil es _einen_ Wahrheitswert auswerten will, aber eine neunelementige Series bekommt → _„The truth value of a Series is ambiguous"_. `&` verknüpft **elementweise**.
- **Klammern sind Pflicht**, weil `&` in Python **stärker bindet** als `>` und `==`. Ohne sie versucht Python zuerst `0.5 & df["Gruppe"]`.

```python
df[df["Protein"] > 0.5 & df["Gruppe"] == "ctrl"]      # ❌ kaputt
df[(df["Protein"] > 0.5) & (df["Gruppe"] == "ctrl")]  # ✅ richtig
```

### Weitere Filter-Werkzeuge

```python
df[df.index.str.startswith("309")]     # alle Messungen von Maus 309
df.query("Protein > 0.5 and Gruppe == 'ctrl'")   # String-Syntax, erlaubt and/or
```

> 💡 `query` parst den String selbst, deshalb sind dort `and`/`or` erlaubt. Bei langen Filtern oft lesbarer.

---

## 🧮 Statistik (auf Series oder DataFrame)

|Methode|Was sie tut|
|---|---|
|`.mean()`|Mittelwert|
|`.median()`|Median|
|`.std()`|**Stichproben**-Standardabweichung (n-1)|
|`.var()`|Varianz|
|`.min()` / `.max()`|Min / Max|
|`.sum()`|Summe|
|`.count()`|Anzahl nicht-leerer Werte|
|`.quantile(0.25)`|Quartil (oder beliebige Quantile)|
|`.corr()`|Korrelationsmatrix (DataFrame)|
|`.describe()`|Mehrere Statistiken auf einmal|
|`.isna().sum()`|**Anzahl NaN pro Spalte** (Boolean→Summe)|

Anwendung:

```python
df["Temperatur"].mean()       # auf einer Spalte
df.mean()                     # auf allen numerischen Spalten
df.mean(axis=1)               # zeilenweise (selten)
```

> 💡 **Achsen-Logik wie in NumPy:** `.sum()` reduziert per Default über `axis=0` (über die Zeilen hinweg) → ein Wert pro Spalte. `df.isna().sum()` zählt so die NaN pro Spalte. Nur sind die Achsen hier **benannt** statt anonym nummeriert.

---

## ➕ Neue Spalten / Spalten ändern

|Aufgabe|Befehl|
|---|---|
|Neue Spalte aus alter|`df["doppelt"] = df["x"] * 2`|
|Spalte aus mehreren|`df["summe"] = df["A"] + df["B"]`|
|Spalte mit Funktion|`df["log_x"] = df["x"].apply(np.log)`|
|Spalte umbenennen|`df = df.rename(columns={"alt": "neu"})`|
|Spalte löschen|`df = df.drop(columns="x")`|
|Spalten umordnen|`df = df[["B", "A", "C"]]`|

---

## 🧹 Daten bereinigen

|Aufgabe|Befehl|
|---|---|
|Zeilen mit NaN entfernen|`df.dropna()`|
|Nur Spalten, in denen NaN sind|`df.dropna(subset=["x"])`|
|NaN durch 0 ersetzen|`df.fillna(0)`|
|NaN durch Mittelwert|`df["x"].fillna(df["x"].mean())`|
|Duplikate entfernen|`df.drop_duplicates()`|
|Typ umwandeln|`df["x"] = df["x"].astype(float)`|
|**Kategorie-Typ setzen**|`df["gen"] = df["gen"].astype("category")`|
|Strings vereinheitlichen|`df["s"] = df["s"].str.lower().str.strip()`|

> 💡 `NaN` (Not a Number) ist Pandas' eingebauter „Fehlt"-Marker. Statistik-Methoden ignorieren ihn standardmäßig.

> 💡 **`category`-dtype** für Spalten mit wenigen Ausprägungen (z.B. `control`/`Ts65Dn`): spart Speicher und schaltet später `groupby` und die `coords`/`dims`-Logik in PyMC frei.

### Die Natur von NaN

`NaN` stammt aus dem **IEEE-754-Fließkommastandard**, nicht aus Pandas. Drei Konsequenzen:

```python
5 + np.nan        # nan   → NaN ist ansteckend
np.nan * 0        # nan   → auch hier, nicht 0
np.nan == np.nan  # False → NaN ist nie gleich sich selbst!
```

> ⚠️ **`NaN` findet man mit `.isna()`, niemals mit `== NaN`.** `df["x"] == np.nan` liefert überall `False`. Das ist der wichtigste Satz zum Thema.

> 💡 **NaN lebt nur im Float-Raum.** Eine Integer-Spalte, die ein NaN bekommt, wird automatisch zu `float64`. Deshalb bricht `astype(int)` auf Spalten mit NaN.

> ⚠️ **Statistik überspringt NaN still.** `df["x"].mean()` rechnet über die vorhandenen Werte – der Mittelwert einer zu 90 % leeren Spalte sieht genauso „gültig" aus wie der einer vollen. **Erst das Ausmaß messen, dann behandeln.**

### NaN-Muster kartieren (immer der erste Schritt)

```python
df.isna().sum()                                    # NaN pro Spalte (absolut)
df.isna().sum().sort_values(ascending=False).head(10)   # Top 10 – ERST sortieren, DANN köpfen!
df.isna().mean()                                   # NaN-ANTEIL pro Spalte (0..1)

anteil = df.isna().mean()
(anteil > 0.2).sum()      # WIE VIELE Spalten sind >20 % leer  → Maske zählen
anteil[anteil > 0.2].sum()  # ⚠ falsch für "wie viele": summiert die ANTEILE
```

> ⚠️ **`sum` zählt, `mean` anteilt.** `maske.sum()` = wie viele erfüllen die Bedingung; `werte[maske].sum()` = Summe ihrer Werte. Zwei verschiedene Fragen.

> 💡 **Boolean-Mittelwert = relative Häufigkeit.** Bei $b_i \in {0,1}$ ist $\bar b = \frac{1}{n}\sum b_i$ die Trefferquote – zugleich der ML-Schätzer $\hat p$ einer Bernoulli-Verteilung. `isna().mean() = 0.002778` heißt: geschätzte Fehlrate $3/1080$. (Achtung: als _Schätzer mit Unsicherheit_ setzt das unabhängige Ziehungen voraus – bei genesteten Daten braucht es ein hierarchisches Modell.)

### NaN-Strategien: wegwerfen, füllen, lassen

```python
df.dropna()                    # ⚠ jede Zeile mit IRGENDEINEM NaN fliegt
df.dropna(subset=["x"])        # nur wo DIESE Spalte fehlt  ← meist richtig
df.dropna(axis=1)              # ganze SPALTEN mit NaN entfernen
df.dropna(thresh=70)           # Zeile behalten, wenn ≥70 Werte vorhanden
```

Beispiel Mäuse-Datensatz (1080 Zeilen, 77 Proteinspalten):

|Aufruf|verbleibende Zeilen|
|---|---|
|`df.dropna()`|552 – **528 Zeilen weg**, weil eine fehlende Proteinmessung reicht|
|`df.dropna(subset=["DYRK1A_N"])`|1077 – nur 3 weg|

> ⚠️ Blankes `dropna()` löscht **nicht zufällig**, sondern systematisch die Zeilen, in denen bestimmte Proteine nicht gemessen wurden.

**Füllen heißt Daten erfinden** – es ist eine Modellannahme, keine Reinigung:

|Füllung|Wirkung|
|---|---|
|`fillna(0)`|nur wenn 0 inhaltlich „nicht vorhanden" heißt. Bei Expression **falsch** (0 = nicht exprimiert ≠ nicht gemessen)|
|`fillna(x.mean())`|**schrumpft die Varianz**: gefüllte Werte sitzen exakt auf dem Mittel|
|`fillna(x.median())`|robuster bei Schiefe / Ausreißern|
|`fillna(method="ffill")`|letzter gültiger Wert davor (Zeitreihen)|

```python
s = df["DYRK1A_N"]
s.std()                      # vorher
s.fillna(s.mean()).std()     # nachher → kleiner
```

Faustregel nach NaN-Anteil:

|Anteil|Typische Wahl|
|---|---|
|< ~5 %|füllen (Median) oder Zeilen droppen – kaum Einfluss|
|~5–40 %|bewusst füllen + begründen, oder mitmodellieren|
|> ~50 %|Spalte verwerfen (`axis=1`) – Auffüllen erfindet mehr als es rettet|

> 💡 **Lassen ist oft die beste Wahl.** Ein NaN ist ehrliche Information („hier wissen wir nichts"). Viele Verfahren – besonders Bayes-Modelle – können fehlende Werte mitmodellieren.

> ⚠️ **Kein `inplace=True`.** Wird abgekündigt und verhält sich bei verketteten Zugriffen tückisch. Immer explizit zuweisen: `df["x"] = df["x"].fillna(...)`.

### Duplikate

```python
df.duplicated()                       # True ab dem ZWEITEN Vorkommen
df.duplicated(keep=False)             # ALLE Vorkommen markieren  ← zum Inspizieren
df.duplicated().sum()                 # wie viele
df.drop_duplicates(subset=["MouseID"])  # Duplikate NUR anhand des Schlüssels
df.index.duplicated().sum()           # doppelte Index-Labels
```

> 💡 **Was ist überhaupt „doppelt"?** Zweimal dieselbe `MouseID` = Fehler (Messung doppelt eingelesen). Zwei Zeilen mit gleichen Proteinwerten, aber verschiedenen IDs = Zufall, **nicht löschen**. Deshalb ist `subset=` fast immer nötig.

> ⚠️ Bei 77 Fließkommaspalten ist `df.duplicated().sum() == 0` fast garantiert – zwei exakt identische Zeilen wären astronomisch unwahrscheinlich. Der **Index-Check ist der schärfere**: er prüft inhaltliche Eindeutigkeit. Ein eindeutiger Index ist stille Grundannahme vieler Pandas-Operationen (auch der Ausrichtung!).

---

## 🔃 Sortieren und Gruppieren

|Aufgabe|Befehl|
|---|---|
|Nach Spalte sortieren|`df.sort_values("Temp")`|
|Absteigend|`df.sort_values("Temp", ascending=False)`|
|NaN nach vorne|`df.sort_values("Temp", na_position="first")`|
|Nach mehreren Spalten|`df.sort_values(["A", "B"])`|
|Nach Index|`df.sort_index()`|
|Gruppieren|`df.groupby("Kategorie").mean()`|

---

## 🗂️ `groupby` – Split · Apply · Combine

`groupby` ist kein Befehl, sondern ein **Denkmuster** in drei Schritten:

1. **Split** – DataFrame anhand einer Schlüsselspalte zerlegen
2. **Apply** – auf jede Gruppe _unabhängig_ dieselbe Operation
3. **Combine** – Teilergebnisse zu einem neuen Objekt

```python
df.groupby("Gruppe")["Protein"].mean()
# Gruppe
# ctrl    0.552
# ts      0.496
```

Die Gruppennamen werden zum **Index** des Ergebnisses – wieder der Index als benannte Koordinate.

### Das GroupBy-Objekt ist faul

```python
g = df.groupby("Gruppe")     # rechnet noch NICHTS, hält nur die Aufteilungsvorschrift
g.groups                     # dict: Gruppenname → Index-Labels
g.get_group("ctrl")          # eine Gruppe als DataFrame
g.size()                     # Zeilen pro Gruppe  ← Standard-Erstblick
```

### Mehrere Schlüssel – der Faktorplan

```python
df.groupby(["Genotype","Treatment","Behavior"], observed=False).size()
```

Bei einem 2×2×2-Design: **8 Gruppen**, Ergebnis mit MultiIndex. Das ist die **Designmatrix-Kontrolle**: Sind alle Zellen besetzt? Ist der Plan balanciert?

> ⚠️ Eine Spalte wie `class`, die die Kombination der drei Faktoren bereits enthält, ist **redundant** – sie mitzugruppieren gibt keine 16 Gruppen, nur eine informationslose Index-Ebene mehr.

### `agg` – mehrere Statistiken, auch eigene

```python
df.groupby("Gruppe")["Protein"].agg(["mean", "std", "count"])

df.groupby("Genotype").agg({              # spaltenweise unterschiedlich
    "DYRK1A_N": ["mean", "std"],
    "ITSN1_N":  "median"
})

df.groupby("Gruppe")["Protein"].agg(lambda s: s.max() - s.min())   # eigene Funktion
```

> 💡 **`count` immer mitführen.** Ein Mittelwert über 3 Zeilen und einer über 300 sehen im Output gleich aus, sind aber völlig verschieden belastbar.

### `agg` vs. `transform` vs. `apply`

|Methode|Rückgabe pro Gruppe|Ergebnis-Länge|
|---|---|---|
|`agg`|ein **Skalar**|eine Zeile pro Gruppe|
|`transform`|so viele Werte wie Zeilen|**wie Original**|
|`apply`|beliebig|beliebig (langsam, letzte Wahl)|

```python
df["Protein_zentriert"] = df["Protein"] - df.groupby("Gruppe")["Protein"].transform("mean")
```

Jede Zeile wird um den Mittelwert **ihrer eigenen Gruppe** zentriert. Weil `transform` die Originallänge behält, passt es direkt als neue Spalte.

**Kontrolle, ob `transform` wirklich gruppenweise wirkte:**

```python
df.groupby("Gruppe")["Protein_zentriert"].mean()   # → ~0 (bis auf 1e-17)
```

Das ist per Konstruktion null:

$$\frac{1}{n_g}\sum_{i \in g}(x_i - \bar{x}_g) = \bar{x}_g - \bar{x}_g = 0$$

> 💡 Faustregel: **`agg` zum Zusammenfassen, `transform` wenn das Ergebnis zurück in die Originaltabelle soll.** Gruppenweise Normalisierung ist Feature Engineering.

### Was `groupby` NICHT kann

`groupby` verarbeitet jede Gruppe **isoliert**. Alles, was Gruppen _miteinander_ vergleicht (ANOVA, Tests), liegt außerhalb:

```python
from scipy import stats
gruppen = [g["Protein"].values for _, g in df.groupby("Gruppe")]
stats.f_oneway(*gruppen)          # groupby liefert nur die Zerlegung
```

### Vorteile von `category` bei `groupby`

- **Performance:** gruppiert über Integer-Codes statt String-Vergleiche
- **Reihenfolge:** bei `ordered=True` in _deiner_ Reihenfolge statt alphabetisch
- **Vollständigkeit:** `observed=False` zeigt auch **leere** Kategorien – bei einem Faktorplan ist eine unbesetzte Zelle eine _Information_, kein Grund zu verschwinden

### Bezug: DoE und Bayes

|Ebene|Werkzeug|
|---|---|
|Design prüfen, Zellen zählen, Zellmittel|`groupby`|
|Varianzzerlegung, Tests, Effektschätzung|`statsmodels` (`ols` + `anova_lm`)|
|Effekte mit Unsicherheit, Partial Pooling|PyMC (hierarchisches Modell)|

```python
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
modell = smf.ols("DYRK1A_N ~ Genotype * Treatment * Behavior", data=df).fit()
print(anova_lm(modell, typ=2))     # * erzeugt Haupteffekte UND alle Interaktionen
```

> 💡 `groupby(...).mean()` ist ein **No-Pooling**-Schätzer: jede Gruppe steht für sich, kleine Gruppen bekommen genauso viel Vertrauen wie große. Das hierarchische Modell macht dasselbe mit **Partial Pooling** – Gruppen leihen sich Information, kleine werden zum Gesamtmittel geschrumpft. ANOVA und hierarchisches Modell beantworten dieselbe Frage: das Verhältnis von Zwischen- zu Innerhalb-Gruppen-Varianz – einmal als Testentscheidung, einmal als kontinuierliches Shrinkage.

---

## 🔗 Tabellen zusammenführen: `merge`, `join`, `concat`

|Werkzeug|Verbindet über|Richtung|
|---|---|---|
|`join`|**Index**|nebeneinander (Spalten)|
|`merge`|**Spalten** (oder Index)|nebeneinander (Spalten)|
|`concat`|Index|**untereinander** (Zeilen), `axis=1` für nebeneinander|

> 💡 Faustregel: **gleiche Spalten, mehr Zeilen → `concat`. Gleiche Zeilen, mehr Spalten → `merge`.**

### Schlüssel angeben

```python
pd.merge(links, rechts, on="EC")                          # gleiche Spalte in beiden
pd.merge(links, rechts, left_on="ID", right_on="mouse")   # unterschiedlich benannt
pd.merge(links, rechts, left_index=True, right_on="ID")   # Index gegen Spalte
```

> ⚠️ **Schlüssel immer explizit mit `on=` angeben.** Ohne `on` merged Pandas über _alle_ gleichnamigen Spalten – bei breiten Tabellen böse Überraschungen.

### Die vier Join-Typen

Entscheidende Frage: **Was passiert mit Zeilen ohne Partner?**

|`how=`|Behält|Bedeutung|
|---|---|---|
|`"inner"`|nur wo **beide** einen Partner haben|Schnittmenge (**Default**)|
|`"left"`|**alle** aus links|linke Tabelle bleibt vollständig|
|`"right"`|**alle** aus rechts|spiegelbildlich|
|`"outer"`|**alle** aus beiden|Vereinigung|

Minibeispiel (Enzym-Szenario):

```python
kinetik = pd.DataFrame({"EC": ["1.1.1.1","2.7.1.1","3.1.1.3"], "kcat": [12.5, 340.0, 8.2]})
seq     = pd.DataFrame({"EC": ["1.1.1.1","2.7.1.1","4.2.1.1"], "laenge": [374, 416, 260]})
```

|`how=`|Zeilen|Ergebnis|
|---|---|---|
|`inner`|2|nur 1.1.1.1 und 2.7.1.1|
|`left`|3|3.1.1.3 bekommt `NaN` bei `laenge`|
|`outer`|4|zusätzlich 4.2.1.1 mit `NaN` bei `kcat`|

> ⚠️ **`how` ist eine inhaltliche Entscheidung, keine technische.** Der Default `inner` wirft **lautlos** Zeilen weg. `left` heißt: „meine linke Tabelle ist die Wahrheit, rechts ist Zusatz."

> 💡 Ein Merge ist eine der **häufigsten NaN-Quellen** – damit schließt sich der Kreis zum Bereinigen.

### Der wichtigste Reflex: Zeilenzahl prüfen

```python
print(len(links), len(rechts), len(ergebnis))
```

Kleiner als erwartet → `inner` hat geschluckt. **Größer** → Vervielfachungs-Falle:

### Vervielfachungs-Falle

Ist der Schlüssel **nicht eindeutig**, multipliziert der Merge die Zeilen:

```python
kinetik_faul = pd.DataFrame({"EC": ["1.1.1.1","1.1.1.1","2.7.1.1","3.1.1.3"],
                             "kcat": [12.5, 12.4, 340.0, 8.2]})
pd.merge(kinetik_faul, seq, on="EC", how="inner")   # → 3 Zeilen, 1.1.1.1 doppelt
```

Bei beidseitig doppelten Schlüsseln entsteht das **kartesische Produkt** innerhalb der Schlüsselgruppe (3×3 → 9 Zeilen).

```python
pd.merge(a, b, on="k", validate="one_to_one")    # MergeError, wenn nicht 1:1
pd.merge(a, b, on="k", validate="one_to_many")
```

> 💡 **`validate=` macht die Kardinalitätsannahme explizit und prüfbar**, statt sie zu hoffen. Der Fehler zeigt direkt, _wo_ eine Dopplung sitzt. Bei BRENDA↔UniProt-Merges unverzichtbar.

### Passung diagnostizieren mit `indicator`

```python
erg = pd.merge(kinetik, seq, on="EC", how="outer", indicator=True)
erg["_merge"].value_counts()
# both          2
# left_only     1
# right_only    1
```

Die schnellste Antwort auf „wie gut haben meine Tabellen zusammengepasst?".

### `concat` – stapeln

```python
pd.concat([df_jan, df_feb])                  # untereinander (axis=0, Default)
pd.concat([df_a, df_b], axis=1)              # nebeneinander über Index
pd.concat([df_a, df_b], ignore_index=True)   # Index neu durchnummerieren
```

---

## 🔄 Reshaping: wide ↔ long, `pivot`/`melt`, MultiIndex

**Der Kernsatz:** In **wide** steckt Information in den **Spaltennamen**, in **long** in den **Daten**.

||Wide|Long||
|---|---|---|
|Zeile =|Beobachtungseinheit|einzelner Messwert|
|Variable steckt in|Spaltennamen|Datenspalte|
|Gut für|Lesen, Korrelationsmatrix, Feature-Matrix (ML)|`groupby`, Plotting, statistische Modelle, PyMC||

### `melt` – wide → long (Spalten werden zu Zeilen)

```python
long = wide.melt(
    id_vars=["MouseID", "Genotype"],      # bleiben stehen, werden wiederholt
    value_vars=["DYRK1A_N", "ITSN1_N"],   # werden eingeschmolzen (weglassen = alle übrigen)
    var_name="Protein",                    # Spalte, die die alten Spaltennamen aufnimmt
    value_name="Wert"                      # Spalte, die die Werte aufnimmt
)
# Zeilenzahl: n_zeilen × n_value_vars. Der Spaltenname wird zum Zellwert.
```

> ⚠️ **`melt` sieht nur Spalten, nicht den Index.** Steht die ID im Index, erst `df.reset_index()`, dann `melt`. Merksatz: index-basierte Operationen (`join`, Ausrichtung) wollen den Index; spalten-basierte (`melt`, `groupby` per Name) wollen Spalten. `reset_index`/`set_index` ist der Umschalter.

### `pivot` – long → wide (die verlustfreie Umkehrung von `melt`)

```python
wide = long.pivot(index="MouseID", columns="Protein", values="Wert")
# index   = was wird zur ZEILE
# columns = was wird zur SPALTE (die Werte dieser Spalte werden Spaltenköpfe)
# values  = was steht in den ZELLEN
```

> ⚠️ **`pivot` verlangt eindeutige index/columns-Kombinationen.** Kommt eine doppelt vor → `ValueError: Index contains duplicate entries`. `pivot` behält außerdem nur, was in `index`/`columns`/`values` genannt wird – Rest fällt weg (ggf. mit in den `index` nehmen).

### `pivot_table` – umformen UND aggregieren

```python
long.pivot_table(index="Genotype", columns="Protein", values="Wert", aggfunc="mean")
```

> 💡 **`pivot` vs. `pivot_table`:** `pivot` formt nur um (Reißverschluss, verlustfrei). `pivot_table` fasst mehrere Werte pro Zelle zusammen – nötig, sobald der Index gröber ist als die Daten eindeutig erlauben. Die Aggregation ist eine inhaltliche Entscheidung, kein technischer Umweg.

### MultiIndex – Index mit mehreren Ebenen

Entsteht bei `groupby` mit mehreren Schlüsseln, `agg`, und `pivot` mit mehreren `index`-Spalten. Label pro Zeile ist ein **Tupel**.

```python
s.index.names       # ['Protein', 'Genotype']
s.index.nlevels     # 2
s.loc["DYRK1A_N"]              # ganze erste Ebene → Rest als Series
s.loc[("DYRK1A_N", "Control")] # eine Zelle → Tupel-Zugriff
s.xs("Control", level="Genotype")   # Querschnitt durch eine Ebene
```

**`stack`/`unstack` = `melt`/`pivot` für Index-Ebenen:**

```python
s.unstack("Genotype")   # Index-Ebene → Spalten (macht groupby-Ergebnis lesbar)
s.stack()               # Umkehrung: Spalten → Index-Ebene
```

|Richtung|Auf Spalten|Auf Index-Ebenen|
|---|---|---|
|breiter|`pivot`|`unstack`|
|länger|`melt`|`stack`|

> 💡 **`reset_index()` ist der pragmatische Ausweg.** MultiIndex ist gut zum _Anschauen_, oft unhandlich zum _Weiterrechnen_. `groupby(...).mean().reset_index()` bringt dich zurück zu normalen Spalten, wo `merge`, Filter und Plotting einfach funktionieren.

---

## 🤖 Brücke ML: DataFrame → Feature-Matrix

Ein ML-Modell will eine **rein numerische Matrix** `X` + Zielvektor `y` – keine Strings, keine benannten Spalten. Drei Schritte: Encoding, Skalierung, Split.

### Encoding: Kategorien → Zahlen

```python
pd.get_dummies(df, columns=["Genotype","Treatment"], drop_first=True)
```

`get_dummies` **ersetzt** jede kategoriale Spalte durch 0/1-Spalten (eine pro Ausprägung). Bei $k$ Ausprägungen und `drop_first=True` → $k-1$ Spalten.

> ⚠️ **Label-Encoding (`cat.codes`) erfindet eine Ordnung** (`A=0,B=1,C=2` behauptet C ist „doppelt so weit" wie B). Nur bei _echt geordneten_ Kategorien (Dosisstufen) korrekt. Sonst **One-Hot** nehmen.

> 💡 **`drop_first=True` vermeidet die Dummy-Falle:** Alle $k$ Spalten sind linear abhängig (sie summieren sich zu 1), das macht $X^TX$ **singulär** (nicht invertierbar) → Normalengleichung unlösbar. Eine Spalte weglassen behebt es; die weggelassene Kategorie wird zur Referenz. (Für Bäume egal, für lineare/Bayes-Modelle wichtig.)

### Skalierung (z-Score)

$$z = \frac{x - \mu}{\sigma} \quad\Rightarrow\quad \text{danach } \mu=0,\ \sigma=1$$

```python
z = (df[proteine] - df[proteine].mean()) / df[proteine].std()
z.mean()   # ~1e-16 (Fließkomma-Rauschen, praktisch 0)
z.std()    # 1.0
```

> 💡 $\mu=0, \sigma=1$ ist das **Ergebnis**, nicht die Eingabe: geteilt wird durch die _echte_ Streuung der Rohdaten. Jede Spalte durch ihre eigene → danach alle gleich skaliert und vergleichbar.

> 💡 **z-Score (linear, Form bleibt) vs. Log (nichtlinear, entschieft).** Log für rechtsschiefe Größen über mehrere Dekaden ($k_{cat}$, $K_M$, Konzentrationen): `np.log10(x)`, nur für positive Werte (`np.log1p` bei Nullen). Typische Reihenfolge: erst log (Form), dann z-Score (Skala).

> ⚠️ **Leakage:** $\mu, \sigma$ (und Füll-/Imputationswerte!) **nur aus dem Trainingsset** berechnen und auf beide Sets anwenden. Sonst sieht das Modell heimlich Testdaten-Statistik → geschöntes, unehrliches Ergebnis. $\mu,\sigma$ sind die suffizienten Statistiken der Normalverteilung – ihr Leck überträgt _alle_ parameterrelevante Info.

### Train/Test-Split auf Gruppenebene

Bei genesteten Daten (15 Messungen pro Maus) darf **keine Maus in beiden Sets** landen – sonst Leakage.

```python
maus_pro_zeile = df.index.str.split("_").str[0]   # "309_1" → "309"
unique_mäuse   = maus_pro_zeile.unique()          # aus den 72, NICHT den 1080 Zeilen ziehen!

rng = np.random.default_rng(42)                   # fester Seed → reproduzierbar
test_mäuse = rng.choice(unique_mäuse, size=int(0.2*len(unique_mäuse)), replace=False)

maske_test = maus_pro_zeile.isin(test_mäuse)
test  = df[maske_test]        # ECHTER DataFrame, nicht die Nummern-Liste
train = df[~maske_test]
```

> ⚠️ **`rng.choice` aus den `unique()`-Mäusen**, nicht aus der wiederholten Zeilen-Liste – sonst kann `replace=False` dieselbe Maus doch mehrfach ziehen (sie steht 15× drin).

> ⚠️ **Split-Prüfung: worüber iteriert `set()`?** `set(dataframe)` iteriert über **Spaltennamen**, nicht Zeilen → `set(train) & set(test)` prüft Spalten-, nicht Maus-Überlapp (falscher Alarm). Korrekt über die Nummern:
> 
> ```python
> set(train.index.str.split("_").str[0]) & set(test.index.str.split("_").str[0])   # soll set() sein
> ```

### num + kategorial getrennt behandeln, dann zusammenfügen

Das Muster, um einen ganzen DataFrame in einem Rutsch modellfertig zu machen:

```python
num = df.select_dtypes(include="number")       # Spalten nach Typ greifen
cat = df.select_dtypes(include="category")
num_z   = (num - num.mean()) / num.std()        # numerisch: skalieren
dummies = pd.get_dummies(cat, drop_first=True)  # kategorial: encoden
fertig  = pd.concat([num_z, dummies], axis=1)   # nebeneinander über Index
```

Leakage-freie Skalierungsfunktion (splitten → train-$\mu,\sigma$ → beide skalieren):

```python
def standardisiere_leakagefrei(train, test):
    num_cols = train.select_dtypes(include="number").columns
    mu, sigma = train[num_cols].mean(), train[num_cols].std()   # NUR train
    train, test = train.copy(), test.copy()
    train[num_cols] = (train[num_cols] - mu) / sigma
    test[num_cols]  = (test[num_cols]  - mu) / sigma            # dieselben mu, sigma
    return train, test
```

> 💡 **Beweis der Leakage-Freiheit:** Danach ist das **train**-Mittel ~0 (per Konstruktion), das **test**-Mittel _nicht_ exakt 0 – es wurde ja mit fremden Werten skaliert. Wäre es exakt 0, hättest du geleakt.

---

## 🧩 Brücke PyMC: `category` → `coords`/`dims`

Der Schlussstein: `coords`/`dims` geben Modell-Achsen **Namen statt Nummern** – dieselbe Idee wie der Pandas-Index, eine Ebene höher. Eine `category`-Spalte liefert die Bausteine direkt.

**Was die category-Spalte liefert:**

||Länge|Inhalt|Rolle im Modell|
|---|---|---|---|
|`.cat.categories`|Anzahl **Kategorien**|die **Namen** (Legende)|→ **`coords`**|
|`.cat.codes`|Anzahl **Zeilen**|**Zahlen**, die in categories zeigen|→ **Index** `mu[idx]`|

`dims` dagegen ist bloß der **Name der Achse**, den du selbst vergibst (`"genotype"`) – kommt _nicht_ aus der Spalte.

```python
# categories[codes] baut die Originalstrings zurück (Fancy Indexing):
s.cat.categories[s.cat.codes]     # ['Control','Ts65Dn','Control',...]
```

**Im Modell:**

```python
import pymc as pm

coords = {"genotype": df["Genotype"].cat.categories}   # Namen → coords
idx    = df["Genotype"].cat.codes.values                # Verweise → Index

with pm.Model(coords=coords) as modell:
    mu    = pm.Normal("mu", 0.5, 0.5, dims="genotype")  # benannte Achse statt shape=2
    sigma = pm.HalfNormal("sigma", 0.5)
    y = pm.Normal("y", mu=mu[idx],                       # codes greifen die PARAMETER
                  sigma=sigma, observed=df["Protein"].values)

with modell:
    idata = pm.sample(1000, tune=1000, chains=4)         # bei Container-Fehler: cores=1
```

**Die Auszahlung – das Posterior trägt die Namen mit:**

```python
idata.posterior["mu"]                        # Dimension "genotype": ['Control','Ts65Dn']
idata.posterior["mu"].sel(genotype="Control")  # Zugriff per LABEL = loc, nicht iloc!
```

> 💡 `mu[idx]` ist dasselbe `categories[codes]`-Fancy-Indexing – nur greifen die Codes die **Parameter** statt der Namen. Für 1080 Zeilen entsteht ein 1080-Vektor, jede Zeile trägt den `mu` ihrer Gruppe.

> 💡 **No-Pooling-Bayes ≈ `groupby().mean()`.** Ein Modell mit einem freien `mu` pro Gruppe und vagen Priors liefert praktisch die Gruppenmittelwerte zurück – aber als **Verteilung mit Unsicherheit** statt als Punktschätzer. `groupby` ist die frequentistische Vorstufe.

> ⚠️ **ArviZ-1.x-Drift** (in der `lern`-Umgebung relevant): `az.summary(..., hdi_prob=)` → `ci_prob=`; Intervallspalten heißen `eti89_lb/ub`. Für Plots **nie** `az.plot_*`, sondern Matplotlib direkt aus `idata.posterior[...].sel(...).values.flatten()`. Kein `matplotlib.use("Agg")` auf dem Mac (unterdrückt die Anzeige).

---

## 🔁 Iteration (selten nötig!)

```python
for index, zeile in df.iterrows():
    print(zeile["Tag"], zeile["Temperatur"])
```

> ⚠️ Iteration ist **langsam**. Versuche zuerst, das Problem mit Vektoroperationen, Masken oder `.apply()` zu lösen. Iteration ist die letzte Wahl.

---

## ⚖️ Pandas vs. Liste – das Vergleichs-Pattern

Was du in **purem Python** schreiben würdest vs. in Pandas:

|Was|Pure Python|Pandas|
|---|---|---|
|Mittelwert|`sum(werte) / len(werte)`|`serie.mean()`|
|Filtern|`[w for w in werte if w > 22]`|`serie[serie > 22]`|
|Doppeln|`[w * 2 for w in werte]`|`serie * 2`|
|Statistik|mehrere Schleifen|`serie.describe()`|
|Spalten kombinieren|doppelte Schleife|`df["A"] + df["B"]`|

---

## 🔌 Bezug zum FuE-Projekt

|Pandas-Konzept|Entspricht in PyTorch|
|---|---|
|`DataFrame`|2D-Tensor|
|`Series`|1D-Tensor|
|`serie.mean()`|`tensor.mean()`|
|`serie * 2`|`tensor * 2`|
|`df[maske]`|`tensor[maske]`|
|`df.values`|`→ np.array → torch.tensor`|

Die APIs sind absichtlich ähnlich – wer Pandas verinnerlicht, ist auf PyTorch vorbereitet.

> 🔗 **Index → `coords`/`dims`:** Der benannte Index ist genau der Gedanke, der in PyMC als `coords`/`dims` wiederkommt. Statt `shape=(72,)` schreibst du `dims="maus"`, und die Posterior-Samples tragen die Mausnamen mit. Labels statt Positionen – eine Ebene höher.

---

## ⚠️ Häufige Stolperfallen

|Problem|Ursache|Lösung|
|---|---|---|
|`df["x"]` vs. `df[["x"]]`|Einfache Klammer = Series, doppelte = DataFrame|Sich des Unterschieds bewusst sein|
|`df[df["x"] > 5 and df["y"] < 3]` crasht|Pandas mag kein `and`/`or`|`&` und `\|` mit Klammern verwenden|
|`series.iloc(6)` → `No axis named 6`|Runde statt eckige Klammern|`series.iloc[6]` – Indexer nutzen `[...]`|
|`print(df.groupby(...)["x"]).mean()` → `'NoneType' has no attribute 'mean'`|Schließende `print`-Klammer zu früh – `.mean()` läuft auf dem Rückgabewert von `print` (= `None`)|Erst die Kette fertig rechnen, dann in `print()` hineinreichen|
|`.head(10)` zeigt nicht die größten Werte|`head` schneidet **blind von oben** ab|**Erst sortieren, dann köpfen:** `.sort_values(ascending=False).head(10)`|
|„wie viele erfüllen X?" liefert Kommazahl|`werte[maske].sum()` summiert Werte statt zu zählen|`maske.sum()` – die Maske selbst zählen|
|Zeilenzahl nach `merge` gestiegen|Schlüssel nicht eindeutig → Vervielfachung|`validate="one_to_one"`, Schlüssel prüfen|
|`groupby`-Ergebnis erscheint nicht im Notebook|Nur der **letzte** Ausdruck einer Zelle wird automatisch angezeigt|`print()` um jedes Zwischenergebnis|
|`The truth value of a Series is ambiguous`|`and`/`or` auf einer Boolean-Series|`&`/`\|` mit Klammern um jede Teilbedingung|
|Nach Sortieren „falsche" Zeile|`loc`/`iloc` verwechselt|`loc` = Label, `iloc` = Position|
|`a + b` liefert überall NaN|Indizes richten sich aus, Labels passen nicht|Index prüfen (`.index`), ggf. `.reset_index(drop=True)`|
|Excel hat eine Index-Spalte links|`index=False` vergessen|`df.to_excel("x.xlsx", index=False)`|
|`KeyError: "spalte"`|Spaltenname falsch geschrieben|`df.columns` prüfen, Groß-/Kleinschreibung|
|Datei bleibt von Excel gesperrt|Datei noch in Excel geöffnet|Excel schließen, neu schreiben|
|`df` ändert sich nicht trotz Aufruf|Methode gibt **neuen** DataFrame zurück|`df = df.dropna()` (Zuweisung!)|

---

## ✅ Mini-Cheatsheet

```python
import pandas as pd

# Erstellen
df = pd.DataFrame({"Tag": ["Mo", "Di"], "Temp": [21.5, 22.0]})

# Einlesen
df = pd.read_excel("daten.xlsx")
df = pd.read_csv("daten.csv")

# Index setzen (erst zusammenfügen, DANN set_index!)
df = df.set_index("MouseID")

# Inspizieren
df.head()
df.describe()
df.shape
df.dtypes.value_counts()
df["x"].isna().sum()

# Auswählen
df["Temp"]              # Series
df[["Tag", "Temp"]]     # DataFrame
df.iloc[0]              # Zeile per Position (Ende exklusiv)
df.loc["Mo"]            # Zeile per Label (Slice-Ende inklusive)
df.loc[df["Temp"] > 22] # Zeilen per Bedingung

# Statistik
df["Temp"].mean()
df.describe()

# Neue Spalte
df["Temp_F"] = df["Temp"] * 9/5 + 32

# Filter
maske = df["Temp"] > 22
maske.sum()             # wie viele
df_warm = df[maske]
df[(df["A"] > 0) & (df["B"] == "x")]    # kombiniert: & und Klammern

# NaN kartieren
df.isna().mean().sort_values(ascending=False).head(10)
(df.isna().mean() > 0.2).sum()
df.dropna(subset=["Temp"])              # gezielt, nicht blank

# Gruppieren
df.groupby(["A","B"], observed=False).size()
df.groupby("A")["Temp"].agg(["mean","std","count"])
df["zentriert"] = df["Temp"] - df.groupby("A")["Temp"].transform("mean")

# Zusammenführen
erg = pd.merge(links, rechts, on="ID", how="left", validate="one_to_one")
print(len(links), len(rechts), len(erg))    # immer prüfen!

# Speichern
df.to_excel("ergebnis.xlsx", index=False)
```

# Referenzen