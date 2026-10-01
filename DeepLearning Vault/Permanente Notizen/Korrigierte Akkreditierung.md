### Vorschlag zur anstehenden Akkreditierung des Bachelorstudiengangs sowie des Lehrplans Bachelor und Master BT/BPT

von Paul Medebach

---

# Einführung: Studiengänge modernisieren

Wie man sich vorstellen kann, ist die Akkreditierung eines ganzen Studiengangs keine triviale Sache, noch eine willkürliche. Es gilt, sich an den Herausforderungen der Gegenwart und der Zukunft anzupassen – seien es didaktische Unsicherheiten durch Künstliche Intelligenz, die breiter werdende Wissenslücke zwischen schulischem und vorausgesetztem Wissen von Studierenden oder die politisch induzierte Streichung von Hochschulgeldern. Eine Akkreditierung ist jedoch nicht nur Anpassung, sondern auch ein Bruch im Kontinuum, um Chancen wahrzunehmen. Sie ist einer der wenigen Momente, den Kairos am Schopfe zu packen und die Studierenden von heute zu den Studierenden von morgen auszubilden. Daher möchte ich hier einen Aspekt dieser Chance vorstellen. Einen Vorschlag unterbreiten. Es ist nicht weniger als das Verhältnis zwischen Wissenschaft, Statistik und kausaler Logik.

Es geht um die Fragen:

- Welches Werkzeug ist so spezifisch wie möglich, so universell wie nötig für wissenschaftliches Arbeiten und wächst organisch mit den Lerninhalten und den Studierenden mit?
- Mit welchem Werkzeug wird _Verstehen_ gelehrt, anstatt Verständnisbarrieren vorauszusetzen?
- Durch welches Werkzeug werden Studierende für ihr Studium und für zukünftige Tätigkeiten gewappnet?

Die Antwort ist einfach – und kostengünstig. Python, die vielseitige Programmiersprache, von der man das eine oder andere Mal gehört hat, kann dieses Werkzeug sein. Im Folgenden erläutere ich eine Integration von Python-basiertem Programmieren in den Studiengang, die Vor- und Nachteile einer solchen Integration und warum sie den oben beschriebenen Anforderungen mit Zuversicht entgegentritt.

---

# Chancen

Excel ist momentan das Werkzeug, der tabellarische Modus für Datenanalyse, deskriptive und induktive Statistik, Mathematik und Datenvisualisierung. Rudimentär wird in einigen Mastermodulen Matlab/SciLab für solche Aufgaben verwendet (Numerik, Upstream Processing). Dabei ist Excel – wie jeder weiß – weder ein wissenschaftliches Programm, noch ist es in seiner Anwendung verträglich mit naturwissenschaftlichen Anforderungen (Wurzeln in Finanz- und Geschäftsdatenanalyse).

Wichtige Einschränkungen von MS Excel:

- Datenanalysen werden bei großen `.xlsx`-Dateien für Dritte nicht mehr nachvollziehbar, bedingt durch die verzweigte Verweisstruktur.
- Unausgereifte Cloud-Synchronisation lässt bei Gruppenarbeiten `.xlsx`-Dateien abstürzen.
- Automatische Datenkorruption (siehe Gennamen-Skandal: Excel wandelte Gensymbole wie _SEPT2_ in Datumsangaben um) [1](https://doi.org/10.1186/s13059-016-1044-7).
- Stille Fehler: Ein abgeschnittener Zellbereich oder eine nicht mitgezogene Formel produziert ein falsches Ergebnis _ohne Warnung_. Code hingegen ist _auditierbar und reproduzierbar_ – der Fehler ist im Skript auffindbar und nachvollziehbar, während er in der Excel-Verweisstruktur verborgen bleibt.
- Kein Ökosystem für moderne Statistik: keine gemischten Modelle, keine kausale Inferenz, keine ordentliche Survival-Analyse, keine robuste Behandlung von Multiplizität.
- Keine ernsthafte numerische Mathematik: lineare Algebra jenseits trivialer Matrizen, numerische Optimierung, ODE-Löser, FFT, symbolische Mathematik – nicht vorhanden oder nur als fragile Krücke.
- Simulation skaliert nicht: Monte-Carlo, Prozesssimulation, Molekulardynamik, FEM – Excel ist weder ausdrucksstark noch performant genug. Schon mittelgroße Datensätze (> 1 Mio. Zeilen) sprengen das Format.

**Der wichtigste Punkt**: Studierende lernen ein proprietäres Klick-Werkzeug statt der konzeptuellen und algorithmischen Grundlagen, die zwischen Werkzeugen übertragbar sind.

Ja, Excel ist eine anfängerfreundliche Software. Was jedoch wie Zugänglichkeit aussieht, ist in Wahrheit verschobener Aufwand. Und ja, für höhere Statistik und Simulationen gibt es Software (COMSOL, GraphPad Prism etc.) – hinter Lizenzkosten. Python und alle wichtigen Bibliotheken sind hingegen Open Source. Nicht ohne Grund wird Python als anfängerfreundliche Programmiersprache gesehen. Um kurz aus meinem persönlichen Bezug zu sprechen: Mir ging es nicht anders. Im Rahmen meines Forschungs- und Entwicklungsprojekts (FuE) am Institut für Biochemische Verfahren und Analysen (IBVA) zur Erforschung von Deep-Learning-Algorithmen für die Vorhersage kinetischer Enzymparameter aus Aminosäuresequenzen musste ich zuletzt Python von Grund auf lernen. Was anfänglich komplex und ungewohnt daherkam, wurde über einen Monat hinweg vertraut und anwendbar. Aufgaben, die in Excel mehrere Klicks und Formatierungsschritte gebraucht haben, waren in Python wenige Zeilen Code. Wenn man Python als universellen Baukasten für spezifische Anwendungen sieht, ergeben sich einige Vorteile gegenüber Excel:

- Schneller Aufbau operativer Komplexität mit wenig Aufwand.
- Schlanke Programmiersprache: Bibliotheken bieten "Bausteine" für spezifische Anwendungen.
- Mehrere Anwendungen über eine Software: Datenverwaltung und Auswertung sind in einem einzigen Python-Notebook möglich.
- Bibliotheken decken Excel-Anwendungen vollständig ab und erweitern die Anwendung auf fortgeschrittene Statistik, Computersimulationen via Differenzialgleichungssysteme, Datenanalyse, generative Modellierung u. v. m.
- Bietet ein breites Spektrum an wissenschaftlichen Graphen.
- Datenexport in `.docx`, `.xlsx`, `.csv`, `.png`, `.jpeg`, `.pdf` u. v. m. möglich.
- Hält Schritt bei Datensätzen mit n > 1 Mio.
- Einlesen vieler Dateitypen (`.md`, `.txt`, `.csv`, `.xlsx` u. v. m.).

Der wichtigste Punkt: Coding _zwingt_ den/die Anwender:in, explizit in Kausalketten zu denken, anstatt es nur zu erlauben. Excel ist eine Point-and-Click-Anwendung, Coding hingegen zwingt – in gewissem Rahmen – dazu, die Anwendung zu verstehen: Jeder Schritt von den Rohdaten bis zum Ergebnis muss explizit und in korrekter Reihenfolge formuliert sein. Es gibt also etwas für die Studierenden persönlich zu gewinnen. Klar: Coding ist nichts für Unkonzentrierte – doch sollte eine Auswertung per Tabellenkalkulation kopflos erfolgen? Oder wäre der Studiengang nicht generell die falsche Wahl, wenn Berührungsängste mit Mathematik und Logik vorliegen?

Wie bereits erwähnt, ist Python eine schlanke Programmiersprache. Die Grundlagen sind in wenigen Tagen gelernt, die Logik von Schleifen und If-else-Verzweigungen schnell durchdrungen. Was das Medium so mächtig macht, ist die Auswahl an Bibliotheken:

- **NumPy** [2](https://numpy.org/): Array-Grundlage; Vektorisierung, lineare Algebra.
- **pandas** [3](https://pandas.pydata.org/): tabellarische Daten (DataFrames); der direkte, mächtigere Excel-Ersatz für Einlesen, Filtern, Gruppieren, Mergen und Export (`.csv`, `.xlsx`).
- **SciPy** [4](https://scipy.org/): numerische Mathematik: Optimierung, Integration, ODE-Löser, Signalverarbeitung, statistische Funktionen.
- **statistics/statsmodels** [5](https://www.statsmodels.org/): klassische/frequentistische Statistik: Regression, GLMs, ANOVA, Zeitreihen, mit detailreichen Modell-Zusammenfassungen.
- **Matplotlib** [6](https://matplotlib.org/): Basis-Plotting, volle Kontrolle, Standard für Publikationsgrafiken.
- **reportlab** [7](https://www.reportlab.com/): Erstellen von PDFs, Einbindung von Ergebnissen und Grafiken.
- **openpyxl** [8](https://openpyxl.readthedocs.io/): Erstellen von Excel-Exporten.
- **Jupyter** [9](https://jupyter.org/): interaktive Notebooks; verbinden Code, Ergebnis, Grafik und Erklärung in einem reproduzierbaren oder gar automatisierbaren Dokument.
- **PyMC** [10](https://www.pymc.io/): Bayes'sche Modellierung / probabilistisches Programmieren (u. a. via Markov-Chain-Monte-Carlo – MCMC).
- **ArviZ** [11](https://www.arviz.org/): Diagnostik und Visualisierung für Bayes'sche Modelle (Posterior, Trace-Plots, ROPE – Region of Practical Equivalence); Begleiter zu PyMC.
- **Biopython** [12](https://biopython.org/): das Allround-Werkzeug für Biodatenanalyse: Sequenzen (FASTA/GenBank), BLAST, Alignment, Zugriff auf NCBI-Datenbanken, Strukturparsing.
- **OpenMM** [13](https://openmm.org/): Molekülmechanik-Simulationen (GPU-beschleunigt).

Die Liste ist erschlagend, könnte jedoch noch weitergehen. All diese Bibliotheken sind validiert und erfreuen sich bei Anwendenden in der Wissenschaft durch ihre einfache Anwendung großer Beliebtheit. Eine Integration in den Bachelorstudiengang heißt nicht, all diese Bibliotheken zu lehren. Das wäre auch nicht zielführend. Es sollen in verschiedenen Semestern unter verschiedenen Anwendungen pragmatische Lösungen aufgezeigt werden – und das alles unter dem Dach **einer** Software. Coding wird oft missverstanden als "Software von Grund auf aufbauen". Bei Python mit seinem Bausteinprinzip ist das nicht so. Glauben Sie jemandem, der Anfang April 2026 noch keine Erfahrung mit Coding gemacht hatte. Glauben Sie mir. Es ist lernbar, schnell, mit steiler Lernkurve. Wie man Coding in den Bachelorstudiengang BT/BPT einbetten könnte, zeige ich im Folgenden.

---

# Empfehlung: Konkrete Integration in den Bachelorstudiengang BT/BPT

Als ich mit BT/BPT angefangen habe, gab es im Modul "Einführung in das Studium und Berufsfeld BT/BPT" neben Ringvorlesungen einen Excel-Kurs mit anschließender Prüfung. Im zweiten Semester folgte mit "Auswertung wissenschaftlicher Daten" eine Einführung in die Aufbereitung von Daten und die Darstellung von Ergebnissen – zugegebenermaßen eher rudimentär als tiefgreifend. Es gibt viele Punkte, an denen eine Integration von Coding Sinn macht. Ein konkreter Vorschlag wäre dieser:

|Semester|Modul|Integration|
|---|---|---|
|1.|Einführung in das Studium/Berufsfeld|Vermittlung von Excel-Grundkenntnissen + Python-Grundkurs: Datentypen, Objekte, Schleifen, Blöcke, Funktionen, Versionierung und Anwendung in Notebooks (siehe Jupyter Notebook)|
|1.|Biologie|Python-Grundlagen für Statistik|
|1.|Physik I|Python-Grundlagen für lineare Algebra mit NumPy und Matplotlib|
|2.|Statistik/Auswertung Wissenschaftlicher Daten (AWD)|Erweiterte Statistik mit statistics/statsmodels + Matplotlib|
|2.|Physik II|SciPy-Grundlagen zur numerischen Problemlösung + NumPy-Vertiefung|
|4.|Circuit, Signals and Systems|Systemsimulationen in Python mittels SciPy|
|4.|Bioanalytik|Datenanalyse und Auswertung via pandas + statsmodels|
|5.|Bioverfahrenstechnik Upstream|Vertiefung Simulation mittels SciPy: Prozesssimulation + Fitting|

Für die beiden Mathematikmodule soll der Fokus auf der Theorie bleiben. Die Physikmodule und die Biologie gelten daher als praktischer Einstieg in Python. In AWD soll das Hauptaugenmerk auf frequentistischer Statistik (Verteilungen, p-Tests, Konfidenzintervalle usw.) und der Darstellung von Ergebnissen als Graphen liegen. Bei Modulen, die nicht explizit in der Liste vorkommen, sollen Auswertungen im Rahmen der Praktika ebenfalls mittels Python-Toolkit erfolgen. Zudem können Zusammenhänge wie Bilanzgleichungen idiomatisch in mathematischen Gleichungen sowie in Python-Code in Vorlesungen dargestellt werden. Das fördert Transferwissen und festigt bereits Gelerntes. Man sieht: Der Werkzeugkasten wächst im Rahmen des Studiums mit. Studierende können Vorwissen aus anderen Modulen in ihrem Code nutzen, um Problemstellungen zu verstehen und zu lösen. Programmier"_sprachen_" heißen nicht nur so, weil sie die Möglichkeit aufzeigen, Computern Anweisungen zu geben, sondern weil es sich um eine Sprache handelt. Etwas, was sich ab einem gewissen Grad natürlich anfühlt.

---

# Anpassung im Masterstudiengang BT/BPT

Auch wenn es hier um die Akkreditierung des Bachelors geht, sollte darüber hinaus über die Anpassung bestimmter Module im Master nachgedacht werden, um das Medium Python in seinen Möglichkeiten voll auszureizen und den Studierenden eine Art der Statistik näherzubringen, die lange Zeit als unkonventionell und unpraktisch galt und heute ihr Revival erlebt: nämlich die **Bayes'sche Statistik**. Die statistischen Modelle, die auf dem bekannten Satz von Bayes beruhen, können die Antworten erbringen, die viele fälschlicherweise in Signifikanztests und Konfidenzintervalle hineininterpretieren: "Mit der Wahrscheinlichkeit $p$ (z. B. $95,%$), gegeben die Daten $D$, gilt meine Hypothese $H$." Während die Bayes'sche Statistik einen Wahrscheinlichkeitsbegriff für einen Estimand liefert, definiert die frequentistische Statistik Wahrscheinlichkeit als Häufigkeit: "Wie häufig komme ich auf dieselben Ergebnisse, wenn ich den Versuch unendlich oft wiederhole?" Das ist das Kernparadigma. Dass eine Kernaussage über eine Hypothese damit verboten ist, ist vielen Wissenschaffenden nicht klar. Die Bayes'sche Statistik hingegen gibt eine ganze Verteilung über Hypothesen zurück und zeigt mittels _A-priori_-Annahmen, welche Hypothese wahrscheinlicher ist als andere. Durch ihren Rechenaufwand (unlösbare Integrale der Evidenz im Satz von Bayes) wurde der bayes'sche Ansatz Mitte des 20. Jahrhunderts durch den frequentistischen abgelöst. Computergestützte Bayes-Statistik hebt diesen Nachteil auf. Es gibt also keine Entschuldigung, sie nicht zu lehren.

Der Wechsel des statistischen Paradigmas ist dabei keine akademische Spielerei, sondern eine breit getragene methodische Debatte. So riefen 2019 über 800 Wissenschaftlerinnen und Wissenschaftler in einem vielbeachteten Kommentar in _Nature_ dazu auf, das Konzept der "statistischen Signifikanz" aufzugeben, da der ritualisierte Umgang mit p-Werten systematisch zu Fehlschlüssen führt [14](https://doi.org/10.1038/d41586-019-00857-9). Eng damit verbunden ist die Einsicht, dass die klassische Trennung in "subjektive" und "objektive" Statistik in die Irre führt und durch transparentere methodische Tugenden ersetzt werden sollte [15](https://doi.org/10.1111/rssa.12276). Beide Arbeiten unterstreichen, warum ein bewusster, modellierender Umgang mit Unsicherheit – wie ihn die Bayes'sche Statistik einfordert – in die Ausbildung gehört.

Wie bereits zu Anfang erwähnt, kann die Bayes-Statistik durch die Python-Bibliothek **PyMC** [10](https://www.pymc.io/) realisiert werden. Studierende können hier lernen, wie kausale Strukturen mittels DAG (Directed Acyclic Graphs) gebildet werden, wie man generative Modelle baut und validiert und wie Zielgrößen **vor** einer Untersuchung definiert werden. (Anzumerken ist, dass DAGs und kausale Inferenz nach Pearl strenggenommen unabhängig vom bayes'schen Inferenzparadigma sind; McElreath verbindet beide Stränge jedoch didaktisch zu einem kohärenten Arbeitsablauf.) Es geht um Verständnis: nicht weniger als die Statistik zu verstehen, die man selbst für eine bestimmte wissenschaftliche Fragestellung zuschneidet und in Code übersetzt. Damit Wissenschaft eben nicht mehr Zweck für schöne Statistik ist – für Stopp-Kriterien, um aus einem p-Wert noch die Signifikanz herauszupressen –, sondern andersherum.

Die Bayes-Statistik via PyMC sollte nach meiner Empfehlung in das Numerik/Statistik-Modul integriert werden. In Numerik werden numerische Berechnungen und Methoden vorgestellt. Es werden per Hand Matrizen zerlegt und berechnet, Algorithmen zu Regressionsmodellen durchgerechnet sowie optionale Coding-Aufgaben in Matlab gestellt, die eben nur _optional_ sind. Das ist ganz nett, hat aber keinen praktischen Nutzen im weiteren Verlauf des Studiums. Ein ehrlicherer Umgang wäre daher ein pragmatischer: Im neuen Numerik/Statistik-Modul wird numerische Theorie direkt mit der Anwendung in Python (PyMC) verbunden, anstelle des optionalen Matlab, wie es momentan der Fall ist. Im Statistikteil soll nicht mehr der Stoff des ersten Semesters durchgekaut, sondern die Theorie zur Bayes-Statistik gelehrt werden. Für die Prüfungsleistung empfehle ich eine 45-minütige Klausur zur allgemeinen Theorie plus eine Hausarbeit in Gruppenarbeit, in der jede Gruppe einen Datensatz mittels bayes'schen Arbeitsablaufs durcharbeitet:

1. Wissenschaftliche Fragestellung definieren.
2. Generatives Modell/DAG erstellen und validieren (Prior Predictive Check).
3. Statistisches Modell erstellen, Daten auswerten und validieren (Posterior Predictive Check).
4. Aus Posterior-Verteilung und Annahmen kausale Antworten ableiten.

Damit wäre das Modul (dann Numerik/Bayes-Statistik) anwendungsbezogener und bereitet die Studierenden auf die Statistik von morgen vor. Zudem: Wer sich mit bayes'scher Statistik auskennt, hat es nicht mehr weit zum probabilistischen maschinellen Lernen – einem Feld, das in den angewandten Naturwissenschaften zunehmend an Bedeutung gewinnt [16](https://www.bishopbook.com/). Um eine Probe der bayes'schen Statistik zu bekommen, empfehle ich die mitgeschnittene Vorlesung "[Statistical Rethinking](https://www.youtube.com/watch?v=ztbYkBPDOgU&list=PLDcUM9US4XdNOlqSyhe38US8mFgmqzI14)" von Richard McElreath, Direktor am Max-Planck-Institut für evolutionäre Anthropologie und Honorarprofessor an der Universität Leipzig [17](https://xcelab.net/rm/statistical-rethinking/).

---

# Hindernisse

Unabhängig von der Frage, welche Vorteile die Integration einer Programmiersprache wie Python in einem ingenieurtechnischen und naturwissenschaftlichen Studiengang für die Studierenden bietet, gibt es einige Hürden. Die größte betrifft das Lehrpersonal selbst. Viele der Dozierenden, Professorinnen und Professoren sowie wissenschaftlichen Mitarbeitenden haben, was Coding betrifft, rudimentäre bis keine Berührungspunkte. Das gilt jedoch nur für die Gegenwart und sicherlich nicht für alle Dozierenden: Gerade an den Schnittstellen zu anderen Fachbereichen und Studiengängen sehe ich einiges an qualifiziertem Personal, das eine solche Integration unterstützen kann. Zudem wären Weiterbildungen der einzige Kostenfaktor im Zusammenhang mit Python (da Open Source) – das heißt: Lernen für das Lehrpersonal. Ja, das ist eine Voraussetzung; wie man jedoch weiß, ist naturwissenschaftliches Arbeiten per se nicht voraussetzungsfrei: Auch ein Biologe muss die Grundlagen der Hard Sciences (Mathematik, Chemie, Physik) in einem bestimmten Umfang beherrschen. Es muss also einen Mindeststandard unter den Lehrenden geben.

Weitere Hürden:

- Studierende, die sich bewusst für BT/BPT entschieden haben, haben sich möglicherweise bewusst gegen einen informatischen Studiengang entschieden. Das kann zu frühem Frust führen. Hier sollten Studierende im ersten Semester umfänglich unterstützt werden. Auch KI kann als Tutorentool nützlich sein (ich spreche aus eigener Erfahrung).
- Installationshürde: Der Aufbau virtueller Umgebungen und die Integration in eine Entwicklungsumgebung können zeitaufwendig sein. Abhilfe könnten gerade im ersten Semester Cloud-Notebooks wie JupyterHub schaffen.
- Prüfungsformate: Klausuren passen schlecht zu Coding-Kompetenzen. Es braucht daher neue oder gemischte Formate.

---

# Ein Appell

Wäre computergestütztes wissenschaftliches Arbeiten ein Taschenrechner, wäre Excel ein Abakus. Die Microsoft-Anwendung wurde einmal aus Bequemlichkeit aus dem Finanzsektor gezerrt – in ein Feld, in dem sie, seien wir ehrlich, nichts zu suchen hat. Klar ist, dass das Tabellenprogramm vieles an Kompetenz gedämpft hat, um die Studierenden durch das Studium zu bringen. Diese Bequemlichkeit sollte man sich nicht mehr leisten dürfen, denn sie geht auf Kosten der Kompetenz derjenigen, die etwas lernen wollen. Aber man kennt es sicher aus dem naturwissenschaftlichen Umfeld: Progressive Veränderungen sind anstrengend – und zahlen sich aus. Wissenschaft basiert auf Disruption, nicht auf dem Status quo. Irgendwo muss man anfangen, irgendwann muss die Disruption kommen – warum nicht dann, wenn sie für alle am günstigsten ist?

---

# Quellenverzeichnis

[1] Ziemann, M., Eren, Y., & El-Osta, A. (2016). Gene name errors are widespread in the scientific literature. _Genome Biology, 17_(1), 177. https://doi.org/10.1186/s13059-016-1044-7

[2] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. _Nature, 585_(7825), 357–362. https://numpy.org

[3] The pandas development team. (2024). _pandas: Powerful Python data analysis toolkit_ [Computer software]. https://pandas.pydata.org

[4] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … Vázquez-Baeza, Y. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. _Nature Methods, 17_(3), 261–272. https://scipy.org

[5] Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. In _Proceedings of the 9th Python in Science Conference (SciPy 2010)_ (pp. 92–96). https://www.statsmodels.org

[6] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering, 9_(3), 90–95. https://matplotlib.org

[7] ReportLab. (2024). _ReportLab PDF Toolkit_ [Computer software]. https://www.reportlab.com

[8] Gazoni, E., & Clark, C. (2024). _openpyxl: A Python library to read/write Excel 2010 xlsx/xlsm files_ [Computer software]. https://openpyxl.readthedocs.io

[9] Kluyver, T., Ragan-Kelley, B., Pérez, F., Granger, B., Bussonnier, M., Frederic, J., Kelley, K., Hamrick, J., Grout, J., Corlay, S., Ivanov, P., Avila, D., Abdalla, S., & Willing, C. (2016). Jupyter Notebooks – A publishing format for reproducible computational workflows. In F. Loizides & B. Schmidt (Eds.), _Positioning and Power in Academic Publishing: Players, Agents and Agendas_ (pp. 87–90). IOS Press. https://jupyter.org

[10] Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T., & Zinkov, R. (2023). PyMC: A modern, and comprehensive probabilistic programming framework in Python. _PeerJ Computer Science, 9_, e1516. https://www.pymc.io

[11] Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. A. (2019). ArviZ a unified library for exploratory analysis of Bayesian models in Python. _Journal of Open Source Software, 4_(33), 1143. https://www.arviz.org

[12] Cock, P. J. A., Antao, T., Chang, J. T., Chapman, B. A., Cox, C. J., Dalke, A., Friedberg, I., Hamelryck, T., Kauff, F., Wilczynski, B., & de Hoon, M. J. L. (2009). Biopython: Freely available Python tools for computational molecular biology and bioinformatics. _Bioinformatics, 25_(11), 1422–1423. https://biopython.org

[13] Eastman, P., Swails, J., Chodera, J. D., McGibbon, R. T., Zhao, Y., Beauchamp, K. A., Wang, L.-P., Simmonett, A. C., Harrigan, M. P., Stern, C. D., Wiewiora, R. P., Brooks, B. R., & Pande, V. S. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. _PLOS Computational Biology, 13_(7), e1005659. https://openmm.org

[14] Amrhein, V., Greenland, S., & McShane, B. (2019). Scientists rise up against statistical significance. _Nature, 567_(7748), 305–307. https://doi.org/10.1038/d41586-019-00857-9

[15] Gelman, A., & Hennig, C. (2017). Beyond subjective and objective in statistics. _Journal of the Royal Statistical Society: Series A (Statistics in Society), 180_(4), 967–1033. https://doi.org/10.1111/rssa.12276

[16] Bishop, C. M., & Bishop, H. (2024). _Deep learning: Foundations and concepts_. Springer. https://www.bishopbook.com

[17] McElreath, R. (2020). _Statistical rethinking: A Bayesian course with examples in R and Stan_ (2nd ed.). CRC Press. https://xcelab.net/rm/statistical-rethinking/