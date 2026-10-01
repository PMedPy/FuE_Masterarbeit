### Vorschlag zur anstehenden Akkreditierung des Bachelorstudiengangs, sowie des Lehrplans Bachelor und Master BT BPT

von Paul Medebach

---
# Einführung: Studiengänge Modernisieren

Wie man sich vorstellen kann, ist eine Akkreditierung eines ganzen Studiengangs keine triviale Sache, noch eine willkürliche. Es gilt sich an den Herausforderungen der Gegenwart und der Zukunft anzupassen; sei es didaktische Unsicherheiten durch Künstliche Intelligenz; die breiter werdende Wissenslücke zwischen schulischen und vorausgesetzten Wissen von Studierenden; oder die von Politik induzierte Streichung von Hochschulgeldern. Eine Akkreditierung ist jedoch nicht nur Anpassung, sondern auch ein Bruch im Kontinuum, um Chancen wahrzunehmen. Es ist eine der wenigen Momente den Kairos am Schopfe zu packen und den Studierenden von Heute zu den Studierenden von Morgen auszubilden. Daher möchte ich hier einen Aspekt dieser Chance vorstellen. Einen Vorschlag unterbreiten. Es ist nicht weniger als das Verhältnis zwischen Wissenschaft und Statistik und kausaler Logik.

Es geht um die Fragen:
- Welches Werkzeug ist spezifisch wie möglich, universell wie nötig für wissenschaftliches Arbeiten und wächst organisch mit den Lerninhalten und den Studierenden? 
- Mit welchen Werkzeug wird *Verstehen* gelehrt anstelle Verständnisbarrieren vorausgesetzt?
- Durch welches Werkzeug werden Studierende für ihr Studium und für zukünftige Tätigkeiten, gewappnet?

Die Antwort ist einfach - und Kostengünstig. 
Python, die vielseitige Programmiersprache, von der man das ein oder andere Mal gehört hat, kann dieses Werkzeug sein. Im folgenden erläutere ich eine Integration von Python basiertes Programmieren in den Studiengang , die die Vor- und Nachteile einer solchen Integration, und warum es die oben beschriebenen Anforderungen mit Zuversicht entgegentretet. 

---
# Chancen 
Excel ist momentan das Werkzeug, der tabellarische Modus für Datenanalyse, deskriptive und induktive Statistik, Mathematik und Datenvisualisierung. Rudimentär wird in einigen Mastermodulen Matlab/SciLab für solche Aufgaben verwendet (Numerik, Upstream Processing). Dabei ist Excel - wie jeder weiß - weder ein Wissenschaftliches Programm, noch ist es in seiner Anwendung verträglich mit naturwissenschaftlichen Anwendungen (Wurzeln in Finanz und Geschäftsdatenanalyse). 
Wichtige Einschränkung von MS Excel:
- Datenanalysen werden bei großen xlsx. für dritte nicht mehr nachvollziehbar, durch verzweigte Verweisstruktur
- Unausgereifte Cloud-Synchronisation lässt bei Gruppenarbeiten xlsx-Dateien crashen.
- Automatische Datenkorruption (Siehe Gennamen-Skandal – Excel wandelte Gensymbole wie _SEPT2_ in Datumsangaben um ##)
- Stille Fehler: Ein abgeschnittener Zellbereich oder eine nicht mitgezogene Formel produziert ein falsches Ergebnis _ohne Warnung_. Code hingegen ist _auditierbar und reproduzierbar_ – der Fehler ist im Skript auffindbar und nachvollziehbar, während er in der Excel-Verweisstruktur verborgen bleibt. 
- Kein Ökosystem für moderne Statistik: keine gemischten Modelle, keine Kausal Inferenz , keine ordentliche Survival-Analyse, keine robuste Behandlung von Multiplizität.
- Keine ernsthafte numerische Mathematik: lineare Algebra jenseits trivialer Matrizen, numerische Optimierung, ODE-Löser, FFT, symbolische Mathematik – nicht vorhanden oder nur als fragile Krücke.
- Simulation skaliert nicht: Monte-Carlo, Prozesssimulation, Molekulardynamik, FEM – Excel ist weder ausdrucksstark noch performant genug. Schon mittelgroße Datensätze (>1 Mio. Zeilen) sprengen das Format.
**Der wichtigste Punkt**: Studierende lernen ein proprietäres Klick-Werkzeug statt der konzeptuellen und algorithmischen Grundlagen, die zwischen Werkzeugen übertragbar sind.

Ja, Excel ist eine anfängerfreundliche Software. Was jedoch wie Zugänglichkeit aussieht, ist in Wahrheit verschobener Aufwand. Und ja, für höhere Statistik und Simulationen gibt es Software (ComSol, Prism Graphpad etc.) - hinter Lizenzkosten.  
Python und alle wichtigen Bibliotheken sind open source. Nicht ohne Grund wird Python als Anfängerfreundliche Programmiersprache gesehen. Um kurz in aus meinem Persönlichen Bezug zu sprechen: Mir ging es nicht anders. Im Rahmen meines Forschungs-& Entwicklungs-Projektes (FuE) am Institut für Biochemische Verfahren und Analysen (IBVA) zur Forschung an DeepLearning Algorithmen für die Vorhersage von kinetischen Enzymparametern aus Aminosäuresequenzen, musste ich zuletzt Python lernen, von Grund auf. Was anfänglich komplex und ungewohnt daher kam, wurde über einen Monat hinweg vertraut und anwendbar. Aufgaben, die in Excel mehrere Klicks und Formatierung gebraucht haben, waren in Python wenige Zeilen code. Wenn man Python als universellen Baukasten für spezifische Anwendung sieht, ergeben sich einige Vorteile gegenüber Excel:
- Schneller Aufbau von operativer Komplexität mit wenig Aufwand.
- Schlanke Programmiersprache: Bibliotheken bieten "Bausteine" für spezifische Anwendungen.
- Mehrere Anwendungen über eine Software: Datenverwaltung als auch Auswertung in einem Python-Notebook möglich.
- Bibliotheken decken Excel-Anwendungen komplett ab und erweitern die Anwendung auf erweiterte Statistik, Computersimulationen via Differenzialgleichungsysteme, Datenanalyse, generative Modellierung uvm. 
- Bietet breites Spektrum an wissenschaftlichen Graphen
- Datenexport in docx, xlsx, csv, png, jpeg, pdf uvm. möglich
- Hält Schritt bei Datensätze n > 1 Mio. 
- Einlesen von vielen Dateitypen (md, txt, csv, xlsx uvm.)
Der wichtigste Punkt: Coding *zwingt* den/die Anwender'in explizit in Kausalketten zu denken, anstatt es nur zu erlauben. Excel ist Point-and-Click-Anwendung, Coding hingegen zwingt, im gewissen Rahmen, die Anwendung zu verstehen: Jeder Schritt von Rohdaten bis zum Ergebnis muss explizit und in korrekter Reihenfolge verstanden sein. Es gibt also etwas für die Studierende persönlich zu gewinnen. Klar: Coding ist nichts für Unkonzentrierte, doch sollte eine Auswertung per Tabellenkalkulation Kopflos erfolgen? Oder wäre der Studiengang generell nicht die falsche Wahl, wenn Berührungsängste mit Mathematik und Logik vorliegen? 

Wie bereits erwähnt ist die Python eine schlanke Programmiersprache. Die Grundlagen sind in wenigen Tagen gelernt, die Logik von Schleifen und if-else-Verzweigungen schnell durchdrungen. Was das Medium so mächtig macht, ist die Auswahl an Bibliotheken: 
- **NumPy**: Array-Grundlage; Vektorisierung, lineare Algebra.
- **pandas**: tabellarische Daten (DataFrames); der direkte, mächtigere Excel-Ersatz für Einlesen, Filtern, Gruppieren, Mergen und Export (csv, xlsx)
- **SciPy**: Numerische Mathematik: Optimierung, Integration, ODE-Löser, Signalverarbeitung, statistische Funktionen.
- **statistics/statsmodels** — klassische/frequentistische Statistik: Regression, GLMs, ANOVA, Zeitreihen, mit detailreicher Modell-Zusammenfassungen
- **Matplotlib**: Basis-Plotting, volle Kontrolle, Standard für Publikationsgrafiken
- **reportlab**: Erstellen von PDFs, einbindung von Ergebnissen und Grafiken. 
- **openpyxl**: Erstellen von Excel-Exporten
- **Jupyter** — interaktive Notebooks; verbinden Code, Ergebnis, Grafik und Erklärung in einem reproduzierbaren oder gar automatisierbaren Dokument
- **PyMC**: Bayessche Modellierung / probabilistisches Programmieren (Markov-Chain Montecarlo - MCMC)
- **ArviZ**: Diagnostik und Visualisierung für Bayessche Modelle (Posterior, Trace-Plots, ROPE (Regions of Practical Euquivalence)); Begleiter zu PyMC
- **Biopython** — das Allround-Werkzeug für Biodatenanalyse: Sequenzen (FASTA/GenBank), BLAST, Alignment, Zugriff auf NCBI-Datenbanken, Strukturparsing
- **OpenMM** — Molekülmechanik-Simulationen selbst (GPU-beschleunigt)

Die Liste ist erschlagend, könnte jedoch noch weitergehen. All diese Bibliotheken sind validiert und erfreuen sich bei Ausführenden in der Wissenschaft durch ihre einfache Anwendung. Eine Integration in den Bachelorstudiengang heißt nicht, all diese Bibliotheken zu lehren. Das wäre auch nicht zielführend. Es soll in verschiedenen Semestern unter verschiedenen Anwendungen pragmatische Lösungen aufgezeigt werden - und das alles unter dem Dach **einer** Software. Coding wird oft missverstanden als "Software von Grund auf aufbauen". Bei Python mit seinem Bausteinprinzip ist das nicht so. Glauben sie jemanden, der Anfang April 2026 noch keine Erfahrung mit Coding gemacht hat. Glauben sie mir. Es ist lernbar, schnell, mit steiler Lernkurve.
Wie man Coding in den Bachelorstudiengang BT/BPT einbetten könnte, zeige ich im folgenden. 

---
# Empfehlung: Konkrete Integration in den Bachelorstudiengang BT/BPT
Als ich mit BT/BPT angefangen habe, gab es im Modul "Einführung in das Studium und Berufsfeld BT/BPT" neben Ringvorlesungen einen Excel-Kurs, mit anschließender Prüfung. Im zweiten Semester dann mit "Auswertung wissenschaftlicher Daten" eine Einführung in Aufbereitung von Daten und die Darstellung von Ergebnis, zugegebenermaßen eher rudimentär als tiefgreifend. Es gibt viele Punkte, in denen eine Integration von Coding sinn macht. Ein konkreter Vorschlag wäre dieser:

| Semester | Modul                                               | Integration                                                                                                                                                                    |
| -------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1.       | Einführung in das Studium/Berufsfeld                | Vermittlung von Excelgrundkenntnissen + Pythongrundkurs: Datentypen, Objekte, Schleifen, Blöcke, Funktionen, Versionierung und Anwendung in Notebooks (siehe Jupyter Notebook) |
| 1.       | Biologie                                            | Python grundlagen für Statistik                                                                                                                                                |
| 1.       | Physik I                                            | Python grundlagen für Lineare Algebra mit NumPy und Mathplotlib                                                                                                                |
| 2.       | Statistik/Auswertung Wissenschaftlicher Daten (AWD) | Erweiterte Statistik mit statistics/statsmodels + Mathplotlib                                                                                                                  |
| 2.       | Physik II                                           | SciPy Grundlagen zu numerischen Problemlösung + NumPy vertiefung                                                                                                               |
| 4.       | Circuit, Signals and Systems                        | Systemsimulationen in Python mittels SciPy                                                                                                                                     |
| 4.       | Bioanalytik                                         | Datenanalyse und Auswertung via Pandas + statsmodels                                                                                                                           |
| 5.       | Bioverfahrenstechnik Upstream                       | Vertiefung Simulation mittels SciPy: Prozesssimulation + Fitting von                                                                                                           |
Für die beiden Mathematikmodule soll der Fokus auf Theorie bleiben. Die Physikmodule und Biologie gelten daher als praktischer Einstieg in Python.
In AWD soll das Hauptaugenmerk auf frequentistische Statistik (Verteilungen, p-Tests, Konfidenzintervalle usw.) und der Darstellung als Graphen von Ergebnissen liegen. Bei Modulen, die nicht explizit in der Liste vorkommen sollen jedoch Auswertungen im Rahmen der Praktika mittels Pythontoolkit erfolgen. Zudem können  Zusammenhänge wie Bilanzgleichungen idiomatisch in mathematischen Gleichungen, sowie Python-Code in Vorlesungen dargestellt werden. Das fördert Transferwissen, als auch festigen von bereits Gelerntem. 
Man sieht: Der Werkzeugkasten wächst im Rahmen des Studiums mit. Studierenden können Vorwissen aus anderen Modulen in ihren Code nutzen, um Problemstellungen zu verstehen und zu lösen. 
Programmier"*sprachen*" heißen nicht nur so, da sie die Möglichkeit aufzeigen, Computern Anweisungen zu geben, sondern weil es sich um eine Sprache handelt. Etwas, was sich ab einem gewissen Grad natürlich anfühlt. 

---
# Anpassung im Masterstudiengang BT/BPT
Auch wenn es hier um die Akkreditierung des Bachelors geht, sollte darüber hinaus über Anpassung bestimmter Module im Master nachgedacht werden, um das Medium Python in seinen Möglichkeiten voll auszureizen und den Studierenden eine  Art der Statistik näher zu bringen, die lange Zeit als unkonventionell und unpraktisch galt und heute ihr Revival erlangt; Nämlich **Bayesische Statistik**. Die statistischen Modelle, die auf dem bekannten Satz von Bayes beruhen, können die Antworten erbringen, die viele fälschlicherweise bei Signifikanztest und Konfidenzintervalle hineininterpretieren: "Mit der Wahrscheinlichkeit $p$ (z.B $95\%$), gegeben durch Daten $D$, ist meine Hypothese $= H$". Während Bayesische Statistik eine Wahrscheinlichkeitsbegriff für einen Estimand erbringt, definiert frequentistische Statistik Wahrscheinlichkeit als Häufigkeit: "wie häufig komme ich auf die gleichen Ergebnisse, wenn ich den Versuch unendlich oft wiederhole". Das ist das Kernparadigma. Das eine Kernaussage über eine Hypothese damit verboten ist, ist vielen Wissenschaffenden nicht klar. Bayesische Statistik hingegen gibt eine ganze Verteilung an Hypothesen zurück und zeigt mittels *a priori* Annahmen, welche Hypothese am wahrscheinlicher als andere ist. Durch ihren Rechenaufwand (unlösliche Integrale der Evidenz im Satz von Bayes), wurde der bayesische Ansatz Mitte des 20. Jahrhundert durch den frequentistischen Abgelöst. Computergestützte Bayesstatistik enthebelt diesen Nachteil. Es gibt also keine Entschuldigung sie nicht zu lehren.
Wie bereits zu Anfang erwähnt, kann die Bayesstatistik durch die Python-Bibliothek **PyMC** realisiert werden. Studierende können hier lernen, wie kausale Ketten gebildet werden können mittels DAG (Directed Acyclic Graphs), wie man generative Modelle baut und validiert, wie Zielgrößen **vor** einer Untersuchung definiert werden. Es geht um Verständnis: Nicht weniger als die Statistik verstehen, die man selbst für eine bestimmte Wissenschaftliche Fragestellung zuschneidet und in Code Übersetzt. Damit Wissenschaft eben nicht mehr Zweck für schöne Statistik ist, für Stopp-kriterien um aus einem p-Wert noch die Signifikanz herauszupressen, sondern andersherum.
Die Bayestsatistik via PyMC sollte nach meiner Empfehlung in das Numerik/Statistik-Modul integriert werden. In Numerik werden numerische Berechnungen und Methoden vorgestellt. Es wird per Hand Matrizen zerlegt, berechnet, Algorithmen zu Regressionsmodellen berechnet, sowie optionale Codingaufgaben in Matlab, die eben nur *optional* sind. Das ist ganz Nett, hat aber keine praktischen Nutzen im Verlauf des weiteren Studiums. Ein ehrlicherer Umgang wäre daher ein pragmatischer: Im neuen Numerik/Statistikmodul wird numerische Theorie direkt mit der Anwendung in Python (PyMC) verbunden, anstelle von optionalen Matlab, wie es Momentan der Fall ist. Im Statistikteil soll nicht mehr der Stoff des ersten Semesters durchgekaut werden, sondern die Theorie zur Bayesstatistik gelehrt werden. Für die Prüfungsleistung empfehle ich eine 45 Min Klausur zur allgemeinen Theorie + eine Hausarbeit in Gruppenarbeit, in dem jede der Gruppen einen Datensatz mittels Bayesischen Arbeitsablauf durcharbeitet: 
1. Wissenschaftliche Fragestellung definieren,
2. Generatives Modell/DAG erstellen, validieren (prior predictive check)
3. Statistisches Modell erstellen, Daten auswerten und validieren (posterior predictive check)
4. Aus Posterior-Verteilung und Annahmen kausale Antworten definieren.
Damit wäre das Modul (dann Numerik/Bayesstatistik) anwendungsbezogener und bereitet die Studierende auf die Statistik von Morgen vor. 
Zudem: Wer sich mit bayesischer Statistik auskennt, der hat es nicht mehr leicht zu probalistischen maschinellen Lernen, ein Feld, dass in den angewandten Naturwissenschaften zunehmend an Bedeutung gewinnt (#Bishop hier zitieren)
Um eine Probe von bayesischer Statistik zu bekommen, empfehle ich die mitgeschnittene Vorlesung "[Statistical Rethinking](https://www.youtube.com/watch?v=ztbYkBPDOgU&list=PLDcUM9US4XdNOlqSyhe38US8mFgmqzI14)" von Richard McElreath, Direktor am Max Planck Institut für evolutionäre Anthropologie, Honorarprofessor an der Universität Leibzig. 

---
# Hindernisse
Unabhängig zur Frage, welche Vorteile die Integration einer Programmiersprache wie Python in einem Ingenieurstechnischen und naturwissenschaftlichen Studiengang für die Studierenden bietet, gibt es einige Hürden. Die größte betrifft das Lehrpersonal selbst. Viele der Dozierenden, Professoren/Professorinnen und wissenschaftlichen Mitarbeitenden haben was Coding betrifft rudimentäre bis keine Berührungspunkte. Das gilt jedoch nur für die Gegenwart und sicherlich nicht für alle Dozierenden: Gerade an den Schnittstellen zu anderen Fachbereichen und Studiengängen sehe ich einiges an Qualifizierten Personal, dass eine solche Integration unterstützen kann. Zudem wären Weiterbildungen der einzige Kostenfaktor im Zusammenhang mit Python (da open source). Das heißt Lernen für das Lehrpersonal. Ja, das ist eine Voraussetzung, wie man jedoch weiß ist naturwissenschaftliches Arbeiten per se nicht Voraussetzungsfrei: Auch ein Biologe muss die grundlagen der hard science (Mathe, Chemie, Physik) in einem bestimmten Umfang wissen. Es muss also ein Mindeststandard unter den Lehrenden geben. 

Weitere Hürden:
- Studenten, die sich bewusst für BT/BPT entschieden haben, haben sich eventuell bewusst gegen einen Informatischen Studiengang entschieden. Das kann zu frühen Frust führen. Hier sollten Studierende umfänglich im ersten Semester unterstützt werden. Auch KI kann hier als Tutorentool nützlich sein (ich spreche aus eigener Erfahrung)
- Installationshürde: Den Aufbau von virtual enviroments, der Integration in ein Codeprogramm kann Zeitaufwendig sein. Abhilfe könnte gerade im ersten Semester Cloudnotebooks schaffen, wie Jupyter Hub. 
- Prüfungsformate: Klausuren passen schlecht zu Codingkompentenzen. Es braucht daher neue oder gemixte Formate. 

---
# Ein Appell

Wenn Computergestütztes wissenschaftliches Arbeiten ein Taschenrechner wäre, wäre Excel ein Abakus. Die Microsoftanwendung wurde einmal aus Bequemlichkeit aus dem Finanzsektor gezerrt, in einem Feld, in dem es, sind wir ehrlich, nichts zu suchen hat. Klar ist, dass das Tabellenprogramm vieles an Kompetenz gedämpft hat, um die Studierenden durch das Studium zu bringen. Diese Bequemlichkeit sollte man sich nicht mehr leisten dürfen, denn sie geht auf die Kosten der Kompetenz derjenigen, die etwas lernen wollen.
Aber man kennt es sicher aus dem naturwissenschaftlichen Umfeld: progressive Veränderungen sind Anstrengend - und zahlen sich aus. Wissenschaft basiert auf Disruption, nicht auf den Status quo. Irgendwo muss man anfangen, irgendwann muss es die Disruption geben, warum nicht dann, wenn sie am günstigsten für Alle ist?
.






[[Korrigierte Akkreditierung]]