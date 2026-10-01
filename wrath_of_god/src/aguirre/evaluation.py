"""Abschnitt 8 -- Metriken und Plots.

Billig und lokal -- das ist der Pflichtteil:
    uniqueness            Anteil einzigartiger Sequenzen
    aa_frequencies        Aminosaeure-Verteilung vs. Trainingsset
    js_divergence         Jensen-Shannon-Divergenz der AA-Verteilungen
    kmer_diversity        einzigartige 3-mere / 6-mere, Jaccard zum Training
    length_distribution

Plots: Lernkurven, AA-Frequenz-Balken (generiert vs. Training),
Noise-Schedule (alpha_bar gegen t), Laengenhistogramm.

Abschnitt 9 (Stretch, NICHT jetzt): APEX-MIC, HydrAMP-Klassifikatoren,
ProGen2-Perplexitaet. Drei fremde Repos mit eigenen Gewichten und
Abhaengigkeiten -- realistisch mehrere Tage. Erst anfassen, wenn 1-8 laufen.
Vergleichszahlen fuer alle Metriken stehen in Torres_Model/note/README.md.
"""


class Evaluator:
    """Intrinsische Metriken auf generierten Sequenzen."""

    # TODO


class Plotter:
    """Alle Figures des Projekts, einheitlich gestylt."""

    # TODO
