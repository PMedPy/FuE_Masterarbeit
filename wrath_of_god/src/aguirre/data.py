"""Abschnitt 1 + 3 -- Rohdaten und Dataset/Dataloader.

Abschnitt 1: Sequenzen laden und filtern.
    Eingang  : CSV/FASTA
    Ausgang  : list[str], NUR AMPs (keine Non-AMPs -- ein unkonditioniertes
               Diffusionsmodell lernt die Verteilung seiner Trainingsdaten)
    Schritte : Laengenfilter, nur die 20 Standard-Aminosaeuren,
               Duplikate entfernen, Train/Val-Split

Abschnitt 3: Dataset, das vorberechnete Embeddings ausliefert.
    __getitem__ gibt zurueck:
        emb           (L, D)   float32
        padding_mask  (L,)     bool   -- True = Padding, wird maskiert
    Nach dem Collate im Dataloader:
        emb           (B, L, D)
        padding_mask  (B, L)

Wichtig: Embeddings EINMAL vorberechnen und cachen, nicht im __getitem__
berechnen. Sonst laeuft ESM in jedem Epoch neu und jeder Testlauf wird zaeh.

Notiz Paul: Das muss nochmal geändert werden. 
"""
import torch
from torch.utils.data import Dataset
from aguirre.config import DATA_DIR, PROJECT_ROOT, CACHE_DIR, TRACKED_DIR, Config
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
config = Config()


def write_fasta(data: list[str], path: str | Path):
    """Baut aus einer Liste von Peptidsequenzen eine Fasta-Datei"""

    with open(path, "w") as datei:
        for i, seq in enumerate(data): 
            header = f">AMP_{i}\n" # "_" Für späteren Split bei der Wiederfindung
            sequence = f"{seq}\n"
            datei.write(header)
            datei.write(sequence)
    return path

def read_fasta(path) -> tuple[list[str], list[str]]:
    """Liest eine FASTA GIbt (ids, Sequences in dateireihenfolge zurueck)"""
    ids = []
    seqs = [] #ids, Sequences
    with open(path, "r") as data:
        for col in data:
            if col.startswith(">"):
                id_sequence = col.strip().removeprefix(">")
                ids.append(id_sequence)
            else:
                sequence = col.strip()
                seqs.append(sequence)
    if len(ids) != len(seqs):
        raise ValueError("Achtung, Verschiebung durch Zeilenumbrüche in der Fasta")
    return ids, seqs

def run_mmseqs_search(
    fasta_path: str | Path,
    out_path: str | Path,
    tmp_dir: str | Path,
    sensitivity: float = 7.5,
    evalue: float = 10000,
    max_seqs: int = 10000,
    kmer_len: int = 5,
    spaced_kmer_mode: int = 0,
    coverage_rate: float = 0.5,
    mask: int = 0,
    cov_mode: int = 5,
    extra_args: list[str] | None = None,
) -> Path:
    """All-gegen-alle-Suche mit MMseqs2. Gibt den Pfad zur .m8-Datei zurueck."""
    #==Input in Paths ändern==
    fasta_path = Path(fasta_path)
    out_path = Path(out_path) # Pfad muss auf eine Datei zeigen. z.B DATA_DIR / "test_hits.m8"
    tmp_dir = Path(tmp_dir)

    tmp_dir.mkdir(parents = True, exist_ok = True)
    out_path.parent.mkdir(parents= True, exist_ok= True)

    #==Command==
    # mmseqs easy-search  <fasta>  <fasta>  <out.m8>  <tmp_dir>  -s 7.5  -e 10000  --max-seqs 2000
    
    cmd = [
        "mmseqs", 
        "easy-search", 
        str(fasta_path), 
        str(fasta_path), 
        str(out_path), 
        str(tmp_dir), 
        "-s",
        str(sensitivity),
        "-e",
        str(evalue),
        "--max-seqs",
        str(max_seqs),
        "-k",
        str(kmer_len),
        "--spaced-kmer-mode",
        str(spaced_kmer_mode),
        "--mask", 
        str(mask),
        "--cov-mode",
        str(cov_mode),
        "-c",
        str(coverage_rate),
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",]
    
    if extra_args:
        cmd += extra_args
    
    #==run & return==
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mmseqs fehlgeschlagen:\n{res.stdout}\n{res.stderr}")
    return out_path

class ClusterSplitter():
    """Clustert AMP-Sequenzen ueber Zusammenhangskomponenten und
        verteilt ganze Cluster auf train/val/test."""
    def __init__(self, pairs: pd.DataFrame, ids, min_ident=0.3):
        """pairs: DataFrame mit query, target, ident (bereits gefiltert).
        ids:   alle Sequenz-IDs aus der FASTA, auch partnerlose."""
        pairs = pairs[pairs["ident"] >= min_ident]
        self.pairs = pairs # Dataframe aus M8-Datei, mit query, target, ident
        self.ids = ids # Liste aus read_fasta
        self.min_ident = min_ident

        self.labels = None # Clusterlabels (jedes Cluster eine ID)
        self.threshold = None 
        self.id_to_idx, self.idx_to_id = self._build_index()

    
    def _build_index(self):
        """IDs <-> 0..n-1. Setzt self.id_to_idx und self.idx_to_id."""
        idx_to_id = sorted(self.ids) # Von zeilennummer zu AMP-ID
        id_to_idx = {key: item for item, key in enumerate(sorted(self.ids))} #Dicitonairy für Übersetzung {AMP: i}
        return id_to_idx, idx_to_id    

    def build_graph(self, t):
        """Kanten bauen -> csr_matrix Gibt die Adjavanzmatrix zurück"""
        if t < self.min_ident: #Guard, damit Kanten nicht fehlen
            raise ValueError(f"t={t} liegt unter min_ident={self.min_ident} - Kanten fehlen")
        
        n = len(self.idx_to_id)
        shape = (n,n) # Shape für csr_matrix, sonst wird größte ID verwendet
        kanten = self.pairs[self.pairs["ident"] >= t] # DataFrame gefiltert mit threshold 
        #Übersetzung:
        rows = kanten["query"].map(self.id_to_idx).to_numpy() #Map ersetzt jeden Wert durch das Dict-Item
        cols = kanten["target"].map(self.id_to_idx).to_numpy()

        data = np.ones(len(rows), dtype= np.int8)
        A = csr_matrix((data, (rows, cols)), shape= shape) # Sparsematrix in größe shape = (n,n)
        A = A.maximum(A.T) #Symmetrisieren da MMseqs2 Suche nicht symmetrisch ist
        return A

    def cluster(self, t):
        """Setzt self.labels, self.threshold."""
        A = self.build_graph(t)
        n_comp, labels = connected_components(csgraph = A, directed = False, return_labels = True) # Directed False ist hier redundant, weil Graph schon symmetrisch in build_graph
        self.labels = labels # Cluster labels
        self.threshold = t 
        return n_comp, labels

    def summary(self):
        """Bestimmung der Clustergrößen"""
        #==Guardian==
        if self.labels is None:
            raise RuntimeError("erst cluster(t) aufrufen")

        sizes = np.bincount(self.labels) # Für jeden Cluster die Anzahl an AMPs im Cluster

        summary_dict = {
            "threshold": self.threshold,
            "n_components": len(sizes),
            "largest": sizes.max(),
            "largest_share": sizes.max() / sizes.sum(),
            "n_singletons": (sizes == 1).sum(),
            "median": np.median(sizes),
            "top10": np.sort(sizes)[::-1][:10],
        }
        return summary_dict

    def assign_split(self, ratios=(0.8, 0.1, 0.1), mode="greedy", seed=0):
        """Ordnet die Cluster nach First-Fit Decreasing in Train/Test/Val Sets ein."""

        #==Guardian==
        if self.labels is None:
            raise RuntimeError("erst cluster(t) aufrufen")
        
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"ratios summieren zu {sum(ratios)}, nicht 1.0")

        sizes = np.bincount(self.labels) # Für jeden Cluster die Anzahl an AMPs im Cluster
        n_total = sizes.sum()
        targets = np.array(ratios) * n_total # Wie viele AMPs pro Set?

        rng = np.random.default_rng(seed)
        tiebreak = rng.random(len(sizes))
        order = np.lexsort((tiebreak, -sizes)) #Reihenfolge Index der Cluster nach ihrer Größe

        #Buchhaltung:
        filled = np.zeros(len(ratios)) #Startwert für gefüllte Sets
        cluster_to_split = np.zeros(len(sizes), dtype = np.int8) #Zeros, Anzahl = Cluster

        #Schleife:
        for c in order:
            deficit = targets - filled # Wie viel AMPs pro Set sind noch übrig?
            j = np.argmax(deficit) # Welches Set hat die meisten AMPs zu besetzen
            cluster_to_split[c] = j # Jeder Cluster wird dem Set mit dem meisten AMPs zugeordnet
            filled[j] += sizes[c] # Set mit höchsten Bedarf wird gefüllt mit ganzen Cluster
        split_per_node = cluster_to_split[self.labels] # 
        # DataFrame
        namen = np.array(["train", "val", "test"])
        split_namen = namen[split_per_node]

        df = pd.DataFrame({
            "id": self.idx_to_id,
            "cluster": self.labels,
            "split": split_namen,
        })
        return df

class Data_coordinator(Dataset):
    """Liefert vorberechnete ESM-Embeddings + Padding-Maske."""

    def __init__(self, split: str, cfg: Config, standardize: bool = False):

        #Data Laden
        EMBEDDING_PATH = CACHE_DIR / f"{split}-Set_{cfg.esm_model_name}_L{cfg.pep_max_len}.pt"
        daten = torch.load(EMBEDDING_PATH)
        emb, lengths, ids = daten["emb"], daten["lengths"], daten["ids"]
        assert emb.shape[2] == cfg.embed_dim, "Embedding-Dimensionen stimmen nicht überein."
        assert lengths.shape[0] == len(ids), "Dateien bei Embedding verrutscht."

        # Standardiesierung
        if standardize is True:
            STANDARDS_PATH = TRACKED_DIR / f"standards_{cfg.esm_model_name}_L{cfg.pep_max_len}.csv"
            standards = pd.read_csv(filepath_or_buffer= STANDARDS_PATH, header = 0)
            mean = torch.tensor(standards["mean"].to_numpy(), dtype=torch.float32) #Nur die Werte des dataframe aufrufen
            std = torch.tensor(standards["std"].to_numpy(), dtype= torch.float32)
            assert emb.shape[2] == mean.shape[0], "Standards passen nicht zu Embeddings"

            emb_s =emb.sub_(mean).div_(std)
            # Padding durch Maske zurücksetzen
            maske = torch.arange(cfg.pep_max_len).unsqueeze(0) < lengths.unsqueeze(-1)
            emb_s *= maske.unsqueeze(-1)
            embedding = emb_s
        else: 
            embedding = emb
            mean = None
            std = None

        #Instanzen
        self.split = split
        self.cfg = cfg
        self.standardize = standardize
        self.emb = embedding
        self.lengths = lengths
        self.ids = ids
        self.mean = mean
        self.std = std


    def __getitem__(self, i):
        return self.emb[i], self.lengths[i]

    def __len__(self):
        return self.emb.shape[0]

    def destandardize(self, x: torch.Tensor):
        if self.standardize is True:
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)
            out = x * std + mean
            return out.cpu()
        else:
            return x.cpu()
