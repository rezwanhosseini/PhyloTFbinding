import numpy as np
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import warnings
warnings.filterwarnings('ignore')


def extract_species(name):
    """Strip tree/FASTA name artifacts to get a clean species identifier."""
    return name.strip("'").split('_[')[0]


def compute_phylo_conservation(aln_path, ref_idx=0):
    """
    Compute phylogenetic conservation per alignment column.

    Score = fraction of pairwise branch-length-weighted identity (non-gap pairs only).
    Concretely, for each alignment column:
        score = Σ_{i<j} dist(i,j) * [nuc_i == nuc_j] * [both non-gap]
              / Σ_{i<j} dist(i,j) * [both non-gap]

    Where dist(i,j) is the patristic (tree) distance between species i and j
    from a Neighbor-Joining tree built on the alignment. Highly diverged species
    pairs contribute more weight, so conserved columns among distant species
    score higher than conservation among closely related ones.

    Parameters
    ----------
    aln_path : str
        Path to a multiple sequence alignment (FASTA format).
    ref_idx : int
        Index of the reference sequence (default 0 = first sequence).
        Output scores are mapped to ungapped positions of this sequence.

    Returns
    -------
    cons_ref : np.ndarray, shape (n_ref_ungapped,)
        Conservation score per ungapped position of the reference sequence.
        Values in [0, 1]; NaN where a column had <2 non-gap sequences.
    ref_to_aln : list of int
        Mapping from reference position index → alignment column index.
    species_names : list of str
        Species names in alignment row order.
    """
    aln = AlignIO.read(aln_path, "fasta")
    n_cols = aln.get_alignment_length()
    species_names = [extract_species(rec.id) for rec in aln]

    # Build Neighbor-Joining tree from the alignment
    calculator = DistanceCalculator('identity')
    dm = calculator.get_distance(aln)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    # Map tree terminals → alignment row indices
    terminals = tree.get_terminals()
    term_species = [extract_species(t.name) for t in terminals]
    species_to_aln_idx = {s: i for i, s in enumerate(species_names)}

    valid_ti, valid_ai = [], []
    for ti, ts in enumerate(term_species):
        if ts in species_to_aln_idx:
            valid_ti.append(ti)
            valid_ai.append(species_to_aln_idx[ts])
    valid_ti = np.array(valid_ti)
    valid_ai = np.array(valid_ai)

    # Precompute all pairwise patristic distances
    n_valid = len(valid_ti)
    pair_dist = np.zeros((n_valid, n_valid))
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            d = tree.distance(terminals[valid_ti[i]], terminals[valid_ti[j]])
            pair_dist[i, j] = d
            pair_dist[j, i] = d

    # Alignment matrix (valid rows only)
    aln_matrix = np.array([[c.upper() for c in str(rec.seq)] for rec in aln])
    sub_matrix = aln_matrix[valid_ai, :]  # (n_valid, n_cols)

    # Score each alignment column
    cons_aln = np.full(n_cols, np.nan)
    for col in range(n_cols):
        nucs = sub_matrix[:, col]
        non_gap = nucs != '-'
        if non_gap.sum() < 2:
            continue
        both_non_gap = non_gap[:, None] & non_gap[None, :]
        same_nuc = (nucs[:, None] == nucs[None, :]) & both_non_gap
        denom = (pair_dist * both_non_gap).sum()
        if denom == 0:
            continue
        cons_aln[col] = (pair_dist * same_nuc).sum() / denom

    # Project alignment coords → reference sequence (ungapped) coords
    ref_seq = str(aln[ref_idx].seq).upper()
    ref_to_aln = [col for col, c in enumerate(ref_seq) if c != '-']
    cons_ref = cons_aln[ref_to_aln]

    return cons_ref, ref_to_aln, species_names
