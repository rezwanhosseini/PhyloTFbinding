import numpy as np
import torch
import torch.nn.functional as F
from MotifDiff.util import returnonehot, MEME_probNorm, MCspline_fitting


# ── 1. Load and parse the PFM file ───────────────────────────────────────────

def pfm_name_to_short(name):
    """Convert 'GM.5.0.Nuclear_receptor.0090' -> 'Nuclear_receptor.0090'"""
    parts = name.split('.')
    return '.'.join(parts[3:]) if len(parts) >= 5 else name


def load_motifs(pfm_path):
    """
    Parse a gimme.vertebrate PFM file and fit probNorm splines.

    Returns
    -------
    kernels      : torch.Tensor, shape (2*n_motifs, 4, max_len)
                   Paired fwd/rev log-probability kernels for conv1d.
    spline_list  : list of PchipInterpolator, length n_motifs
                   Maps raw log-likelihood score -> CDF probability [0,1].
    name_to_idx  : dict  short_name -> kernel index k
                   Use kernels[2*k] / kernels[2*k+1] for fwd/rev.
    """
    motif_obj = MEME_probNorm()
    kernels, _ = motif_obj.parse(pfm_path, nuc="mono", transform=False,
                                 strand_specific=False)
    spline_list = MCspline_fitting(kernels, nuc="mono")
    name_to_idx = {pfm_name_to_short(n): i for i, n in enumerate(motif_obj.names)}
    return kernels, spline_list, name_to_idx


# ── 2. Per-position binding probability scanner ───────────────────────────────

def scan_binding_probs_per_position(seq, motif_id, kernels, spline_list, name_to_idx):
    """
    Compute per-position probNorm binding probability for one motif on one sequence.

    Follows the MotifDiff probNorm/max pipeline:
      1. One-hot encode the sequence (N -> zero vector).
      2. Convolve with fwd and rev log-probability kernels.
      3. Map raw scores through the per-motif CDF spline -> probability [0,1].
      4. Take max(fwd, rev) at each position.
      5. N-MASKING: set NaN at any position whose motif window overlaps an N.

    Why N-masking is necessary
    --------------------------
    The kernel stores log(p_motif), so all real-sequence scores are in (-inf, 0].
    An N encodes as the zero vector, giving score = 0.0 — which is ABOVE the
    maximum achievable score for any real sequence. The spline CDF(0) therefore
    returns ~1.0 for highly specific motifs, producing spurious near-1.0 binding
    probabilities at N-containing positions. Masking these to NaN is the fix.

    Parameters
    ----------
    seq        : str   Raw (ungapped) DNA sequence, any length >= motif_len.
    motif_id   : str   Short motif name, e.g. 'Nuclear_receptor.0090'.
    kernels    : torch.Tensor  From load_motifs().
    spline_list: list          From load_motifs().
    name_to_idx: dict          From load_motifs().

    Returns
    -------
    result : np.ndarray, shape (seq_len,), dtype float32
        Binding probability [0,1] at each position (center of motif window).
        NaN at edge positions where the motif window doesn't fit.
        NaN at positions where the motif window overlaps any N.
    """
    seq_len = len(seq)
    k = name_to_idx[motif_id]

    fwd_kernel = kernels[2*k:2*k+1]    # (1, 4, kernel_len)
    rev_kernel = kernels[2*k+1:2*k+2]  # (1, 4, kernel_len)
    kernel_len = fwd_kernel.shape[2]   # max_len (zero-padded), e.g. 30

    # Actual motif length = number of non-zero columns in the fwd kernel
    # (the kernel is right-zero-padded to max_len)
    col_norms = fwd_kernel[0].abs().sum(dim=0)
    motif_len = int((col_norms != 0).sum().item())
    if motif_len == 0:
        return np.full(seq_len, np.nan, dtype=np.float32)

    half = motif_len // 2

    # One-hot encode: N positions become all-zero columns (4, seq_len)
    one_hot = returnonehot(seq, dinucleotide=False)
    seq_tensor = torch.from_numpy(one_hot).unsqueeze(0)  # (1, 4, seq_len)

    # Convolve: output length = seq_len - kernel_len + 1
    fwd_scores = F.conv1d(seq_tensor, fwd_kernel).squeeze(0).squeeze(0)  # (n_pos,)
    rev_scores = F.conv1d(seq_tensor, rev_kernel).squeeze(0).squeeze(0)  # (n_pos,)

    n_pos = seq_len - kernel_len + 1
    if n_pos <= 0:
        return np.full(seq_len, np.nan, dtype=np.float32)

    # Apply probNorm spline: raw log-likelihood -> CDF probability [0,1]
    spl = spline_list[k]
    fwd_prob = np.clip(spl(fwd_scores.numpy()), 0, 1)
    rev_prob = np.clip(spl(rev_scores.numpy()), 0, 1)

    # Max over strands at each position
    best_prob = np.maximum(fwd_prob, rev_prob)  # (n_pos,)

    # Map output positions to center positions in the original sequence.
    # conv1d output position p corresponds to seq window [p, p+kernel_len).
    # The actual motif occupies [p, p+motif_len), so its center = p + half.
    result = np.full(seq_len, np.nan, dtype=np.float32)
    centers = np.arange(n_pos) + half
    valid = (centers >= 0) & (centers < seq_len)
    result[centers[valid]] = best_prob[valid]

    # ── N-MASKING ────────────────────────────────────────────────────────────
    # For each N at position n, the contaminated center range is:
    #   [n - (motif_len - 1 - half), n + half]
    # because any window whose range [p, p+motif_len) contains n
    # has center p+half in that interval.
    n_positions = np.where(np.array(list(seq.upper())) == 'N')[0]

    if len(n_positions) > 0:
        contaminated = set()
        for n in n_positions:
            c_min = max(n - motif_len + 1 + half, half)
            c_max = min(n + half,                 half + n_pos - 1)
            if c_min <= c_max:
                contaminated.update(range(c_min, c_max + 1))

        if contaminated:
            mask = np.array(list(contaminated), dtype=np.int64)
            mask = mask[(mask >= 0) & (mask < seq_len)]
            result[mask] = np.nan

    return result


# ── 3. Batch scan: all species × all motifs for one gene ─────────────────────

def scan_gene(fasta_path, motif_ids, kernels, spline_list, name_to_idx):
    """
    Scan all sequences in a FASTA file against a list of motifs.

    Returns
    -------
    dict  motif_id -> {'species': [str, ...], 'matrix': np.ndarray (n_species, seq_len)}
    """
    from Bio import SeqIO
    records = list(SeqIO.parse(fasta_path, "fasta"))
    species_list = [r.id for r in records]
    seqs         = [str(r.seq) for r in records]
    seq_len      = len(seqs[0])

    results = {}
    for motif_id in motif_ids:
        mat = np.full((len(seqs), seq_len), np.nan, dtype=np.float32)
        for si, seq in enumerate(seqs):
            mat[si] = scan_binding_probs_per_position(
                seq, motif_id, kernels, spline_list, name_to_idx
            )
        results[motif_id] = {'species': species_list, 'matrix': mat}
    return results


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from Bio import SeqIO

    PFM_PATH  = "gimme.vertebrate.v5.0.pfm"
    FASTA_DIR = "promoterSeqs_byGene_combiend"

    # gene -> list of motif IDs to scan
    gene_to_motifs = {
        "ABCC10": ["Homeodomain.0015", "Homeodomain.0156",
                   "Nuclear_receptor.0090", "Unknown.0007"],
        "ALDOA":  ["C2H2_ZF.0071"],
        # ... add more genes/motifs as needed
    }

    print("Loading motifs and fitting splines...")
    kernels, spline_list, name_to_idx = load_motifs(PFM_PATH)
    print(f"  {len(name_to_idx)} motifs loaded.")

    all_results = {}
    for gene, motifs in sorted(gene_to_motifs.items()):
        fasta_path = os.path.join(FASTA_DIR, f"{gene}_combined_names.fa")
        if not os.path.exists(fasta_path):
            print(f"  {gene}: FASTA not found, skipping")
            continue
        all_results[gene] = scan_gene(fasta_path, motifs, kernels, spline_list, name_to_idx)
        n_sp = len(all_results[gene][motifs[0]]['species'])
        print(f"  {gene}: {n_sp} species × {len(motifs)} motifs done")

    # Access results:
    # all_results['ALDOA']['C2H2_ZF.0071']['matrix']  -> (n_species, 500) float32
    # all_results['ALDOA']['C2H2_ZF.0071']['species'] -> list of FASTA IDs
