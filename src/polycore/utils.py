import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import psutil

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

IUPAC_BITS: Dict[str, int] = {
    "A": 1, "C": 2, "G": 4, "T": 8,
    "R": 1 | 4, "Y": 2 | 8, "S": 4 | 2, "W": 1 | 8, "K": 4 | 8, "M": 1 | 2,
    "B": 2 | 4 | 8, "D": 1 | 4 | 8, "H": 1 | 2 | 8, "V": 1 | 2 | 4,
}
MIXED_IUPAC = np.array(list("RYSWKMBDHV"))
ALLELES: List[int] = [1, 2, 4, 8]
POPCOUNT16 = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)
COUNT_DENOM = 12  # lcm(1..4): makes denom*ploidy/k integral for any popcount k

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    min_gf: float = 0.9
    min_cf: float = 0.95
    progressive: bool = False
    ploidy: Optional[int] = None
    chunk_size: Optional[int] = None
    split: bool = False
    ref_by_name: bool = False
    bed_file: Optional[str] = None


@dataclass
class WorkflowState:
    # Config
    cfg: Optional[Config] = None

    # Raw input
    files: Optional[List[str]] = None
    sequences: Any = None
    names: Optional[List[str]] = None
    contigs: Optional[List[str]] = None

    # Encoding
    ploidy: Optional[int] = None
    bit_map: Optional[Dict[str, int]] = None
    bit_lut: Optional[Dict[int, str]] = None
    valid_bases: Optional[List[str]] = None

    # Stacks & maps (per contig)
    stacks: Optional[List[Any]] = None
    idx_maps: Optional[Dict[str, Any]] = None
    filt_maps: Optional[Dict[str, Any]] = None

    # Sample info
    gfs: Any = None
    names_filt: Optional[List[str]] = None
    cfs: Any = None
    core_order: Any = None
    full_stacks: Any = None
    full_masked_stacks: Any = None
    core_masked_stacks: Any = None

    # Site info
    n_total: int = 0
    n_mask: int = 0
    n_miss_ref: int = 0
    n_valid: int = 0
    n_miss: Any = None
    n_mix: Any = None

    # Site masks
    ref_masks: Any = None
    core_masks: Any = None
    const_masks: Any = None

    # Variant distances
    dist: Any = None

    # BED mask (per contig; list aligned to contigs/stacks)
    bed_mask: Any = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


import logging
import sys

NAME_COLORS = {
    "polycore.cli":   "\033[37m",  # white
    "polycore.core":  "\033[36m",  # cyan
    "polycore.io":    "\033[32m",  # green
    "polycore.utils": "\033[90m",  # gray
}
RESET = "\033[0m"

class NameColorFormatter(logging.Formatter):
    def format(self, record):
        color = NAME_COLORS.get(record.name, "")
        record.colored_name = f"{color}{record.name}{RESET}"
        return super().format(record)

def set_up_logging(level=logging.INFO):
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            NameColorFormatter(
                "%(asctime)s %(levelname)s %(colored_name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)

    root.setLevel(level)


def ambiguity_size(char: str) -> int:
    """Return number of concrete alleles represented by an IUPAC base."""
    return bin(IUPAC_BITS.get((char or "").upper(), 0)).count("1")


def auto_chunk_size(n_samples: int, safety_fraction: float = 0.8, min_chunk: int = 1000) -> int:
    """Compute a conservative chunk size based on available memory."""
    logger = logging.getLogger(__name__)

    if n_samples <= 0:
        logger.warning(f"auto_chunk_size called with n_samples={n_samples}; returning min_chunk={min_chunk}")
        return min_chunk

    avail_bytes = psutil.virtual_memory().available
    max_bytes = int(avail_bytes * safety_fraction)

    # Heuristic: operations scale ~ n_samples^2 per site
    cost_per_site = n_samples * n_samples
    chunk = max(min_chunk, int(max_bytes / max(cost_per_site, 1)))

    logger.info(
        f"Auto chunk size: n_samples={n_samples} avail={avail_bytes / (1024**3):.2f}GB "
        f"safety={safety_fraction:.2f} -> chunk={chunk}"
    )
    return chunk


def build_config(args) -> Config:
    """Build Config from argparse args."""

    print(args.snippy, args.split, args.ref_by_name)
    cfg = Config(
        min_gf=args.min_gf,
        min_cf=args.min_cf,
        progressive=args.progressive,
        ploidy=args.ploidy,
        chunk_size=args.chunk_size,
        split=args.snippy or args.split,
        ref_by_name=args.snippy or args.ref_by_name,
        bed_file=getattr(args, "mask", None),
    )

    logging.getLogger(__name__).debug(f"Config built: {cfg}")
    return cfg


def log_state_summary(logger: logging.Logger, state: WorkflowState) -> None:
    """Log lightweight summary fields; avoid dumping large arrays."""
    files = state.files or []
    logger.info(f"Inputs: {len(files)} file(s)")

    if state.names is not None:
        logger.info(f"Samples: {len(state.names)}")

    if state.contigs is not None:
        logger.info(f"Contigs: {len(state.contigs)}")

    if state.ploidy is not None:
        logger.info(f"Ploidy: {state.ploidy}")

    if state.stacks is not None:
        try:
            shapes = [tuple(getattr(st, "shape", (None, None))) for st in state.stacks[:3]]
            suffix = "" if len(state.stacks) <= 3 else " (showing first 3)"
            logger.info(f"Stacks: {len(state.stacks)} contig stack(s){suffix} shapes={shapes}")
        except Exception:
            logger.debug("Stacks present but could not summarize shapes")


def collapse_sequences(
    sequences: List[List[str]], names: List[str]
) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    Collapse identical sequences (string-equal) while preserving reference at index 0.

    Returns:
      - unique_sequences: list[str] (sequence strings)
      - idx_map: dict[rep_index] -> list[original_indices]
    """
    logger = logging.getLogger(__name__)

    if not sequences or not names:
        raise ValueError("collapse_sequences: 'sequences' and 'names' must be non-empty")

    if len(sequences) != len(names):
        raise ValueError(f"collapse_sequences: sequences ({len(sequences)}) != names ({len(names)})")

    seen: Dict[str, int] = {}
    idx_map: Dict[int, List[int]] = {}
    unique_sequences: List[str] = []

    # Always keep the reference (index 0)
    ref_seq = "".join(sequences[0])
    unique_sequences.append(ref_seq)
    seen[ref_seq] = 0
    idx_map[0] = [0]

    n_collapsed_groups = 0
    n_collapsed_total = 0

    for i, seq_list in enumerate(sequences[1:], start=1):
        seq = "".join(seq_list)
        rep_idx = seen.get(seq)

        if rep_idx is not None:
            idx_map[rep_idx].append(i)
            n_collapsed_total += 1
        else:
            rep_idx = len(unique_sequences)
            seen[seq] = rep_idx
            unique_sequences.append(seq)
            idx_map[rep_idx] = [i]

    # Log summary + a few examples (avoid huge logs)
    for rep_idx, orig_idxs in idx_map.items():
        if len(orig_idxs) > 1:
            n_collapsed_groups += 1
            if n_collapsed_groups <= 10:
                group_names = [names[j] for j in orig_idxs]
                logger.info(f"Identical samples collapsed: rep={names[rep_idx]} <- {', '.join(group_names)}")

    if n_collapsed_total > 0:
        logger.info(
            f"Collapsed identical sequences: {n_collapsed_total} duplicate sample(s) "
            f"into {n_collapsed_groups} group(s) (ref preserved)"
        )
    else:
        logger.info("No identical sequences detected; no collapsing performed.")

    return unique_sequences, idx_map


def expand_distances(diffs, filter_map, idx_map, orig_names, names_filt):
    """
    Expand a rep-level distance matrix to original samples, but ONLY include
    original samples whose name is in `names_filt`, and only from reps where
    filter_map[rep] is True.

    Returns:
        expanded (n x n): expanded distance matrix
        expanded_names (len n): names corresponding to expanded rows/cols
    """
    names_filt_set = set(names_filt)

    # Reps kept by filter_map, in rep-index order (aligns with diffs a,b)
    kept_reps = [r for r, keep in enumerate(filter_map) if keep]

    # For each kept rep, keep only the original indices whose name is allowed
    groups = [[i for i in idx_map[rep] if orig_names[i] in names_filt_set] for rep in kept_reps]

    # Flatten groups to names (canonical expanded order)
    expanded_names = [orig_names[i] for g in groups for i in g]

    # Allocate and fill by blocks using FILTERED group sizes
    n = len(expanded_names)
    expanded = np.zeros((n, n), dtype=diffs.dtype)

    r0 = 0
    for a, ga in enumerate(groups):
        ra = len(ga)
        c0 = 0
        for b, gb in enumerate(groups):
            rb = len(gb)
            if ra and rb:
                block = expanded[r0:r0+ra, c0:c0+rb]
                block[:] = diffs[a, b]
                if a == b:
                    d = np.arange(ra)
                    block[d, d] = 0
            c0 += rb
        r0 += ra

    return expanded, expanded_names


def to_bits(sequences: np.ndarray, lut) -> np.ndarray:
    """Convert sequence array to bit-encoded representation using lookup table."""

    logger = logging.getLogger(__name__)
    
    logger.info("Converting variants to bits")

    up = np.char.upper(sequences)
    codes_obj = np.frompyfunc(ord, 1, 1)(up)
    codes_int = codes_obj.astype(np.int32)

    if np.any(codes_int > 127):
        bad_chars = np.unique(up[codes_int > 127])
        bad_desc = ", ".join(f"{repr(c)} (U+{ord(c):04X})" for c in bad_chars)
        raise ValueError(
            f"Non-ASCII character(s) detected in sequences: {bad_desc}. "
            "Please sanitize the FASTA to ASCII IUPAC letters (A,C,G,T and ambiguity codes) and '-' only."
        )

    codes = codes_int.astype(np.uint8)
    return lut[codes]


def _mask_to_counts(mask: int, ploidy: int, denom: int = COUNT_DENOM) -> np.ndarray:
    """Length-4 allele count vector, scaled by `denom`, splitting copies
    uniformly across the alleles present in the mask."""
    k = int(POPCOUNT16[mask])
    counts = np.zeros(4, dtype=np.int64)
    if k == 0:
        return counts
    per = (denom * ploidy) // k          # exact: k divides denom
    for i, bit in enumerate(ALLELES):
        if mask & bit:
            counts[i] = per
    return counts


def build_match_table(ploidy: int, denom: int = COUNT_DENOM) -> np.ndarray:
    """table[m1, m2] = matching copies, scaled by `denom`."""
    table = np.zeros((16, 16), dtype=np.int64)   # was uint8; scaled values overflow it
    cnt = [_mask_to_counts(m, ploidy, denom) for m in range(16)]
    for m1 in range(16):
        for m2 in range(16):
            table[m1, m2] = np.minimum(cnt[m1], cnt[m2]).sum()
    return table


def calculate_distances(bits: np.ndarray, ploidy: int, chunk_size: int) -> np.ndarray:
    """
    Calculate pairwise distances from bit-encoded sequences.

    Args:
        bits: (n_samples, n_sites) uint8 bitmasks (0=unknown, A=1, C=2, G=4, T=8, combos via OR)
        ploidy: ploidy level
        chunk_size: process sites in chunks of this size

    Returns:
        (n_samples, n_samples) distance matrix
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Calculating pairwise distances in {chunk_size} bp chunks")

    n, L = bits.shape
    diffs = np.zeros((n, n), dtype=np.int64)
    match_table = build_match_table(ploidy)
    scaled_ploidy = COUNT_DENOM * ploidy

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk = bits[:, start:end]
        logger.info(f"  Processing sites {start}-{end}")

        for i in range(n):
            bi = chunk[i]
            for j in range(i, n):
                bj = chunk[j]

                # Only compare where both known
                both_known = (bi > 0) & (bj > 0)

                if not np.any(both_known):
                    continue

                # Matches per site from lookup; vectorized gather
                matches = match_table[bi[both_known], bj[both_known]]

                # Mismatches per site = ploidy - matches
                d = (scaled_ploidy - matches).sum()

                diffs[i, j] += d
                if i != j:                 # don't double-count the self pair
                    diffs[j, i] += d

    return diffs