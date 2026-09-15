import sys
import time
import argparse
import logging

from .io import load_sequences, stacks_to_fasta, stacks_to_csv, write_summary
from .utils import (
    set_up_logging,
    IUPAC_BITS,
    Config,
    WorkflowState,
    build_config
)
from .core import (
    set_ploidy,
    find_core,
    find_const,
    find_diffs,
    filter_samples,
    calculate_genome_fraction,
    expand_stacks,
    find_valid_sites,
    build_stacks,
)

__version__ = "1.0.0"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PolyCore - Core genome analysis on polyploid organisms"
    )
    p.add_argument("input", nargs="+", help="Input sequences")
    p.add_argument("--ref", help="Reference FASTA file")
    p.add_argument("--mask", help="Bed file with coordinates of sites to exclude.")

    p.add_argument("--min-gf", type=float, default=0.9, help="Minimum genome fraction per input")
    p.add_argument("--min-cf", type=float, default=0.95, help="Minimum fraction with valid data per site")

    p.add_argument("--progressive", action="store_true")
    p.add_argument("--ploidy", type=int)
    p.add_argument("--chunk-size", type=int, help="Sites per chunk for pairwise diffs (controls memory)")

    p.add_argument("--split", action="store_true",
                   help="Treat each contig in a multi-fasta file as a separate sample.")
    p.add_argument("--ref-by-name", action="store_true",
                   help="Treat file/contig labeled 'reference' as the reference (case-insensitive).")
    p.add_argument("--snippy", action="store_true",
                   help="Shortcut for Snippy *.full.aln input (sets split and ref-by-name).")

    p.add_argument("--version", action="version", version=__version__)
    return p


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv

    set_up_logging()
    logger = logging.getLogger(__name__)

    args = build_parser().parse_args(argv)
    start = time.time()

    logger.info("PolyCore v%s starting", __version__)

    state = WorkflowState()
    state.cfg = build_config(args)

    # Resolve input list once
    state.files = ([args.ref] + args.input) if args.ref else list(args.input)

    if len(state.files) == 1 and not state.cfg.split:
        logger.error("Only one FASTA file provided. Re-run with the '--split' parameter to treat each contig as a separate sample.")
        exit()

    try:
        # ---- Load / encode -------------------------------------------------
        t0 = time.time()
        logger.info("Loading sequences...")
        load_sequences(state)
        logger.info("Loaded sequences in %.2fs", time.time() - t0)

        t0 = time.time()
        logger.info("Setting ploidy / IUPAC encoding...")
        set_ploidy(state, IUPAC_BITS)
        logger.info("Encoding ready in %.2fs", time.time() - t0)

        # ---- Build stacks --------------------------------------------------
        t0 = time.time()
        logger.info("Building stacks...")
        build_stacks(state)
        logger.info("Stacks built in %.2fs", time.time() - t0)

        # ---- Valid sites + genome fraction --------------------------------
        t0 = time.time()
        logger.info("Finding valid sites...")
        find_valid_sites(state)
        logger.info("Valid sites computed in %.2fs", time.time() - t0)

        t0 = time.time()
        logger.info("Calculating genome fraction...")
        calculate_genome_fraction(state)
        logger.info("Genome fraction computed in %.2fs", time.time() - t0)

        # ---- Filter + core -------------------------------------------------
        t0 = time.time()
        logger.info("Filtering samples (min_gf=%.3f)...", state.cfg.min_gf)
        filter_samples(state)
        logger.info("Filtering done in %.2fs", time.time() - t0)

        t0 = time.time()
        logger.info("Finding core sites (min_cf=%.3f, progressive=%s)...",
                    state.cfg.min_cf, state.cfg.progressive)
        find_core(state)
        logger.info("Core computed in %.2fs", time.time() - t0)

        # ---- Const + diffs -------------------------------------------------
        t0 = time.time()
        logger.info("Finding constant sites...")
        find_const(state)
        logger.info("Constant sites computed in %.2fs", time.time() - t0)

        t0 = time.time()
        logger.info("Finding differences...")
        find_diffs(state)
        logger.info("Differences computed in %.2fs", time.time() - t0)

        # ---- Expand + outputs ---------------------------------------------
        t0 = time.time()
        logger.info("Expanding stacks for output alignments...")
        expand_stacks(state)
        logger.info("Expanded stacks in %.2fs", time.time() - t0)

        logger.info("Writing outputs...")
        stacks_to_fasta(state.full_mask_stacks, "core.full.aln")
        stacks_to_fasta(state.core_mask_stacks, "core.aln")
        stacks_to_csv(state)

        write_summary(state)

        logger.info("Completed successfully in %.1fs", time.time() - start)

    except Exception:
        logger.exception("Failed after %.1fs", time.time() - start)
        sys.exit(1)
