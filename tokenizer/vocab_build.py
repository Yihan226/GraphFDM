import argparse
import os
import pathlib
import sys
from collections import Counter
from functools import partial


CURRENT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[0]

sys.path = [entry for entry in sys.path if pathlib.Path(entry or ".").resolve() != CURRENT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import multiprocessing as mp


def parse_int_list(raw_value, default_values):
    if raw_value is None:
        return list(default_values)
    values = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def load_smiles(csv_path, smiles_col, sample_limit=None, sample_seed=42):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in {csv_path}. Available columns: {list(df.columns)}"
        )

    df = df[df[smiles_col].notna()].copy()
    if sample_limit is not None and sample_limit < len(df):
        df = df.sample(n=sample_limit, random_state=sample_seed)
    smiles_list = df[smiles_col].astype(str).tolist()
    return df, smiles_list


def canonicalize_smiles_list(smiles_list, kekulize):
    from tqdm import tqdm
    from tokenizer.token import canonicalize

    canonical_smiles = []
    invalid_smiles = []
    for smiles in tqdm(smiles_list, desc="Canonicalizing SMILES"):
        try:
            canonical_smiles.append(canonicalize(smiles, kekulize=kekulize))
        except Exception:
            invalid_smiles.append(smiles)
    return canonical_smiles, invalid_smiles


def summarize_atoms(smiles_list):
    from tokenizer.token import ELEMENTS, count_atom

    atom_counter = Counter()
    for smiles in smiles_list:
        atom_counts = count_atom(smiles, return_dict=True)
        for atom, count in atom_counts.items():
            if count > 0:
                atom_counter[atom] += count
    present_elements = [atom for atom in ELEMENTS if atom_counter.get(atom, 0) > 0]
    return atom_counter, present_elements


def collect_molecule_info(smiles_list, kekulize, num_processors):
    from tqdm import tqdm
    from tokenizer.graph_bpe import count_subgraph_frequency, read_single_molecule

    with mp.Pool(num_processors) as pool:
        results = pool.map(
            partial(read_single_molecule, kekulize=kekulize, simple_mode=False),
            tqdm(smiles_list, desc="Reading molecules"),
        )

        molecule_list = []
        ring_counts = Counter()
        for mol, rings in results:
            if mol is None or rings is None:
                continue
            molecule_list.append(mol)
            ring_counts.update(rings)

        subgraph_results = pool.map(
            count_subgraph_frequency,
            tqdm(molecule_list, desc="Counting initial subgraphs"),
        )

    global_freq = Counter()
    for freq_dict, _ in subgraph_results:
        global_freq.update(freq_dict)
    return molecule_list, ring_counts, global_freq


def format_top_items(counter_obj, limit):
    return counter_obj.most_common(limit)


def print_analysis_report(
    total_rows,
    canonical_smiles,
    invalid_smiles,
    atom_counter,
    present_elements,
    ring_counts,
    global_freq,
    candidate_vocab_lens,
    candidate_ring_lens,
    top_k,
):
    print("\n=== Dataset Summary ===")
    print(f"Total rows read: {total_rows}")
    print(f"Valid canonical SMILES: {len(canonical_smiles)}")
    print(f"Invalid SMILES skipped: {len(invalid_smiles)}")
    print(f"Unique canonical SMILES: {len(set(canonical_smiles))}")

    print("\n=== Atom Summary ===")
    print(f"Elements present: {len(present_elements)}")
    for atom, count in format_top_items(atom_counter, top_k):
        print(f"{atom}\t{count}")

    sorted_rings = ring_counts.most_common()
    total_ring_occurrences = sum(ring_counts.values())
    print("\n=== Ring Summary ===")
    print(f"Unique ring systems: {len(sorted_rings)}")
    print(f"Total ring occurrences: {total_ring_occurrences}")
    for ring_smiles, count in sorted_rings[:top_k]:
        print(f"{ring_smiles}\t{count}")

    print("\n=== Candidate Ring Settings ===")
    if total_ring_occurrences == 0:
        print("No ring systems detected in this dataset.")
    else:
        for ring_len in candidate_ring_lens:
            selected_count = min(ring_len, len(sorted_rings))
            used_ring_count = sum(count for _, count in sorted_rings[:selected_count])
            coverage = used_ring_count / total_ring_occurrences
            initial_vocab_size = len(present_elements) + selected_count
            print(
                f"vocab_ring_len={ring_len}\tselected_rings={selected_count}\t"
                f"coverage={coverage:.2%}\tinitial_vocab={initial_vocab_size}"
            )

    print("\n=== Candidate vocab_len / vocab_ring_len Pairs ===")
    for ring_len in candidate_ring_lens:
        selected_count = min(ring_len, len(sorted_rings))
        initial_vocab_size = len(present_elements) + selected_count
        for vocab_len in candidate_vocab_lens:
            extra_budget = max(0, vocab_len - initial_vocab_size)
            status = "ok" if vocab_len >= initial_vocab_size else "too_small"
            print(
                f"vocab_len={vocab_len}\tvocab_ring_len={ring_len}\t"
                f"initial_vocab={initial_vocab_size}\textra_merges={extra_budget}\t{status}"
            )

    print("\n=== Top Initial Merge Candidates ===")
    if len(global_freq) == 0:
        print("No mergeable subgraphs found.")
    else:
        for subgraph_smiles, count in format_top_items(global_freq, top_k):
            print(f"{subgraph_smiles}\t{count}")


def build_and_save_vocab(
    smiles_list,
    kekulize,
    vocab_len,
    vocab_ring_len,
    num_processors,
    out_dir,
    output_name,
):
    from tqdm import tqdm
    from tokenizer.graph_bpe import MolecularGraphTokenizer

    tokenizer = MolecularGraphTokenizer(
        kekulize=kekulize,
        name=output_name,
        simple_mode=False,
    )
    tokenizer.train_node(
        smiles_list=smiles_list,
        vocab_len=vocab_len,
        vocab_ring_len=vocab_ring_len,
        num_processors=num_processors,
    )

    for smiles in tqdm(smiles_list, desc="Building edge vocab"):
        try:
            tokenizer.encode(smiles, update_vocab_edge=True)
        except Exception:
            continue

    os.makedirs(out_dir, exist_ok=True)
    file_prefix = os.path.join(out_dir, output_name)
    tokenizer.save(file_prefix)

    print("\n=== Saved Vocabulary ===")
    print(f"file_prefix: {file_prefix}")
    print(f"node_vocab_size: {len(tokenizer.vocab_node)}")
    print(f"edge_vocab_size: {len(tokenizer.vocab_edge)}")
    print(f"max_atom_in_token: {tokenizer.max_atom_in_token}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze a SMILES dataset and optionally build MolecularGraphTokenizer vocab assets."
    )
    parser.add_argument("--csv-path", required=True, help="Path to .csv or .csv.gz file.")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Randomly sample at most this many rows.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed used with --sample-limit.")
    parser.set_defaults(kekulize=True)
    parser.add_argument(
        "--kekulize",
        dest="kekulize",
        action="store_true",
        help="Canonicalize and tokenize with kekulization.",
    )
    parser.add_argument(
        "--no-kekulize",
        dest="kekulize",
        action="store_false",
        help="Disable kekulization during canonicalization and tokenization.",
    )
    parser.add_argument("--num-processors", type=int, default=max(1, mp.cpu_count() - 3), help="Worker count.")
    parser.add_argument("--top-k", type=int, default=20, help="How many top atoms/rings/subgraphs to print.")
    parser.add_argument(
        "--candidate-vocab-lens",
        default="1000,2000,3000,5000",
        help="Comma-separated vocab_len values to compare in analysis output.",
    )
    parser.add_argument(
        "--candidate-ring-lens",
        default="50,100,300,500",
        help="Comma-separated vocab_ring_len values to compare in analysis output.",
    )
    parser.add_argument("--build", action="store_true", help="Actually train and save tokenizer assets.")
    parser.add_argument("--vocab-len", type=int, default=None, help="Final node vocab size for build mode.")
    parser.add_argument("--vocab-ring-len", type=int, default=None, help="Initial ring vocab size for build mode.")
    parser.add_argument("--out-dir", default=None, help="Directory for saved tokenizer assets in build mode.")
    parser.add_argument("--output-name", default="pretrain-token", help="Saved file prefix name in build mode.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    candidate_vocab_lens = parse_int_list(args.candidate_vocab_lens, [1000, 2000, 3000, 5000])
    candidate_ring_lens = parse_int_list(args.candidate_ring_lens, [50, 100, 300, 500])

    df, raw_smiles = load_smiles(
        csv_path=args.csv_path,
        smiles_col=args.smiles_col,
        sample_limit=args.sample_limit,
        sample_seed=args.sample_seed,
    )
    canonical_smiles, invalid_smiles = canonicalize_smiles_list(raw_smiles, kekulize=args.kekulize)
    atom_counter, present_elements = summarize_atoms(canonical_smiles)
    _, ring_counts, global_freq = collect_molecule_info(
        canonical_smiles,
        kekulize=args.kekulize,
        num_processors=args.num_processors,
    )

    print_analysis_report(
        total_rows=len(df),
        canonical_smiles=canonical_smiles,
        invalid_smiles=invalid_smiles,
        atom_counter=atom_counter,
        present_elements=present_elements,
        ring_counts=ring_counts,
        global_freq=global_freq,
        candidate_vocab_lens=candidate_vocab_lens,
        candidate_ring_lens=candidate_ring_lens,
        top_k=args.top_k,
    )

    if not args.build:
        return

    if args.vocab_len is None or args.vocab_ring_len is None or args.out_dir is None:
        parser.error("--build requires --vocab-len, --vocab-ring-len, and --out-dir")

    build_and_save_vocab(
        smiles_list=canonical_smiles,
        kekulize=args.kekulize,
        vocab_len=args.vocab_len,
        vocab_ring_len=args.vocab_ring_len,
        num_processors=args.num_processors,
        out_dir=args.out_dir,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
