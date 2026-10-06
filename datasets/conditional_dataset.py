import json
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import utils
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.unconditional_dataset import load_or_build_meta
from diffusion.distributions import DistributionNodes
from shared import (
    get_normalization_stats,
    get_normalization_stats_path,
    get_objective_names,
    get_objective_stem,
    get_objective_weights,
    get_value_normalization_mode,
    normalize_objective_value,
    save_normalization_stats,
)


MOTIF_DATA_VERSION = 3


def canonicalize_smiles(smiles):
    smiles = "" if pd.isna(smiles) else str(smiles).strip()
    if not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def resolve_csv_path(root_dir, stem):
    stem = str(stem).strip()
    candidate_stems = []
    for item in [stem, stem.replace("_", "-"), stem.replace("-", "_")]:
        if item and item not in candidate_stems:
            candidate_stems.append(item)
    for candidate in candidate_stems:
        for suffix in [".csv", ".csv.gz"]:
            path = os.path.join(root_dir, f"{candidate}{suffix}")
            if os.path.exists(path):
                return path
    return None


def resolve_stats_path(cfg, dataset_root):
    stats_path = get_normalization_stats_path(cfg)
    if stats_path:
        return stats_path
    return os.path.join(dataset_root, "raw", f"{get_objective_stem(cfg)}.stats.json")


def _resolve_required_columns(df, objective_names, *, require_smiles):
    columns_by_key = {str(col).strip().casefold(): col for col in df.columns}
    resolved = {}
    if require_smiles:
        smiles_col = columns_by_key.get("smiles")
        if smiles_col is None:
            raise ValueError("Multiobjective CSV must contain a smiles column.")
        resolved["smiles"] = smiles_col
    for name in objective_names:
        col = columns_by_key.get(str(name).strip().casefold())
        if col is None:
            raise ValueError(f"Multiobjective CSV is missing required objective column {name!r}.")
        resolved[name] = col
    return resolved


def normalize_multiobjective_dataframe(df, objective_names, *, require_smiles=True):
    resolved = _resolve_required_columns(df, objective_names, require_smiles=require_smiles)
    output_columns = []
    if require_smiles:
        output_columns.append("smiles")
    out = pd.DataFrame()
    if require_smiles:
        out["smiles"] = df[resolved["smiles"]].astype(str)
    for name in objective_names:
        values = pd.to_numeric(df[resolved[name]], errors="coerce")
        if values.isna().any():
            bad_count = int(values.isna().sum())
            raise ValueError(f"Objective column {name!r} contains {bad_count} missing or non-numeric values.")
        out[name] = values.astype(float)
        output_columns.append(name)
    return out[output_columns].reset_index(drop=True)


def load_multiobjective_dataframe(dataset_root, cfg, *, require_smiles=True):
    raw_root = os.path.join(dataset_root, "raw")
    source_path = resolve_csv_path(raw_root, get_objective_stem(cfg))
    if source_path is None:
        raise FileNotFoundError(
            f"Could not resolve multiobjective CSV {get_objective_stem(cfg)!r} under {raw_root}; "
            "expected .csv or .csv.gz."
        )
    df = pd.read_csv(source_path)
    return normalize_multiobjective_dataframe(df, get_objective_names(cfg), require_smiles=require_smiles), source_path


def _validate_split_fractions(train_fraction, val_fraction, test_fraction):
    fractions = [float(train_fraction), float(val_fraction), float(test_fraction)]
    if fractions[0] <= 0 or fractions[1] < 0 or fractions[2] < 0:
        raise ValueError("Split fractions require train > 0 and val/test >= 0.")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    return fractions


def _split_key_list(keys, seed, train_fraction=0.8, val_fraction=0.1, test_fraction=0.1):
    keys = list(keys)
    train_fraction, val_fraction, test_fraction = _validate_split_fractions(
        train_fraction,
        val_fraction,
        test_fraction,
    )
    if len(keys) == 0:
        return [], [], []
    rng = np.random.RandomState(seed)
    perm = list(rng.permutation(len(keys)))
    keys = [keys[i] for i in perm]
    n = len(keys)
    if n == 1:
        return keys, [], []
    if n == 2:
        return keys[:1], keys[1:], []
    n_test = max(1, int(round(n * test_fraction))) if test_fraction > 0 else 0
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        if n_val + n_test > n - 1:
            if n_val >= n_test:
                n_val = max(0, n - 1 - n_test)
            else:
                n_test = max(0, n - 1 - n_val)
    train_keys = keys[:n_train]
    val_keys = keys[n_train : n_train + n_val]
    test_keys = keys[n_train + n_val : n_train + n_val + n_test]
    return train_keys, val_keys, test_keys


def _load_or_create_split_keys(
    unique_keys,
    seed,
    train_fraction,
    val_fraction,
    test_fraction,
    manifest_path,
):
    fractions = _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = {
            "seed": int(seed),
            "train_fraction": fractions[0],
            "val_fraction": fractions[1],
            "test_fraction": fractions[2],
        }
        for key, value in expected.items():
            if not np.isclose(float(manifest[key]), float(value)):
                raise ValueError(f"Split manifest {key} does not match the requested split.")
        split_keys = [manifest["train"], manifest["val"], manifest["test"]]
    else:
        split_keys = _split_key_list(unique_keys, seed, *fractions)
        if manifest_path:
            manifest_path = os.path.abspath(manifest_path)
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "seed": int(seed),
                        "train_fraction": fractions[0],
                        "val_fraction": fractions[1],
                        "test_fraction": fractions[2],
                        "train": split_keys[0],
                        "val": split_keys[1],
                        "test": split_keys[2],
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
    flattened = [key for split in split_keys for key in split]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(unique_keys):
        raise ValueError("Split manifest does not partition the current canonical SMILES set exactly.")
    return split_keys


def split_multiobjective_dataframe(
    df,
    seed=42,
    train_fraction=0.8,
    val_fraction=0.1,
    test_fraction=0.1,
    manifest_path=None,
):
    df = df.copy()
    df["_smiles_key"] = df["smiles"].map(canonicalize_smiles)
    unique_keys = list(dict.fromkeys(df["_smiles_key"].tolist()))
    train_keys, val_keys, test_keys = _load_or_create_split_keys(
        unique_keys,
        seed,
        train_fraction,
        val_fraction,
        test_fraction,
        manifest_path,
    )
    train_set = set(train_keys)
    val_set = set(val_keys)
    test_set = set(test_keys)
    columns = [col for col in df.columns if not col.startswith("_")]
    return (
        df[df["_smiles_key"].isin(train_set)][columns].reset_index(drop=True),
        df[df["_smiles_key"].isin(val_set)][columns].reset_index(drop=True),
        df[df["_smiles_key"].isin(test_set)][columns].reset_index(drop=True),
    )


def split_multiobjective_dataframe_from_config(df, cfg):
    dataset_cfg = cfg.dataset
    return split_multiobjective_dataframe(
        df,
        seed=int(getattr(dataset_cfg, "split_seed", 42)),
        train_fraction=float(getattr(dataset_cfg, "train_fraction", 0.8)),
        val_fraction=float(getattr(dataset_cfg, "val_fraction", 0.1)),
        test_fraction=float(getattr(dataset_cfg, "test_fraction", 0.1)),
        manifest_path=getattr(dataset_cfg, "split_manifest_path", None),
    )


def compute_train_stats(train_df, objective_names):
    stats = {}
    if len(train_df) == 0:
        return stats
    for name in objective_names:
        values = train_df[name].astype(float).to_numpy()
        stats[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "count": int(len(values)),
        }
    return stats


def maybe_build_and_load_stats(train_df, cfg, dataset_root):
    stats_path = resolve_stats_path(cfg, dataset_root)
    stats = get_normalization_stats(cfg)
    if len(stats) == 0:
        stats = compute_train_stats(train_df, get_objective_names(cfg))
        save_normalization_stats(stats_path, stats)
    objectives_cfg = getattr(cfg, "objectives", None)
    if objectives_cfg is not None:
        objectives_cfg.normalization_stats_path = stats_path
    return stats_path, stats


def normalize_objective_vector(row, objective_names, cfg, stats):
    if get_value_normalization_mode(cfg) in {None, "none"}:
        return [float(row[name]) for name in objective_names]
    return [normalize_objective_value(row[name], name, cfg=cfg, stats=stats) for name in objective_names]


class ConditionalDataset(TorchDataset):
    def __init__(self, dataframe, tokenizer, cfg, stats):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.stats = stats
        self.objective_names = list(get_objective_names(cfg))
        self.data_list = self._build()

    def _build(self):
        RDLogger.DisableLog("rdApp.*")
        data_list = []
        for idx, row in self.df.iterrows():
            raw_values = [float(row[name]) for name in self.objective_names]
            norm_values = normalize_objective_vector(row, self.objective_names, self.cfg, self.stats)
            data_list.append(self.smiles_to_graph(str(row["smiles"]), norm_values, raw_values, idx))
        return data_list

    def smiles_to_graph(self, smiles, norm_values, raw_values, idx):
        node_types, adj_combined = self.tokenizer.encode(smiles, update_vocab_edge=False)
        bond_adj, pos_adj = self.tokenizer.get_bond_position_by_vocab(adj_combined)
        bond_adj = np.array(bond_adj, dtype=np.int64) + 1
        pos_adj = np.array(pos_adj, dtype=np.int64) + 1
        rows, cols = np.nonzero(bond_adj)
        edge_index = (
            torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
            if len(rows) > 0
            else torch.zeros((2, 0), dtype=torch.long)
        )
        edge_attr = torch.tensor(bond_adj[rows, cols], dtype=torch.long) if len(rows) > 0 else torch.zeros((0,), dtype=torch.long)
        edge_pos = torch.tensor(pos_adj[rows, cols], dtype=torch.long) if len(rows) > 0 else torch.zeros((0,), dtype=torch.long)
        x = torch.tensor(node_types, dtype=torch.long)
        if x.numel() == 0:
            raise ValueError(f"Tokenizer returned an empty motif graph for row_idx={idx}")
        y = torch.tensor([norm_values], dtype=torch.float32)
        y_raw = torch.tensor([raw_values], dtype=torch.float32)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_pos=edge_pos,
            y=y,
            y_raw=y_raw,
            idx=idx,
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.data_family = str(cfg.dataset.data).strip()
        self.data_dir = getattr(cfg.dataset, "data_dir", None)
        self.tokenizer_dir = getattr(cfg.dataset, "tokenizer_dir", None)
        self.root_path = utils.get_dataset_root(self.data_family, self.data_dir)
        self.objective_stem = get_objective_stem(cfg)
        self.pretrain_data_source = self.objective_stem
        self.meta_source = getattr(cfg.dataset, "meta_source", self.objective_stem)
        self.tokenizer_root = utils.get_tokenizer_root(self.data_family, self.data_dir, self.tokenizer_dir)
        self.tokenizer = utils.load_motif_tokenizer(self.data_family, self.data_dir, self.tokenizer_dir)
        self.objective_names = get_objective_names(cfg)
        self.condition_dim = len(self.objective_names)
        self.base_example_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.training_iterations = 0
        self.stats_path = None
        self.normalization_stats = {}
        self.train_split_df = None
        self.val_split_df = None
        self.test_split_df = None
        self.train_dataset_full = None

    def prepare_data(self):
        full_df, source_path = load_multiobjective_dataframe(self.root_path, self.cfg)
        print(f"[data] multiobjective source: {source_path}")
        train_df, val_df, test_df = split_multiobjective_dataframe_from_config(full_df, self.cfg)
        if len(train_df) == 0:
            raise ValueError(f"No multiobjective training samples were built from source {self.objective_stem!r}.")
        self.stats_path, self.normalization_stats = maybe_build_and_load_stats(train_df, self.cfg, self.root_path)
        train_dataset = ConditionalDataset(train_df, self.tokenizer, self.cfg, self.normalization_stats)
        val_dataset = ConditionalDataset(val_df, self.tokenizer, self.cfg, self.normalization_stats)
        test_dataset = ConditionalDataset(test_df, self.tokenizer, self.cfg, self.normalization_stats)
        self.base_example_dataset = train_dataset
        self.train_dataset_full = train_dataset
        self.train_split_df = train_df
        self.val_split_df = val_df
        self.test_split_df = test_df
        self.pt_train_index = list(range(len(train_df)))
        self.pt_val_index = list(range(len(val_df)))
        self.pt_test_index = list(range(len(test_df)))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            pin_memory=self.cfg.dataset.pin_memory,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        self.training_iterations = max(1, len(train_dataset) // self.cfg.train.batch_size)

    def get_train_smiles(self):
        train_smiles = [canonicalize_smiles(smi) for smi in self.train_split_df["smiles"].tolist()]
        test_smiles = [canonicalize_smiles(smi) for smi in self.test_split_df["smiles"].tolist()]
        return train_smiles, test_smiles

    def get_data_split(self):
        return self.train_split_df, self.val_split_df, self.test_split_df

    def example_batch(self):
        dataset = self.train_dataset_full if len(self.val_loader.dataset) == 0 else self.val_loader.dataset
        loader = DataLoader(
            dataset,
            batch_size=min(self.cfg.train.batch_size, len(dataset)),
            shuffle=False,
        )
        return next(iter(loader))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class DataInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.cfg = cfg
        self.pretrain_data_source = datamodule.objective_stem
        self.data_family = str(cfg.dataset.data).strip()
        self.meta_source = getattr(cfg.dataset, "meta_source", self.pretrain_data_source)
        self.condition_mode = "multiobjective_vector"
        self.task = self.pretrain_data_source
        self.objective_names = list(get_objective_names(cfg))
        self.objective_weights = list(get_objective_weights(cfg))
        self.normalization_stats = datamodule.normalization_stats
        self.task_type = "regression"
        self.ensure_connected = cfg.model.ensure_connected
        self.data_root = datamodule.root_path
        self.data_dir = getattr(datamodule, "data_dir", getattr(cfg.dataset, "data_dir", None))
        self.evaluator_dir = utils.get_evaluator_root(self.data_dir)
        self.tokenizer_root = datamodule.tokenizer_root
        self.tokenizer = datamodule.tokenizer
        meta_dict = load_or_build_meta(self.data_root, self.meta_source, self.tokenizer, self.tokenizer_root, [])
        self.max_node_type = meta_dict["max_node_type"]
        self.max_bond_type = meta_dict["max_bond_type"]
        self.max_position_type = meta_dict["max_position_type"]
        self.max_n_nodes = meta_dict["max_node"]
        self.original_max_n_nodes = self.max_n_nodes
        self.n_nodes = torch.tensor(meta_dict["n_nodes"], dtype=torch.float32)
        self.node_types = torch.tensor(meta_dict["node_type_dist"], dtype=torch.float32)
        self.edge_types = torch.tensor(meta_dict["bond_type_dist"], dtype=torch.float32)
        self.pos_types = torch.tensor(meta_dict["pos_type_dist"], dtype=torch.float32)
        self.transition_E = torch.tensor(meta_dict["transition_E"], dtype=torch.float32)
        self.atom_decoder = list(self.tokenizer.vocab_node)
        self.active_atoms = list(self.tokenizer.vocab_node)
        self.nodes_dist = DistributionNodes(self.n_nodes)
        self.active_index = torch.arange(self.max_node_type)
        self.valency_distribution = torch.zeros(1)
        self.y_prior = None
        self.train_ymin = []
        self.train_ymax = []
