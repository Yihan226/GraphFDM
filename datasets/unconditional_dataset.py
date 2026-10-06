import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import utils
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from diffusion.distributions import DistributionNodes


MOTIF_META_VERSION = 1
MOTIF_DATA_VERSION = 3


def get_meta_path(dataset_root, task):
    return os.path.join(dataset_root, "raw", f"{task}.motif.meta.json")


def load_or_build_meta(dataset_root, task, tokenizer, tokenizer_root, test_index):
    meta_filename = get_meta_path(dataset_root, task)
    data_root = os.path.join(dataset_root, "raw")
    if os.path.exists(meta_filename):
        with open(meta_filename, "r") as f:
            meta_dict = json.load(f)
        if meta_dict.get("schema_version") == MOTIF_META_VERSION and meta_dict.get("tokenizer_root") == tokenizer_root:
            print(f"motif meta cache hit: {meta_filename}")
            return meta_dict
        print(f"motif meta cache invalidated: {meta_filename}")
    meta_dict = compute_motif_meta(data_root, task, tokenizer, tokenizer_root, test_index)
    with open(meta_filename, "w") as f:
        json.dump(meta_dict, f)
    return meta_dict


def normalize_dist(values):
    total = float(np.sum(values))
    if total <= 0:
        out = np.zeros_like(values, dtype=np.float64)
        out[0] = 1.0
        return out.tolist()
    return (values / total).tolist()


def normalize_transition(values):
    denom = values.sum(axis=-1, keepdims=True)
    denom[denom == 0] = 1
    return (values / denom).tolist()


def compute_motif_meta(root, source_name, tokenizer, tokenizer_root, test_index):
    filename = f"{source_name}.csv.gz"
    if not os.path.exists(os.path.join(root, filename)):
        filename = f"{source_name}.csv"
    df = pd.read_csv(os.path.join(root, filename))
    all_index = list(range(len(df)))
    non_test_index = list(set(all_index) - set(test_index))
    df = df.iloc[non_test_index]
    smiles_list = df["smiles"].tolist()

    max_node_type = tokenizer.max_node_type + 1
    max_bond_type = tokenizer.max_bond_type + 1
    max_position_type = tokenizer.max_atom_in_token + 1

    node_count_list = np.zeros(max_node_type, dtype=np.float64)
    edge_count_list = np.zeros(max_bond_type, dtype=np.float64)
    pos_count_list = np.zeros(max_position_type, dtype=np.float64)
    transition_E = np.zeros((max_node_type, max_node_type, max_bond_type), dtype=np.float64)
    node_num_frequency = np.zeros(512, dtype=np.float64)
    max_node = 1

    for smiles in smiles_list:
        try:
            node_types, adj_combined = tokenizer.encode(smiles, update_vocab_edge=False)
            bond_adj, pos_adj = tokenizer.get_bond_position_by_vocab(adj_combined)
        except Exception:
            continue

        node_types = np.asarray(node_types, dtype=np.int64)
        bond_adj = np.asarray(bond_adj, dtype=np.int64) + 1
        pos_adj = np.asarray(pos_adj, dtype=np.int64) + 1
        n = len(node_types)
        if n == 0:
            continue
        max_node = max(max_node, n)
        if n >= len(node_num_frequency):
            node_num_frequency = np.pad(node_num_frequency, (0, n - len(node_num_frequency) + 32))
        node_num_frequency[n] += 1

        for node in node_types:
            node_count_list[node] += 1

        rows, cols = np.nonzero(bond_adj)
        for row, col in zip(rows.tolist(), cols.tolist()):
            edge = int(bond_adj[row, col])
            pos = int(pos_adj[row, col])
            edge_count_list[edge] += 1
            pos_count_list[pos] += 1
            transition_E[node_types[row], node_types[col], edge] += 1

        node_counts = np.bincount(node_types, minlength=max_node_type)
        potential_edges = node_counts.reshape(-1, 1) * node_counts.reshape(1, -1)
        potential_edges = potential_edges - np.diag(node_counts)
        transition_E[:, :, 0] += np.maximum(potential_edges * 2 - transition_E[:, :, 1:].sum(axis=-1), 0)
        null_edge_count = n * n - n - len(rows)
        edge_count_list[0] += null_edge_count
        pos_count_list[0] += null_edge_count

    node_type_dist = normalize_dist(node_count_list)
    bond_type_dist = normalize_dist(edge_count_list)
    pos_type_dist = normalize_dist(pos_count_list)
    transition_E = normalize_transition(transition_E)
    n_nodes = normalize_dist(node_num_frequency[: max_node + 1])

    return {
        "schema_version": MOTIF_META_VERSION,
        "tokenizer_root": tokenizer_root,
        "max_node_type": max_node_type,
        "max_bond_type": max_bond_type,
        "max_position_type": max_position_type,
        "max_node": max_node,
        "node_type_dist": node_type_dist,
        "bond_type_dist": bond_type_dist,
        "pos_type_dist": pos_type_dist,
        "transition_E": transition_E,
        "n_nodes": n_nodes,
    }


class MotifDataset(InMemoryDataset):
    def __init__(self, source, root, tokenizer, transform=None, pre_transform=None, pre_filter=None):
        self.source = source
        self.tokenizer = tokenizer
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        for filename in [f"{self.source}.csv.gz", f"{self.source}.csv"]:
            if os.path.exists(os.path.join(self.raw_dir, filename)):
                return [filename]
        return [f"{self.source}.csv.gz"]

    @property
    def processed_file_names(self):
        return [f"{self.source}.uncond.motif.v{MOTIF_DATA_VERSION}.pt"]

    def process(self):
        RDLogger.DisableLog("rdApp.*")
        data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        data_df = pd.read_csv(data_path)
        data_list = []
        with tqdm(total=len(data_df)) as pbar:
            for i, (_, row) in enumerate(data_df.iterrows()):
                smiles = row["smiles"]
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
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    edge_pos=edge_pos,
                    y=torch.zeros((1, 0), dtype=torch.float32),
                    y_raw=torch.zeros((1, 0), dtype=torch.float32),
                    idx=i,
                )
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                pbar.update(1)
        torch.save(self.collate(data_list), self.processed_paths[0])


class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.data_family = str(cfg.dataset.data).strip()
        self.data_dir = getattr(cfg.dataset, "data_dir", None)
        self.tokenizer_dir = getattr(cfg.dataset, "tokenizer_dir", None)
        self.root_path = utils.get_dataset_root(self.data_family, self.data_dir)
        self.pretrain_data_source = cfg.dataset.pretrain_data_source
        self.tokenizer_root = utils.get_tokenizer_root(self.data_family, self.data_dir, self.tokenizer_dir)
        self.tokenizer = utils.load_motif_tokenizer(self.data_family, self.data_dir, self.tokenizer_dir)
        self.base_example_dataset = None
        self.condition_dim = 0
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset_full = None
        self.training_iterations = 0

    def prepare_data(self):
        dataset = MotifDataset(
            source=self.pretrain_data_source,
            root=self.root_path,
            tokenizer=self.tokenizer,
            transform=None,
        )
        full_index = torch.LongTensor(list(range(len(dataset))))
        train_dataset = dataset[full_index]
        empty_dataset = dataset[torch.LongTensor([])]
        self.base_example_dataset = train_dataset
        self.train_dataset_full = train_dataset
        self.pt_train_index = list(range(len(dataset)))
        self.pt_val_index = list(range(len(dataset)))
        self.pt_test_index = []
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            pin_memory=self.cfg.dataset.pin_memory,
        )
        self.val_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        self.test_loader = DataLoader(
            empty_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        self.training_iterations = max(1, len(train_dataset) // self.cfg.train.batch_size)

    def get_train_smiles(self):
        filename = f"{self.pretrain_data_source}.csv.gz"
        if not os.path.exists(f"{self.root_path}/raw/{filename}"):
            filename = f"{self.pretrain_data_source}.csv"
        full_df = pd.read_csv(f"{self.root_path}/raw/{filename}")
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in full_df["smiles"].tolist()]
        return smiles_list, smiles_list

    def get_data_split(self):
        filename = f"{self.pretrain_data_source}.csv.gz"
        if not os.path.exists(f"{self.root_path}/raw/{filename}"):
            filename = f"{self.pretrain_data_source}.csv"
        df = pd.read_csv(f"{self.root_path}/raw/{filename}")
        return df, df, df.iloc[[]]

    def example_batch(self):
        loader = DataLoader(
            self.train_dataset_full,
            batch_size=min(self.cfg.train.batch_size, len(self.train_dataset_full)),
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
        self.pretrain_data_source = cfg.dataset.pretrain_data_source
        self.data_family = str(cfg.dataset.data).strip()
        self.meta_source = getattr(cfg.dataset, "meta_source", self.pretrain_data_source)
        self.condition_mode = "none"
        self.task = self.pretrain_data_source
        self.objective_names = []
        self.objective_weights = []
        self.normalization_stats = {}
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
