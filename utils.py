import copy
import os
import sys

import torch
from omegaconf import OmegaConf
from torch_geometric.utils import to_dense_adj, to_dense_batch


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool, device=E.device).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


class PlaceHolder:
    def __init__(self, X, E, y=None, pos=None):
        self.X = X
        self.E = E
        self.y = y
        self.pos = pos

    def type_as(self, x):
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        if self.pos is not None:
            self.pos = self.pos.type_as(x)
        return self

    def to(self, device):
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.pos is not None:
            self.pos = self.pos.to(device)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        edge_mask = e_mask1 * e_mask2
        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = -1
            self.E[edge_mask.squeeze(-1) == 0] = -1
            if self.pos is not None:
                self.pos = torch.argmax(self.pos, dim=-1)
                self.pos[edge_mask.squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * edge_mask
            if self.pos is not None:
                self.pos = self.pos * edge_mask
        return self


def to_dense(x, edge_index, edge_attr, batch, max_num_nodes=None, y=None, edge_pos=None):
    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
    if max_num_nodes is None:
        max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)
    pos = None
    if edge_pos is not None:
        pos = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_pos, max_num_nodes=max_num_nodes)
        pos = encode_no_edge(pos)
    dense_data = PlaceHolder(X=X, E=E, y=y, pos=pos)
    return dense_data, node_mask


def batch_to_dense_graph(data, x_classes, e_classes, pos_classes, max_n_nodes):
    data_x = torch.nn.functional.one_hot(data.x, num_classes=x_classes).float()
    data_edge_attr = torch.nn.functional.one_hot(data.edge_attr, num_classes=e_classes).float()
    data_edge_pos = torch.nn.functional.one_hot(data.edge_pos, num_classes=pos_classes).float()
    dense_data, node_mask = to_dense(
        data_x,
        data.edge_index,
        data_edge_attr,
        data.batch,
        max_n_nodes,
        edge_pos=data_edge_pos,
    )
    return dense_data.mask(node_mask), node_mask


def update_config_with_new_keys(cfg, saved_cfg):
    cfg_container = OmegaConf.to_container(cfg, resolve=False) if not isinstance(cfg, dict) else copy.deepcopy(cfg)
    saved_container = OmegaConf.to_container(saved_cfg, resolve=False) if not isinstance(saved_cfg, dict) else copy.deepcopy(saved_cfg)

    def merge(dst, src):
        for key, value in src.items():
            if key not in dst:
                dst[key] = value
                continue
            if isinstance(dst[key], dict) and isinstance(value, dict):
                merge(dst[key], value)

    merge(cfg_container, saved_container)
    return OmegaConf.create(cfg_container)


def get_data_root(data_dir=None):
    if data_dir is not None and str(data_dir).strip():
        return str(data_dir).strip()
    env_data_dir = os.environ.get("GRAPHFDM_DATA_ROOT")
    if env_data_dir:
        return env_data_dir
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def get_dataset_root(data_family, data_dir=None):
    return os.path.join(get_data_root(data_dir), str(data_family).strip())


def get_evaluator_root(data_dir=None):
    return os.path.join(get_data_root(data_dir), "evaluator")


def get_tokenizer_root(data_family, data_dir=None, tokenizer_dir=None):
    if tokenizer_dir is not None and str(tokenizer_dir).strip():
        return os.path.abspath(str(tokenizer_dir).strip())
    return os.path.join(get_dataset_root(data_family, data_dir), "tokenizer", "vocab1000ring80")


def load_motif_tokenizer(data_family, data_dir=None, tokenizer_dir=None):
    from tokenizer.graph_bpe import MolecularGraphTokenizer

    tokenizer = MolecularGraphTokenizer(kekulize=True, name="pretrain", simple_mode=False)
    token_path = os.path.join(get_tokenizer_root(data_family, data_dir, tokenizer_dir), "pretrain-token")
    if not os.path.exists(f"{token_path}.node"):
        raise FileNotFoundError(f"Tokenizer assets not found: {token_path}")
    tokenizer.load(token_path)
    return tokenizer
