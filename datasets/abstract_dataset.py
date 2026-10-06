import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from diffusion.distributions import DistributionNodes


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None

    def prepare_data(self, datasets):
        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers
        self.dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle="debug" not in self.cfg.general.name,
            )
            for split, dataset in datasets.items()
        }

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders["train"][idx]


class MolecularDataModule(AbstractDataModule):
    pass


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule):
        example_batch = datamodule.example_batch()
        example_batch_x = torch.nn.functional.one_hot(
            example_batch.x,
            num_classes=self.max_node_type,
        ).float()
        example_batch_edge_attr = torch.nn.functional.one_hot(
            example_batch.edge_attr,
            num_classes=self.max_bond_type,
        ).float()
        example_batch_edge_pos = torch.nn.functional.one_hot(
            example_batch.edge_pos,
            num_classes=self.max_position_type,
        ).float()
        y_dim = example_batch["y"].size(1)
        if y_dim == 0:
            y_dim = getattr(datamodule, "condition_dim", 0)

        self.input_dims = {
            "X": example_batch_x.size(1),
            "E": example_batch_edge_attr.size(1),
            "pos": example_batch_edge_pos.size(1),
            "y": y_dim,
        }
        self.output_dims = {
            "X": example_batch_x.size(1),
            "E": example_batch_edge_attr.size(1),
            "pos": example_batch_edge_pos.size(1),
            "y": y_dim,
        }
