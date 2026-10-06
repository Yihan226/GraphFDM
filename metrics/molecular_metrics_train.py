import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection


class AggregateCrossEntropy:
    def __init__(self):
        self.reset()

    def update(self, preds, target):
        target = target.to(preds.device)
        target = target.reshape(-1, target.shape[-1])
        preds = preds.reshape(-1, preds.shape[-1])
        mask = (target != 0.0).any(dim=-1)
        if mask.sum() == 0:
            return
        labels = torch.argmax(target[mask], dim=-1)
        self.total_ce += float(F.cross_entropy(preds[mask], labels, reduction="sum").detach().item())
        self.total_samples += int(mask.sum().item())

    def compute(self):
        if self.total_samples == 0:
            return float("nan")
        return float(self.total_ce / self.total_samples)

    def reset(self):
        self.total_ce = 0.0
        self.total_samples = 0


class CEPerClass(Metric):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds, target):
        target = target.reshape(-1, target.shape[-1])
        preds = preds.reshape(-1, preds.shape[-1])
        mask = (target != 0.0).any(dim=-1)
        if mask.sum() == 0:
            return
        prob = self.softmax(preds)[..., self.class_id][mask]
        target = target[:, self.class_id][mask]
        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        if self.total_samples == 0:
            return torch.tensor(float("nan"))
        return self.total_ce / self.total_samples


class LabelCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DynamicMetricsCE(MetricCollection):
    def __init__(self, labels):
        metrics = []
        for i, label in enumerate(labels):
            metrics.append(type(f"{label}_CE", (LabelCE,), {})(i))
        super().__init__(metrics)


class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        cfg = getattr(dataset_infos, "cfg", None)
        train_cfg = getattr(cfg, "train", None) if cfg is not None else None
        self.log_per_class = bool(getattr(train_cfg, "log_per_class_metrics", False))
        self.train_atom_metrics = AggregateCrossEntropy()
        self.train_bond_metrics = AggregateCrossEntropy()
        self.train_pos_metrics = AggregateCrossEntropy()
        self.per_class_atom_metrics = None
        self.per_class_bond_metrics = None
        self.per_class_pos_metrics = None
        if self.log_per_class:
            self.per_class_atom_metrics = DynamicMetricsCE(dataset_infos.active_atoms)
            bond_labels = [f"bond_{i}" for i in range(dataset_infos.max_bond_type)]
            pos_labels = [f"pos_{i}" for i in range(dataset_infos.max_position_type)]
            self.per_class_bond_metrics = DynamicMetricsCE(bond_labels)
            self.per_class_pos_metrics = DynamicMetricsCE(pos_labels)

    def forward(self, masked_pred_X, masked_pred_E, masked_pred_pos, true_X, true_E, true_pos, log):
        self.train_atom_metrics.update(masked_pred_X, true_X)
        self.train_bond_metrics.update(masked_pred_E, true_E)
        self.train_pos_metrics.update(masked_pred_pos, true_pos)
        if self.log_per_class:
            self.per_class_atom_metrics(masked_pred_X, true_X)
            self.per_class_bond_metrics(masked_pred_E, true_E)
            self.per_class_pos_metrics(masked_pred_pos, true_pos)

    def reset(self):
        self.train_atom_metrics.reset()
        self.train_bond_metrics.reset()
        self.train_pos_metrics.reset()
        if self.log_per_class:
            self.per_class_atom_metrics.reset()
            self.per_class_bond_metrics.reset()
            self.per_class_pos_metrics.reset()

    def log_epoch_metrics(self, current_epoch, log=True):
        if not log:
            return
        atom_log = self.train_atom_metrics.compute()
        bond_log = self.train_bond_metrics.compute()
        pos_log = self.train_pos_metrics.compute()
        print(f"Epoch {current_epoch}: atom_ce={atom_log:.4f} -- bond_ce={bond_log:.4f} -- pos_ce={pos_log:.4f}")
        if not self.log_per_class:
            return
        atom_metrics = {key: round(val.item(), 4) for key, val in self.per_class_atom_metrics.compute().items()}
        bond_metrics = {key: round(val.item(), 4) for key, val in self.per_class_bond_metrics.compute().items()}
        pos_metrics = {key: round(val.item(), 4) for key, val in self.per_class_pos_metrics.compute().items()}
        print(f"Epoch {current_epoch} per-class: {atom_metrics} -- {bond_metrics} -- {pos_metrics}")
