import time

import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

from metrics.abstract_metrics import CrossEntropyMetric


class TrainLossDiscrete(nn.Module):
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.pos_loss = CrossEntropyMetric()
        self.y_loss = MeanSquaredError()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, masked_pred_pos, pred_y, true_X, true_E, true_pos, true_y, node_mask, log):
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        true_pos = torch.reshape(true_pos, (-1, true_pos.size(-1)))
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))
        masked_pred_pos = torch.reshape(masked_pred_pos, (-1, masked_pred_pos.size(-1)))

        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)
        mask_pos = (true_pos != 0.0).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_true_E = true_E[mask_E, :]
        flat_true_pos = true_pos[mask_pos, :]
        flat_pred_X = masked_pred_X[mask_X, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        flat_pred_pos = masked_pred_pos[mask_pos, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if flat_true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if flat_true_E.numel() > 0 else 0.0
        loss_pos = self.pos_loss(flat_pred_pos, flat_true_pos) if flat_true_pos.numel() > 0 else 0.0

        return (
            self.lambda_train[0] * loss_X
            + self.lambda_train[1] * loss_E
            + self.lambda_train[2] * loss_pos
        )

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.pos_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time, log=True):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_pos_loss = self.pos_loss.compute() if self.pos_loss.total_samples > 0 else -1
        if log:
            print(
                f"Epoch {current_epoch} finished: X_CE: {epoch_node_loss :.4f} -- "
                f"E_CE: {epoch_edge_loss :.4f} -- Pos_CE: {epoch_pos_loss :.4f} "
                f"-- Time taken {time.time() - start_epoch_time:.1f}s "
            )
