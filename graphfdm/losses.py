import torch
import torch.nn.functional as F

import utils


def component_ce_per_graph(prediction, truth):
    batch_size = truth.size(0)
    classes = truth.size(-1)
    ce = F.cross_entropy(
        prediction.reshape(-1, classes),
        truth.argmax(dim=-1).reshape(-1),
        reduction="none",
    ).reshape(batch_size, -1)
    mask = (truth != 0).any(dim=-1).reshape(batch_size, -1).type_as(ce)
    return (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def dense_batch(model, batch, device):
    x = batch["x"].to(device=device, dtype=torch.long)
    e = batch["e"].to(device=device, dtype=torch.long)
    pos = batch["pos"].to(device=device, dtype=torch.long)
    y = batch["y"].to(device=device, dtype=torch.float32)
    num_nodes = batch["num_nodes"].to(device=device, dtype=torch.long)
    support = torch.arange(x.size(1), device=device)[None, :]
    node_mask = support < num_nodes[:, None]
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    diagonal = torch.eye(x.size(1), dtype=torch.bool, device=device)[None, :, :]
    pair_mask = pair_mask & ~diagonal
    x_onehot = F.one_hot(x, num_classes=model.Xdim_output).float()
    e_onehot = F.one_hot(e, num_classes=model.Edim_output).float()
    pos_onehot = F.one_hot(pos, num_classes=model.Posdim_output).float()
    x_onehot *= node_mask[:, :, None]
    e_onehot *= pair_mask[:, :, :, None]
    pos_onehot *= pair_mask[:, :, :, None]
    dense = utils.PlaceHolder(X=x_onehot, E=e_onehot, pos=pos_onehot, y=y)
    return dense, node_mask, y


def loss_per_graph(model, prediction, dense):
    weights = model.train_loss.lambda_train
    x = component_ce_per_graph(prediction.X, dense.X)
    e = component_ce_per_graph(prediction.E, dense.E)
    pos = component_ce_per_graph(prediction.pos, dense.pos)
    total = weights[0] * x + weights[1] * e + weights[2] * pos
    return {"total": total, "x": x, "e": e, "pos": pos}


def weighted_group_mean(values, weights, group_ids):
    losses = []
    for group_id in torch.unique(group_ids):
        selected = group_ids == group_id
        losses.append((values[selected] * weights[selected]).sum())
    if not losses:
        return values.new_tensor(0.0)
    return torch.stack(losses).mean()

