import numpy as np
import torch
from torch.nn import functional as F

from utils import PlaceHolder


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, "Variables not masked properly."


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=30, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    p = 4 / 5
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2
    beta_first = 1.2 / (p * num_edges)
    betas[betas < beta_first] = beta_first
    return np.array(betas)


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if variable is not None and len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def merge_graph(X, E):
    bs, n, _ = X.shape
    return torch.cat([X, E.reshape(bs, n, -1)], dim=-1)


def split_graph(XE, X_dim, E_dim):
    bs, n, _ = XE.shape
    X = XE[:, :, :X_dim]
    E = XE[:, :, X_dim:].reshape(bs, n, n, E_dim)
    return X, E


def sample_discrete_features(probX, probE, prob_pos, node_mask, step=None, deterministic=False):
    bs, n, _ = probX.shape
    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)
    probX = probX + 1e-12
    probX = probX / probX.sum(dim=-1, keepdim=True)
    if deterministic:
        X_t = probX.argmax(dim=-1)
    else:
        X_t = probX.multinomial(1)
    X_t = X_t.reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=probE.device).unsqueeze(0).expand(bs, -1, -1).bool()

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)
    probE = probE + 1e-12
    probE = probE / probE.sum(dim=-1, keepdim=True)
    if deterministic:
        E_t = probE.argmax(dim=-1).reshape(bs, n, n)
    else:
        E_t = probE.multinomial(1).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    prob_pos[inverse_edge_mask] = 1 / prob_pos.shape[-1]
    prob_pos[diag_mask] = 1 / prob_pos.shape[-1]
    prob_pos = prob_pos.reshape(bs * n * n, -1)
    prob_pos = prob_pos + 1e-12
    prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True)
    if deterministic:
        pos_t = prob_pos.argmax(dim=-1).reshape(bs, n, n)
    else:
        pos_t = prob_pos.multinomial(1).reshape(bs, n, n)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t), pos=pos_t)


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    X_t = X_t.float()
    Qt_T = Qt.transpose(-1, -2).float()
    left_term = X_t @ Qt_T
    left_term = left_term.unsqueeze(dim=2)
    right_term = Qsb.unsqueeze(1)
    numerator = left_term * right_term
    denominator = Qtb @ X_t.transpose(-1, -2)
    denominator = denominator.transpose(-1, -2)
    denominator = denominator.unsqueeze(-1)
    denominator[denominator == 0] = 1
    return numerator / denominator


def reverse_diffusion(predX_0, X_t, Qt, Qsb, Qtb):
    left_term = torch.einsum("bijd,bdk->bijk", X_t, Qt.transpose(-1, -2))
    right_term = torch.einsum("bijd,bdk->bijk", predX_0, Qsb)
    numerator = left_term * right_term
    denominator = torch.einsum("bdk,bijk->bijd", Qtb, X_t)
    return numerator / denominator.clamp_min(1e-5)


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask, true_pos=None, pred_pos=None):
    pred_X = pred_X.clamp_min(1e-18)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    pred_E = pred_E.clamp_min(1e-18)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    row_X = torch.ones(true_X.size(-1), dtype=true_X.dtype, device=true_X.device)
    row_E = torch.zeros(true_E.size(-1), dtype=true_E.dtype, device=true_E.device).clamp_min(1e-18)
    row_E[0] = 1
    diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
    mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask
    true_X[~node_mask] = row_X
    true_E[~mask, :] = row_E
    pred_X[~node_mask] = row_X
    pred_E[~mask, :] = row_E
    if true_pos is not None and pred_pos is not None:
        row_pos = torch.zeros(true_pos.size(-1), dtype=true_pos.dtype, device=true_pos.device).clamp_min(1e-18)
        row_pos[0] = 1
        true_pos[~mask, :] = row_pos
        pred_pos[~mask, :] = row_pos
        return true_X, true_E, pred_X, pred_E, true_pos, pred_pos
    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, X_t, Qt, Qsb, Qtb, X_dim):
    bs, n, _ = X.shape
    X = X.float()
    Qt_X_T = torch.transpose(Qt.X, -2, -1).float()
    left_term = X_t @ Qt_X_T
    right_term = X @ Qsb.X
    numerator = left_term * right_term
    denominator = X @ Qtb.X
    denominator = denominator * X_t

    num_X = numerator[:, :, :X_dim]
    num_E = numerator[:, :, X_dim:].reshape(bs, n * n, -1)
    deno_X = denominator[:, :, :X_dim]
    deno_E = denominator[:, :, X_dim:].reshape(bs, n * n, -1)
    deno_X = deno_X.sum(dim=-1).unsqueeze(-1)
    deno_E = deno_E.sum(dim=-1).unsqueeze(-1)
    deno_X[deno_X == 0] = 1
    deno_E[deno_E == 0] = 1
    prob_X = num_X / deno_X
    prob_E = num_E / deno_E
    prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True)
    prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True)
    return PlaceHolder(X=prob_X, E=prob_E, y=None)


def sample_discrete_feature_noise(limit_dist, node_mask):
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_X = F.one_hot(U_X.long(), num_classes=x_limit.shape[-1]).float().to(node_mask.device)

    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_E = F.one_hot(U_E.long(), num_classes=e_limit.shape[-1]).float().to(node_mask.device)

    pos_limit = limit_dist.pos[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_pos = pos_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_pos = F.one_hot(U_pos.long(), num_classes=pos_limit.shape[-1]).float().to(node_mask.device)

    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1
    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)
    return PlaceHolder(X=U_X, E=U_E, y=None, pos=U_pos).mask(node_mask)


def get_pair_masks(node_mask):
    n = node_mask.size(1)
    node_mask_f = node_mask.float()
    valid_pairs = node_mask_f.unsqueeze(1) * node_mask_f.unsqueeze(2)
    diag_mask = 1.0 - torch.eye(n, device=node_mask.device, dtype=node_mask_f.dtype).unsqueeze(0)
    directed_pairs = valid_pairs * diag_mask
    upper_mask = torch.triu(torch.ones(n, n, device=node_mask.device, dtype=node_mask_f.dtype), diagonal=1).unsqueeze(0)
    undirected_pairs = directed_pairs * upper_mask
    return node_mask_f, undirected_pairs, directed_pairs


def compute_action_logprob(prob_X, prob_E, prob_pos, X_prev_idx, E_prev_idx, pos_prev_idx, node_mask):
    node_mask_f, undirected_pairs, directed_pairs = get_pair_masks(node_mask)
    X_logprob = prob_X.clamp_min(1e-12).log().gather(-1, X_prev_idx.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    E_logprob = prob_E.clamp_min(1e-12).log().gather(-1, E_prev_idx.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    pos_logprob = prob_pos.clamp_min(1e-12).log().gather(-1, pos_prev_idx.long().clamp_min(0).unsqueeze(-1)).squeeze(-1)
    total_X = (X_logprob * node_mask_f).sum(dim=1)
    total_E = (E_logprob * undirected_pairs).sum(dim=(1, 2))
    total_pos = (pos_logprob * directed_pairs).sum(dim=(1, 2))
    return total_X + total_E + total_pos


def compute_action_entropy(prob_X, prob_E, prob_pos, node_mask):
    node_mask_f, undirected_pairs, directed_pairs = get_pair_masks(node_mask)
    X_entropy = -(prob_X.clamp_min(1e-12) * prob_X.clamp_min(1e-12).log()).sum(dim=-1)
    E_entropy = -(prob_E.clamp_min(1e-12) * prob_E.clamp_min(1e-12).log()).sum(dim=-1)
    pos_entropy = -(prob_pos.clamp_min(1e-12) * prob_pos.clamp_min(1e-12).log()).sum(dim=-1)
    total_X = (X_entropy * node_mask_f).sum(dim=1)
    total_E = (E_entropy * undirected_pairs).sum(dim=(1, 2))
    total_pos = (pos_entropy * directed_pairs).sum(dim=(1, 2))
    return total_X + total_E + total_pos
