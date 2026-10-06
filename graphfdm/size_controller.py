import torch
from torch import nn


class ConditionalSizeController(nn.Module):
    def __init__(self, condition_dim, max_nodes, hidden_dim, node_embedding_dim):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.max_nodes = int(max_nodes)
        self.hidden_dim = int(hidden_dim)
        self.node_embedding_dim = int(node_embedding_dim)
        self.node_embedding = nn.Embedding(self.max_nodes + 1, self.node_embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.condition_dim + self.node_embedding_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, condition, num_nodes):
        embedding = self.node_embedding(num_nodes.long())
        return self.network(torch.cat([condition, embedding], dim=-1)).squeeze(-1)

    def size_distribution(self, condition, base_probability):
        support = torch.arange(base_probability.numel(), device=condition.device, dtype=torch.long)
        conditions = condition[:, None, :].expand(-1, support.numel(), -1)
        sizes = support[None, :].expand(condition.size(0), -1)
        gate = torch.sigmoid(
            self(
                conditions.reshape(-1, condition.size(-1)),
                sizes.reshape(-1),
            )
        ).reshape(condition.size(0), -1)
        unnormalized = gate * base_probability.to(condition.device)[None, :]
        return unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(1.0e-30)

    def config(self):
        return {
            "condition_dim": self.condition_dim,
            "max_nodes": self.max_nodes,
            "hidden_dim": self.hidden_dim,
            "node_embedding_dim": self.node_embedding_dim,
        }


def initialize_controller(condition_dim, max_nodes, hidden_dim, node_embedding_dim):
    model = ConditionalSizeController(condition_dim, max_nodes, hidden_dim, node_embedding_dim)
    torch.nn.init.zeros_(model.network[-1].weight)
    torch.nn.init.zeros_(model.network[-1].bias)
    return model


def save_controller(model, path, round_id):
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model.config(),
            "round_id": int(round_id),
            "initialization": "exact_p0",
        },
        path,
    )


def load_controller(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = ConditionalSizeController(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload

