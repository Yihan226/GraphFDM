import numpy as np
import torch

from graphfdm.size_controller import initialize_controller
from graphfdm.teacher import blockwise_distribution, summarize_group


def test_blockwise_teacher_obeys_kl_chain_rule():
    group = {
        "base": np.asarray([0.2, 0.3, 0.1, 0.4]),
        "cost": np.asarray([1.0, 0.1, 0.8, 0.2]),
        "strata": [
            {
                "num_nodes": 2,
                "positions": np.asarray([0, 1]),
                "marginal": 0.5,
                "conditional": np.asarray([0.4, 0.6]),
                "cost": np.asarray([1.0, 0.1]),
            },
            {
                "num_nodes": 3,
                "positions": np.asarray([2, 3]),
                "marginal": 0.5,
                "conditional": np.asarray([0.2, 0.8]),
                "cost": np.asarray([0.8, 0.2]),
            },
        ],
    }
    teacher = blockwise_distribution(group, eta_n=1.1, eta_s=0.7)
    summary = summarize_group(group, teacher)
    assert np.isclose(teacher.sum(), 1.0)
    assert abs(summary["chain_rule_error"]) < 1.0e-10
    assert summary["weighted_property_cost"] < np.dot(group["base"], group["cost"])


def test_controller_preserves_empirical_support():
    controller = initialize_controller(3, 4, 8, 4)
    condition = torch.randn(5, 3)
    p0 = torch.tensor([0.0, 0.2, 0.0, 0.8, 0.0])
    probability = controller.size_distribution(condition, p0)
    assert torch.all(probability[:, p0 == 0] == 0)
    assert torch.allclose(probability.sum(dim=1), torch.ones(5))
