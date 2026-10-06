import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import logsumexp


def prepare_candidates(frame, objective_names):
    frame = frame.copy()
    columns = [f"final_normalized_abs_error_{name}" for name in objective_names]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing normalized property errors: {missing}")
    errors = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame["property_cost"] = errors.mean(axis=1)
    recorded = pd.to_numeric(frame["final_sample_avg_nmae"], errors="coerce")
    finite = np.isfinite(recorded) & np.isfinite(frame["property_cost"])
    if finite.any() and not np.allclose(
        recorded[finite].to_numpy(np.float64),
        frame.loc[finite, "property_cost"].to_numpy(np.float64),
        atol=1.0e-6,
    ):
        raise ValueError("Teacher cost does not match final_sample_avg_nmae.")
    frame["eligible"] = (
        frame["validity"].astype(bool)
        & frame["final_property_evaluable"].astype(bool)
        & np.isfinite(frame["property_cost"])
    )
    controller = pd.to_numeric(frame["controller_probability"], errors="raise")
    proposal = pd.to_numeric(frame["proposal_probability"], errors="raise")
    if bool((proposal <= 0).any()):
        raise ValueError("Proposal probability must be positive.")
    frame["controller_to_proposal_ratio"] = controller / proposal
    frame["group_weight"] = 0.0
    frame["sdc_group_id"] = -1
    return frame


def build_groups(frame):
    groups = []
    skipped = []
    sampled_strata = 0
    zero_valid_strata = 0
    for group_id, (key, group) in enumerate(frame.groupby(["target_row_id", "seed"], sort=True)):
        frame.loc[group.index, "sdc_group_id"] = int(group_id)
        cell_validity = group.groupby("num_nodes")["eligible"].any()
        sampled_strata += int(len(cell_validity))
        zero_valid_strata += int((~cell_validity).sum())
        valid = group[group["eligible"]]
        if valid.empty:
            skipped.append(
                {
                    "target_row_id": int(key[0]),
                    "seed": int(key[1]),
                    "candidate_count": int(len(group)),
                }
            )
            continue
        indices = valid.index.to_numpy()
        base = valid["controller_to_proposal_ratio"].to_numpy(np.float64, copy=True)
        base /= base.sum()
        costs = valid["property_cost"].to_numpy(np.float64)
        nodes = valid["num_nodes"].to_numpy(np.int64)
        strata = []
        for value in np.unique(nodes):
            positions = np.flatnonzero(nodes == value)
            marginal = float(base[positions].sum())
            strata.append(
                {
                    "num_nodes": int(value),
                    "positions": positions,
                    "marginal": marginal,
                    "conditional": base[positions] / marginal,
                    "cost": costs[positions],
                }
            )
        groups.append(
            {
                "indices": indices,
                "base": base,
                "cost": costs,
                "strata": strata,
            }
        )
    support = {
        "sampled_strata": sampled_strata,
        "zero_valid_strata": zero_valid_strata,
        "zero_valid_strata_fraction": zero_valid_strata / max(sampled_strata, 1),
    }
    return groups, skipped, support


def blockwise_distribution(group, eta_n, eta_s):
    conditionals = []
    log_partition = []
    marginal = []
    for stratum in group["strata"]:
        log_weight = np.log(stratum["conditional"]) - float(eta_s) * stratum["cost"]
        current_partition = float(logsumexp(log_weight))
        conditionals.append(np.exp(log_weight - current_partition))
        log_partition.append(current_partition)
        marginal.append(stratum["marginal"])
    log_marginal = np.log(np.asarray(marginal, dtype=np.float64))
    log_marginal += (float(eta_n) / float(eta_s)) * np.asarray(log_partition, dtype=np.float64)
    log_marginal -= logsumexp(log_marginal)
    teacher_marginal = np.exp(log_marginal)
    teacher = np.zeros_like(group["base"])
    for mass, conditional, stratum in zip(teacher_marginal, conditionals, group["strata"]):
        teacher[stratum["positions"]] = mass * conditional
    return teacher


def summarize_group(group, teacher):
    base = group["base"]
    kl_total = float(np.sum(teacher * (np.log(teacher) - np.log(base))))
    teacher_marginal = []
    base_marginal = []
    kl_s = 0.0
    normalized_ess = 0.0
    for stratum in group["strata"]:
        positions = stratum["positions"]
        mass = float(teacher[positions].sum())
        conditional = teacher[positions] / mass
        base_conditional = stratum["conditional"]
        kl_s += mass * float(
            np.sum(conditional * (np.log(conditional) - np.log(base_conditional)))
        )
        ess = float(1.0 / np.square(conditional).sum())
        normalized_ess += mass * ess / len(positions)
        teacher_marginal.append(mass)
        base_marginal.append(stratum["marginal"])
    teacher_marginal = np.asarray(teacher_marginal, dtype=np.float64)
    base_marginal = np.asarray(base_marginal, dtype=np.float64)
    kl_n = float(
        np.sum(teacher_marginal * (np.log(teacher_marginal) - np.log(base_marginal)))
    )
    return {
        "kl_total": kl_total,
        "kl_n": kl_n,
        "kl_s": kl_s,
        "chain_rule_error": kl_total - kl_n - kl_s,
        "weighted_property_cost": float(np.sum(teacher * group["cost"])),
        "weighted_mean_num_nodes": float(
            sum(
                mass * stratum["num_nodes"]
                for mass, stratum in zip(teacher_marginal, group["strata"])
            )
        ),
        "effective_sample_size": float(1.0 / np.square(teacher).sum()),
        "teacher_size_entropy": float(
            -np.sum(teacher_marginal * np.log(np.maximum(teacher_marginal, 1.0e-30)))
        ),
        "within_n_normalized_ess": normalized_ess,
    }


def evaluate_groups(groups, eta_n, eta_s, include_weights=False):
    rows = []
    weights = []
    for group in groups:
        teacher = blockwise_distribution(group, eta_n, eta_s)
        rows.append(summarize_group(group, teacher))
        if include_weights:
            weights.append((group["indices"], teacher))
    metrics = pd.DataFrame(rows)
    return (
        {column: float(metrics[column].mean()) for column in metrics.columns},
        metrics,
        weights,
    )


def solve_budgets(groups, target_n, target_s, initial_eta_n, initial_eta_s):
    target_n = float(target_n)
    target_s = float(target_s)

    def residual(log_eta):
        eta_n, eta_s = np.exp(log_eta)
        summary, _, _ = evaluate_groups(groups, eta_n, eta_s)
        return [
            np.log(max(summary["kl_n"], 1.0e-12) / target_n),
            np.log(max(summary["kl_s"], 1.0e-12) / target_s),
        ]

    solution = least_squares(
        residual,
        np.log([float(initial_eta_n), float(initial_eta_s)]),
        bounds=(np.log([1.0e-5, 1.0e-5]), np.log([100.0, 100.0])),
        max_nfev=100,
    )
    eta_n, eta_s = np.exp(solution.x)
    summary, _, _ = evaluate_groups(groups, eta_n, eta_s)
    if abs(summary["kl_n"] - target_n) > 5.0e-4:
        raise ValueError("Marginal KL budget could not be reached.")
    if abs(summary["kl_s"] - target_s) > 5.0e-4:
        raise ValueError("Conditional KL budget could not be reached.")
    return float(eta_n), float(eta_s)


def build_teacher(frame, objective_names, cfg):
    frame = prepare_candidates(frame, objective_names)
    groups, skipped, support = build_groups(frame)
    if not groups:
        raise ValueError("No target has feasible teacher support.")
    eta_n, eta_s = solve_budgets(
        groups,
        cfg.trust_region.marginal_kl,
        cfg.trust_region.conditional_kl,
        cfg.trust_region.initial_eta_n,
        cfg.trust_region.initial_eta_s,
    )
    summary, diagnostics, weights = evaluate_groups(
        groups,
        eta_n,
        eta_s,
        include_weights=True,
    )
    for indices, values in weights:
        frame.loc[indices, "group_weight"] = values
    summary.update(support)
    summary.update(
        {
            "target_kl_n": float(cfg.trust_region.marginal_kl),
            "target_kl_s": float(cfg.trust_region.conditional_kl),
            "eta_n": eta_n,
            "eta_s": eta_s,
        }
    )
    return frame, skipped, diagnostics, summary
