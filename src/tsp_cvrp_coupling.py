"""Coupling plans for TSP -> CVRP mapping.

The source and target vertices are never compared directly.  Structural
relations are computed internally in each solution space and then compared by
their matrix values.  The relation term uses the usual matrix-form GW
majorization update; no four-dimensional tensor is constructed.
"""
import copy
import hashlib

import numpy as np
import torch

from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions
from src.tsp_mapping_controls import decoder_parameters


ARMS = ("rank_pairing", "quality_coupling", "quality_relation_coupling")


def _finite_matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(name + " must be a finite matrix")
    return value


def _edge_set(sequence, include_padding=False):
    values = np.asarray(sequence, dtype=np.int64)
    if not include_padding:
        values = values[values >= 0]
    if len(values) < 2:
        raise ValueError("solution has too few active tokens")
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(values[:-1], values[1:]) if int(a) != int(b)}


def tsp_relation_matrix(tours):
    """Return 1-Jaccard of undirected TSP cycle edge sets, including closure."""
    tours = np.asarray(tours)
    if tours.ndim != 2 or tours.shape[1] < 3 or tours.dtype.kind not in "iu":
        raise ValueError("tours must be an integer matrix")
    edge_sets = []
    for tour in tours:
        if len(np.unique(tour)) != len(tour):
            raise ValueError("TSP tours must have unique vertices")
        closed = np.r_[tour, tour[0]]
        edge_sets.append(_edge_set(closed))
    return _relation_from_sets(edge_sets)


def cvrp_relation_matrix(solutions):
    """Return 1-Jaccard of CVRP route edge sets, excluding -1 padding.

    Edges are sets: repeated edges have no multiplicity, and vehicle loads are
    intentionally not represented by this relation.
    """
    return _relation_from_sets([_edge_set(solution) for solution in solutions])


def _relation_from_sets(edge_sets):
    size = len(edge_sets)
    if size == 0:
        raise ValueError("at least one solution is required")
    # Build solution x edge incidence once. This avoids the old O(n^2) Python
    # set-intersection loop; the dense products can be moved to Torch/CUDA by
    # the coupling solver.
    universe = sorted(set().union(*edge_sets))
    index = {edge: i for i, edge in enumerate(universe)}
    incidence = np.zeros((size, len(universe)), dtype=np.int32)
    for row, edges in enumerate(edge_sets):
        incidence[row, [index[edge] for edge in edges]] = 1
    intersection = incidence @ incidence.T
    counts = incidence.sum(1).astype(np.int64)
    union = counts[:, None] + counts[None, :] - intersection
    result = 1.0 - intersection / np.maximum(union, 1)
    np.fill_diagonal(result, 0.0)
    return result.astype(np.float64, copy=False)


def percentile(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("quality values must be finite and non-empty")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    left = np.searchsorted(sorted_values, values, side="left")
    right = np.searchsorted(sorted_values, values, side="right")
    return (left + right) / (2.0 * len(values))


def _sinkhorn(cost, epsilon, max_iter=500, tolerance=1e-8):
    cost = _finite_matrix(cost, "cost")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    n, m = cost.shape
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    kernel = np.exp(-np.clip(cost / epsilon, 0.0, 700.0))
    u, v = np.ones(n), np.ones(m)
    objective_trace, coupling_trace = [], []
    previous_gamma = None
    converged = False
    for iteration in range(1, max_iter + 1):
        u = a / np.maximum(kernel @ v, np.finfo(np.float64).tiny)
        v = b / np.maximum(kernel.T @ u, np.finfo(np.float64).tiny)
        gamma = (u[:, None] * kernel) * v[None, :]
        entropy = np.sum(gamma * np.log(np.maximum(gamma, np.finfo(np.float64).tiny)) - gamma)
        objective_trace.append(float(np.sum(cost * gamma) + epsilon * entropy))
        coupling_trace.append(None if previous_gamma is None else float(
            np.max(np.abs(gamma - previous_gamma)) / max(np.max(np.abs(previous_gamma)), np.finfo(np.float64).tiny)))
        previous_gamma = gamma
        relative_residual = max(
            float(np.max(np.abs(gamma.sum(1) - a) / np.maximum(a, np.finfo(np.float64).tiny))),
            float(np.max(np.abs(gamma.sum(0) - b) / np.maximum(b, np.finfo(np.float64).tiny))))
        if relative_residual <= tolerance:
            converged = True
            break
    gamma = (u[:, None] * kernel) * v[None, :]
    row_error = float(np.max(np.abs(gamma.sum(1) - a)))
    col_error = float(np.max(np.abs(gamma.sum(0) - b)))
    relative_residual = max(row_error / np.min(a), col_error / np.min(b))
    if not np.isfinite(gamma).all() or np.min(gamma) < -1e-12:
        raise RuntimeError("Sinkhorn produced a nonfinite or negative coupling")
    return gamma, dict(iterations=iteration, row_marginal_error=row_error,
                       column_marginal_error=col_error, relative_marginal_residual=float(relative_residual),
                       stopping_criterion="relative_marginal_residual<=tolerance",
                       converged=converged, stopped_due_to_max_iter=not converged,
                       objective_trace=objective_trace, coupling_change_trace=coupling_trace)


@torch.no_grad()
def _sinkhorn_torch(cost, epsilon, device, max_iter=500, tolerance=1e-8, dtype=torch.float64):
    """Torch Sinkhorn backend; all large matrix products stay on device."""
    target = torch.device(device)
    cost_t = torch.as_tensor(cost, dtype=dtype, device=target)
    if not torch.isfinite(cost_t).all():
        raise ValueError("cost contains nonfinite values")
    n, m = cost_t.shape
    a = torch.full((n,), 1.0 / n, dtype=dtype, device=target)
    b = torch.full((m,), 1.0 / m, dtype=dtype, device=target)
    kernel = torch.exp((-cost_t / float(epsilon)).clamp(min=-80., max=0.))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    objective_trace, coupling_trace = [], []
    previous_gamma = None
    converged = False
    for iteration in range(1, max_iter + 1):
        u = a / (kernel @ v).clamp_min(torch.finfo(dtype).tiny)
        v = b / (kernel.T @ u).clamp_min(torch.finfo(dtype).tiny)
        gamma = u[:, None] * kernel * v[None, :]
        entropy = torch.sum(gamma * torch.log(gamma.clamp_min(torch.finfo(dtype).tiny)) - gamma)
        objective_trace.append(float((torch.sum(cost_t * gamma) + epsilon * entropy).item()))
        coupling_trace.append(None if previous_gamma is None else float(
            (torch.max(torch.abs(gamma - previous_gamma)) /
             torch.max(torch.abs(previous_gamma)).clamp_min(torch.finfo(dtype).tiny)).item()))
        previous_gamma = gamma
        relative_residual = max(
            (torch.abs(gamma.sum(1) - a) / a).max().item(),
            (torch.abs(gamma.sum(0) - b) / b).max().item())
        if relative_residual <= tolerance:
            converged = True
            break
    gamma = u[:, None] * kernel * v[None, :]
    row_error = torch.max(torch.abs(gamma.sum(1) - a)).item()
    col_error = torch.max(torch.abs(gamma.sum(0) - b)).item()
    relative_residual = max(row_error / (1. / n), col_error / (1. / m))
    if not torch.isfinite(gamma).all() or gamma.min().item() < -1e-7:
        raise RuntimeError("Torch Sinkhorn produced a nonfinite or negative coupling")
    return gamma, dict(iterations=iteration, row_marginal_error=float(row_error),
                       column_marginal_error=float(col_error), relative_marginal_residual=float(relative_residual),
                       stopping_criterion="relative_marginal_residual<=tolerance",
                       converged=converged, stopped_due_to_max_iter=not converged,
                       objective_trace=objective_trace, coupling_change_trace=coupling_trace)


def _gw_cost(source_relation, target_relation, gamma):
    """Matrix-form GW linearization; equivalent to the two-index update."""
    source_relation, target_relation = _finite_matrix(source_relation, "source_relation"), _finite_matrix(target_relation, "target_relation")
    gamma = _finite_matrix(gamma, "gamma")
    if source_relation.shape != (len(gamma), len(gamma)) or target_relation.shape != (gamma.shape[1], gamma.shape[1]):
        raise ValueError("relation/coupling shape mismatch")
    source_sq = source_relation ** 2
    target_sq = target_relation ** 2
    a = gamma.sum(1)
    b = gamma.sum(0)
    # Terms constant under fixed marginals are retained for transparent audit;
    # the cross term is the only part that changes the next transport plan.
    constant = (source_sq @ a)[:, None] + (target_sq @ b)[None, :]
    return constant - 2.0 * source_relation @ gamma @ target_relation.T


def relation_cost(source_relation, target_relation, gamma):
    """Evaluate the full relation objective on any coupling, without training."""
    sr, tr = _finite_matrix(source_relation, "source_relation"), _finite_matrix(target_relation, "target_relation")
    g = _finite_matrix(gamma, "gamma")
    if sr.shape != (len(g), len(g)) or tr.shape != (g.shape[1], g.shape[1]):
        raise ValueError("relation/coupling shape mismatch")
    a, b = g.sum(1), g.sum(0)
    return float(a @ (sr ** 2) @ a + b @ (tr ** 2) @ b -
                 2. * np.sum((sr @ g @ tr.T) * g))


def average_total_variation(gamma_q, gamma_qr, marginal_tolerance=1e-5):
    """Average TV between P(target|source) rows under equal/explicit marginals."""
    q, qr = _finite_matrix(gamma_q, "quality_coupling"), _finite_matrix(gamma_qr, "relation_coupling")
    if q.shape != qr.shape:
        raise ValueError("coupling shape mismatch")
    a = q.sum(1)
    if np.any(a <= 0) or not np.allclose(a, qr.sum(1), rtol=marginal_tolerance, atol=marginal_tolerance):
        raise ValueError("couplings must have matching positive source marginals")
    return float(np.sum(a * (.5 * np.abs(q / a[:, None] - qr / a[:, None]).sum(1))))


def solve_coupling(source_quality, target_quality, source_relation=None, target_relation=None,
                   alpha=0.5, epsilon=0.05, max_iter=50, tolerance=1e-7, device=None,
                   quality_weight=None, relation_weight=None, sinkhorn_max_iter=500):
    """Solve quality OT or quality+relation GW-regularized coupling.

    The quality-only arm solves <Cq,Gamma> + epsilon*sum Gamma(log Gamma-1)
    with uniform marginals.  The relation arm uses fixed-point GW
    majorization: each iteration solves the entropic OT subproblem with
    (1-alpha)Cq + alpha*GW_linearized_cost.  The returned audit recomputes the
    exact quality and relation terms from the final Gamma.
    """
    sq, tq = np.asarray(source_quality, dtype=np.float64), np.asarray(target_quality, dtype=np.float64)
    if sq.ndim != 1 or tq.ndim != 1 or len(sq) == 0 or len(tq) == 0 or not np.isfinite(sq).all() or not np.isfinite(tq).all():
        raise ValueError("quality arrays must be finite and non-empty")
    if not 0 <= alpha <= 1 or not np.isfinite(alpha):
        raise ValueError("alpha must be in [0, 1]")
    has_relation = source_relation is not None or target_relation is not None
    if quality_weight is None:
        quality_weight = 1. - alpha if has_relation else 1.
    if relation_weight is None:
        relation_weight = alpha if has_relation else 0.
    if quality_weight < 0 or relation_weight < 0 or not np.isfinite(quality_weight + relation_weight):
        raise ValueError("coupling objective weights must be finite and nonnegative")
    source_rank, target_rank = percentile(sq), percentile(tq)
    quality_cost = (source_rank[:, None] - target_rank[None, :]) ** 2
    weighted_quality_cost = quality_weight * quality_cost
    if source_relation is None and target_relation is not None or source_relation is not None and target_relation is None:
        raise ValueError("provide both relation matrices or neither")
    if type(sinkhorn_max_iter) is not int or sinkhorn_max_iter < 1:
        raise ValueError("sinkhorn_max_iter must be a positive integer")
    if device is None:
        sinkhorn = lambda cost, eps: _sinkhorn(cost, eps, max_iter=sinkhorn_max_iter, tolerance=tolerance)
    else:
        sinkhorn = lambda cost, eps: _sinkhorn_torch(
            cost, eps, device, max_iter=sinkhorn_max_iter, tolerance=tolerance)
    objective_trace, coupling_change_trace = [], []
    initial_sinkhorn_audit = None
    if source_relation is None:
        gamma, audit = sinkhorn(weighted_quality_cost, epsilon)
        initial_sinkhorn_audit = audit.copy()
        relation_cost = 0.0
        relation_iterations = 0
        relation_converged = None
        outer_delta = None
    else:
        sr, tr = _finite_matrix(source_relation, "source_relation"), _finite_matrix(target_relation, "target_relation")
        if sr.shape != (len(sq), len(sq)) or tr.shape != (len(tq), len(tq)):
            raise ValueError("relation matrix shape mismatch")
        gamma, audit = sinkhorn(weighted_quality_cost, epsilon)
        initial_sinkhorn_audit = audit.copy()
        if device is not None:
            source_t = torch.as_tensor(sr, dtype=torch.float64, device=device)
            target_t = torch.as_tensor(tr, dtype=torch.float64, device=device)
        relation_iterations = 0
        relation_converged = False
        outer_delta = None
        for relation_iterations in range(1, max_iter + 1):
            if device is None:
                linearized = _gw_cost(sr, tr, gamma)
                updated, sub_audit = sinkhorn(quality_weight * quality_cost + relation_weight * linearized, epsilon)
                delta = float(np.max(np.abs(updated - gamma)) / max(np.max(np.abs(gamma)), np.finfo(np.float64).tiny))
                gamma = updated
            else:
                gamma_t = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, dtype=torch.float64, device=device)
                source_sq = source_t.square()
                target_sq = target_t.square()
                a_t, b_t = gamma_t.sum(1), gamma_t.sum(0)
                linearized_t = (source_sq @ a_t)[:, None] + (target_sq @ b_t)[None, :] - 2. * source_t @ gamma_t @ target_t.T
                quality_t = torch.as_tensor(quality_cost, dtype=torch.float64, device=device)
                updated, sub_audit = sinkhorn(quality_weight * quality_t + relation_weight * linearized_t, epsilon)
                delta = (torch.max(torch.abs(updated - gamma_t)) /
                         torch.max(torch.abs(gamma_t)).clamp_min(torch.finfo(torch.float32).tiny)).item()
                gamma = updated
            audit = sub_audit
            outer_delta = float(delta)
            gamma_iter = gamma.cpu().numpy() if torch.is_tensor(gamma) else gamma
            q_iter = float(np.sum(quality_cost * gamma_iter))
            a_iter, b_iter = gamma_iter.sum(1), gamma_iter.sum(0)
            r_iter = float(a_iter @ sr ** 2 @ a_iter + b_iter @ tr ** 2 @ b_iter -
                           2. * np.sum((sr @ gamma_iter @ tr.T) * gamma_iter))
            e_iter = float(np.sum(gamma_iter * np.log(np.maximum(gamma_iter, np.finfo(np.float64).tiny)) - gamma_iter))
            objective_trace.append(float(quality_weight * q_iter + relation_weight * r_iter + epsilon * e_iter))
            coupling_change_trace.append(float(delta))
            if delta < tolerance:
                relation_converged = True
                break
        if device is None:
            source_sq = sr ** 2
            target_sq = tr ** 2
            a, b = gamma.sum(1), gamma.sum(0)
            relation_cost = float(a @ source_sq @ a + b @ target_sq @ b -
                                  2. * np.sum((sr @ gamma @ tr.T) * gamma))
        else:
            gamma_t = gamma if torch.is_tensor(gamma) else torch.as_tensor(gamma, dtype=torch.float64, device=device)
            a_t, b_t = gamma_t.sum(1), gamma_t.sum(0)
            relation_cost = float((a_t @ source_t.square() @ a_t + b_t @ target_t.square() @ b_t -
                                   2. * torch.sum((source_t @ gamma_t @ target_t.T) * gamma_t)).item())
    if torch.is_tensor(gamma):
        gamma = gamma.cpu().numpy()
    quality_value = float(np.sum(quality_cost * gamma))
    entropy = float(np.sum(gamma * np.log(np.maximum(gamma, np.finfo(np.float64).tiny)) - gamma))
    objective_value = float(quality_weight * quality_value + relation_weight * relation_cost + epsilon * entropy)
    if not objective_trace:
        objective_trace.append(objective_value)
    effective_targets = float(np.exp(-np.sum((gamma / gamma.sum(1, keepdims=True)) *
                                             np.log(np.maximum(gamma / gamma.sum(1, keepdims=True), np.finfo(np.float64).tiny)), axis=1)).mean())
    audit.update(dict(alpha=float(alpha), epsilon=float(epsilon), quality_weight=float(quality_weight),
                      relation_weight=float(relation_weight), quality_cost=quality_value,
                      relation_cost=relation_cost, entropy=entropy, objective=objective_value,
                      objective_trace=(objective_trace if source_relation is not None else audit.get("objective_trace", objective_trace)),
                      coupling_change_trace=(coupling_change_trace if source_relation is not None else audit.get("coupling_change_trace", coupling_change_trace)),
                      sinkhorn_objective_trace=(initial_sinkhorn_audit or {}).get("objective_trace", []),
                      relation_iterations=relation_iterations,
                      relation_relative_coupling_change=outer_delta,
                      relation_stopping_criterion="relative_coupling_change<=tolerance",
                      relation_converged=relation_converged,
                      relation_stopped_due_to_max_iter=(relation_converged is False),
                      solver_dtype="float64",
                      solver_device="cpu" if device is None else str(device),
                      sinkhorn_inner_audit=(initial_sinkhorn_audit if source_relation is not None else audit.copy()),
                      mean_effective_targets_per_source=effective_targets,
                      min_coupling=float(gamma.min()), max_coupling=float(gamma.max()),
                      finite=bool(np.isfinite(gamma).all())))
    return gamma, audit


def prepare_coupling(source_latent, source_quality, source_tours, target_quality,
                     target_solutions, seed, max_pairs, alpha=0.5, epsilon=0.05, device=None):
    """Split banks first, then build rank/quality/relation plans independently."""
    source = np.asarray(source_latent, dtype=np.float32)
    sq, tq = np.asarray(source_quality, dtype=np.float64), np.asarray(target_quality, dtype=np.float64)
    tours = np.asarray(source_tours)
    if source.ndim != 2 or sq.shape != (len(source),) or tours.shape[0] != len(source):
        raise ValueError("source bank shape mismatch")
    if len(target_solutions) != len(tq) or type(max_pairs) is not int or max_pairs < 8:
        raise ValueError("target bank/max_pairs mismatch")
    rng = np.random.default_rng(seed)
    source_order = rng.permutation(len(source))
    target_order = rng.permutation(len(target_solutions))
    count = min(max_pairs, len(source_order), len(target_order))
    if count < 8:
        raise ValueError("at least 8 source and target rows are required")
    source_order, target_order = source_order[:count], target_order[:count]
    cut = int(0.8 * count)
    train, valid = np.arange(cut), np.arange(cut, count)
    source_order[train] = source_order[train][np.argsort(-sq[source_order[train]], kind="stable")]
    source_order[valid] = source_order[valid][np.argsort(-sq[source_order[valid]], kind="stable")]
    target_order[train] = target_order[train][np.argsort(-tq[target_order[train]], kind="stable")]
    target_order[valid] = target_order[valid][np.argsort(-tq[target_order[valid]], kind="stable")]
    paired_source = source[source_order]
    mean, std = paired_source[train].mean(0), paired_source[train].std(0)
    active = std > 1e-6
    scale = np.where(active, std, 1.)
    normalized = (paired_source - mean) / scale
    normalized[:, ~active] = 0
    plans = {}
    audits = {}
    for name in ("train", "valid"):
        ids = train if name == "train" else valid
        source_ids, target_ids = source_order[ids], target_order[ids]
        source_r = tsp_relation_matrix(tours[source_ids])
        target_r = cvrp_relation_matrix([target_solutions[i] for i in target_ids])
        gamma_q, audit_q = solve_coupling(sq[source_ids], tq[target_ids], epsilon=epsilon, device=device,
                                          quality_weight=0.5, relation_weight=0.)
        gamma_r, audit_r = solve_coupling(sq[source_ids], tq[target_ids], source_r, target_r,
                                          alpha=alpha, epsilon=epsilon, device=device,
                                          quality_weight=1. - alpha, relation_weight=alpha)
        rank_target = np.argsort(-tq[target_ids], kind="stable")
        plans[name] = dict(rank=rank_target, quality=gamma_q, relation=gamma_r,
                           source_ids=source_ids, target_ids=target_ids,
                           source_global_ids=source_ids.copy(), target_global_ids=target_ids.copy(),
                           local_target_ids=np.arange(len(target_ids), dtype=np.int64))
        audits[name] = dict(quality=audit_q, relation=audit_r)
    conditions = normalized
    pairing = dict(source_order=source_order.tolist(), target_order=target_order.tolist(),
                   train_source_order=source_order[train].tolist(), valid_source_order=source_order[valid].tolist(),
                   train_target_order=target_order[train].tolist(), valid_target_order=target_order[valid].tolist(),
                   train_indices=train.tolist(), valid_indices=valid.tolist(), source_mean=mean.tolist(),
                   source_scale=scale.tolist(), active_source_dimensions=int(active.sum()),
                   quality_definition="TSP/CVRP quality=-length; higher is better",
                   relation_definition="1-Jaccard of undirected edge sets; TSP closure included; CVRP depot edges included; padding removed; multiplicity/load ignored",
                   coupling_constraints="uniform source/target marginals; split before coupling; train and validation solved independently",
                   alpha=float(alpha), epsilon=float(epsilon), audits=audits)
    pairing["split_checksum"] = hashlib.sha256(
        np.asarray(source_order, dtype=np.int64).tobytes() +
        np.asarray(target_order, dtype=np.int64).tobytes()).hexdigest()
    return conditions, plans, pairing


def sample_target_indices(plan, arm, rows, rng):
    rows = np.asarray(rows, dtype=np.int64)
    if arm == "rank_pairing":
        return plan["rank"][rows]
    probabilities = plan["quality" if arm == "quality_coupling" else "relation"][rows]
    cumulative = np.cumsum(probabilities / probabilities.sum(1, keepdims=True), axis=1)
    draws = rng.random(len(rows))[:, None]
    return (cumulative < draws).sum(1).clip(max=probabilities.shape[1] - 1)


def make_epoch_targets(plan, arm, epochs, train_count, seed):
    rng = np.random.default_rng(seed)
    return [sample_target_indices(plan, arm, np.arange(train_count), rng) for _ in range(epochs)]


def train_coupled(model, bridge, instance, conditions,
                  train_target_padded, train_target_sizes, valid_target_padded, valid_target_sizes,
                  train_target_global_ids, valid_target_global_ids,
                  train_plan, valid_plan, arm, train, valid, config):
    params = decoder_parameters(model) + ([] if bridge is None else list(bridge.parameters()))
    optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
    device = instance[0].device
    x = torch.tensor(conditions, dtype=torch.float32, device=device)
    train_target = torch.tensor(train_target_padded, device=device)
    train_sizes = torch.tensor(train_target_sizes, device=device)
    valid_target = torch.tensor(valid_target_padded, device=device)
    valid_sizes = torch.tensor(valid_target_sizes, device=device)
    epoch_targets = make_epoch_targets(train_plan, arm, config["epochs"], len(train), config["pair_seed"])
    valid_choice = sample_target_indices(valid_plan, arm, np.arange(len(valid)),
                                         np.random.default_rng(config["validation_pair_seed"]))
    if np.any(valid_choice < 0) or np.any(valid_choice >= len(valid_target_padded)):
        raise AssertionError("validation target choice escaped validation target pool")
    if set(valid_plan["target_global_ids"]) & set(train_plan["target_global_ids"]):
        raise AssertionError("train and validation target pools overlap")
    history, best = [], None
    batch_rng = torch.Generator().manual_seed(config["seed"])
    batch_trace, train_trace = [], []
    for epoch in range(config["epochs"]):
        record = {"epoch": epoch + 1}
        for update, rows, key, selected, target, sizes in (
                (True, train, "train_nll", epoch_targets[epoch], train_target, train_sizes),
                (False, valid, "valid_nll", valid_choice, valid_target, valid_sizes)):
            model.train(update)
            if bridge is not None:
                bridge.train(update)
            order = rows[torch.randperm(len(rows), generator=batch_rng).numpy()] if update else rows
            if update:
                batch_trace.append(order.copy())
                train_trace.append(selected.copy())
            total = tokens = 0
            for start in range(0, len(order), config["batch_size"]):
                batch = order[start:start + config["batch_size"]]
                local = np.searchsorted(rows, batch) if update else np.arange(start, min(start + len(batch), len(rows)))
                targets = selected[local]
                target_global_ids = (train_target_global_ids if update else valid_target_global_ids)[targets]
                expected_ids = train_plan["target_global_ids"] if update else valid_plan["target_global_ids"]
                if not np.all(np.isin(target_global_ids, expected_ids)):
                    raise AssertionError("mapped target ID escaped the active train/validation pool")
                with torch.set_grad_enabled(update):
                    z = bridge(x[batch]) if bridge is not None else torch.zeros(len(batch), model.latent_dim, device=device)
                    coords, demands, capacity = (t.expand(len(batch), *t.shape[1:]) for t in instance)
                    logits = model.decode(coords, demands, capacity, z, target[targets], sizes[targets])[2]
                    loss = CVRPVAE.reconstruction_nll(logits, target[targets])
                    if update:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, config["grad_clip"])
                        optimizer.step()
                count = int((sizes[targets] - 1).sum().item())
                total += float(loss.item()) * count
                tokens += count
            record[key] = total / tokens
        history.append(record)
        if best is None or record["valid_nll"] < best["valid_nll"]:
            best = dict(model=copy.deepcopy(model.state_dict()), bridge=None if bridge is None else copy.deepcopy(bridge.state_dict()),
                        epoch=epoch + 1, valid_nll=record["valid_nll"])
    model.load_state_dict(best["model"])
    if bridge is not None:
        bridge.load_state_dict(best["bridge"])
    return history, best, dict(
        train_target_choices=train_trace,
        validation_target_choices=valid_choice.copy(),
        train_target_global_choices=np.asarray(train_target_global_ids)[np.asarray(train_trace, dtype=np.int64)],
        validation_target_global_choices=np.asarray(valid_target_global_ids)[valid_choice],
        target_membership_audit=dict(
            train_pool_size=int(len(train_target_global_ids)),
            validation_pool_size=int(len(valid_target_global_ids)),
            train_choices_checked=int(np.asarray(train_trace).size),
            validation_choices_checked=int(np.asarray(valid_choice).size),
            train_all_in_pool=True,
            validation_all_in_pool=True,
            train_pool_disjoint_validation_pool=True),
        batch_orders=batch_trace,
        train_target_choice_hash=hashlib.sha256(
            np.asarray(train_target_global_ids, dtype=np.int64)[np.asarray(train_trace, dtype=np.int64)].tobytes()).hexdigest(),
        validation_target_choice_hash=hashlib.sha256(np.asarray(valid_target_global_ids)[valid_choice].tobytes()).hexdigest(),
        batch_order_hash=hashlib.sha256(np.asarray(batch_trace, dtype=np.int64).tobytes()).hexdigest())
