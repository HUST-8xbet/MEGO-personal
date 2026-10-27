"""Fixed-checkpoint source-quality probe; no target labels enter selection/decode."""
import numpy as np
import torch

from src.cvrp_within_rank import percentile_from_train


class InsufficientGroupError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('A source validation quintile has fewer than {} rows: {}. '
                         'STOP; no replacement/resampling or group change authorized.'.format(
                             audit['selected_per_group'], audit['validation_counts']))


def quality_plan(source_quality, train, validation, per_group, selection_seed):
    """Train-fit quintiles, sampling WITHOUT replacement from validation only."""
    q = np.asarray(source_quality, dtype=np.float64)
    train, validation = np.asarray(train), np.asarray(validation)
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError('source quality must be a finite vector')
    for ids in (train, validation):
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids):
            raise ValueError('partitions must be nonempty integer vectors')
    if not np.array_equal(np.sort(np.r_[train, validation]), np.arange(len(q))):
        raise ValueError('source train/validation must partition all rows exactly once')
    if type(per_group) is not int or per_group < 1:
        raise ValueError('per_group must be a positive integer')
    cuts = np.quantile(q[train], np.arange(1, 5)/5.)
    groups = np.searchsorted(cuts, q, side='right')
    percentile = percentile_from_train(q, q[train])
    counts = [int(np.sum(groups[validation] == g)) for g in range(5)]
    audit = dict(group_order='Q1 worst source quality -> Q5 best; quality=-TSP length',
                 quality_cutpoints=cuts.tolist(), boundary_fit='source_train_only',
                 quantile_method='linear', boundary_side='right', ties='kept together',
                 percentile_definition='midpoint source-train empirical CDF, 0..100',
                 train_counts=[int(np.sum(groups[train] == g)) for g in range(5)],
                 validation_counts=counts, selected_per_group=per_group,
                 selection_seed=selection_seed,
                 selection='uniform without replacement; source IDs only; fixed across arms/training seeds')
    if min(counts) < per_group:
        raise InsufficientGroupError(audit)
    selected = []
    for g in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([selection_seed, g]))
        selected.append(rng.permutation(validation[groups[validation] == g])[:per_group])
    return np.asarray(selected), groups, percentile, audit


def sampling_seeds(base_seed, training_seed, groups=5, rounds=5):
    """Same group/round action RNG seed for all arms; independent across rounds."""
    return np.array([[int(np.random.SeedSequence([base_seed, training_seed, g, r]).generate_state(1)[0])
                      for r in range(rounds)] for g in range(groups)], dtype=np.int64)


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Spearman inputs must be finite vectors')
    order = np.argsort(values, kind='stable')
    ranked = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranked[order[i:j]] = (i+j-1)/2. + 1
        i = j
    return ranked


def spearman(percentiles, per_latent_mean_length):
    x, y = average_ranks(percentiles), average_ranks(per_latent_mean_length)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError('Spearman needs equal vectors with >=2 observations')
    x, y = x-x.mean(), y-y.mean()
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    rho = float(np.clip(np.dot(x, y)/denominator, -1., 1.)) if denominator > 0 else None
    return dict(rho=rho, latent_count=len(x), desired_direction='negative',
                unit='one selected source latent; target length averaged across sampling rounds',
                ties='average ranks', undefined_reason='constant percentile or mean length' if rho is None else None,
                inference='descriptive; no iid p-value or significance claim')


@torch.no_grad()
def decode_probe(model, bridge, instance, source_conditions, action_seeds, problem, sample):
    """No target solution, target quality, incumbent, or selection based on cost."""
    model.eval(); bridge.eval()
    groups, count, _ = source_conditions.shape
    seeds = np.asarray(action_seeds)
    if seeds.ndim != 2 or seeds.shape[0] != groups or seeds.shape[1] < 1:
        raise ValueError('action seeds must have shape groups x rounds')
    device = instance[0].device
    padded, sizes = [], []
    for g in range(groups):
        group_padded, group_sizes = [], []
        z = bridge(torch.as_tensor(source_conditions[g], dtype=torch.float32, device=device))
        for seed in seeds[g]:
            torch.manual_seed(int(seed))
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(int(seed))
            coords, demands, cap = (t.expand(count, *t.shape[1:]) for t in instance)
            seq, length, _ = model.decode(coords, demands, cap, z, sample=sample)
            group_padded.append(seq.cpu().numpy()); group_sizes.append(length.cpu().numpy())
        padded.append(group_padded); sizes.append(group_sizes)
    padded, sizes = np.asarray(padded), np.asarray(sizes)
    shape = sizes.shape
    flat_padded, flat_sizes = padded.reshape(-1, padded.shape[-1]), sizes.ravel()
    valid = np.zeros(len(flat_sizes), dtype=bool)
    lengths = np.full(len(flat_sizes), np.nan)
    routes = np.full(len(flat_sizes), np.nan)
    # Evaluate every attempted candidate once, including rejected infeasible candidates.
    # Never repair, replace, deduplicate or retry a candidate.
    for i, (seq, length) in enumerate(zip(flat_padded, flat_sizes)):
        try:
            solution = problem.unpack_solutions(seq[None], np.array([length]))[0]
            lengths[i] = problem.evaluate(solution)
            routes[i] = np.count_nonzero(solution == 0)-1
            valid[i] = True
        except ValueError:
            pass
    return dict(padded=padded, sequence_lengths=sizes, valid=valid.reshape(shape),
                lengths=lengths.reshape(shape), route_counts=routes.reshape(shape))


def summarize_mode(arrays, incumbent):
    lengths, routes, valid = arrays['lengths'], arrays['route_counts'], arrays['valid']
    if lengths.ndim != 2 or routes.shape != lengths.shape or valid.shape != lengths.shape:
        raise ValueError('need rounds x selected-latents result matrices')
    best = [float(row[mask].min()) if mask.any() else None for row, mask in zip(lengths, valid)]
    complete = all(x is not None for x in best)
    best_all = float(lengths[valid].min()) if valid.any() else None
    wins = int(np.sum(valid & (lengths < incumbent)))
    return dict(attempted_candidates=int(lengths.size), valid_candidates=int(valid.sum()),
                valid_rate=float(valid.mean()), mean_length=float(lengths[valid].mean()) if valid.any() else None,
                mean_route_count=float(routes[valid].mean()) if valid.any() else None,
                means_denominator='valid candidates only; valid_rate denominator=all attempted candidates',
                round_best_lengths=best,
                best_of_k_mean=float(np.mean(best)) if complete else None,
                best_of_k_std_across_sampling_rounds=float(np.std(best, ddof=1)) if complete and len(best)>1 else None,
                sampling_round_count=len(best), selected_latent_count=lengths.shape[1], std_ddof=1,
                best_generated_length=best_all, better_than_adaptation_count=wins,
                better_than_adaptation_rate=float(wins/lengths.size),
                improvement_definition='strict length < exact prior adaptation incumbent; invalid counts as no improvement',
                best_final_including_adaptation=min(incumbent, best_all) if best_all is not None else incumbent)


def report_probe(greedy, sampled, percentile, incumbent):
    groups = []
    for g in range(sampled['lengths'].shape[0]):
        groups.append(dict(group=g+1, greedy=summarize_mode({k:v[g] for k,v in greedy.items()}, incumbent),
                           sampled=summarize_mode({k:v[g] for k,v in sampled.items()}, incumbent)))
    all_valid = bool(sampled['valid'].all())
    mean_per_latent = sampled['lengths'].mean(axis=1)
    correlation = spearman(percentile.ravel(), mean_per_latent.ravel()) if all_valid else dict(
        rho=None, latent_count=int(percentile.size), undefined_reason='invalid sampled candidates; do not silently drop latents')
    def flatten(arrays):
        return {k:v.reshape(-1, v.shape[-1]) for k,v in arrays.items() if k in ('lengths','route_counts','valid')}
    result = dict(groups=groups, spearman=correlation,
                greedy_all_groups=summarize_mode(flatten(greedy), incumbent),
                sampled_all_groups=summarize_mode(flatten(sampled), incumbent),
                best_final_greedy_and_sampled_union=min([incumbent] +
                    [x for x in (float(greedy['lengths'][greedy['valid']].min()) if greedy['valid'].any() else None,
                                  float(sampled['lengths'][sampled['valid']].min()) if sampled['valid'].any() else None)
                      if x is not None]))
    for name in ('greedy_all_groups', 'sampled_all_groups'):
        for key in ('round_best_lengths','best_of_k_mean','best_of_k_std_across_sampling_rounds',
                    'sampling_round_count','selected_latent_count','std_ddof'):
            result[name].pop(key)
    return result, mean_per_latent
