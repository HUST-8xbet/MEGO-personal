"""Quality-stratified latent permutation and label-free generation protocol."""
import hashlib

import numpy as np
import torch

ARMS = ('source', 'within_rank_shuffle', 'rank_only')


def percentile_from_train(quality, train_reference):
    reference = np.sort(np.asarray(train_reference, dtype=np.float64))
    values = np.asarray(quality, dtype=np.float64)
    if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError('quality reference must be a finite nonempty vector')
    return (np.searchsorted(reference, values, side='left') +
            np.searchsorted(reference, values, side='right')) / (2. * len(reference))


def _drift_report(ids, permutation, groups, percentiles, latent, group_count):
    moved = permutation[ids] != ids
    delta = 100. * (percentiles[permutation[ids]] - percentiles[ids])
    changed = np.any(latent[permutation[ids]] != latent[ids], axis=1)
    report = dict(count=len(ids), moved_count=int(moved.sum()), moved_fraction=float(moved.mean()),
                  changed_latent_fraction=float(changed.mean()),
                  mean_abs_percentile_change_pp=float(np.abs(delta).mean()),
                  max_abs_percentile_change_pp=float(np.abs(delta).max()),
                  p95_abs_percentile_change_pp=float(np.quantile(np.abs(delta), .95)),
                  mean_signed_percentile_change_pp=float(delta.mean()),
                  rmse_percentile_change_pp=float(np.sqrt(np.mean(delta**2))))
    table = []
    for group in range(group_count):
        local = np.flatnonzero(groups[ids] == group)
        table.append(dict(group=group, count=len(local),
                          moved_fraction=float(moved[local].mean()) if len(local) else None,
                          mean_abs_percentile_change_pp=float(np.abs(delta[local]).mean()) if len(local) else None,
                          max_abs_percentile_change_pp=float(np.abs(delta[local]).max()) if len(local) else None))
    report['groups'] = table
    return report


def within_rank_permutation(latent, quality, train_ids, valid_ids, shuffle_seed,
                            training_seed, group_count=10):
    latent, quality = np.asarray(latent), np.asarray(quality, dtype=np.float64)
    train_ids, valid_ids = np.asarray(train_ids), np.asarray(valid_ids)
    if latent.ndim != 2 or quality.shape != (len(latent),) or not np.isfinite(quality).all() or not np.isfinite(latent).all():
        raise ValueError('invalid latent / quality arrays')
    if (train_ids.dtype.kind not in 'iu' or valid_ids.dtype.kind not in 'iu'
            or not len(train_ids) or not len(valid_ids)):
        raise ValueError('need nonempty integer train and validation partitions')
    union = np.concatenate([train_ids, valid_ids])
    if not np.array_equal(np.sort(union), np.arange(len(quality))):
        raise ValueError('partitions must cover rows exactly once and never overlap')
    if type(group_count) is not int or group_count < 1:
        raise ValueError('group_count must be positive')
    reference = quality[train_ids]
    # NumPy default linear quantiles, fitted ONLY on train. Ties stay together.
    cutpoints = np.quantile(reference, np.arange(1, group_count) / group_count)
    groups = np.searchsorted(cutpoints, quality, side='right')
    percentiles = percentile_from_train(quality, reference)
    permutation = np.arange(len(quality))
    seed_specs = {}
    for split_number, (name, ids) in enumerate((('train', train_ids), ('validation', valid_ids))):
        spec = [int(shuffle_seed), int(training_seed), split_number]
        rng = np.random.default_rng(np.random.SeedSequence(spec))
        for group in range(group_count):
            members = ids[groups[ids] == group]
            permutation[members] = rng.permutation(members)
        seed_specs[name] = spec
    if not np.array_equal(groups, groups[permutation]):
        raise RuntimeError('shuffle crossed quality groups')
    for ids in (train_ids, valid_ids):
        if not np.array_equal(np.sort(permutation[ids]), np.sort(ids)):
            raise RuntimeError('shuffle crossed train/validation')
    report = dict(group_count=group_count, group_fit='source_train_only',
                  quality_cutpoints=cutpoints.tolist(), quantile_method='linear', boundary_side='right',
                  train_quality_min=float(reference.min()), train_quality_max=float(reference.max()),
                  percentile_boundaries=(100*np.linspace(0, 1, group_count+1)).tolist(),
                  percentile_definition='midpoint_empirical_CDF_of_source_train; pp units 0..100',
                  ties='kept_together; empty/singleton_bins_allowed; no_forced_derangement',
                  rng='NumPy SeedSequence [shuffle_seed, training_seed, split_number]',
                  seed_sequences=seed_specs,
                  train=_drift_report(train_ids, permutation, groups, percentiles, latent, group_count),
                  validation=_drift_report(valid_ids, permutation, groups, percentiles, latent, group_count))
    arrays = dict(permutation=permutation, group_ids=groups,
                  percentile_before=100*percentiles, percentile_after=100*percentiles[permutation])
    return latent[permutation].copy(), report, arrays


def generation_plan(source_pool_positions, rounds, k, selection_seed):
    """Pure source-ID selection. No target solution/quality is accepted."""
    pool = np.asarray(source_pool_positions)
    if pool.ndim != 1 or pool.dtype.kind not in 'iu' or len(np.unique(pool)) != len(pool):
        raise ValueError('source pool must contain distinct integer IDs')
    if type(rounds) is not int or type(k) is not int or min(rounds, k) < 1:
        raise ValueError('rounds and k must be positive integers')
    if len(pool) < rounds*k:
        raise ValueError('source validation pool is too small for distinct rounds*K candidates')
    return np.random.default_rng(selection_seed).permutation(pool)[:rounds*k].reshape(rounds, k)


def generation_conditions(original_latent, original_rank, plan):
    """The shuffled-training arm receives ORIGINAL source IDs at inference."""
    return dict(source=original_latent[plan], within_rank_shuffle=original_latent[plan],
                rank_only=original_rank[plan])


def derived_seed(base, training_seed, round_number):
    return int(np.random.SeedSequence([int(base), int(training_seed), int(round_number)]).generate_state(1)[0])


def state_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def batch_order_digest(train_ids, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    digest = hashlib.sha256()
    for _ in range(epochs):
        order = train_ids[torch.randperm(len(train_ids), generator=generator).numpy()]
        digest.update(np.asarray(order, dtype=np.int64).tobytes())
    return digest.hexdigest()


def summarize_generation(lengths, route_counts, best_adaptation=None):
    lengths, route_counts = np.asarray(lengths), np.asarray(route_counts)
    if lengths.ndim != 2 or lengths.shape != route_counts.shape or not np.isfinite(lengths).all():
        raise ValueError('generation arrays must be finite rounds x K matrices')
    best = lengths.min(1)
    result = dict(mean_length=float(lengths.mean()), mean_route_count=float(route_counts.mean()),
                valid_rate=1., objective_queries=int(lengths.size), rounds=lengths.shape[0], k=lengths.shape[1],
                round_best_lengths=best.tolist(), best_of_k_mean=float(best.mean()),
                best_of_k_std=float(best.std(ddof=1)) if len(best)>1 else None,
                best_of_k_std_ddof=1, best_of_all=float(best.min()))
    if best_adaptation is not None:
        result.update(best_adaptation_length=float(best_adaptation),
                      best_final_including_adaptation=float(min(best_adaptation, best.min())),
                      round_best_including_adaptation=np.minimum(best, best_adaptation).tolist())
    return result


@torch.no_grad()
def generate_rounds(model, bridge, instance, common_source_conditions, problem,
                    sampling_base_seed, training_seed, sample):
    """Decode without teacher targets, target qualities or objective feedback."""
    model.eval()
    bridge.eval()
    device = instance[0].device
    rounds, k, _ = common_source_conditions.shape
    arrays, sizes, lengths, counts, round_seeds = [], [], [], [], []
    for r in range(rounds):
        action_seed = derived_seed(sampling_base_seed, training_seed, r)
        round_seeds.append(action_seed)
        torch.manual_seed(action_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(action_seed)
        z = bridge(torch.tensor(common_source_conditions[r], dtype=torch.float32, device=device))
        coords, demands, capacity = (t.expand(k, *t.shape[1:]) for t in instance)
        output, out_sizes, _ = model.decode(coords, demands, capacity, z, sample=sample)
        output, out_sizes = output.cpu().numpy(), out_sizes.cpu().numpy()
        solutions = problem.unpack_solutions(output, out_sizes)  # raises on any infeasibility
        costs = problem.evaluate_batch(solutions)
        arrays.append(output)
        sizes.append(out_sizes)
        lengths.append(costs)
        counts.append([np.count_nonzero(sequence == 0)-1 for sequence in solutions])
    arrays, sizes, lengths, counts = map(np.asarray, (arrays, sizes, lengths, counts))
    report = summarize_generation(lengths, counts)
    report['action_seeds_by_round'] = round_seeds
    return report, dict(padded=arrays, sequence_lengths=sizes, lengths=lengths, route_counts=counts)


@torch.no_grad()
def validation_nll(model, bridge, instance, conditions, targets, sizes, batch_size):
    model.eval()
    bridge.eval()
    device = instance[0].device
    total, tokens = 0., 0
    for start in range(0, len(targets), batch_size):
        sl = slice(start, start+batch_size)
        count = len(targets[sl])
        z = bridge(torch.tensor(conditions[sl], dtype=torch.float32, device=device))
        coords, demands, cap = (t.expand(count, *t.shape[1:]) for t in instance)
        target = torch.tensor(targets[sl], dtype=torch.long, device=device)
        lens = torch.tensor(sizes[sl], dtype=torch.long, device=device)
        logits = model.decode(coords, demands, cap, z, target, lens)[2]
        nll = model.reconstruction_nll(logits, target).item()
        n = int((lens-1).sum().item())
        total += nll*n
        tokens += n
    return total/tokens


def summarize_seeds(seed_reports):
    def stats(values):
        values = np.asarray(values, dtype=float)
        return dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values)>1 else None,
                    std_ddof=1, training_seed_count=len(values))
    output = {}
    for arm in ARMS:
        records = [seed['arms'][arm] for seed in seed_reports]
        out = dict(validation_nll=stats([r['best_validation_nll'] for r in records]))
        for mode in ('greedy', 'sampled'):
            out[mode] = {key: stats([r[mode][key] for r in records]) for key in
                         ('mean_length', 'mean_route_count', 'valid_rate', 'best_of_all',
                          'best_of_k_mean', 'best_of_k_std', 'best_final_including_adaptation')}
        output[arm] = out
    return output
