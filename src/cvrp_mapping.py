"""Rank-matched TSP -> CVRP mapping with separately trained controls."""
import copy

import numpy as np
import torch

from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions
from src.tsp_mapping_controls import ARMS, decoder_parameters


def prepare_pairs(problem, source_latent, source_quality, target_solutions, target_quality, seed, max_pairs):
    source = np.asarray(source_latent, dtype=np.float32)
    sq, tq = np.asarray(source_quality, dtype=np.float64), np.asarray(target_quality, dtype=np.float64)
    canonical = canonicalize_solutions(problem, target_solutions)
    if source.ndim != 2 or source.shape[1] < 1 or sq.shape != (len(source),):
        raise ValueError('source latent/quality shape mismatch')
    if tq.shape != (len(canonical),) or not all(np.isfinite(x).all() for x in (source, sq, tq)):
        raise ValueError('invalid target quality or nonfinite data')
    if type(max_pairs) is not int or max_pairs < 8:
        raise ValueError('max_pairs must be an integer >= 8')
    seen, unique = set(), []
    for i, sequence in enumerate(canonical):
        key = tuple(sequence)
        if key not in seen:
            seen.add(key)
            unique.append(i)
    rng = np.random.default_rng(seed)
    source_ids, target_ids = rng.permutation(len(source)), rng.permutation(unique)
    count = min(len(source_ids), len(target_ids), max_pairs)
    if count < 8:
        raise ValueError('need >=8 source rows and distinct target solutions')
    source_ids, target_ids = source_ids[:count], target_ids[:count]
    cut = int(.8 * count)
    for part in (slice(0, cut), slice(cut, count)):
        source_ids[part] = source_ids[part][np.argsort(-sq[source_ids[part]], kind='stable')]
        target_ids[part] = target_ids[part][np.argsort(-tq[target_ids[part]], kind='stable')]
    train, valid = np.arange(cut), np.arange(cut, count)
    paired = source[source_ids]
    mean, std = paired[train].mean(0), paired[train].std(0)
    active, scale = std > 1e-6, np.where(std > 1e-6, std, 1.)
    normalized = (paired - mean) / scale
    normalized[:, ~active] = 0
    noise = rng.standard_normal(normalized.shape).astype(np.float32) * active
    permutation = np.arange(count)
    for ids in (train, valid):
        permutation[ids] = rng.permutation(ids)
    reference = np.sort(sq[source_ids[train]])
    values = sq[source_ids]
    percentile = (np.searchsorted(reference, values, side='left') +
                  np.searchsorted(reference, values, side='right')) / (2. * len(reference))
    rank = np.zeros_like(normalized)
    rank[:, 0] = 2 * percentile - 1
    conditions = dict(source=normalized, shuffled_train=normalized[permutation],
                      noise_train=noise, rank_only=rank, no_source=np.zeros_like(normalized))
    padded, sizes = problem.pack_solutions([canonical[i] for i in target_ids])
    pairing = dict(source_order=source_ids.tolist(), target_order=target_ids.tolist(),
                   shuffled_order=permutation.tolist(), source_mean=mean.tolist(), source_scale=scale.tolist(),
                   active_source_dimensions=int(active.sum()), rank_reference=reference.tolist(),
                   source_quality_definition='TSP -length; higher is better',
                   target_quality_definition='CVRP -total_route_length; higher is better',
                   source_sorted_within_each_split=all(bool(np.all(np.diff(sq[source_ids[ids]]) <= 0)) for ids in (train, valid)),
                   target_sorted_within_each_split=all(bool(np.all(np.diff(tq[target_ids[ids]]) <= 0)) for ids in (train, valid)),
                   shuffled='fixed_uniform_permutation_within_each_split; trained_independently',
                   noise='independent_standard_normal_in_active_standardized_dimensions; covariance_not_matched',
                   split='canonical_target_solutions_and_source_rows_split_before_rank_matching')
    return conditions, padded, sizes, train, valid, pairing


def train_mapping(model, bridge, instance, conditions, sequences, sizes, train, valid, config):
    params = decoder_parameters(model) + ([] if bridge is None else list(bridge.parameters()))
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    device = instance[0].device
    x = torch.tensor(conditions, dtype=torch.float32, device=device)
    target, target_sizes = torch.tensor(sequences, device=device), torch.tensor(sizes, device=device)
    generator = torch.Generator().manual_seed(config['seed'])
    history, best, state = [], float('inf'), None
    for epoch in range(1, config['epochs'] + 1):
        record = dict(epoch=epoch)
        for update, ids, key in ((True, train, 'train_nll'), (False, valid, 'valid_nll')):
            model.train(update)
            if bridge is not None:
                bridge.train(update)
            order = ids[torch.randperm(len(ids), generator=generator).numpy()] if update else ids
            total, tokens = 0., 0
            for start in range(0, len(order), config['batch_size']):
                batch = order[start:start+config['batch_size']]
                with torch.set_grad_enabled(update):
                    z = bridge(x[batch]) if bridge is not None else torch.zeros(len(batch), model.latent_dim, device=device)
                    coords, demands, capacity = (t.expand(len(batch), *t.shape[1:]) for t in instance)
                    logits = model.decode(coords, demands, capacity, z, target[batch], target_sizes[batch])[2]
                    loss = CVRPVAE.reconstruction_nll(logits, target[batch])
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite mapping loss')
                    if update:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, config['grad_clip'])):
                            raise RuntimeError('nonfinite mapping gradient')
                        optimizer.step()
                count = int((target_sizes[batch] - 1).sum().item())
                total += loss.item() * count
                tokens += count
            record[key], record[key + '_real_actions'] = total / tokens, tokens
        history.append(record)
        if record['valid_nll'] < best:
            best = record['valid_nll']
            state = dict(model=copy.deepcopy(model.state_dict()),
                         bridge=None if bridge is None else copy.deepcopy(bridge.state_dict()),
                         epoch=epoch, valid_nll=best)
    model.load_state_dict(state['model'])
    if bridge is not None:
        bridge.load_state_dict(state['bridge'])
    return history, state


@torch.no_grad()
def evaluate_mapping(model, bridge, instance, conditions, sequences, sizes, problem, seed, sample):
    model.eval()
    if bridge is not None:
        bridge.eval()
    device = instance[0].device
    x = torch.tensor(conditions, dtype=torch.float32, device=device)
    z = bridge(x) if bridge is not None else torch.zeros(len(x), model.latent_dim, device=device)
    coordinates, demands, capacity = (t.expand(len(x), *t.shape[1:]) for t in instance)
    target, lengths = torch.tensor(sequences, device=device), torch.tensor(sizes, device=device)
    logits = model.decode(coordinates, demands, capacity, z, target, lengths)[2]
    nll = model.reconstruction_nll(logits, target).item()
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    padded, out_sizes, _ = model.decode(coordinates, demands, capacity, z, sample=sample)
    padded, out_sizes = padded.cpu().numpy(), out_sizes.cpu().numpy()
    solutions = problem.unpack_solutions(padded, out_sizes)
    costs = problem.evaluate_batch(solutions)
    canonical, truth = canonicalize_solutions(problem, solutions), problem.unpack_solutions(sequences, sizes)
    keys = [tuple(sequence) for sequence in canonical]
    report = dict(valid_rate=1., teacher_forced_nll=nll, mean_length=float(costs.mean()),
                  best_length=float(costs.min()), mean_route_count=float(np.mean([np.count_nonzero(s == 0)-1 for s in solutions])),
                  unique_solutions=len(set(keys)),
                  exact_solution_rate=float(np.mean([a == tuple(b) for a, b in zip(keys, truth)])),
                  objective_queries=len(costs))
    return report, padded, out_sizes, costs
