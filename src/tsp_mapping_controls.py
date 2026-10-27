"""Independently trained controls for the Max-Cut -> TSP functional baseline."""
import copy

import numpy as np
import torch
from torch.nn import functional as F

from src.regression.tsp_vae import canonicalize_tours

ARMS = ('source', 'shuffled_train', 'noise_train', 'rank_only', 'no_source')


def prepare_controls(source_latent, source_quality, target_tours, target_quality, seed, max_pairs):
    source = np.asarray(source_latent, dtype=np.float32)
    sq = np.asarray(source_quality, dtype=np.float64)
    tours = canonicalize_tours(target_tours)
    tq = np.asarray(target_quality, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 1 or sq.shape != (len(source),):
        raise ValueError('invalid source latent/quality shapes')
    if tq.shape != (len(tours),):
        raise ValueError('invalid target quality shape')
    if not all(np.isfinite(x).all() for x in (source, sq, tq)):
        raise ValueError('nonfinite data')
    if type(max_pairs) is not int or max_pairs < 8:
        raise ValueError('max_pairs must be an integer >= 8')
    _, unique = np.unique(tours, axis=0, return_index=True)
    rng = np.random.default_rng(seed)
    # Choose both banks randomly, then split BEFORE matching by quality.
    target_ids = rng.permutation(np.sort(unique))
    source_ids = rng.permutation(len(source))
    count = min(len(source_ids), len(target_ids), max_pairs)
    if count < 8:
        raise ValueError('need >= 8 source rows and distinct target cycles')
    source_ids, target_ids = source_ids[:count], target_ids[:count]
    cut = int(0.8 * count)
    for part in (slice(0, cut), slice(cut, count)):
        source_ids[part] = source_ids[part][np.argsort(-sq[source_ids[part]], kind='stable')]
        target_ids[part] = target_ids[part][np.argsort(-tq[target_ids[part]], kind='stable')]
    train_ids, valid_ids = np.arange(cut), np.arange(cut, count)
    paired_source = source[source_ids]
    mean, raw_std = paired_source[train_ids].mean(0), paired_source[train_ids].std(0)
    active = raw_std > 1e-6
    scale = np.where(active, raw_std, 1.)
    normalized = (paired_source - mean) / scale
    normalized[:, ~active] = 0
    # Independent noise during TRAINING and validation, in standardized space.
    noise = rng.standard_normal(normalized.shape).astype(np.float32) * active
    permutation = np.arange(count)
    for ids in (train_ids, valid_ids):
        permutation[ids] = rng.permutation(ids)
    # Source quality only; empirical percentile fitted on source TRAINING rows.
    reference = np.sort(sq[source_ids[train_ids]])
    values = sq[source_ids]
    percentile = (np.searchsorted(reference, values, side='left') +
                  np.searchsorted(reference, values, side='right')) / (2. * len(reference))
    rank = np.zeros_like(normalized)
    rank[:, 0] = 2 * percentile - 1
    conditions = dict(source=normalized, shuffled_train=normalized[permutation],
                      noise_train=noise, rank_only=rank, no_source=np.zeros_like(normalized))
    metadata = dict(source_mean=mean.tolist(), source_scale=scale.tolist(),
                    active_source_dimensions=int(active.sum()),
                    noise='independent_standard_normal_on_active_standardized_dimensions; covariance_not_matched',
                    shuffled='fixed_uniform_random_permutation_within_each_split_during_training_and_evaluation',
                    split='unique_target_cycles; disjoint_source_rows; split_before_rank_matching',
                    source_order=source_ids.tolist(), target_order=target_ids.tolist(),
                    shuffled_order=permutation.tolist(), rank_reference=reference.tolist())
    return conditions, tours[target_ids], train_ids, valid_ids, metadata


def decoder_parameters(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in (model.city_embedding, model.decoder_init, model.decoder_cell,
                   model.key, model.query, model.pointer):
        for p in module.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def train_control(model, bridge, coordinates, conditions, tours, train_ids, valid_ids, config):
    params = decoder_parameters(model) + ([] if bridge is None else list(bridge.parameters()))
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    device = coordinates.device
    condition_tensor = torch.tensor(conditions, dtype=torch.float32, device=device)
    targets = torch.tensor(tours, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(config['seed'])
    history, best, state = [], float('inf'), None
    for epoch in range(1, config['epochs'] + 1):
        row = dict(epoch=epoch)
        for update, ids, key in ((True, train_ids, 'train_nll'), (False, valid_ids, 'valid_nll')):
            model.train(update)
            if bridge is not None:
                bridge.train(update)
            order = ids[torch.randperm(len(ids), generator=generator).numpy()] if update else ids
            total = 0.
            for start in range(0, len(order), config['batch_size']):
                batch = order[start:start + config['batch_size']]
                with torch.set_grad_enabled(update):
                    z = bridge(condition_tensor[batch]) if bridge is not None else torch.zeros(
                        len(batch), model.latent_dim, device=device)
                    logits = model.decode(coordinates.expand(len(batch), -1, -1), z,
                                          teacher_tours=targets[batch])[1]
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[batch, 1:].reshape(-1))
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite loss')
                    if update:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params, config['grad_clip'])):
                            raise RuntimeError('nonfinite gradient')
                        optimizer.step()
                total += loss.item() * len(batch)
            row[key] = total / len(order)
        history.append(row)
        if row['valid_nll'] < best:
            best = row['valid_nll']
            state = dict(model=copy.deepcopy(model.state_dict()),
                         bridge=None if bridge is None else copy.deepcopy(bridge.state_dict()),
                         epoch=epoch, valid_nll=best)
    model.load_state_dict(state['model'])
    if bridge is not None:
        bridge.load_state_dict(state['bridge'])
    return history, state


@torch.no_grad()
def evaluate_control(model, bridge, coordinates, conditions, tours, problem, seed, sample):
    model.eval()
    if bridge is not None:
        bridge.eval()
    device = coordinates.device
    x = torch.tensor(conditions, dtype=torch.float32, device=device)
    z = bridge(x) if bridge is not None else torch.zeros(len(x), model.latent_dim, device=device)
    coords = coordinates.expand(len(x), -1, -1)
    target = torch.tensor(tours, dtype=torch.long, device=device)
    teacher = model.decode(coords, z, teacher_tours=target)[1]
    nll = F.cross_entropy(teacher.reshape(-1, problem.dimension), target[:, 1:].reshape(-1)).item()
    # Common action-sampling seed across arms. No objective-based selection.
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    generated = model.decode(coords, z, sample=sample)[0].cpu().numpy()
    costs = problem.evaluate_batch(generated)
    canonical = canonicalize_tours(generated)
    return dict(valid_rate=1., teacher_forced_nll=nll, mean_length=float(costs.mean()),
                best_length=float(costs.min()), exact_cycle_rate=float(np.all(canonical == tours, axis=1).mean()),
                unique_cycles=len(np.unique(canonical, axis=0)), objective_queries=len(costs)), generated
