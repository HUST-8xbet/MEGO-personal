"""H_residual preprocessing and mapping helpers for the TSP -> CVRP pilot."""
import copy
import hashlib
import json

import numpy as np
import torch

from src.cvrp_mapping import decoder_parameters
from src.cvrp_within_rank import summarize_generation
from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions


ARMS = ('rank_residual', 'rank_residual_shuffle', 'rank_residual_zero')


class ResidualBridge(torch.nn.Module):
    """Explicit rank plus residual bridge; residual branch has no bias."""
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.rank = torch.nn.Linear(3, target_dim)
        self.residual = torch.nn.Linear(source_dim, target_dim, bias=False)

    def forward(self, rank, residual):
        return self.rank(rank) + self.residual(residual)


def array_sha256(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def percentile_from_train(quality, train_ids):
    quality = np.asarray(quality, dtype=np.float64); train_ids = np.asarray(train_ids, dtype=np.int64)
    reference = np.sort(quality[train_ids])
    return (np.searchsorted(reference, quality, side='left') +
            np.searchsorted(reference, quality, side='right')) / (2.0 * len(reference))


def residual_preprocess(latent, quality, train_ids, valid_ids, bin_count=10, std_floor=1e-8):
    latent = np.asarray(latent, dtype=np.float32); quality = np.asarray(quality, dtype=np.float64)
    train_ids = np.asarray(train_ids, dtype=np.int64); valid_ids = np.asarray(valid_ids, dtype=np.int64)
    if latent.ndim != 2 or quality.shape != (len(latent),) or not np.isfinite(latent).all() or not np.isfinite(quality).all():
        raise ValueError('invalid latent/quality arrays')
    if np.intersect1d(train_ids, valid_ids).size or not np.array_equal(np.sort(np.r_[train_ids, valid_ids]), np.arange(len(latent))):
        raise ValueError('train and validation positions must partition the bank')
    percentile = percentile_from_train(quality, train_ids)
    bins = np.minimum((percentile * bin_count).astype(np.int64), bin_count - 1)
    global_mean = latent[train_ids].mean(0)
    means = np.empty((bin_count, latent.shape[1]), dtype=np.float32); counts = []
    fallback = []
    for b in range(bin_count):
        ids = train_ids[bins[train_ids] == b]; counts.append(int(len(ids)))
        if len(ids): means[b] = latent[ids].mean(0)
        else: means[b] = global_mean; fallback.append(b)
    raw = latent - means[bins]
    scale = raw[train_ids].std(0)
    active = scale >= std_floor
    safe_scale = np.where(active, scale, 1.0)
    residual = raw / safe_scale
    residual[:, ~active] = 0.0
    rank = np.column_stack([percentile, percentile ** 2, percentile ** 3]).astype(np.float32)
    audit = dict(bin_count=bin_count, bin_counts_train=counts, fallback_bins=fallback,
                 percentile_definition='midpoint empirical CDF fit on source train only',
                 rank_bins='fixed percentile bins [0,.1),...,[.9,1]', std_floor=std_floor,
                 active_dimensions=np.flatnonzero(active).astype(int).tolist(), active_dimension_count=int(active.sum()),
                 latent_mean_train=global_mean.tolist(), bin_means=means.tolist(), residual_scales=safe_scale.tolist(),
                 residual_raw_sha256=array_sha256(raw), residual_sha256=array_sha256(residual),
                 rank_sha256=array_sha256(rank), train_positions=train_ids.tolist(), validation_positions=valid_ids.tolist())
    return rank, residual.astype(np.float32), bins, percentile, audit


def shuffle_residual_within_bins(residual, bins, train_ids, valid_ids, seed):
    residual = np.asarray(residual, dtype=np.float32); bins = np.asarray(bins, dtype=np.int64)
    permutation = np.arange(len(residual), dtype=np.int64); specs = {}
    for split_number, (name, ids) in enumerate((('train', train_ids), ('validation', valid_ids))):
        specs[name] = [int(seed), split_number]
        for b in range(int(bins.max()) + 1):
            members = np.asarray(ids)[bins[np.asarray(ids)] == b]
            if len(members) > 1:
                permutation[members] = np.random.default_rng(np.random.SeedSequence(specs[name] + [b])).permutation(members)
    if not np.array_equal(bins, bins[permutation]): raise RuntimeError('residual shuffle crossed rank bins')
    if not np.array_equal(np.sort(permutation[train_ids]), np.sort(train_ids)) or not np.array_equal(np.sort(permutation[valid_ids]), np.sort(valid_ids)):
        raise RuntimeError('residual shuffle crossed partitions')
    moved = permutation != np.arange(len(residual))
    audit = dict(seed=seed, permutation=permutation.tolist(), permutation_sha256=array_sha256(permutation),
                 train_moved_count=int(moved[train_ids].sum()), validation_moved_count=int(moved[valid_ids].sum()),
                 train_swap_rate=float(moved[train_ids].mean()), validation_swap_rate=float(moved[valid_ids].mean()),
                 singleton_bins_train=[int(b) for b in range(int(bins.max())+1) if np.sum(bins[train_ids] == b) <= 1],
                 singleton_bins_validation=[int(b) for b in range(int(bins.max())+1) if np.sum(bins[valid_ids] == b) <= 1],
                 rank_bins_unchanged=True)
    return residual[permutation].copy(), permutation, audit


def assert_loss_partition(target_global_ids, train_positions, valid_positions):
    target_global_ids = np.asarray(target_global_ids, dtype=np.int64)
    train_global = set(target_global_ids[np.asarray(train_positions)])
    valid_global = set(target_global_ids[np.asarray(valid_positions)])
    if train_global & valid_global: raise AssertionError('target train/validation global IDs overlap')
    if len(train_global) != len(train_positions) or len(valid_global) != len(valid_positions):
        raise AssertionError('target global IDs are not unique within partition')
    return dict(train_global_ids=[int(x) for x in sorted(train_global)],
                validation_global_ids=[int(x) for x in sorted(valid_global)], disjoint=True)


def train_mapping(model, bridge, instance, rank, residual, target_padded, target_sizes, target_global_ids,
                  train_positions, valid_positions, config):
    audit = assert_loss_partition(target_global_ids, train_positions, valid_positions)
    params = decoder_parameters(model) + list(bridge.parameters())
    optimizer = torch.optim.Adam(params, lr=config['learning_rate'])
    device = instance[0].device; rank_t = torch.tensor(rank, dtype=torch.float32, device=device)
    residual_t = torch.tensor(residual, dtype=torch.float32, device=device)
    target = torch.tensor(target_padded, dtype=torch.long, device=device); sizes = torch.tensor(target_sizes, dtype=torch.long, device=device)
    target_global_ids = np.asarray(target_global_ids, dtype=np.int64); generator = torch.Generator().manual_seed(config['seed'])
    history=[]; best=float('inf'); state=None; train_epoch_ids=[]; valid_epoch_ids=[]
    for epoch in range(1, config['epochs'] + 1):
        record=dict(epoch=epoch)
        for update, positions, key in ((True, train_positions, 'train_nll'), (False, valid_positions, 'valid_nll')):
            model.train(update); bridge.train(update); order=np.asarray(positions)
            if update: order=order[torch.randperm(len(order), generator=generator).numpy()]
            epoch_ids=[]; total=tokens=0
            for start in range(0, len(order), config['batch_size']):
                batch=order[start:start+config['batch_size']]; epoch_ids.extend(target_global_ids[batch].tolist())
                with torch.set_grad_enabled(update):
                    z=bridge(rank_t[batch], residual_t[batch]); coords,demands,capacity=(t.expand(len(batch),*t.shape[1:]) for t in instance)
                    logits=model.decode(coords,demands,capacity,z,target[batch],sizes[batch])[2]
                    loss=CVRPVAE.reconstruction_nll(logits,target[batch])
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite residual mapping loss')
                    if update:
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(params,config['grad_clip'])): raise RuntimeError('nonfinite residual mapping gradient')
                        optimizer.step()
                count=int((sizes[batch]-1).sum().item()); total += loss.item()*count; tokens += count
            record[key]=total/tokens; record[key+'_real_actions']=tokens
            if update: train_epoch_ids.append(epoch_ids)
            else: valid_epoch_ids.append(epoch_ids)
        history.append(record)
        if record['valid_nll'] < best:
            best=record['valid_nll']; state=dict(model=copy.deepcopy(model.state_dict()),bridge=copy.deepcopy(bridge.state_dict()),epoch=epoch,valid_nll=best)
    model.load_state_dict(state['model']); bridge.load_state_dict(state['bridge'])
    audit.update(train_epoch_target_global_ids=train_epoch_ids, valid_epoch_target_global_ids=valid_epoch_ids,
                 train_epoch_target_id_sha256=[array_sha256(np.asarray(x,dtype=np.int64)) for x in train_epoch_ids],
                 valid_epoch_target_id_sha256=[array_sha256(np.asarray(x,dtype=np.int64)) for x in valid_epoch_ids])
    return history, state, audit


@torch.no_grad()
def evaluate_mapping(model, bridge, instance, rank, residual, target_padded, target_sizes, problem, seed, sample):
    model.eval(); bridge.eval(); device=instance[0].device; n=len(rank)
    rank_t=torch.tensor(rank,dtype=torch.float32,device=device); residual_t=torch.tensor(residual,dtype=torch.float32,device=device)
    target=torch.tensor(target_padded,dtype=torch.long,device=device); sizes=torch.tensor(target_sizes,dtype=torch.long,device=device)
    coords,demands,capacity=(t.expand(n,*t.shape[1:]) for t in instance); z=bridge(rank_t,residual_t)
    nll=CVRPVAE.reconstruction_nll(model.decode(coords,demands,capacity,z,target,sizes)[2],target).item()
    torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    padded,out_sizes,_=model.decode(coords,demands,capacity,z,sample=sample); padded,out_sizes=padded.cpu().numpy(),out_sizes.cpu().numpy()
    solutions=problem.unpack_solutions(padded,out_sizes); costs=problem.evaluate_batch(solutions)
    keys=[tuple(x) for x in canonicalize_solutions(problem,solutions)]
    return dict(valid_rate=1.,teacher_forced_nll=nll,mean_length=float(costs.mean()),best_length=float(costs.min()),
                mean_route_count=float(np.mean([np.count_nonzero(x==0)-1 for x in solutions])),unique_solutions=len(set(keys)),objective_queries=len(costs)), padded,out_sizes,costs


def generation_rounds(model, bridge, instance, rank, residual, plan, problem, seed_base, training_seed, sample):
    reports=[]; all_padded=[]; all_sizes=[]; all_lengths=[]; source_ids=[]
    device=instance[0].device
    for round_number, ids in enumerate(plan):
        action_seed=int(np.random.SeedSequence([seed_base,training_seed,round_number]).generate_state(1)[0]); torch.manual_seed(action_seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(action_seed)
        count=len(ids); r=torch.tensor(rank[ids],dtype=torch.float32,device=device); u=torch.tensor(residual[ids],dtype=torch.float32,device=device)
        z=bridge(r,u); coords,demands,capacity=(t.expand(count,*t.shape[1:]) for t in instance)
        output,sizes,_=model.decode(coords,demands,capacity,z,sample=sample); output,sizes=output.cpu().numpy(),sizes.cpu().numpy()
        solutions=problem.unpack_solutions(output,sizes); lengths=problem.evaluate_batch(solutions)
        all_padded.append(output); all_sizes.append(sizes); all_lengths.append(lengths); source_ids.append(np.asarray(ids))
    padded=np.asarray(all_padded); sizes=np.asarray(all_sizes); lengths=np.asarray(all_lengths); routes=np.asarray([[np.count_nonzero(s==0)-1 for s in row] for row in [problem.unpack_solutions(padded[r],sizes[r]) for r in range(len(padded))]])
    report=summarize_generation(lengths,routes); report['action_seeds_by_round']=[int(np.random.SeedSequence([seed_base,training_seed,r]).generate_state(1)[0]) for r in range(len(plan))]
    return report,dict(padded=padded,sequence_lengths=sizes,lengths=lengths,route_counts=routes,source_pair_positions=np.asarray(source_ids))
