"""Small, label-free behavioral-profile probe for the existing TSP/CVRP banks.

The probe is deliberately independent of the VAE/coupling pipeline.  It starts
every move from the original solution, keeps train/validation IDs explicit,
and uses independent A/B/C random streams.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem


TOLERANCE = 1e-12
ALPHA = 1.0
MAX_MOVE_TRIALS = 128


def sha256_array(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def percentile_from_train(values, train_ids):
    ref = np.sort(np.asarray(values, dtype=np.float64)[np.asarray(train_ids)])
    values = np.asarray(values, dtype=np.float64)
    return (np.searchsorted(ref, values, side='left') +
            np.searchsorted(ref, values, side='right')) / (2.0 * len(ref))


def select_spread(ids, percentiles, limit, seed):
    """Select at most limit rows at evenly spaced train-CDF percentiles."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= limit:
        return ids.copy()
    order = ids[np.argsort(percentiles[ids], kind='stable')]
    positions = np.floor((np.arange(limit) + .5) * len(order) / limit).astype(int)
    positions = np.minimum(positions, len(order) - 1)
    # The seed is part of provenance even though the balanced selection itself is fixed.
    rng = np.random.default_rng(seed)
    jitter = rng.integers(0, 1, size=limit)
    return order[np.clip(positions + jitter, 0, len(order)-1)]


def _two_opt(tour, i, j):
    """Apply the 2-opt exchange for edges (i,i+1) and (j,j+1)."""
    candidate = np.array(tour, copy=True)
    candidate[i + 1:j + 1] = candidate[i + 1:j + 1][::-1]
    return candidate


def _cycle_equivalent(first, second):
    """True for rotation/reversal equivalents of an undirected tour cycle."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        return False
    return any(np.array_equal(np.roll(first, shift), second) or
               np.array_equal(np.roll(first[::-1], shift), second)
               for shift in range(len(first)))


def _relocate(solution, source_route, source_pos, destination_route, insert_pos, problem):
    routes = [list(r.astype(np.int64, copy=False)) for r in problem.split_routes(solution)]
    customer = int(routes[source_route][source_pos])
    del routes[source_route][source_pos]
    if len(routes[source_route]) == 2:
        del routes[source_route]
        if source_route < destination_route:
            destination_route -= 1
    routes[destination_route].insert(insert_pos, customer)
    candidate = np.concatenate([np.asarray(r, dtype=np.int64) if k == 0 else np.asarray(r[1:], dtype=np.int64)
                                for k, r in enumerate(routes)])
    try:
        problem.validate_solution(candidate)
        return candidate
    except ValueError:
        return None


def _moves_tsp(original, k, rng, problem):
    moves = []; metadata = []; trials = 0; seen = set(); n = problem.dimension
    while len(moves) < k and trials < MAX_MOVE_TRIALS:
        trials += 1
        i, j = sorted(rng.choice(n, size=2, replace=False))
        # Edges e_i=(i,i+1) and e_j=(j,j+1), modulo n, must be disjoint.
        if (j - i) in (1, n - 1):
            continue
        candidate = _two_opt(original, int(i), int(j))
        key = candidate.tobytes()
        if key in seen or _cycle_equivalent(candidate, original): continue
        seen.add(key); moves.append(candidate)
        metadata.append(dict(operator='2-opt', edge_pair=[int(i), int(j)],
                             exchanged_edges=[[int(i), int((i+1) % n)], [int(j), int((j+1) % n)]]))
    return moves, trials, metadata


def _moves_cvrp(original, k, rng, problem):
    routes = [list(route.astype(np.int64, copy=False)) for route in problem.split_routes(original)]
    customers = np.concatenate([route[1:-1] for route in routes]).astype(np.int64)
    moves = []; metadata = []; trials = 0; seen = set()
    while len(moves) < k and trials < MAX_MOVE_TRIALS:
        trials += 1
        customer = int(rng.choice(customers))
        src = next(i for i, route in enumerate(routes) if customer in route)
        source_pos = routes[src].index(customer)
        reduced = [list(route) for route in routes]
        del reduced[src][source_pos]
        if len(reduced[src]) == 2:
            del reduced[src]
        valid_positions = []
        for dst, route in enumerate(reduced):
            for insert_pos in range(1, len(route)):
                original_dst = dst if len(reduced) == len(routes) else (dst if dst < src else dst + 1)
                candidate = _relocate(original, src, source_pos, original_dst, insert_pos, problem)
                if candidate is not None and not np.array_equal(candidate, original):
                    valid_positions.append((dst, original_dst, insert_pos, candidate))
        if not valid_positions:
            continue
        dst, original_dst, insert_pos, candidate = valid_positions[int(rng.integers(len(valid_positions)))]
        key = candidate.tobytes()
        if key in seen:
            continue
        seen.add(key); moves.append(candidate)
        metadata.append(dict(operator='relocate', customer=customer, source_route=int(src),
                             source_position=int(source_pos), sampling='customer_uniform_then_valid_insert_position',
                             destination_route_after_removal=int(dst), destination_route_original=int(original_dst),
                             insert_position=int(insert_pos),
                             valid_insert_position_count=len(valid_positions)))
    return moves, trials, metadata


def _probe_set(original, k, seed, domain, problem, objective):
    rng = np.random.default_rng(seed)
    if domain == 'tsp': moves, trials, metadata = _moves_tsp(original, k, rng, problem)
    else: moves, trials, metadata = _moves_cvrp(original, k, rng, problem)
    values = np.full(k, np.nan, dtype=np.float64)
    keys = []
    for i, move in enumerate(moves):
        values[i] = objective(move); keys.append(hashlib.sha256(move.tobytes()).hexdigest())
    return values, trials, len(moves), keys, metadata, moves


def _features(deltas):
    finite = np.isfinite(deltas)
    if not finite.all(): return np.array([np.nan, np.nan])
    return np.array([np.mean(deltas > TOLERANCE), np.mean(np.maximum(deltas, 0.0))], dtype=np.float64)


def _rank(values):
    order = np.argsort(values, kind='stable'); out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]: j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0; i = j
    return out


def spearman(x, y):
    x, y = _rank(x), _rank(y)
    dx, dy = x - x.mean(), y - y.mean(); den = np.sqrt(np.sum(dx*dx) * np.sum(dy*dy))
    return None if den == 0 else float(np.dot(dx, dy) / den)


def _standardize_fit_predict(x_train, y_train, x_eval):
    xm, xs = x_train.mean(0), x_train.std(0); xs[xs < 1e-15] = 1.0
    ym, ys = y_train.mean(), y_train.std(); ys = ys if ys >= 1e-15 else 1.0
    z = (x_train - xm) / xs; ze = (x_eval - xm) / xs
    # Intercept is handled by centering; alpha is fixed and not tuned.
    coef = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ ((y_train - ym) / ys))
    return ze @ coef * ys + ym, dict(input_mean=xm.tolist(), input_scale=xs.tolist(), outcome_mean=float(ym), outcome_scale=float(ys))


def _metrics(y, pred):
    error = pred - y; ss = np.sum((y-y.mean())**2)
    return dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                r2=None if ss == 0 else float(1 - np.sum(error**2) / ss), unit='raw normalized outcome')


def _bootstrap_difference(y, base, expanded, seed, draws=2000):
    rng = np.random.default_rng(seed); n = len(y); diffs = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean(np.abs(y[idx]-base[idx])) - np.mean(np.abs(y[idx]-expanded[idx]))
    return dict(estimate=float(np.mean(np.abs(y-base))-np.mean(np.abs(y-expanded))),
                ci95=[float(np.quantile(diffs,.025)), float(np.quantile(diffs,.975))],
                bootstrap_draws=draws, seed=seed,
                interpretation='positive means expanded model lowers MAE; instance-level CI only')


def regression_report(percentile, features, outcome, train_mask, valid_mask, seed):
    rank = percentile * 100.0
    base = np.column_stack([rank, rank**2, rank**3])
    expanded = np.column_stack([base, features])
    report = {}
    for name, target in zip(('p_improve', 'mean_positive_gain'), outcome.T):
        row = dict(outcome=name, train_variance=float(np.var(target[train_mask])), sufficient_signal=bool(np.var(target[train_mask]) > 0))
        if not row['sufficient_signal']:
            row['reason'] = 'training outcome variance is zero'; report[name] = row; continue
        bp, bscale = _standardize_fit_predict(base[train_mask], target[train_mask], base[valid_mask])
        ep, escale = _standardize_fit_predict(expanded[train_mask], target[train_mask], expanded[valid_mask])
        row['baseline'] = _metrics(target[valid_mask], bp); row['rank_plus_features'] = _metrics(target[valid_mask], ep)
        row['paired_mae_bootstrap'] = _bootstrap_difference(target[valid_mask], bp, ep, seed)
        row['fit_standardization'] = dict(baseline=bscale, expanded=escale)
        report[name] = row
    return report


def _group_summary(percentile, features, outcome):
    groups = np.minimum((percentile * 5).astype(int), 4)
    result = []
    for g in range(5):
        idx = groups == g
        result.append(dict(group=g+1, percentile_range=[g*20, (g+1)*20], count=int(idx.sum()),
                           feature_mean=features[idx].mean(0).tolist(), feature_std=features[idx].std(0, ddof=1).tolist() if idx.sum()>1 else [None,None],
                           outcome_mean=outcome[idx].mean(0).tolist(), outcome_std=outcome[idx].std(0, ddof=1).tolist() if idx.sum()>1 else [None,None]))
    return result


def run_domain(name, data_path, split_path, max_solutions, k, seed, smoke=False):
    data = np.load(data_path, allow_pickle=False); split = np.load(split_path, allow_pickle=False)
    if name == 'tsp':
        problem = TSPProblem(coordinates=data['coordinates']); solutions = data['tours']; objective = problem.evaluate
    else:
        problem = CVRPProblem(coordinates=data['coordinates'], demands=data['demands'], capacity=int(data['capacity'].item()))
        solutions = problem.unpack_solutions(data['routes_padded'], data['sequence_lengths']); objective = problem.evaluate
    n = len(solutions); train_ids = np.asarray(split['train_indices'], dtype=np.int64); valid_ids = np.asarray(split['valid_indices'], dtype=np.int64)
    quality = np.asarray(data['qualities'], dtype=np.float64); lengths = np.asarray(data['lengths'], dtype=np.float64)
    percentile = percentile_from_train(quality, train_ids)
    limit = min(max_solutions, len(train_ids), len(valid_ids)); selected = np.r_[select_spread(train_ids, percentile, limit, seed), select_spread(valid_ids, percentile, limit, seed+1)]
    partition = np.r_[np.repeat('train', limit), np.repeat('validation', limit)]
    train_selected = selected[:limit]; valid_selected = selected[limit:]
    support_lengths = np.array([objective(s) for s in solutions], dtype=np.float64)
    scale = float(np.std(support_lengths[train_ids], ddof=1))
    if not np.isfinite(scale) or scale < 1e-15: raise ValueError('{} train objective std is zero/unstable'.format(name))
    feature_a = np.full((len(selected), 2), np.nan); feature_b = np.full_like(feature_a, np.nan); outcome = np.full_like(feature_a, np.nan)
    move_metadata = [[None, None, None] for _ in selected]; move_keys_by_set = [[None, None, None] for _ in selected]
    initial_index = int(train_ids[np.argmin(support_lengths[train_ids])])
    best_probe_info = dict(source_solution_id=initial_index, partition='train', set='incumbent_initial', move_index=None)
    best_probe_solution = np.array(solutions[initial_index], copy=True)
    trials = np.zeros((len(selected),3), dtype=np.int64); valid_moves = np.zeros_like(trials); objective_calls = len(support_lengths)
    best_probe_train = float(np.min(support_lengths[train_ids])); duplicate_moves = 0
    for row, original in enumerate([solutions[i] for i in selected]):
        base = float(support_lengths[selected[row]])
        values = []; move_keys = []
        for set_index in range(3):
            vals, attempted, valid, keys, metadata, moves = _probe_set(original, k, int(np.random.SeedSequence([seed, row, set_index]).generate_state(1)[0]), name, problem, objective)
            delta = (base - vals) / scale; values.append(delta); trials[row,set_index] = attempted; valid_moves[row,set_index] = valid; objective_calls += valid
            move_keys.append(keys)
            move_metadata[row][set_index] = metadata
            move_keys_by_set[row][set_index] = keys
            for move_index, value in enumerate(vals):
                if partition[row] == 'train' and np.isfinite(value) and value < best_probe_train:
                    best_probe_train = float(value); best_probe_solution = np.array(moves[move_index], copy=True)
                    best_probe_info = dict(source_solution_id=int(selected[row]), partition='train',
                                           set='ABC'[set_index], move_index=int(move_index),
                                           move=metadata[move_index], move_sha256=keys[move_index])
        duplicate_moves += len(set(move_keys[0]) & set(move_keys[1])) + len(set(move_keys[0]) & set(move_keys[2])) + len(set(move_keys[1]) & set(move_keys[2]))
        if valid_moves[row].min() == k:
            feature_a[row] = _features(values[0]); feature_b[row] = _features(values[1]); outcome[row] = _features(values[2])
    complete = np.isfinite(feature_a).all(1) & np.isfinite(feature_b).all(1) & np.isfinite(outcome).all(1)
    train_mask = (partition == 'train') & complete; valid_mask = (partition == 'validation') & complete
    if train_mask.sum() < 2 or valid_mask.sum() < 2: raise ValueError('{} has too few complete profiles'.format(name))
    profiles = dict(domain=name, ids=selected, partition=partition, quality=quality[selected], percentile=percentile[selected]*100,
                    feature_a=feature_a, feature_b=feature_b, outcome=outcome, complete=complete,
                    trials=trials, valid_moves=valid_moves, objective=support_lengths[selected], moves=move_metadata,
                    move_keys=move_keys_by_set)
    diagnostics = dict(domain=name, selected_count=int(len(selected)), train_count=int(train_mask.sum()), validation_count=int(valid_mask.sum()),
        quality_direction='quality=-length; higher is better', objective_scale_train_std=scale,
        tolerance=TOLERANCE, operator=('sample unordered non-adjacent edge pairs; apply 2-opt; reject cycle rotation/reversal equivalents' if name=='tsp' else 'sample customer uniformly; remove it; enumerate capacity-valid insertion positions; sample one uniformly; delete empty source routes'),
        max_move_trials=MAX_MOVE_TRIALS, k=k, complete_profile_count=int(complete.sum()), incomplete_ids=selected[~complete].tolist(),
        feature_ab=['p_improve','mean_positive_gain'], feature_definition="delta=(f(x)-f(x'))/s_I; minimization; positive delta improves",
        feature_stability=dict(spearman_p_improve=spearman(feature_a[complete,0],feature_b[complete,0]), spearman_mean_positive_gain=spearman(feature_a[complete,1],feature_b[complete,1]),
                               mean_abs_difference= np.mean(np.abs(feature_a[complete]-feature_b[complete]),0).tolist(), zero_feature_rate=np.mean(feature_a[complete]==0,0).tolist()),
        grouped=_group_summary(percentile[selected][complete], feature_a[complete], outcome[complete]),
        costs=dict(source_objective_calls=int(objective_calls if name == 'tsp' else 0), target_objective_calls=int(objective_calls if name == 'cvrp' else 0),
                   tsp_source_objective_calls=int(objective_calls if name == 'tsp' else 0), cvrp_target_objective_calls=int(objective_calls if name == 'cvrp' else 0),
                   cache_hits=0, move_trials=int(trials.sum()), valid_moves=int(valid_moves.sum()),
                   selected_solutions=int(len(selected)), complete_solutions=int(complete.sum()), duplicate_moves=int(duplicate_moves),
                   note='TSP objective evaluations are classified as source calls; CVRP objective evaluations are classified as target calls'))
    diagnostics['incumbent'] = dict(initial_train_best=float(np.min(support_lengths[train_ids])),
                                     new_best_after_probe=float(best_probe_train),
                                     probe_improvement=float(np.min(support_lengths[train_ids]) - best_probe_train),
                                     unit='objective length; lower is better')
    diagnostics['regression'] = regression_report(percentile[selected][complete], feature_a[complete], outcome[complete], partition[complete]=='train', partition[complete]=='validation', seed+77)
    profiles['incumbent'] = best_probe_train
    profiles['incumbent_info'] = best_probe_info
    profiles['incumbent_solution'] = best_probe_solution
    return profiles, diagnostics, lengths, support_lengths


def write_outputs(output, domains, config, elapsed):
    output.mkdir(parents=True, exist_ok=False)
    rows=[]
    for name, (p, _, _, _) in domains.items():
        for i, sid in enumerate(p['ids']):
            rows.append(dict(domain=name, solution_id=int(sid), partition=p['partition'][i], quality=float(p['quality'][i]), percentile=float(p['percentile'][i]),
                solution_role='TSP source' if name == 'tsp' else 'CVRP target',
                objective=float(p['objective'][i]), feature_a_p_improve=p['feature_a'][i,0], feature_a_mean_positive_gain=p['feature_a'][i,1],
                feature_b_p_improve=p['feature_b'][i,0], feature_b_mean_positive_gain=p['feature_b'][i,1], outcome_c_p_improve=p['outcome'][i,0], outcome_c_mean_positive_gain=p['outcome'][i,1],
                complete=bool(p['complete'][i]), trials_a=int(p['trials'][i,0]), trials_b=int(p['trials'][i,1]), trials_c=int(p['trials'][i,2]),
                valid_moves_a=int(p['valid_moves'][i,0]), valid_moves_b=int(p['valid_moves'][i,1]), valid_moves_c=int(p['valid_moves'][i,2]),
                move_a_json=json.dumps(p['moves'][i][0], sort_keys=True), move_b_json=json.dumps(p['moves'][i][1], sort_keys=True), move_c_json=json.dumps(p['moves'][i][2], sort_keys=True),
                move_a_sha256_json=json.dumps(p['move_keys'][i][0]), move_b_sha256_json=json.dumps(p['move_keys'][i][1]), move_c_sha256_json=json.dumps(p['move_keys'][i][2]),
                id_sha256=sha256_array(np.asarray([sid],dtype=np.int64))))
    with (output/'per_solution_profiles.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    prediction={name: d[1]['regression'] for name,d in domains.items()}
    diagnostics=dict(config=config, elapsed_seconds=elapsed, domains={name:d[1] for name,d in domains.items()},
                     provenance={name:dict(data_sha256=sha256_file(config['domains'][name]['data']), split_sha256=sha256_file(config['domains'][name]['split']),
                                           selected_id_sha256=sha256_array(d[0]['ids'])) for name,d in domains.items()})
    (output/'config.json').write_text(json.dumps(config, indent=2, allow_nan=False)+'\n')
    (output/'prediction_metrics.json').write_text(json.dumps(prediction,indent=2,allow_nan=False)+'\n')
    (output/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')
    costs={name:d[1]['costs'] for name,d in domains.items()}; (output/'costs.json').write_text(json.dumps(costs,indent=2,allow_nan=False)+'\n')
    with (output/'incumbent.npz').open('wb') as f: np.savez_compressed(f, **{name: np.array([float(d[0]['incumbent'])]) for name,d in domains.items()})
    incumbent_json={name:dict(objective=float(d[0]['incumbent']), provenance=d[0]['incumbent_info'], solution=np.asarray(d[0]['incumbent_solution'], dtype=np.int64).tolist(), solution_sha256=sha256_array(d[0]['incumbent_solution'])) for name,d in domains.items()}
    (output/'incumbent.json').write_text(json.dumps(incumbent_json, indent=2, allow_nan=False)+'\n')
    lines=['# Behavioral profile probe','', 'No expert, decoder, loss, or coupling was changed. This is an in-domain diagnostic.', '', '| Domain | Complete profiles | Train | Validation | TSP source calls | CVRP target calls | Move trials |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,(_,diag,_,_) in domains.items(): lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(name,diag['complete_profile_count'],diag['train_count'],diag['validation_count'],diag['costs']['tsp_source_objective_calls'],diag['costs']['cvrp_target_objective_calls'],diag['costs']['move_trials']))
    lines += ['', '## Answers', '', '1. Stability is reported by A/B Spearman, mean absolute difference, and zero-feature rate; it is descriptive for this instance.', '2. Rank-only versus rank-plus-A regression, raw-unit MAE/RMSE/R², and paired bootstrap MAE intervals are in `prediction_metrics.json`.', '3. The probe uses one instance per domain, at most 128 train + 128 validation solutions, K=16, and different operators: random 2-opt for TSP versus capacity-checked relocate for CVRP.', '4. These results can screen the behavioral signal, but do not justify replacing Jaccard or claiming transfer. Use the predeclared decision rule and replicate before changing coupling.', '', 'Probe budget maximum: 2 domains × 256 solutions × 3 sets × 16 moves = 24,576 valid neighbor evaluations, plus bounded failed move trials and support verification.', 'C is never used to select solutions, operators, or parameters. Validation rows never update train data.']
    for name,(_,diag,_,_) in domains.items():
        st=diag['feature_stability']; rp=diag['regression']
        lines += ['', '### {} observed diagnostic'.format(name.upper()),
                  'A/B Spearman: p_improve={:.4f}, mean_positive_gain={:.4f}; mean absolute A-B difference={}.'.format(st['spearman_p_improve'], st['spearman_mean_positive_gain'], [round(x,6) for x in st['mean_abs_difference']])]
        for outcome in ('p_improve','mean_positive_gain'):
            r=rp[outcome]; b=r['baseline']; e=r['rank_plus_features']; ci=r['paired_mae_bootstrap']['ci95']
            lines.append('{}: validation MAE rank={:.6f}, rank+A={:.6f}, paired MAE difference={:.6f}, CI95=[{:.6f},{:.6f}].'.format(outcome,b['mae'],e['mae'],r['paired_mae_bootstrap']['estimate'],ci[0],ci[1]))
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    import argparse
    root=Path(__file__).resolve().parents[1]; parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--tsp-data',type=Path,default=root/'data/tsp_milestone_a/tsp20_seed1088.npz'); parser.add_argument('--cvrp-data',type=Path,default=root/'data/cvrp_milestone_a/cvrp20_seed2088.npz'); parser.add_argument('--tsp-split',type=Path,default=root/'logs_tsp/tsp20_expert_smoke_01/split.npz'); parser.add_argument('--cvrp-split',type=Path,default=root/'logs_cvrp/cvrp20_expert_smoke_01/split.npz'); parser.add_argument('--max-solutions',type=int,default=128); parser.add_argument('--k',type=int,default=16); parser.add_argument('--seed',type=int,default=321001); parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError('output exists; choose a new directory')
    max_solutions, k = (4,3) if args.smoke_test else (args.max_solutions,args.k)
    config=dict(protocol='behavioral_profile_probe_v1', tolerance=TOLERANCE, ridge_alpha=ALPHA, bootstrap_draws=2000, max_move_trials=MAX_MOVE_TRIALS, k=k, max_solutions_per_partition=max_solutions, selection_seed=args.seed, independent_rng='SeedSequence [selection_seed,row,set]', execution_mode='smoke' if args.smoke_test else 'full',
                operators={'tsp':'sample unordered non-adjacent edge pairs; apply 2-opt; reject cycle rotation/reversal equivalents',
                           'cvrp':'sample customer uniformly; remove it; enumerate capacity-valid insertion positions; sample one uniformly; delete empty source routes'},
                domains={'tsp':{'data':str(args.tsp_data),'split':str(args.tsp_split)},'cvrp':{'data':str(args.cvrp_data),'split':str(args.cvrp_split)}})
    start=time.perf_counter(); domains={}
    for offset,(name,data,split) in enumerate((('tsp',args.tsp_data,args.tsp_split),('cvrp',args.cvrp_data,args.cvrp_split))): domains[name]=run_domain(name,data,split,max_solutions,k,args.seed+offset,args.smoke_test)
    write_outputs(args.output,domains,config,time.perf_counter()-start); print('BEHAVIORAL PROFILE PROBE COMPLETED'); print('output:',args.output)


if __name__ == '__main__': main()
