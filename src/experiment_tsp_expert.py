"""Train a single-instance TSP expert for functional validation, not a benchmark."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE, canonicalize_tours


def load_data(path, seed, max_samples=None):
    with np.load(path, allow_pickle=False) as saved:
        problem = TSPProblem(coordinates=saved['coordinates'])
        tours = saved['tours'].copy()
        lengths = saved['lengths'].copy()
        qualities = saved['qualities'].copy()
    if lengths.shape != (len(tours),) or qualities.shape != lengths.shape:
        raise ValueError('lengths and qualities must have shape (sample_num,)')
    np.testing.assert_allclose(problem.evaluate_batch(tours), lengths, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(qualities, -lengths, rtol=1e-7, atol=1e-9)
    # Deduplicate equivalence classes BEFORE splitting, avoiding cycle leakage.
    tours = canonicalize_tours(tours)
    _, unique_indices = np.unique(tours, axis=0, return_index=True)
    unique_indices.sort()
    indices = np.random.default_rng(seed).permutation(unique_indices)
    if max_samples is not None:
        if max_samples < 8:
            raise ValueError('--max-samples must be >= 8')
        indices = indices[:max_samples]
    if len(indices) < 8:
        raise ValueError('need at least 8 distinct undirected cycles')
    cut = max(1, int(0.8 * len(indices)))
    train_indices, valid_indices = indices[:cut], indices[cut:]
    mean = float(qualities[train_indices].mean())
    raw_std = float(qualities[train_indices].std())
    scale = raw_std if raw_std > 1e-8 else 1.0
    normalized = (qualities - mean) / scale
    return (problem, tours, normalized.astype(np.float32), lengths,
            train_indices, valid_indices, dict(mean=mean, scale=scale, raw_std=raw_std))


def run_epoch(model, loader, coordinates, device, config, optimizer=None):
    model.train(optimizer is not None)
    totals, count = {}, 0
    for tours, scores in loader:
        tours, scores = tours.to(device), scores.to(device)
        coords = coordinates.expand(len(tours), -1, -1)
        with torch.set_grad_enabled(optimizer is not None):
            result = model(coords, tours, stochastic=optimizer is not None)
            metrics = model.loss(result, tours, scores, config['beta'], config['score_weight'])
            if not all(torch.isfinite(value).item() for value in metrics.values()):
                raise RuntimeError('nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                metrics['loss'].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                if not torch.isfinite(grad_norm):
                    raise RuntimeError('nonfinite gradient')
                optimizer.step()
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.) + value.item() * len(tours)
        count += len(tours)
    return {name: value / count for name, value in totals.items()}


def edge_overlap(first, second):
    def edges(tour):
        return {tuple(sorted((int(a), int(b)))) for a, b in zip(tour, np.roll(tour, -1))}
    return float(np.mean([len(edges(a) & edges(b)) / len(a) for a, b in zip(first, second)]))


@torch.no_grad()
def diagnostics(model, problem, coordinates, tours, lengths, indices, stats, seed):
    model.eval()
    target = torch.as_tensor(tours[indices], dtype=torch.long, device=coordinates.device)
    coords = coordinates.expand(len(target), -1, -1)
    mu, logvar = model.encode(coords, target)
    # Roll the latent assignment, holding instance and teacher prefix fixed.
    shuffled = mu.roll(1, 0)
    cases = dict(aligned=mu, shuffled=shuffled, zero=torch.zeros_like(mu))
    results, probabilities, decoded_arrays = {}, {}, {}
    for name, z in cases.items():
        _, logits = model.decode(coords, z, teacher_tours=target)
        probabilities[name] = logits.softmax(-1)
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, problem.dimension),
                                                target[:, 1:].reshape(-1)).item()
        decoded, _ = model.decode(coords, z)
        decoded_np = decoded.cpu().numpy()
        decoded_arrays[name] = decoded_np
        costs = problem.evaluate_batch(decoded_np)  # raises on invalid tours
        canonical = canonicalize_tours(decoded_np)
        results[name] = dict(teacher_forced_nll=nll,
                             exact_cycle_rate=float(np.mean(np.all(canonical == tours[indices], axis=1))),
                             undirected_edge_overlap=edge_overlap(decoded_np, tours[indices]),
                             valid_rate=1.0, mean_length=float(costs.mean()), best_length=float(costs.min()))
    prediction = model.score_predictor(mu).squeeze(-1).cpu().numpy() * stats['scale'] + stats['mean']
    true_quality = -lengths[indices]
    correlation = (float(np.corrcoef(prediction, true_quality)[0, 1])
                   if prediction.std() > 1e-8 and true_quality.std() > 1e-8 else None)
    results['score'] = dict(mae_length_units=float(np.mean(np.abs(prediction - true_quality))),
                            train_mean_baseline_mae=float(np.mean(np.abs(stats['mean'] - true_quality))),
                            pearson=correlation)
    results['latent'] = dict(mu_std_across_tours=float(mu.std(dim=0, unbiased=False).mean().item()),
                            posterior_std_mean=float((0.5 * logvar).exp().mean().item()))
    for name in ('shuffled', 'zero'):
        results['latent'][name + '_mean_abs_probability_change'] = float(
            (probabilities['aligned'] - probabilities[name]).abs().mean().item())
        results['latent'][name + '_greedy_cycle_changed_rate'] = float(np.mean(np.any(
            canonicalize_tours(decoded_arrays['aligned']) != canonicalize_tours(decoded_arrays[name]), axis=1)))
    generator = torch.Generator(device=coordinates.device).manual_seed(seed)
    prior = torch.randn(len(target), model.latent_dim, generator=generator, device=coordinates.device)
    prior_tours, _ = model.decode(coords, prior, sample=False)
    prior_np = prior_tours.cpu().numpy()
    prior_costs = problem.evaluate_batch(prior_np)
    random_tours = problem.sample_tours(len(target), seed=seed)
    random_costs = problem.evaluate_batch(random_tours)
    results['prior_greedy'] = dict(valid_rate=1.0, mean_length=float(prior_costs.mean()),
                                 best_length=float(prior_costs.min()),
                                 unique_cycles=len(np.unique(canonicalize_tours(prior_np), axis=0)))
    results['random_reference'] = dict(mean_length=float(random_costs.mean()), best_length=float(random_costs.min()))
    results['diagnostic_tour_count'] = len(target)
    results['generated_objective_queries'] = 5 * len(target)  # 3 latent cases + prior + random
    return results, prior_np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/tsp_expert_smoke.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    for name in ('epochs', 'batch_size', 'hidden_dim', 'latent_dim', 'diagnostic_samples', 'cpu_threads'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('learning_rate', 'grad_clip'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    for name in ('beta', 'score_weight'):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu')
    problem, tours, scores, lengths, train_ids, valid_ids, stats = load_data(args.data, seed, args.max_samples)
    coordinates = torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None]
    def loader(ids, shuffle):
        dataset = TensorDataset(torch.tensor(tours[ids]), torch.tensor(scores[ids]))
        return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle,
                          generator=torch.Generator().manual_seed(seed), num_workers=0)
    train_loader, valid_loader = loader(train_ids, True), loader(valid_ids, False)
    model = TSPVAE(config['hidden_dim'], config['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                     ('src/regression/tsp_vae.py', 'src/experiment_tsp_expert.py', 'src/problem/tsp_problem.py')}
    config.update(device=str(device), torch_version=str(torch.__version__), numpy_version=np.__version__,
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), data_path=str(args.data.resolve()),
                  base_commit=commit, source_sha256=source_hashes,
                  dimension=problem.dimension, train_count=len(train_ids), valid_count=len(valid_ids),
                  split='unique_canonical_cycles_random_80_20_single_instance',
                  max_samples=args.max_samples, quality_normalization=stats)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'split.npz', train_indices=train_ids, valid_indices=valid_ids)
    history = []
    initial = run_epoch(model, valid_loader, coordinates, device, config)
    print('INITIAL valid_nll={:.5f} valid_score_mse={:.5f}'.format(initial['nll'], initial['score_mse']), flush=True)
    best, best_epoch = float('inf'), None
    for epoch in range(1, config['epochs'] + 1):
        train = run_epoch(model, train_loader, coordinates, device, config, optimizer)
        valid = run_epoch(model, valid_loader, coordinates, device, config)
        history.append(dict(epoch=epoch, train=train, valid=valid))
        if valid['loss'] < best:
            best, best_epoch = valid['loss'], epoch
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            validation=valid, quality_normalization=stats), args.output / 'best_model.pt')
        print('epoch={:03d} train_nll={:.5f} valid_nll={:.5f} valid_score_mse={:.5f} kl={:.5f}'.format(
            epoch, train['nll'], valid['nll'], valid['score_mse'], valid['kl']), flush=True)
    checkpoint = torch.load(args.output / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    diag_ids = valid_ids[:config['diagnostic_samples']]
    report, generated = diagnostics(model, problem, coordinates, tours, lengths, diag_ids, stats, seed + 1)
    report.update(best_epoch=best_epoch, initial_validation=initial,
                  best_validation=checkpoint['validation'],
                  status='functional_run_completed', mapping_run=False,
                  input_revalidation_queries=len(lengths),
                  saved_tour_revalidation_queries=len(diag_ids),
                  total_run_objective_queries=len(lengths) + 6 * len(diag_ids),
                  note='Single-instance smoke run; diagnostic scores are not acceptance thresholds or transfer evidence.')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
    (args.output / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'generated.npz', coordinates=problem.coordinates,
                        tours=generated, lengths=problem.evaluate_batch(generated), diagnostic_indices=diag_ids)
    print('TSP EXPERT FUNCTIONAL RUN COMPLETED', flush=True)
    print('best_epoch:', best_epoch)
    print('prior_greedy_valid_rate:', report['prior_greedy']['valid_rate'])
    print('checkpoint:', args.output / 'best_model.pt')
    print('diagnostics:', args.output / 'diagnostics.json')


if __name__ == '__main__':
    main()
