"""Train a single-instance CVRP expert; functional validation, not a benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions


def solution_key(problem, solution):
    return tuple(canonicalize_solutions(problem, [solution])[0])


def load_data(path, seed, max_samples=None):
    with np.load(path, allow_pickle=False) as saved:
        problem = CVRPProblem(coordinates=saved['coordinates'], demands=saved['demands'],
                              capacity=int(saved['capacity'].item()))
        solutions = problem.unpack_solutions(saved['routes_padded'], saved['sequence_lengths'])
        lengths, qualities = saved['lengths'].copy(), saved['qualities'].copy()
    if lengths.shape != (len(solutions),) or qualities.shape != lengths.shape:
        raise ValueError('objective arrays must have shape (sample_num,)')
    np.testing.assert_allclose(problem.evaluate_batch(solutions), lengths, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(qualities, -lengths, rtol=1e-7, atol=1e-9)
    canonical = canonicalize_solutions(problem, solutions)
    seen, unique = set(), []
    for index, sequence in enumerate(canonical):
        key = tuple(sequence)
        if key not in seen:
            seen.add(key)
            unique.append(index)
    indices = np.random.default_rng(seed).permutation(unique)
    if max_samples is not None:
        if max_samples < 8:
            raise ValueError('--max-samples must be >= 8')
        indices = indices[:max_samples]
    if len(indices) < 8:
        raise ValueError('need >= 8 distinct canonical CVRP solutions')
    cut = int(.8 * len(indices))
    train, valid = indices[:cut], indices[cut:]
    mean, std = float(qualities[train].mean()), float(qualities[train].std())
    scale = std if std > 1e-8 else 1.
    stats = dict(mean=mean, scale=scale, raw_std=std)
    padded, sizes = problem.pack_solutions(canonical)
    return (problem, padded, sizes, ((qualities - mean) / scale).astype(np.float32),
            lengths, train, valid, stats)


def run_epoch(model, loader, instance, device, config, optimizer=None):
    model.train(optimizer is not None)
    nll_sum, token_count, kl_sum, mse_sum, count = 0., 0, 0., 0., 0
    for sequences, sizes, quality in loader:
        sequences, sizes, quality = sequences.to(device), sizes.to(device), quality.to(device)
        batch = len(sequences)
        coordinates, demands, capacity = (tensor.expand(batch, *tensor.shape[1:]) for tensor in instance)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(coordinates, demands, capacity, sequences, sizes, stochastic=optimizer is not None)
            losses = model.loss(output, sequences, quality, config['beta'], config['score_weight'])
            if not all(torch.isfinite(value).item() for value in losses.values()):
                raise RuntimeError('nonfinite CVRP loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                if not torch.isfinite(norm):
                    raise RuntimeError('nonfinite CVRP gradient')
                optimizer.step()
        tokens = int((sizes - 1).sum().item())
        nll_sum += losses['nll'].item() * tokens
        token_count += tokens
        kl_sum += losses['kl'].item() * batch
        mse_sum += losses['score_mse'].item() * batch
        count += batch
    nll, kl, mse = nll_sum / token_count, kl_sum / count, mse_sum / count
    return dict(loss=nll + config['beta'] * kl + config['score_weight'] * mse,
                nll=nll, kl=kl, score_mse=mse, real_action_count=token_count)


def edge_f1(prediction, target):
    def edges(sequence):
        return Counter(tuple(sorted((int(a), int(b)))) for a, b in zip(sequence[:-1], sequence[1:]))
    scores = []
    for first, second in zip(prediction, target):
        a, b = edges(first), edges(second)
        scores.append(2. * sum((a & b).values()) / (sum(a.values()) + sum(b.values())))
    return float(np.mean(scores))


@torch.no_grad()
def diagnostics(model, problem, instance, padded, sizes, lengths, indices, stats, seed):
    model.eval()
    device = instance[0].device
    target = torch.tensor(padded[indices], dtype=torch.long, device=device)
    target_sizes = torch.tensor(sizes[indices], dtype=torch.long, device=device)
    coordinates, demands, capacity = (tensor.expand(len(target), *tensor.shape[1:]) for tensor in instance)
    mu, logvar = model.encode(coordinates, demands, capacity, target, target_sizes)
    generator = torch.Generator(device=device).manual_seed(seed)
    order = torch.randperm(len(target), generator=generator, device=device)
    prior = torch.randn(mu.shape, generator=generator, device=device)
    cases = dict(aligned=mu, shuffled=mu[order], zero=torch.zeros_like(mu), prior=prior)
    true_solutions = problem.unpack_solutions(padded[indices], sizes[indices])
    true_keys = [tuple(sequence) for sequence in true_solutions]
    real_actions = (target[:, 1:] != -1)[..., None]
    probabilities, report, generated = {}, {}, {}
    for name, z in cases.items():
        entry = {}
        if name != 'prior':
            logits = model.decode(coordinates, demands, capacity, z, target, target_sizes)[2]
            probabilities[name] = logits.softmax(-1)
            entry['teacher_forced_nll'] = model.reconstruction_nll(logits, target).item()
        for mode, sample in (('greedy', False), ('sampled', True)):
            torch.manual_seed(seed + 1)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed + 1)
            output, output_sizes, _ = model.decode(coordinates, demands, capacity, z, sample=sample)
            arrays, size_array = output.cpu().numpy(), output_sizes.cpu().numpy()
            solutions = problem.unpack_solutions(arrays, size_array)
            costs = problem.evaluate_batch(solutions)  # raises for invalid capacity/service/termination
            keys = [solution_key(problem, sequence) for sequence in solutions]
            route_counts = np.array([int(np.count_nonzero(sequence == 0) - 1) for sequence in solutions])
            entry[mode] = dict(valid_rate=1., mean_length=float(costs.mean()), best_length=float(costs.min()),
                               mean_route_count=float(route_counts.mean()),
                               route_count_mae=float(np.mean(np.abs(route_counts -
                                    np.array([np.count_nonzero(row == 0) - 1 for row in true_solutions])))),
                               exact_solution_rate=float(np.mean([a == b for a, b in zip(keys, true_keys)])),
                               undirected_edge_multiset_f1=edge_f1(solutions, true_solutions),
                               unique_solutions=len(set(keys)), objective_queries=len(costs))
            prefix = name + '_' + mode
            generated[prefix + '_padded'] = arrays
            generated[prefix + '_sequence_lengths'] = size_array
            generated[prefix + '_lengths'] = costs
        report[name] = entry
    quality = model.score_predictor(mu).squeeze(-1).cpu().numpy() * stats['scale'] + stats['mean']
    true_quality = -lengths[indices]
    correlation = (float(np.corrcoef(quality, true_quality)[0, 1])
                   if quality.std() > 1e-8 and true_quality.std() > 1e-8 else None)
    report['score'] = dict(mae_length_units=float(np.abs(quality - true_quality).mean()), pearson=correlation,
                           train_mean_baseline_mae=float(np.abs(stats['mean'] - true_quality).mean()))
    report['latent'] = dict(mu_std_across_solutions=float(mu.std(0, unbiased=False).mean().item()),
                            posterior_std_mean=float((.5 * logvar).exp().mean().item()))
    denominator = int(real_actions.sum().item()) * demands.shape[1]
    for name in ('shuffled', 'zero'):
        report['latent'][name + '_valid_action_probability_change'] = float(
            ((probabilities['aligned'] - probabilities[name]).abs() * real_actions).sum().item() / denominator)
    reference = problem.sample_solutions(len(target), seed=seed + 2)
    costs = problem.evaluate_batch(reference)
    report['random_reference'] = dict(mean_length=float(costs.mean()), best_length=float(costs.min()),
                                       sampling='milestone_A_nonuniform_feasible_sampler', objective_queries=len(costs))
    report['diagnostic_solution_count'] = len(target)
    report['generated_objective_queries'] = 9 * len(target)  # 4 latent cases x 2 modes + reference
    return report, generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--data', type=Path, default=root / 'data/cvrp_milestone_a/cvrp20_seed1088.npz')
    parser.add_argument('--config', type=Path, default=root / 'configs/cvrp_expert_smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    for name in ('epochs', 'batch_size', 'hidden_dim', 'latent_dim', 'cpu_threads', 'diagnostic_samples'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('learning_rate', 'grad_clip'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    for name in ('beta', 'score_weight'):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
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
        raise RuntimeError('CUDA requested but unavailable')
    problem, padded, sizes, scores, lengths, train, valid, stats = load_data(args.data, seed, args.max_samples)
    instance = (torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None],
                torch.tensor(problem.demands.copy(), dtype=torch.long, device=device)[None],
                torch.tensor([problem.capacity], dtype=torch.long, device=device))
    def loader(ids, shuffle):
        dataset = TensorDataset(torch.tensor(padded[ids]), torch.tensor(sizes[ids]), torch.tensor(scores[ids]))
        return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    train_loader, valid_loader = loader(train, True), loader(valid, False)
    model = CVRPVAE(config['hidden_dim'], config['latent_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    config.update(device=str(device), torch_version=str(torch.__version__), numpy_version=np.__version__,
                  data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  base_commit=commit, dimension=problem.dimension, capacity=problem.capacity,
                  train_count=len(train), valid_count=len(valid), input_sample_count=len(padded),
                  max_samples=args.max_samples, quality_normalization=stats,
                  split='unique_route_reversal_and_vehicle_order_classes; 80/20; single_instance',
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                      ('src/regression/cvrp_vae.py', 'src/experiment_cvrp_expert.py', 'src/problem/cvrp_problem.py')})
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'split.npz', train_indices=train, valid_indices=valid)
    history, best = [], float('inf')
    initial = run_epoch(model, valid_loader, instance, device, config)
    print('INITIAL valid_nll={:.5f} valid_score_mse={:.5f}'.format(initial['nll'], initial['score_mse']), flush=True)
    for epoch in range(1, config['epochs'] + 1):
        tr = run_epoch(model, train_loader, instance, device, config, optimizer)
        va = run_epoch(model, valid_loader, instance, device, config)
        history.append(dict(epoch=epoch, train=tr, valid=va))
        if va['loss'] < best:
            best = va['loss']
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch,
                            validation=va, quality_normalization=stats), args.output / 'best_model.pt')
        print('epoch={:03d} train_nll={:.5f} valid_nll={:.5f} valid_score_mse={:.5f} kl={:.5f}'.format(
            epoch, tr['nll'], va['nll'], va['score_mse'], va['kl']), flush=True)
    checkpoint = torch.load(args.output / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    report, generated = diagnostics(model, problem, instance, padded, sizes, lengths,
                                    valid[:config['diagnostic_samples']], stats, seed + 1)
    report.update(status='cvrp_expert_functional_run_completed', best_epoch=checkpoint['epoch'],
                  initial_validation=initial, best_validation=checkpoint['validation'], mapping_run=False,
                  validation_scope='held_out_solutions_on_one_instance; checkpoint_selection; not_a_final_test',
                  input_revalidation_queries=len(padded),
                  total_run_objective_queries=len(padded) + report['generated_objective_queries'],
                  prior_data_generation_queries='not_included_in_current_run',
                  note='Input latent perturbations are diagnostics, not separately trained mapping controls.')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
    (args.output / 'diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'generated.npz', coordinates=problem.coordinates,
                        demands=problem.demands, capacity=problem.capacity, **generated)
    print('CVRP EXPERT FUNCTIONAL RUN COMPLETED')
    print('best_epoch:', checkpoint['epoch'])
    print('prior_greedy_valid_rate:', report['prior']['greedy']['valid_rate'])
    print('prior_sampled_valid_rate:', report['prior']['sampled']['valid_rate'])
    print('checkpoint:', args.output / 'best_model.pt')
    print('diagnostics:', args.output / 'diagnostics.json')


if __name__ == '__main__':
    main()
