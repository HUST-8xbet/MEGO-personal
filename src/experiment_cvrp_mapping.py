"""TSP -> CVRP rank mapping, fresh target decoder and five trained controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.regression.cvrp_vae import CVRPVAE
from src.cvrp_mapping import ARMS, prepare_pairs, train_mapping, evaluate_mapping


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def load_source(checkpoint_path, data_path, device, batch_size):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload['config']
    if config.get('data_sha256') is not None and config['data_sha256'] != sha256(data_path):
        raise ValueError('source data hash differs from the TSP expert training data; use its original NPZ')
    model = TSPVAE(config['hidden_dim'], config['latent_dim']).to(device).eval()
    model.load_state_dict(payload['model'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with np.load(data_path, allow_pickle=False) as saved:
        problem = TSPProblem(coordinates=saved['coordinates'])
        tours, qualities = saved['tours'].copy(), saved['qualities'].copy()
    if qualities.shape != (len(tours),):
        raise ValueError('invalid source quality shape')
    np.testing.assert_allclose(-problem.evaluate_batch(tours), qualities, rtol=1e-7, atol=1e-9)
    canonical = canonicalize_tours(tours)
    _, unique = np.unique(canonical, axis=0, return_index=True)
    unique.sort()
    result = []
    coordinates = torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None]
    for start in range(0, len(unique), batch_size):
        ids = unique[start:start+batch_size]
        mu, _ = model.encode(coordinates.expand(len(ids), -1, -1),
                             torch.tensor(canonical[ids], dtype=torch.long, device=device))
        result.append(mu.cpu().numpy())
    if not result:
        raise ValueError('empty source bank')
    return np.concatenate(result), qualities[unique], unique, problem.coordinates, len(tours)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--source-checkpoint', type=Path, default=root / 'logs_tsp/tsp20_expert_smoke_01/best_model.pt')
    parser.add_argument('--source-data', type=Path, default=root / 'data/tsp_milestone_a/tsp20_seed1088.npz')
    parser.add_argument('--target-data', type=Path, default=root / 'data/cvrp_milestone_a/cvrp20_seed2088.npz')
    parser.add_argument('--config', type=Path, default=root / 'configs/cvrp_mapping_smoke.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-pairs', type=int)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms must not contain duplicates')
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.max_pairs is not None:
        config['max_pairs'] = args.max_pairs
    for name in ('epochs', 'batch_size', 'hidden_dim', 'latent_dim', 'cpu_threads', 'max_pairs'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    for name in ('learning_rate', 'grad_clip'):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    seed = config['seed']
    torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source, source_quality, source_rows, source_coords, source_calls = load_source(
        args.source_checkpoint, args.source_data, device, config['batch_size'])
    with np.load(args.target_data, allow_pickle=False) as saved:
        problem = CVRPProblem(coordinates=saved['coordinates'], demands=saved['demands'],
                              capacity=int(saved['capacity'].item()))
        solutions = problem.unpack_solutions(saved['routes_padded'], saved['sequence_lengths'])
        quality = saved['qualities'].copy()
        provenance = json.loads(saved['metadata'].item()) if 'metadata' in saved else {}
    if quality.shape != (len(solutions),):
        raise ValueError('invalid target quality shape')
    np.testing.assert_allclose(-problem.evaluate_batch(solutions), quality, rtol=1e-7, atol=1e-9)
    conditions, padded, sizes, train, valid, pairing = prepare_pairs(
        problem, source, source_quality, solutions, quality, seed, config['max_pairs'])
    template = CVRPVAE(config['hidden_dim'], config['latent_dim']).to(device)
    bridge_template = torch.nn.Linear(source.shape[1], template.latent_dim).to(device)
    instance = (torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None],
                torch.tensor(problem.demands.copy(), dtype=torch.long, device=device)[None],
                torch.tensor([problem.capacity], dtype=torch.long, device=device))
    histories, reports, generated = {}, {}, {}
    args.output.mkdir(parents=True)
    for arm in args.arms:
        model = copy.deepcopy(template)
        bridge = None if arm == 'no_source' else copy.deepcopy(bridge_template)
        history, best = train_mapping(model, bridge, instance, conditions[arm], padded, sizes, train, valid, config)
        histories[arm] = history
        reports[arm] = dict(best_epoch=best['epoch'], best_validation_nll=best['valid_nll'])
        for mode, sample in (('greedy', False), ('sampled', True)):
            result, output, out_sizes, lengths = evaluate_mapping(model, bridge, instance,
                conditions[arm][valid], padded[valid], sizes[valid], problem, seed + 1, sample)
            reports[arm][mode] = result
            prefix = arm + '_' + mode
            generated[prefix + '_padded'], generated[prefix + '_sequence_lengths'] = output, out_sizes
            generated[prefix + '_lengths'] = lengths
        torch.save(dict(model=best['model'], bridge=best['bridge'], epoch=best['epoch'],
                        config=config, arm=arm, source_dim=source.shape[1], target_dim=template.latent_dim,
                        initialization='shared_fresh_CVRP_decoder; no_target_expert_weights'), args.output / (arm + '.pt'))
        print('{} best_epoch={} greedy_mean={:.5f} sampled_mean={:.5f}'.format(
            arm, best['epoch'], reports[arm]['greedy']['mean_length'], reports[arm]['sampled']['mean_length']), flush=True)
    reference = problem.sample_solutions(len(valid), seed=seed + 2)
    reference_lengths = problem.evaluate_batch(reference)
    target_eval = (2 * len(args.arms) + 1) * len(valid)
    costs = dict(source_revalidation_queries_current_run=source_calls,
                 target_existing_label_count=len(solutions), target_generation_queries_current_run=0,
                 target_verification_queries_current_run=len(solutions), target_gradient_training_queries_current_run=0,
                 target_evaluation_queries_current_run=target_eval,
                 target_total_current_run=len(solutions) + target_eval,
                 all_objective_queries_current_run=source_calls + len(solutions) + target_eval,
                 prior_target_data_generation_queries_from_metadata=provenance.get('data_generation_objective_queries', 'unknown'),
                 prior_target_data_reload_queries_from_metadata=provenance.get('reload_verification_objective_queries', 'unknown'),
                 prior_TSP_source_pretraining_queries='unknown; excluded from current run',
                 prior_CVRP_expert_weights='not_used',
                 note='Imported labels and previous runs are separate; gradient steps and saving generated outputs make no objective calls.')
    overlap = int(np.any(np.all(source_coords[:, None, :] == problem.coordinates[None, :, :], axis=-1), axis=1).sum())
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    report = dict(status='cvrp_mapping_functional_run_completed', protocol_version='D-C_fresh_target_trained_controls',
                  base_commit=commit,
                  source_family='TSP', target_family='CVRP', source_latent_dim=source.shape[1], target_latent_dim=template.latent_dim,
                  config=config, evaluated_arms=args.arms, device=str(device), torch_version=str(torch.__version__),
                  numpy_version=np.__version__, pair_count=len(padded), train_count=len(train), valid_count=len(valid),
                  initialization='shared_fresh_CVRP_decoder; no_target_expert_weights',
                  exact_coordinate_overlap=dict(source_vertex_count=len(source_coords), target_vertex_count=len(problem.coordinates),
                                                 source_vertices_also_in_target=overlap),
                  split='unique_target_route_solutions_and_unique_source_cycles; split_before_rank_pairing',
                  quality_pairing=dict(source='TSP -length; maximize', target='CVRP -length; maximize',
                                       source_sorted=pairing['source_sorted_within_each_split'], target_sorted=pairing['target_sorted_within_each_split']),
                  validation_scope='held_out_solutions_on_one_target_instance; checkpoint_selection; not_a_final_test',
                  paired_validation_nll='validation target quality defines diagnostic pairs only; generation uses no target label',
                  conditions=reports, costs=costs,
                  random_reference=dict(mean_length=float(reference_lengths.mean()), best_length=float(reference_lengths.min()),
                                        objective_queries=len(valid), sampling='D-A nonuniform feasible sampler'),
                  file_sha256={str(p): sha256(p) for p in (args.source_checkpoint, args.source_data, args.target_data, args.config)},
                  source_code_sha256={n: sha256(root / n) for n in ('src/cvrp_mapping.py','src/experiment_cvrp_mapping.py',
                                                                 'src/regression/cvrp_vae.py','src/tsp_mapping_controls.py')})
    for name, value in (('diagnostics.json', report), ('history.json', histories), ('costs.json', costs), ('conditioning.json', pairing)):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'split.npz', train_indices=train, valid_indices=valid,
                        source_order=pairing['source_order'], source_original_rows=source_rows[pairing['source_order']],
                        target_order=pairing['target_order'])
    np.savez_compressed(args.output / 'conditioning.npz', **conditions)
    np.savez_compressed(args.output / 'generated.npz', coordinates=problem.coordinates, demands=problem.demands,
                        capacity=problem.capacity, valid_target_padded=padded[valid], valid_target_sequence_lengths=sizes[valid], **generated)
    print('CVRP MAPPING FUNCTIONAL RUN COMPLETED')
    print('pair_count:', len(padded))
    print('diagnostics:', args.output / 'diagnostics.json')


if __name__ == '__main__':
    main()
