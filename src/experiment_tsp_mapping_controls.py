"""C.1: matched TRAINED controls. Original mapping scripts/runs are preserved."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE
from src.tsp_mapping_controls import ARMS, prepare_controls, train_control, evaluate_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    source_root = root / 'logs_smoke_20261020_224736/MaxCutProblem-0-SurrogateVAE'
    source_data = root / 'data/smoke_20261020_224736/train/max_cut_problem_0_30'
    parser.add_argument('--source-checkpoint', type=Path, default=source_root / 'best_model.pt')
    parser.add_argument('--source-config', type=Path, default=source_root / 'HyperParam.yaml')
    parser.add_argument('--source-x', type=Path, default=source_data / 'x.npy')
    parser.add_argument('--source-y', type=Path, default=source_data / 'y.npy')
    parser.add_argument('--source-latents', type=Path, help='Optional precomputed bank; otherwise load original expert')
    parser.add_argument('--target-data', type=Path, default=root / 'data/tsp_milestone_a/tsp20_seed1088.npz')
    parser.add_argument('--target-checkpoint', type=Path, help='Optional warm start; disables clean validation claim')
    parser.add_argument('--config', type=Path, default=root / 'configs/tsp_mapping_controls.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS),
                        help='Train only selected arms; default: all five')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms must not contain duplicates')
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    for key in ('epochs', 'batch_size', 'hidden_dim', 'latent_dim', 'max_pairs', 'cpu_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for key in ('learning_rate', 'grad_clip'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    if args.output.exists():
        raise FileExistsError('choose a new output directory')
    seed = int(config['seed'])
    torch.manual_seed(seed)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.source_latents is not None:
        source_latent = np.load(args.source_latents, allow_pickle=False)
        source_quality = np.load(args.source_y, allow_pickle=False)
        source_paths = [args.source_latents, args.source_y]
        source_provenance = 'precomputed_bank; caller_responsible_for_provenance'
    else:
        # Lazy legacy import: test fixtures and precomputed banks do not need sklearn.
        import yaml
        from src.tsp_mapping import load_maxcut_latent
        source_latent, source_quality, _ = load_maxcut_latent(
            args.source_checkpoint, yaml.safe_load(args.source_config.read_text()),
            args.source_x, args.source_y, device)
        source_paths = [args.source_checkpoint, args.source_config, args.source_x, args.source_y]
        source_provenance = 'legacy_MaxCut_posterior_mean'
    with np.load(args.target_data, allow_pickle=False) as saved:
        problem = TSPProblem(coordinates=saved['coordinates'])
        tours = saved['tours'].copy()
        quality = saved['qualities'].copy()
    input_count = len(tours)
    lengths = problem.evaluate_batch(tours)
    if quality.shape != lengths.shape:
        raise ValueError('target quality shape mismatch')
    np.testing.assert_allclose(quality, -lengths, rtol=1e-7, atol=1e-9)
    conditions, paired_tours, train_ids, valid_ids, pairing = prepare_controls(
        source_latent, source_quality, tours, quality, seed, config['max_pairs'])
    if args.target_checkpoint is None:
        template = TSPVAE(config['hidden_dim'], config['latent_dim']).to(device)
        initialization = 'shared_fresh_random_decoder; no_target_expert_pretraining'
    else:
        payload = torch.load(args.target_checkpoint, map_location=device, weights_only=False)
        template = TSPVAE(payload['config']['hidden_dim'], payload['config']['latent_dim']).to(device)
        template.load_state_dict(payload['model'])
        initialization = 'shared_warm_target_expert; pretraining_overlap_not_certified'
        config['hidden_dim'], config['latent_dim'] = template.hidden_dim, template.latent_dim
    bridge_template = torch.nn.Linear(source_latent.shape[1], template.latent_dim).to(device)
    coordinates = torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None]
    args.output.mkdir(parents=True)
    histories, reports, generated, checkpoints = {}, {}, {}, {}
    for arm in args.arms:
        model = copy.deepcopy(template)
        bridge = None if arm == 'no_source' else copy.deepcopy(bridge_template)
        history, best = train_control(model, bridge, coordinates, conditions[arm], paired_tours,
                                      train_ids, valid_ids, config)
        histories[arm] = history
        reports[arm] = dict(best_epoch=best['epoch'], best_validation_nll=best['valid_nll'])
        for sample, mode in ((False, 'greedy'), (True, 'sampled')):
            result, output = evaluate_control(model, bridge, coordinates,
                conditions[arm][valid_ids], paired_tours[valid_ids], problem, seed + 1, sample)
            reports[arm][mode] = result
            generated[arm + '_' + mode] = output
        torch.save(dict(model=best['model'], bridge=best['bridge'], epoch=best['epoch'],
                        source_dim=source_latent.shape[1], target_dim=template.latent_dim,
                        config=config, arm=arm, initialization=initialization), args.output / (arm + '.pt'))
        checkpoints[arm] = arm + '.pt'
        print('{} best_epoch={} greedy_mean={:.5f} sampled_mean={:.5f}'.format(
            arm, best['epoch'], reports[arm]['greedy']['mean_length'], reports[arm]['sampled']['mean_length']), flush=True)
    random_tours = problem.sample_tours(len(valid_ids), seed=seed + 2)
    random_costs = problem.evaluate_batch(random_tours)
    queries = sum(reports[arm][mode]['objective_queries'] for arm in args.arms for mode in ('greedy', 'sampled')) + len(valid_ids)
    costs = dict(existing_target_label_count=input_count,
                 target_data_generation_queries_this_run=0,
                 target_verification_queries_this_run=input_count,
                 target_training_queries_this_run=0,
                 target_evaluation_queries_this_run=queries,
                 target_total_current_run=input_count + queries,
                 target_total_with_assumed_one_call_per_original_label=input_count * 2 + queries,
                 prior_source_expert_queries='unknown',
                 prior_target_expert_queries='not_used' if args.target_checkpoint is None else 'unknown',
                 note='Verification calls are repeated evaluations. Imported labels were not generated by this run. Cumulative prior-run costs are not reconstructed.')
    paths = source_paths + [args.target_data, args.config]
    if args.target_checkpoint is not None:
        paths += [args.target_checkpoint]
    report = dict(status='mapping_controls_functional_run_completed', config=config,
                  protocol_version='C.1.1_uniform_shuffle', evaluated_arms=args.arms,
                  initialization=initialization, source_provenance=source_provenance,
                  target_validation_unseen_by_decoder_pretraining=args.target_checkpoint is None,
                  validation_scope='held_out_cycles_on_one_instance; used_for_checkpoint_selection; not_a_final_test_set',
                  paired_validation_nll='diagnostic_pairing_uses_validation_target_quality; not_an_inference_input',
                  pair_count=len(paired_tours), train_count=len(train_ids), valid_count=len(valid_ids),
                  conditions=reports, target_costs=costs,
                  random_tour_reference=dict(mean_length=float(random_costs.mean()), best_length=float(random_costs.min())),
                  checkpoints=checkpoints,
                  device=str(device),
                  file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  source_code_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                      ('src/tsp_mapping_controls.py', 'src/experiment_tsp_mapping_controls.py')},
                  torch_version=str(torch.__version__), numpy_version=np.__version__)
    for name, value in (('diagnostics.json', report), ('history.json', histories), ('costs.json', costs),
                        ('conditioning.json', pairing)):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'split.npz', train_indices=train_ids, valid_indices=valid_ids,
                        source_order=pairing['source_order'], target_order=pairing['target_order'])
    np.savez_compressed(args.output / 'conditioning.npz', **conditions)
    np.savez_compressed(args.output / 'generated.npz', coordinates=problem.coordinates,
                        valid_target_tours=paired_tours[valid_ids], random_tours=random_tours, **generated)
    print('TSP MAPPING CONTROLS FUNCTIONAL RUN COMPLETED')
    print('diagnostics:', args.output / 'diagnostics.json')


if __name__ == '__main__':
    main()
