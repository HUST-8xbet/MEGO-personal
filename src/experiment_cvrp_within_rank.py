"""Three-seed, within-quality-band shuffle diagnostic on the existing CVRP20."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from src.cvrp_mapping import prepare_pairs, train_mapping
from src.experiment_cvrp_mapping import load_source, sha256
from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE
from src.cvrp_within_rank import (ARMS, within_rank_permutation, generation_plan,
    generation_conditions, generate_rounds, summarize_generation, summarize_seeds,
    percentile_from_train, state_digest, batch_order_digest, validation_nll)

STATE_START = '<!-- within-rank-protocol:start -->'
STATE_END = '<!-- within-rank-protocol:end -->'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def update_state(path, section):
    content = path.read_text() if path.exists() else '# PROJECT_STATE\n'
    block = STATE_START + '\n' + section.rstrip() + '\n' + STATE_END
    if STATE_START in content:
        start, end = content.index(STATE_START), content.index(STATE_END) + len(STATE_END)
        content = content[:start] + block + content[end:]
    else:
        content = content.rstrip() + '\n\n' + block + '\n'
    path.write_text(content)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def protocol_config(spec, baseline, smoke):
    seeds = spec['training_seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('production protocol requires three distinct nonnegative training seeds')
    if (spec['quality_groups'], spec['sampling_rounds'], spec['candidates_per_round']) != (10, 10, 41):
        raise ValueError('production protocol requires 10 quality bins and 10 rounds x K=41')
    if spec['split_seed'] != baseline['seed']:
        raise ValueError('keep the previous pairing/split seed')
    for name in ('shuffle_seed', 'source_selection_seed', 'sampling_seed'):
        if type(spec[name]) is not int or spec[name] < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    expected = dict(epochs=30, batch_size=128, learning_rate=.001, grad_clip=1.,
                    max_pairs=2048, hidden_dim=64, latent_dim=16, cpu_threads=1, seed=1088)
    if baseline != expected:
        raise ValueError('baseline config differs from the completed CVRP protocol')
    resolved = dict(baseline)
    resolved.update(spec)
    resolved['execution_mode'] = 'SMOKE_ONLY_NOT_EXPERIMENT' if smoke else 'requested_diagnostic'
    if smoke:
        # Software test only: still exercises all THREE seeds and all THREE arms.
        resolved.update(epochs=1, max_pairs=64, sampling_rounds=2, candidates_per_round=3)
    return resolved


def render_summary(report):
    shared = report['shared_adaptation']
    lines = ['# Within-rank shuffle diagnostic', '', 'Execution: ' + report['execution_mode'], '',
             'Single CVRP20 instance; diagnostic only, no novelty claim.', '',
             'Best shared adaptation solution: {:.6f} (train + validation support).'.format(shared['best_adaptation_length']), '',
             '| Seed | Arm | Greedy mean | Sample mean | Best-of-K mean ± std | Best sampled overall | Final incl. adaptation | Val NLL |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in report['seeds']:
        for arm in ARMS:
            r = seed['arms'][arm]; s = r['sampled']
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} ± {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
                seed['training_seed'], arm, r['greedy']['mean_length'], s['mean_length'],
                s['best_of_k_mean'], s['best_of_k_std'], s['best_of_all'],
                s['best_final_including_adaptation'], r['best_validation_nll']))
    lines += ['', 'Within-seed std is across rounds, ddof=1. Seeds are summarized separately in diagnostics.json.',
              'Final column uses ONLY sampling budget plus shared adaptation; greedy+sampling union has a separate combined-budget field.',
              'No conclusions should pool this result with old single-seed protocol outputs.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--source-checkpoint', type=Path, default=root/'logs_tsp/tsp20_expert_smoke_01/best_model.pt')
    parser.add_argument('--source-data', type=Path, default=root/'data/tsp_milestone_a/tsp20_seed1088.npz')
    parser.add_argument('--target-data', type=Path, default=root/'data/cvrp_milestone_a/cvrp20_seed2088.npz')
    parser.add_argument('--config', type=Path, default=root/'configs/cvrp_within_rank.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-state', type=Path, default=root/'PROJECT_STATE.md')
    parser.add_argument('--smoke-test', action='store_true', help='Reduced CPU software check, NOT the requested experiment')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new output directory; no resume or overwrite')
    baseline_path = root/'configs/cvrp_mapping_smoke.json'
    spec = json.loads(args.config.read_text())
    if sha256(baseline_path) != spec['expected_baseline_config_sha256']:
        raise ValueError('baseline config checksum changed; restore the recorded protocol')
    config = protocol_config(spec, json.loads(baseline_path.read_text()), args.smoke_test)
    paths = dict(source_checkpoint=args.source_checkpoint, source_data=args.source_data, target_data=args.target_data)
    input_hashes = {key: sha256(path) for key, path in paths.items()}
    if not args.smoke_test:
        for key, expected in spec['expected_inputs'].items():
            if input_hashes[key] != expected:
                raise ValueError(key + ' differs from the recorded current instance/checkpoint; do not regenerate or replace it')
    for name, expected in spec['expected_baseline_code'].items():
        if sha256(root/name) != expected:
            raise ValueError(name + ' changed from baseline; restore it before running paired controls')
    code_names = list(spec['expected_baseline_code']) + [
        'src/cvrp_within_rank.py', 'src/experiment_cvrp_within_rank.py', 'configs/cvrp_within_rank.json']
    code_hashes = {name: sha256(root/name) for name in code_names}
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    start = time.perf_counter()
    source, sq, source_original_rows, source_coords, source_calls = load_source(
        args.source_checkpoint, args.source_data, device, config['batch_size'])
    source_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with np.load(args.target_data, allow_pickle=False) as data:
        problem = CVRPProblem(coordinates=data['coordinates'], demands=data['demands'], capacity=int(data['capacity'].item()))
        solutions = problem.unpack_solutions(data['routes_padded'], data['sequence_lengths'])
        target_quality = data['qualities'].copy()
        provenance = json.loads(data['metadata'].item()) if 'metadata' in data else {}
    if target_quality.shape != (len(solutions),):
        raise ValueError('target quality shape mismatch')
    checked_lengths = problem.evaluate_batch(solutions)
    np.testing.assert_allclose(target_quality, -checked_lengths, rtol=1e-7, atol=1e-9)
    base_conditions, padded, sizes, train, valid, pairing = prepare_pairs(
        problem, source, sq, solutions, target_quality, config['split_seed'], config['max_pairs'])
    if not args.smoke_test and (problem.dimension, problem.capacity, len(padded), len(valid)) != (20, 30, 2048, 410):
        raise ValueError('production protocol expects current CVRP20/30, 2048 pairs and 410 validation source rows')
    pair_quality = sq[np.asarray(pairing['source_order'])]
    p = percentile_from_train(pair_quality, pair_quality[train])
    np.testing.assert_allclose(base_conditions['rank_only'][:, 0], 2*p-1, atol=1e-6)
    plan = generation_plan(valid, config['sampling_rounds'], config['candidates_per_round'], config['source_selection_seed'])
    gen_conditions = generation_conditions(base_conditions['source'], base_conditions['rank_only'], plan)
    # Adaptation incumbent is metrics-only; never passed to source selection or decode.
    original_target_rows = np.asarray(pairing['target_order'])
    support_lengths = checked_lengths[original_target_rows]
    best_pos = int(support_lengths.argmin())
    adaptation = dict(label_count_shared=len(padded), training_label_count=len(train), validation_label_count=len(valid),
                      definition='entire selected adaptation support INCLUDING validation labels used for checkpoint selection',
                      best_adaptation_length=float(support_lengths.min()),
                      best_training_only_length=float(support_lengths[train].min()),
                      best_validation_support_length=float(support_lengths[valid].min()),
                      best_loaded_dataset_length=float(checked_lengths.min()),
                      best_original_target_row=int(original_target_rows[best_pos]))
    target_prepare_seconds = time.perf_counter()-start
    instance = (torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None],
                torch.tensor(problem.demands.copy(), dtype=torch.long, device=device)[None],
                torch.tensor([problem.capacity], dtype=torch.long, device=device))
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json', config)
    write_json(args.output/'pairing.json', pairing)
    np.savez_compressed(args.output/'shared_plan.npz', train_pair_positions=train, validation_pair_positions=valid,
                        source_bank_rows=pairing['source_order'], target_original_rows=original_target_rows,
                        source_original_rows=source_original_rows[pairing['source_order']],
                        generation_pair_positions=plan,
                        generation_source_bank_rows=np.asarray(pairing['source_order'])[plan],
                        generation_source_original_rows=source_original_rows[np.asarray(pairing['source_order'])[plan]],
                        original_latent_condition=base_conditions['source'], rank_condition=base_conditions['rank_only'])
    np.savez_compressed(args.output/'adaptation_incumbent.npz', coordinates=problem.coordinates, demands=problem.demands,
                        capacity=problem.capacity, padded=padded[best_pos], sequence_length=sizes[best_pos],
                        length=support_lengths[best_pos], original_target_row=original_target_rows[best_pos])
    seed_reports, costs_per_arm = [], []
    for training_seed in config['training_seeds']:
        seed_dir = args.output/('seed_'+str(training_seed)); seed_dir.mkdir()
        shuffled, audit, audit_arrays = within_rank_permutation(base_conditions['source'], pair_quality, train, valid,
            config['shuffle_seed'], training_seed, config['quality_groups'])
        conditions = dict(source=base_conditions['source'], within_rank_shuffle=shuffled, rank_only=base_conditions['rank_only'])
        np.savez_compressed(seed_dir/'shuffle_audit.npz', **audit_arrays,
                            quality_before=pair_quality, quality_after=pair_quality[audit_arrays['permutation']],
                            train_pair_positions=train, validation_pair_positions=valid)
        write_json(seed_dir/'shuffle_audit.json', audit)
        torch.manual_seed(training_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(training_seed)
        template = CVRPVAE(config['hidden_dim'], config['latent_dim']).to(device)
        bridge_template = torch.nn.Linear(source.shape[1], template.latent_dim).to(device)
        initial = dict(decoder_and_unused_heads=state_digest(template), bridge=state_digest(bridge_template))
        order_hash = batch_order_digest(train, config['epochs'], training_seed)
        seed_report = dict(training_seed=training_seed, shuffle_audit=audit, initial_state_sha256=initial,
                           batch_order_sha256=order_hash, arms={})
        train_config = dict(config, seed=training_seed)
        for arm in ARMS:
            arm_dir = seed_dir/arm; arm_dir.mkdir()
            model, bridge = copy.deepcopy(template), copy.deepcopy(bridge_template)
            assert state_digest(model) == initial['decoder_and_unused_heads']
            assert state_digest(bridge) == initial['bridge']
            synchronize(device); start = time.perf_counter()
            history, best = train_mapping(model, bridge, instance, conditions[arm], padded, sizes, train, valid, train_config)
            synchronize(device); train_seconds = time.perf_counter()-start
            nll = validation_nll(model, bridge, instance, conditions[arm][valid], padded[valid], sizes[valid], config['batch_size'])
            if not np.isclose(nll, best['valid_nll'], rtol=1e-6, atol=1e-6):
                raise RuntimeError('reloaded checkpoint NLL differs from selection NLL')
            torch.save(dict(model=best['model'], bridge=best['bridge'], epoch=best['epoch'], validation_nll=best['valid_nll'],
                            config=train_config, arm=arm, initial_state_sha256=initial, batch_order_sha256=order_hash,
                            source_dim=source.shape[1], target_dim=template.latent_dim, input_sha256=input_hashes,
                            code_sha256=code_hashes), arm_dir/'mapping_checkpoint.pt')
            record = dict(best_epoch=best['epoch'], best_validation_nll=nll,
                          validation_condition='within-band validation shuffle' if arm=='within_rank_shuffle' else arm,
                          checkpoint_selection='minimum real-action-weighted validation reconstruction NLL; no generation selection',
                          initial_state_sha256=initial, batch_order_sha256=order_hash,
                          adaptation_training_and_validation_seconds=train_seconds)
            generated = {}
            synchronize(device); start = time.perf_counter()
            for mode, sample in (('greedy', False), ('sampled', True)):
                metrics, arrays = generate_rounds(model, bridge, instance, gen_conditions[arm], problem,
                    config['sampling_seed'], training_seed, sample)
                # Only post-process evaluated costs after ALL fixed-budget generation.
                metrics = dict(metrics, **{k:v for k,v in summarize_generation(arrays['lengths'], arrays['route_counts'],
                                     adaptation['best_adaptation_length']).items() if k not in metrics})
                record[mode] = metrics
                generated.update({mode+'_'+k:v for k,v in arrays.items()})
            synchronize(device); generation_seconds = time.perf_counter()-start
            total_calls = record['greedy']['objective_queries']+record['sampled']['objective_queries']
            record['greedy_plus_sampled_union'] = dict(objective_queries=total_calls,
                best_generated_length=min(record['greedy']['best_of_all'], record['sampled']['best_of_all']),
                best_final_including_adaptation=min(adaptation['best_adaptation_length'],
                    record['greedy']['best_of_all'], record['sampled']['best_of_all']))
            arm_cost = dict(training_seed=training_seed, arm=arm,
                            adaptation_labels_reused_from_shared_pool=True, new_adaptation_label_queries=0,
                            gradient_objective_queries=0, validation_objective_queries=0,
                            greedy_generation_queries=record['greedy']['objective_queries'],
                            sampled_generation_queries=record['sampled']['objective_queries'],
                            generation_queries_total=total_calls,
                            adaptation_training_and_validation_seconds=train_seconds,
                            generation_and_evaluation_seconds=generation_seconds)
            write_json(arm_dir/'config.json', dict(train_config, arm=arm))
            write_json(arm_dir/'history.json', history)
            write_json(arm_dir/'diagnostics.json', record)
            write_json(arm_dir/'costs.json', arm_cost)
            np.savez_compressed(arm_dir/'generated.npz', coordinates=problem.coordinates, demands=problem.demands,
                                capacity=problem.capacity, generation_source_original_rows=source_original_rows[np.asarray(pairing['source_order'])[plan]],
                                **generated)
            seed_report['arms'][arm] = record; costs_per_arm.append(arm_cost)
            print('seed={} arm={} val_nll={:.6f} sampled_best_of_{}={:.6f} +/- {:.6f} best_of_{}={:.6f} final={:.6f}'.format(
                training_seed, arm, nll, config['candidates_per_round'], record['sampled']['best_of_k_mean'],
                record['sampled']['best_of_k_std'], config['sampling_rounds']*config['candidates_per_round'],
                record['sampled']['best_of_all'], record['sampled']['best_final_including_adaptation']), flush=True)
        write_json(seed_dir/'diagnostics.json', seed_report)
        seed_reports.append(seed_report)
    if any(sha256(root/name) != digest for name, digest in code_hashes.items()):
        raise RuntimeError('code changed while experiment was running')
    if any(sha256(paths[key]) != digest for key, digest in input_hashes.items()):
        raise RuntimeError('input data/checkpoint changed while experiment was running')
    gen_calls = sum(c['generation_queries_total'] for c in costs_per_arm)
    costs = dict(shared=dict(existing_adaptation_labels=len(padded), imported_target_labels_in_file=len(solutions),
                            new_target_label_generation_queries=0, target_verification_queries=len(solutions),
                            source_verification_queries=source_calls, source_loading_seconds=source_load_seconds,
                            target_preparation_seconds=target_prepare_seconds,
                            prior_target_data_generation_queries=provenance.get('data_generation_objective_queries', 'unknown'),
                            prior_target_data_reload_queries=provenance.get('reload_verification_objective_queries', 'unknown'),
                            prior_source_expert_pretraining_queries='unknown; not counted as zero'),
                 per_arm_seed=costs_per_arm, generated_target_queries_all_arms_and_seeds=gen_calls,
                 target_objective_queries_current_run=len(solutions)+gen_calls,
                 source_and_target_objective_queries_current_run=source_calls+len(solutions)+gen_calls,
                 note='Shared adaptation counted once; each arm/seed uses greedy {0} + sampled {0}. No random-reference budget or generation-based tuning.'.format(config['sampling_rounds']*config['candidates_per_round']))
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        commit='unavailable'
    paired_deltas = {}
    for control in ('within_rank_shuffle', 'rank_only'):
        paired_deltas[control] = {}
        for mode in ('greedy', 'sampled'):
            paired_deltas[control][mode] = {}
            for metric in ('mean_length', 'best_of_k_mean', 'best_of_all', 'best_final_including_adaptation'):
                values = [r['arms']['source'][mode][metric]-r['arms'][control][mode][metric] for r in seed_reports]
                paired_deltas[control][mode][metric] = dict(source_minus_control_by_seed=values,
                    mean=float(np.mean(values)), std=float(np.std(values,ddof=1)), std_ddof=1, negative_favors_source=True)
    report = dict(status='within_rank_diagnostic_completed' if not args.smoke_test else 'SMOKE_TEST_ONLY_COMPLETED',
                  execution_mode=config['execution_mode'], protocol=config['protocol'], config=config,
                  expected_input_hash_checks_applied=not args.smoke_test, baseline_code_hash_checks_applied=True,
                  base_commit=commit, device=str(device), torch_version=str(torch.__version__), numpy_version=np.__version__,
                  input_sha256=input_hashes, code_sha256=code_hashes,
                  pair_count=len(padded), train_count=len(train), validation_count=len(valid),
                  shared_adaptation=adaptation, source_selection='fixed uniform permutation of SOURCE VALIDATION IDs; partition into rounds*K; no target label',
                  inference_condition='source and within_rank_shuffle use same ORIGINAL latent IDs; rank_only uses their original train-CDF rank',
                  source_selection_seed=config['source_selection_seed'], generation_source_original_rows=source_original_rows[np.asarray(pairing['source_order'])[plan]].tolist(),
                  seeds=seed_reports, across_training_seeds=summarize_seeds(seed_reports), paired_seed_differences=paired_deltas, costs=costs,
                  validation_scope='one instance; same validation support used for checkpoint selection; diagnostic, not benchmark or novelty confirmation')
    write_json(args.output/'diagnostics.json', report)
    write_json(args.output/'costs.json', costs)
    (args.output/'summary.md').write_text(render_summary(report))
    state = ('## Within-rank shuffle: triển khai theo yêu cầu nghiên cứu\n\n'
             'Trạng thái: '+report['status']+'.\n'
             'Ba nhánh source / within_rank_shuffle / rank_only; ba seed 1088/1089/1090; '
             'giữ kiến trúc, optimizer và checkpoint selection theo validation NLL.\n'
             'Biên decile fit source train; shuffle validation độc lập; generation dùng nguồn gốc chung.\n'
             'Kết quả và ngân sách: '+str(args.output.resolve())+'/diagnostics.json và costs.json.\n'
             'Đây là chẩn đoán trên một instance, chưa là benchmark hoặc novelty.\n')
    for arm in ARMS:
        agg = report['across_training_seeds'][arm]
        state += ('- {}: sampled mean length {:.6f}; mean(best-of-K) across seeds {:.6f} +/- {:.6f}; '
                  'mean best-generated overall {:.6f}; mean final including adaptation {:.6f}.\n').format(
            arm, agg['sampled']['mean_length']['mean'], agg['sampled']['best_of_k_mean']['mean'],
            agg['sampled']['best_of_k_mean']['std'], agg['sampled']['best_of_all']['mean'],
            agg['sampled']['best_final_including_adaptation']['mean'])
    (args.output/'project_state_update.md').write_text(state)
    if not args.smoke_test:
        update_state(args.project_state, state)
    print('WITHIN-RANK SHUFFLE DIAGNOSTIC COMPLETED' if not args.smoke_test else 'WITHIN-RANK SMOKE TEST ONLY COMPLETED')
    print('diagnostics:',args.output/'diagnostics.json')
    print('summary:',args.output/'summary.md')


if __name__=='__main__':
    main()
