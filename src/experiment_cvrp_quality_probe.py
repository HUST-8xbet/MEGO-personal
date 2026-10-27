"""Evaluate all nine EXISTING TSP->CVRP checkpoints by source validation quintile."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.cvrp_quality_probe import (quality_plan, sampling_seeds, decode_probe,
                                    report_probe, InsufficientGroupError)
from src.cvrp_within_rank import ARMS, generation_conditions, percentile_from_train, state_digest
from src.experiment_cvrp_mapping import sha256
from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def update_state(path, section):
    start='<!-- source-quality-probe:start -->';end='<!-- source-quality-probe:end -->'
    text=path.read_text() if path.exists() else '# PROJECT_STATE\n'
    block=start+'\n'+section.rstrip()+'\n'+end
    if (start in text)!=(end in text):
        raise ValueError('incomplete source-quality state marker')
    if start in text:
        left,right=text.index(start),text.index(end)+len(end)
        text=text[:left]+block+text[right:]
    else:
        text=text.rstrip()+'\n\n'+block+'\n'
    path.write_text(text)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def validate_checkpoint(payload, parent, seed, arm, width):
    record = next(s for s in parent['seeds'] if s['training_seed']==seed)
    expected_config = dict(parent['config'], seed=seed)
    for key, value in dict(arm=arm, config=expected_config, source_dim=width,
                           target_dim=expected_config['latent_dim'], input_sha256=parent['input_sha256'],
                           code_sha256=parent['code_sha256'], initial_state_sha256=record['initial_state_sha256'],
                           batch_order_sha256=record['batch_order_sha256'],
                           epoch=record['arms'][arm]['best_epoch']).items():
        if payload.get(key) != value:
            raise ValueError('checkpoint metadata mismatch seed={} arm={} key={}'.format(seed,arm,key))
    if not np.isclose(payload['validation_nll'], record['arms'][arm]['best_validation_nll'], rtol=1e-6, atol=1e-6):
        raise ValueError('checkpoint validation NLL mismatch')


def aggregate_seeds(records):
    def stats(values):
        if any(x is None for x in values):
            return dict(mean=None, std_between_training_seeds=None, training_seed_count=len(values),
                        reason='at least one seed has undefined metric')
        return dict(mean=float(np.mean(values)), std_between_training_seeds=float(np.std(values,ddof=1)),
                    training_seed_count=len(values), std_ddof=1)
    result = {}
    for arm in ARMS:
        runs=[r['arms'][arm] for r in records]
        result[arm]=dict(spearman=stats([r['spearman']['rho'] for r in runs]),groups=[])
        for g in range(5):
            row=dict(group=g+1)
            for mode in ('greedy','sampled'):
                row[mode]={key:stats([r['groups'][g][mode][key] for r in runs]) for key in (
                    'mean_length','mean_route_count','valid_rate','best_of_k_mean',
                    'better_than_adaptation_rate','best_generated_length','best_final_including_adaptation')}
            result[arm]['groups'].append(row)
    return result


def table_csv(report):
    buf=io.StringIO(); fields=['training_seed','arm','group','validation_pool_count','selected_count',
        'greedy_mean_length','greedy_mean_route_count','greedy_valid_rate','greedy_better_than_adaptation_rate',
        'sampling_mean_length','sampling_mean_route_count','sampling_valid_rate',
        'best_of_k_mean','best_of_k_std_across_sampling_rounds','sampling_better_than_adaptation_rate',
        'best_sampled','final_sampled_including_adaptation','final_greedy_including_adaptation','final_union']
    writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader()
    for seed in report['seeds']:
        for arm,r in seed['arms'].items():
            for g in r['groups']:
                a,b=g['greedy'],g['sampled'];i=g['group']-1
                writer.writerow(dict(training_seed=seed['training_seed'],arm=arm,group=i+1,
                    validation_pool_count=report['selection_audit']['validation_counts'][i],
                    selected_count=report['config']['per_group'],
                    greedy_mean_length=a['mean_length'],greedy_mean_route_count=a['mean_route_count'],
                    greedy_valid_rate=a['valid_rate'],greedy_better_than_adaptation_rate=a['better_than_adaptation_rate'],
                    sampling_mean_length=b['mean_length'],sampling_mean_route_count=b['mean_route_count'],
                    sampling_valid_rate=b['valid_rate'],best_of_k_mean=b['best_of_k_mean'],
                    best_of_k_std_across_sampling_rounds=b['best_of_k_std_across_sampling_rounds'],
                    sampling_better_than_adaptation_rate=b['better_than_adaptation_rate'],
                    best_sampled=b['best_generated_length'],final_sampled_including_adaptation=b['best_final_including_adaptation'],
                    final_greedy_including_adaptation=a['best_final_including_adaptation'],
                    final_union=min(a['best_final_including_adaptation'],b['best_final_including_adaptation'])))
    return buf.getvalue()


def number(value):
    return 'NA' if value is None else '{:.6f}'.format(value)


def render_summary(report):
    lines=['# Source-quality quintile diagnostic','', 'Execution: '+report['execution_mode'],
        'Q1 worst -> Q5 best source quality; higher train-CDF percentile means better source.',
        'Validation pool counts: '+str(report['selection_audit']['validation_counts']),
        'Exact shared adaptation incumbent: '+number(report['adaptation_incumbent']), '',
        '| Seed | Arm | Q | Greedy length | Sample length | Routes G / S | Valid G / S | Best-of-K mean ± round std | Sample beat incumbent | Final sample + adaptation |',
        '| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |']
    for seed in report['seeds']:
        for arm,r in seed['arms'].items():
            for g in r['groups']:
                a,b=g['greedy'],g['sampled']
                lines.append('| {} | {} | {} | {} | {} | {} / {} | {} / {} | {} ± {} | {} | {} |'.format(
                    seed['training_seed'],arm,g['group'],number(a['mean_length']),number(b['mean_length']),
                    number(a['mean_route_count']),number(b['mean_route_count']),number(a['valid_rate']),number(b['valid_rate']),
                    number(b['best_of_k_mean']),number(b['best_of_k_std_across_sampling_rounds']),
                    number(b['better_than_adaptation_rate']),number(b['best_final_including_adaptation'])))
    lines += ['', 'Round std above uses five sampling rounds (ddof=1), not training seeds.', '',
              '| Training seed | source Spearman | within_rank_shuffle Spearman | rank_only Spearman |',
              '| --- | ---: | ---: | ---: |']
    for seed in report['seeds']:
        lines.append('| {} | {} | {} | {} |'.format(seed['training_seed'],
            *[number(seed['arms'][a]['spearman']['rho']) for a in ARMS]))
    lines += ['', '| Arm | Q | Greedy mean ± seed std | Sampling mean ± seed std | Mean(best-of-K) ± seed std |',
              '| --- | --- | --- | --- | --- |']
    for arm,aggregate in report['across_training_seeds'].items():
        for g in aggregate['groups']:
            cells=[]
            for mode,metric in [('greedy','mean_length'),('sampled','mean_length'),('sampled','best_of_k_mean')]:
                stat=g[mode][metric]
                cells.append(number(stat['mean'])+' ± '+number(stat['std_between_training_seeds']))
            lines.append('| {} | {} | {} | {} | {} |'.format(arm,g['group'],*cells))
    lines += ['', 'Spearman uses one observation per selected latent, target length averaged over its sampling rounds.',
              'Negative is the desired direction. Descriptive on a balanced stratified selection of 100 validation latents.',
              'Across-training-seed group tables (mean and separate seed std) are in diagnostics.json.',
              'Paired per-group source-minus-control differences are also saved; negative length differences favor source.',
              'No target-based source/group selection, checkpoint tuning, temperature tuning or significance/novelty claim.']
    return '\n'.join(lines)+'\n'


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=root/'logs_cvrp/within_rank_shuffle_v1')
    parser.add_argument('--source-data',type=Path,default=root/'data/tsp_milestone_a/tsp20_seed1088.npz')
    parser.add_argument('--target-data',type=Path,default=root/'data/cvrp_milestone_a/cvrp20_seed2088.npz')
    parser.add_argument('--config',type=Path,default=root/'configs/cvrp_quality_probe.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--project-state',type=Path,default=root/'PROJECT_STATE.md')
    parser.add_argument('--smoke-test',action='store_true',help='Software only: existing reduced CPU checkpoints, 1 latent/group x 2 rounds')
    args=parser.parse_args()
    wall_start=time.perf_counter()
    if args.output.exists():
        raise FileExistsError('choose a new output directory; no overwrite or resume')
    spec=json.loads(args.config.read_text())
    if (spec['training_seeds'],spec['groups'],spec['per_group'],spec['sampling_rounds'])!=([1088,1089,1090],5,20,5):
        raise ValueError('locked protocol: seeds 1088/1089/1090, five quintiles, 20 latents, five rounds')
    config=dict(spec,execution_mode='SMOKE_ONLY_NOT_EXPERIMENT' if args.smoke_test else 'requested_diagnostic')
    if args.smoke_test:
        config.update(per_group=1,sampling_rounds=2)
    prior=args.run_dir
    # Every dependency is read-only and hashed before use; rechecked after generation.
    dependencies={}
    def track(path):
        if not path.is_file():
            raise FileNotFoundError('Missing existing-run file: '+str(path))
        dependencies[str(path.resolve())]=sha256(path)
        return path
    parent_path=track(prior/'diagnostics.json')
    parent=json.loads(parent_path.read_text())
    if not args.smoke_test:
        if sha256(parent_path)!=spec['parent_diagnostics_sha256']:
            raise ValueError('parent diagnostics differs from the completed nine-checkpoint run')
        if parent['execution_mode']!='requested_diagnostic' or parent['status']!='within_rank_diagnostic_completed':
            raise ValueError('production requires completed production checkpoints')
        if parent['input_sha256']!=spec['expected_inputs']:
            raise ValueError('parent inputs differ from the recorded current instance')
    elif 'SMOKE' not in parent['execution_mode']:
        raise ValueError('smoke mode requires explicitly marked smoke checkpoints')
    if parent['config']['training_seeds']!=config['training_seeds'] or [s['training_seed'] for s in parent['seeds']]!=config['training_seeds']:
        raise ValueError('need exactly the three completed training seeds')
    for name,digest in parent['code_sha256'].items():
        if sha256(track(root/name))!=digest:
            raise ValueError('original training code/config differs: '+name)
    track(args.config);track(root/'src/cvrp_quality_probe.py');track(Path(__file__))
    for key,path in [('source_data',args.source_data),('target_data',args.target_data)]:
        if sha256(track(path))!=parent['input_sha256'][key]:
            raise ValueError(key+' hash differs from the existing checkpoint run')
    with np.load(track(prior/'shared_plan.npz'),allow_pickle=False) as saved:
        bank={k:saved[k].copy() for k in saved.files}
    pairing=json.loads(track(prior/'pairing.json').read_text())
    train,valid=bank['train_pair_positions'],bank['validation_pair_positions']
    latent,rank=bank['original_latent_condition'],bank['rank_condition']
    ids=bank['source_original_rows']
    if (latent.ndim!=2 or rank.shape!=latent.shape or ids.shape!=(len(latent),) or
            not np.isfinite(latent).all() or len(np.unique(ids))!=len(ids) or
            ids.dtype.kind not in 'iu' or np.any(ids<0)):
        raise ValueError('invalid existing latent bank / source IDs')
    np.testing.assert_array_equal(bank['source_bank_rows'],pairing['source_order'])
    if len(train)!=parent['train_count'] or len(valid)!=parent['validation_count']:
        raise ValueError('existing split size mismatch')
    np.testing.assert_array_equal(ids[bank['generation_pair_positions']],parent['generation_source_original_rows'])
    np.testing.assert_array_equal(bank['generation_source_original_rows'],parent['generation_source_original_rows'])
    if not args.smoke_test:
        np.testing.assert_array_equal(np.sort(ids[valid]),
            np.sort(np.asarray(parent['generation_source_original_rows']).ravel()))
    if not args.smoke_test and (len(latent),len(train),len(valid))!=(2048,1638,410):
        raise ValueError('production requires the existing 2048-row split')
    with np.load(args.source_data,allow_pickle=False) as saved:
        quality=saved['qualities'][ids].copy()
    np.testing.assert_allclose(np.sort(quality[train]),pairing['rank_reference'],rtol=0,atol=0)
    p=percentile_from_train(quality,quality[train])
    expected_rank=np.zeros_like(latent);expected_rank[:,0]=2*p-1
    np.testing.assert_allclose(rank,expected_rank,rtol=0,atol=1e-6)
    for seed in config['training_seeds']:
        with np.load(track(prior/('seed_'+str(seed))/'shuffle_audit.npz'),allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['quality_before'],quality)
            np.testing.assert_array_equal(saved['train_pair_positions'],train)
            np.testing.assert_array_equal(saved['validation_pair_positions'],valid)
    try:
        plan,groups,p,audit=quality_plan(quality,train,valid,config['per_group'],config['selection_seed'])
    except InsufficientGroupError as error:
        print(json.dumps(error.audit,indent=2));raise
    print('SOURCE VALIDATION QUINTILE COUNTS:',audit['validation_counts'],flush=True)
    print('SELECTED PER QUINTILE:',config['per_group'],flush=True)
    conditions=generation_conditions(latent,rank,plan)
    with np.load(args.target_data,allow_pickle=False) as saved:
        problem=CVRPProblem(coordinates=saved['coordinates'],demands=saved['demands'],capacity=int(saved['capacity'].item()))
    with np.load(track(prior/'adaptation_incumbent.npz'),allow_pickle=False) as saved:
        for name,expected in [('coordinates',problem.coordinates),('demands',problem.demands),('capacity',problem.capacity)]:
            np.testing.assert_array_equal(saved[name],expected)
        incumbent_solution=problem.unpack_solutions(saved['padded'][None],saved['sequence_length'].reshape(1))[0]
        incumbent=problem.evaluate(incumbent_solution)  # ONE shared new verification query.
        np.testing.assert_allclose(incumbent,saved['length'].item(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(incumbent,parent['shared_adaptation']['best_adaptation_length'],rtol=0,atol=1e-12)
    if not args.smoke_test:
        np.testing.assert_allclose(incumbent,config['adaptation_incumbent_exact'],rtol=0,atol=1e-12)
        if (problem.dimension,problem.capacity)!=(20,30):
            raise ValueError('expected existing CVRP20, capacity30')
    payloads={};checkpoint_hashes={}
    for seed in config['training_seeds']:
        for arm in ARMS:
            directory=prior/('seed_'+str(seed))/arm
            path=track(directory/'mapping_checkpoint.pt')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            validate_checkpoint(payload,parent,seed,arm,latent.shape[1])
            history=json.loads(track(directory/'history.json').read_text())
            selected=min(history,key=lambda x:x['valid_nll'])
            if selected['epoch']!=payload['epoch'] or not np.isclose(selected['valid_nll'],payload['validation_nll'],rtol=1e-6,atol=1e-6):
                raise ValueError('checkpoint does not match prior minimum validation NLL')
            payloads[(seed,arm)]=payload
            checkpoint_hashes[str(path.resolve())]=sha256(path)
    torch.set_num_threads(parent['config']['cpu_threads'])
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    instance=(torch.tensor(problem.coordinates.copy(),dtype=torch.float32,device=device)[None],
              torch.tensor(problem.demands.copy(),dtype=torch.long,device=device)[None],
              torch.tensor([problem.capacity],dtype=torch.long,device=device))
    shared_preparation_seconds=time.perf_counter()-wall_start
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',config);write_json(args.output/'selection_audit.json',audit)
    np.savez_compressed(args.output/'source_plan.npz',train_pair_positions=train,validation_pair_positions=valid,
        all_source_quality=quality,all_source_group=groups+1,group_numbers=np.arange(1,6),all_source_percentile_pp=100*p,
        selected_pair_positions=plan,selected_source_original_rows=ids[plan],selected_source_bank_rows=bank['source_bank_rows'][plan],
        selected_source_quality=quality[plan],selected_source_percentile_pp=100*p[plan],
        original_latent_condition=latent[plan],rank_condition=rank[plan])
    records=[];cost_records=[]
    for seed in config['training_seeds']:
        seeds=sampling_seeds(config['sampling_seed'],seed,rounds=config['sampling_rounds'])
        seed_report=dict(training_seed=seed,sampling_action_seeds=seeds.tolist(),arms={})
        for arm in ARMS:
            payload=payloads[(seed,arm)];directory=args.output/('seed_'+str(seed))/arm;directory.mkdir(parents=True)
            model=CVRPVAE(payload['config']['hidden_dim'],payload['target_dim']).to(device)
            bridge=torch.nn.Linear(payload['source_dim'],payload['target_dim']).to(device)
            model.load_state_dict(payload['model'],strict=True);bridge.load_state_dict(payload['bridge'],strict=True)
            model.requires_grad_(False);bridge.requires_grad_(False)
            before=[state_digest(model),state_digest(bridge)]
            synchronize(device);start=time.perf_counter()
            greedy=decode_probe(model,bridge,instance,conditions[arm],seeds[:,:1],problem,False)
            sampled=decode_probe(model,bridge,instance,conditions[arm],seeds,problem,True)
            synchronize(device);elapsed=time.perf_counter()-start
            if before!=[state_digest(model),state_digest(bridge)]:
                raise RuntimeError('frozen checkpoint state changed during inference')
            result,per_latent=report_probe(greedy,sampled,100*p[plan],incumbent)
            result.update(checkpoint_sha256=checkpoint_hashes[str((prior/('seed_'+str(seed))/arm/'mapping_checkpoint.pt').resolve())],
                          selected_epoch=payload['epoch'],selected_validation_nll=payload['validation_nll'],
                          model_state_unchanged=True,decoder_settings='original decode(sample=False/True), unmodified categorical logits')
            seed_report['arms'][arm]=result
            costs=dict(training_seed=seed,arm=arm,new_training_epochs=0,new_adaptation_label_queries=0,
                greedy_objective_attempts=int(greedy['valid'].size),sampled_objective_attempts=int(sampled['valid'].size),
                valid_evaluations=int(greedy['valid'].sum()+sampled['valid'].sum()),
                total_objective_attempts=int(greedy['valid'].size+sampled['valid'].size),
                generation_and_evaluation_seconds=elapsed)
            cost_records.append(costs)
            write_json(directory/'diagnostics.json',result);write_json(directory/'costs.json',costs)
            np.savez_compressed(directory/'generated.npz',coordinates=problem.coordinates,demands=problem.demands,
                capacity=problem.capacity,group_numbers=np.arange(1,6),source_original_rows=ids[plan],source_pair_positions=plan,
                source_percentile_pp=100*p[plan],source_quality=quality[plan],sampling_action_seeds=seeds,
                mean_sampled_length_per_latent=per_latent,
                **{mode+'_'+key:value for mode,arrays in [('greedy',greedy),('sampled',sampled)] for key,value in arrays.items()})
            print('seed={} arm={} Spearman={} mean_sampled={} final={}'.format(seed,arm,number(result['spearman']['rho']),
                number(result['sampled_all_groups']['mean_length']),number(result['best_final_greedy_and_sampled_union'])),flush=True)
        records.append(seed_report)
    for path,digest in dependencies.items():
        if sha256(Path(path))!=digest:
            raise RuntimeError('read-only dependency changed during probe: '+path)
    costs=dict(shared=dict(new_adaptation_labels=0,new_training_epochs=0,checkpoint_selection_queries=0,
        source_objective_queries=0,target_dataset_verification_queries=0,adaptation_incumbent_verification_queries=1,
        preflight_and_checkpoint_loading_seconds=shared_preparation_seconds,
        prior_experiment_costs='historical, not charged again; see parent costs.json'),
        per_arm_seed=cost_records,new_generation_objective_attempts=sum(c['total_objective_attempts'] for c in cost_records),
        new_target_objective_attempts_total=1+sum(c['total_objective_attempts'] for c in cost_records),
        overall_wall_seconds_before_final_report=time.perf_counter()-wall_start,
        accounting='one attempt per generated candidate, including invalid; repeats retained; no repair or retries')
    paired={control:[dict(training_seed=r['training_seed'],
        source_minus_control_spearman=(r['arms']['source']['spearman']['rho']-r['arms'][control]['spearman']['rho'])
            if r['arms']['source']['spearman']['rho'] is not None and r['arms'][control]['spearman']['rho'] is not None else None,
        groups=[dict(group=g+1,**{mode+'_source_minus_control_mean_length':
            (r['arms']['source']['groups'][g][mode]['mean_length']-r['arms'][control]['groups'][g][mode]['mean_length'])
                if r['arms']['source']['groups'][g][mode]['mean_length'] is not None and r['arms'][control]['groups'][g][mode]['mean_length'] is not None else None
            for mode in ('greedy','sampled')}) for g in range(5)]) for r in records]
        for control in ('within_rank_shuffle','rank_only')}
    report=dict(status='SOURCE_QUALITY_PROBE_COMPLETED' if not args.smoke_test else 'SMOKE_TEST_ONLY_COMPLETED',
        execution_mode=config['execution_mode'],config=config,selection_audit=audit,adaptation_incumbent=incumbent,
        parent_diagnostics_sha256=sha256(parent_path),checkpoint_sha256=checkpoint_hashes,read_only_dependencies_sha256=dependencies,
        device=str(device),torch_version=str(torch.__version__),numpy_version=np.__version__,
        selected_source_original_rows=ids[plan].tolist(),selected_source_percentile_pp=(100*p[plan]).tolist(),
        source_selection_receives_target_labels=False,inference_uses_original_source_latents=True,
        seeds=records,across_training_seeds=aggregate_seeds(records),paired_source_minus_control=paired,costs=costs,
        interpretation='single instance, fixed checkpoints, descriptive stratified source-validation probe; no tuning or novelty claim')
    write_json(args.output/'diagnostics.json',report);write_json(args.output/'costs.json',costs)
    (args.output/'summary.md').write_text(render_summary(report));(args.output/'group_results.csv').write_text(table_csv(report))
    section=('## TSP -> CVRP: fixed-checkpoint source-quality probe\n\nStatus: '+report['status']+
        '.\nExisting three arms x three training seeds; no retraining or checkpoint/decoder changes.\n'
        'Five train-fit quintiles; validation pool counts '+str(audit['validation_counts'])+'.\n'
        'Selected per group '+str(config['per_group'])+'; sampling rounds '+str(config['sampling_rounds'])+
        '; shared source IDs and action seeds across arms.\n'
        'Incumbent '+repr(incumbent)+'; new generation attempts '+str(costs['new_generation_objective_attempts'])+
        ', plus one shared incumbent verification.\nResults: '+str(args.output.resolve())+
        '/diagnostics.json, costs.json, summary.md and group_results.csv.\n'
        'Round std and training-seed std are separate. Spearman is descriptive, per arm/seed.\n'
        'One instance; no tuning, benchmark or novelty claim.\n')
    for seed in records:
        section+='- seed {}: Spearman source / within_rank_shuffle / rank_only = {}.\n'.format(
            seed['training_seed'],' / '.join(number(seed['arms'][a]['spearman']['rho']) for a in ARMS))
    (args.output/'project_state_update.md').write_text(section)
    if not args.smoke_test:
        update_state(args.project_state,section)
    print(report['status']);print('diagnostics:',args.output/'diagnostics.json');print('summary:',args.output/'summary.md')


if __name__=='__main__':
    main()
