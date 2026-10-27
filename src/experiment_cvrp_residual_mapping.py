"""Pilot H_residual: explicit rank plus train-only residual source latent."""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.cvrp_mapping import prepare_pairs
from src.cvrp_residual_mapping import (ARMS, ResidualBridge, array_sha256, evaluate_mapping,
    generation_rounds, residual_preprocess, shuffle_residual_within_bins, train_mapping)
from src.cvrp_within_rank import generation_plan
from src.experiment_cvrp_mapping import load_source, sha256
from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE


def write_json(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def state_digest(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]; p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint',type=Path,default=root/'logs_tsp/tsp20_expert_smoke_01/best_model.pt'); p.add_argument('--source-data',type=Path,default=root/'data/tsp_milestone_a/tsp20_seed1088.npz'); p.add_argument('--target-data',type=Path,default=root/'data/cvrp_milestone_a/cvrp20_seed2088.npz'); p.add_argument('--config',type=Path,default=root/'configs/cvrp_residual_mapping.json'); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cpu'); p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args();
    if args.output.exists(): raise FileExistsError('choose a new output directory; no overwrite/resume')
    spec=json.loads(args.config.read_text()); config=dict(spec)
    if args.smoke_test: config.update(epochs=1,max_pairs=64,sampling_rounds=2,candidates_per_round=3,execution_mode='SMOKE_TEST_ONLY')
    else: config['execution_mode']='requested_pilot'
    for key in ('epochs','batch_size','max_pairs','sampling_rounds','candidates_per_round','hidden_dim','latent_dim','cpu_threads'):
        if type(config[key]) is not int or config[key] < 1: raise ValueError(key+' must be positive integer')
    torch.set_num_threads(config['cpu_threads']); device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    paths={'source_checkpoint':args.source_checkpoint,'source_data':args.source_data,'target_data':args.target_data}; input_hashes={k:sha256(v) for k,v in paths.items()}
    if not args.smoke_test:
        for key, expected in spec['expected_inputs'].items():
            if input_hashes[key] != expected: raise ValueError(key+' differs from locked within-rank input')
    start=time.perf_counter(); source,source_quality,source_rows,source_coords,source_calls=load_source(args.source_checkpoint,args.source_data,device,config['batch_size']); source_load_seconds=time.perf_counter()-start
    with np.load(args.target_data,allow_pickle=False) as data:
        problem=CVRPProblem(coordinates=data['coordinates'],demands=data['demands'],capacity=int(data['capacity'].item())); solutions=problem.unpack_solutions(data['routes_padded'],data['sequence_lengths']); target_quality=data['qualities'].copy(); provenance=json.loads(data['metadata'].item())
    checked_lengths=problem.evaluate_batch(solutions); np.testing.assert_allclose(target_quality,-checked_lengths,rtol=1e-7,atol=1e-9)
    base,padded,sizes,train,valid,pairing=prepare_pairs(problem,source,source_quality,solutions,target_quality,config['split_seed'],config['max_pairs'])
    if not args.smoke_test and (len(padded),len(valid)) != (2048,410): raise ValueError('locked pilot expects 2048/410 pairs')
    target_global_ids=np.asarray(pairing['target_order'],dtype=np.int64); source_global_ids=np.asarray(pairing['source_order'],dtype=np.int64)
    if set(target_global_ids[train]) & set(target_global_ids[valid]): raise AssertionError('target ID leakage')
    pair_quality=np.asarray(source_quality)[source_global_ids]; pair_latent=np.asarray(source)[source_global_ids]
    rank,residual,bins,percentile,pre_audit=residual_preprocess(pair_latent,pair_quality,train,valid,config['quality_bins'],config['residual_std_floor'])
    plan=generation_plan(valid,config['sampling_rounds'],config['candidates_per_round'],config['source_selection_seed'])
    support_lengths=checked_lengths[target_global_ids]; best_pos=int(support_lengths.argmin()); adaptation=dict(label_count_shared=len(padded),training_label_count=len(train),validation_label_count=len(valid),best_adaptation_length=float(support_lengths.min()),best_original_target_row=int(target_global_ids[best_pos]),definition='same support incumbent as within-rank protocol; no historical probe outputs')
    instance=(torch.tensor(problem.coordinates.copy(),dtype=torch.float32,device=device)[None],torch.tensor(problem.demands.copy(),dtype=torch.long,device=device)[None],torch.tensor([problem.capacity],dtype=torch.long,device=device))
    args.output.mkdir(parents=True); write_json(args.output/'config.json',config); write_json(args.output/'pairing.json',dict(pairing,source_global_ids=source_global_ids.tolist(),target_global_ids=target_global_ids.tolist(),train_target_global_ids=target_global_ids[train].tolist(),validation_target_global_ids=target_global_ids[valid].tolist(),target_partitions_disjoint=True))
    write_json(args.output/'preprocessing.json',pre_audit); np.savez_compressed(args.output/'preprocessing.npz',rank=rank,residual=residual,rank_bins=bins,percentile=percentile,source_global_ids=source_global_ids,target_global_ids=target_global_ids)
    np.savez_compressed(args.output/'shared_plan.npz',train_pair_positions=train,validation_pair_positions=valid,source_global_ids=source_global_ids,target_global_ids=target_global_ids,generation_pair_positions=plan,generation_source_global_ids=source_global_ids[plan],rank=rank,residual=residual)
    np.savez_compressed(args.output/'adaptation_incumbent.npz',coordinates=problem.coordinates,demands=problem.demands,capacity=problem.capacity,padded=padded[best_pos],sequence_length=sizes[best_pos],length=support_lengths[best_pos],target_global_id=target_global_ids[best_pos])
    seed_reports=[]; costs=[]
    for training_seed in config['training_seeds']:
        torch.manual_seed(training_seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(training_seed)
        template=CVRPVAE(config['hidden_dim'],config['latent_dim']).to(device); bridge_template=ResidualBridge(source.shape[1],template.latent_dim).to(device); initial=dict(decoder=state_digest(template),bridge=state_digest(bridge_template))
        shuffled,perm,shuffle_audit=shuffle_residual_within_bins(residual,bins,train,valid,config['shuffle_seed']+training_seed)
        seed_dir=args.output/f'seed_{training_seed}'; seed_dir.mkdir(); write_json(seed_dir/'residual_shuffle_audit.json',shuffle_audit); np.savez_compressed(seed_dir/'residual_shuffle_audit.npz',permutation=perm,bins=bins,source_global_ids=source_global_ids,target_global_ids=target_global_ids)
        reports={}; seed_report=dict(training_seed=training_seed,initial_state_sha256=initial,arms={}); train_config=dict(config, seed=training_seed)
        conds={'rank_residual':(residual,residual),'rank_residual_shuffle':(shuffled,shuffled),'rank_residual_zero':(np.zeros_like(residual),np.zeros_like(residual))}
        for arm in ARMS:
            model=copy.deepcopy(template); bridge=copy.deepcopy(bridge_template); arm_dir=seed_dir/arm; arm_dir.mkdir()
            history,best,loss_audit=train_mapping(model,bridge,instance,rank,conds[arm][0],padded,sizes,target_global_ids,train,valid,train_config)
            rep={'best_epoch':best['epoch'],'best_validation_nll':best['valid_nll'],'initial_state_sha256':initial,'loss_target_id_audit':{'train_expected_ids':loss_audit['train_global_ids'],'validation_expected_ids':loss_audit['validation_global_ids'],'train_epoch_target_global_ids':loss_audit['train_epoch_target_global_ids'],'valid_epoch_target_global_ids':loss_audit['valid_epoch_target_global_ids'],'train_epoch_target_id_sha256':loss_audit['train_epoch_target_id_sha256'],'valid_epoch_target_id_sha256':loss_audit['valid_epoch_target_id_sha256'],'disjoint':True}}
            for mode,sample in (('greedy',False),('sampled',True)):
                result,_,_,_=evaluate_mapping(model,bridge,instance,rank[valid],conds[arm][0][valid],padded[valid],sizes[valid],problem,training_seed+1,sample); rep[mode]=result
            for mode,sample in (('greedy',False),('sampled',True)):
                gen,arrays=generation_rounds(model,bridge,instance,rank,conds[arm][1],plan,problem,config['sampling_seed'],training_seed,sample); gen['best_final_including_adaptation']=min(adaptation['best_adaptation_length'],gen['best_of_all']); rep[mode]=dict(rep[mode],**gen); rep[mode+'_arrays_meta']={'source_global_ids':source_global_ids[arrays['source_pair_positions']].tolist(),'action_seeds':gen['action_seeds_by_round']}
                if mode=='greedy': generated={'greedy_'+k:v for k,v in arrays.items()}
                else: generated.update({'sampled_'+k:v for k,v in arrays.items()})
            rep['greedy_plus_sampled_union']={'objective_queries':rep['greedy']['objective_queries']+rep['sampled']['objective_queries'],'best_generated_length':min(rep['greedy']['best_of_all'],rep['sampled']['best_of_all']),'best_final_including_adaptation':min(adaptation['best_adaptation_length'],rep['greedy']['best_of_all'],rep['sampled']['best_of_all'])}
            torch.save(dict(model=best['model'],bridge=best['bridge'],epoch=best['epoch'],validation_nll=best['valid_nll'],config=train_config,arm=arm,training_seed=training_seed,source_dim=source.shape[1],target_dim=template.latent_dim,source_global_ids=source_global_ids.tolist(),target_global_ids=target_global_ids.tolist()),arm_dir/'mapping_checkpoint.pt')
            write_json(arm_dir/'history.json',history); write_json(arm_dir/'diagnostics.json',rep); np.savez_compressed(arm_dir/'generated.npz',coordinates=problem.coordinates,demands=problem.demands,capacity=problem.capacity,**generated); write_json(arm_dir/'costs.json',dict(training_seed=training_seed,arm=arm,greedy_generation_queries=rep['greedy']['objective_queries'],sampled_generation_queries=rep['sampled']['objective_queries'],generation_queries_total=rep['greedy']['objective_queries']+rep['sampled']['objective_queries'],new_adaptation_label_queries=0))
            reports[arm]=rep
        seed_report['arms']=reports; seed_reports.append(seed_report)
    across_training_seeds={}
    for arm in ARMS:
        across_training_seeds[arm]={}
        for metric in ('mean_length','best_of_k_mean','best_of_all','best_final_including_adaptation'):
            values=np.asarray([s['arms'][arm]['sampled'][metric] for s in seed_reports],dtype=np.float64)
            across_training_seeds[arm][metric]=dict(mean=float(values.mean()),std_between_training_seeds=float(values.std(ddof=1)),std_ddof=1,training_seed_count=len(values))
        values=np.asarray([s['arms'][arm]['best_validation_nll'] for s in seed_reports],dtype=np.float64)
        across_training_seeds[arm]['best_validation_nll']=dict(mean=float(values.mean()),std_between_training_seeds=float(values.std(ddof=1)),std_ddof=1,training_seed_count=len(values))
    gen_calls=sum(r['arms'][a]['greedy']['objective_queries']+r['arms'][a]['sampled']['objective_queries'] for r in seed_reports for a in ARMS)
    costs_root=dict(shared=dict(existing_adaptation_labels=len(padded),source_verification_queries=source_calls,target_verification_queries=len(solutions),new_target_label_generation_queries=0,source_loading_seconds=source_load_seconds),generated_target_queries_all_seeds=gen_calls,target_objective_queries_current_run=len(solutions)+gen_calls,source_and_target_objective_queries_current_run=source_calls+len(solutions)+gen_calls,maximum_generation_evaluations=2*len(config['training_seeds'])*len(ARMS)*config['sampling_rounds']*config['candidates_per_round'],note='target IDs and support are reused; no behavioral probe/incumbent is imported')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(); code_files=['src/cvrp_residual_mapping.py','src/experiment_cvrp_residual_mapping.py','src/cvrp_mapping.py','src/regression/cvrp_vae.py','src/regression/tsp_vae.py']
    report=dict(status='SMOKE_TEST_ONLY_COMPLETED' if args.smoke_test else 'H_RESIDUAL_PILOT_COMPLETED',protocol='H_residual_v1',base_commit=commit,working_tree_code_sha256={n:sha256(root/n) for n in code_files},input_sha256=input_hashes,config=config,pair_count=len(padded),train_count=len(train),validation_count=len(valid),target_train_validation_disjoint=True,preprocessing=pre_audit,source_global_ids=source_global_ids.tolist(),target_global_ids=target_global_ids.tolist(),generation_source_global_ids=source_global_ids[plan].tolist(),adaptation_incumbent=adaptation,seeds=seed_reports,across_training_seeds=across_training_seeds,costs=costs_root,interpretation='pilot only; residual is not called independent of rank; no novelty/transfer claim')
    write_json(args.output/'diagnostics.json',report); write_json(args.output/'costs.json',costs_root)
    lines=['# H_residual pilot','',report['status'],'','Three branches: rank_residual, rank_residual_shuffle, rank_residual_zero. Same target support, decoder/bridge template, 30 epochs and 10x41 generation budget.','Residual is train-bin-detrended latent; it is not called independent of rank.','', '| Seed | Arm | Val NLL | Mean best-of-K | Best-of-all | Final with incumbent |','|---:|---|---:|---:|---:|---:|']
    for s in seed_reports:
        for a in ARMS:
            r=s['arms'][a]; lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(s['training_seed'],a,r['best_validation_nll'],r['sampled']['best_of_k_mean'],r['sampled']['best_of_all'],r['sampled']['best_final_including_adaptation']))
    lines += ['', '## Across training seeds', '', '| Arm | Val NLL mean ± std | Mean best-of-K mean ± std | Best-of-410 mean ± std | Final with incumbent mean ± std |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        a=across_training_seeds[arm]; lines.append('| {} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} | {:.6f} ± {:.6f} |'.format(arm,a['best_validation_nll']['mean'],a['best_validation_nll']['std_between_training_seeds'],a['best_of_k_mean']['mean'],a['best_of_k_mean']['std_between_training_seeds'],a['best_of_all']['mean'],a['best_of_all']['std_between_training_seeds'],a['best_final_including_adaptation']['mean'],a['best_final_including_adaptation']['std_between_training_seeds']))
    lines += ['', 'Interpretation is pre-registered: true must beat both shuffle and zero consistently to justify a new target instance; NLL-only improvement is insufficient; no clear advantage stops this candidate.', '', 'Objective accounting is in `costs.json`; actual target global IDs used at loss time are in per-arm diagnostics, with per-epoch hashes for audit.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n'); print('H_RESIDUAL PILOT COMPLETED' if not args.smoke_test else 'H_RESIDUAL SMOKE TEST ONLY COMPLETED'); print('diagnostics:',args.output/'diagnostics.json')


if __name__=='__main__': main()
