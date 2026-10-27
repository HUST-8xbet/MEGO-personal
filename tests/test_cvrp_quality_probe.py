import copy
import inspect
import json
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch

from src.cvrp_quality_probe import (quality_plan, InsufficientGroupError, sampling_seeds,
    average_ranks, spearman, decode_probe, summarize_mode, report_probe)
from src.cvrp_within_rank import generation_conditions, state_digest
from src.experiment_cvrp_quality_probe import validate_checkpoint, aggregate_seeds, table_csv, update_state
from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE


class QualityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def bank(self):
        quality=np.linspace(-20,-1,2500)
        valid=np.arange(0,2500,5)
        train=np.setdiff1d(np.arange(2500),valid)
        return quality,train,valid

    def test_quintile_train_fit_source_validation_only_and_distinct_twenty(self):
        q,t,v=self.bank();selected,groups,p,a=quality_plan(q,t,v,20,96001)
        np.testing.assert_array_equal(a['quality_cutpoints'],np.quantile(q[t],[.2,.4,.6,.8]))
        self.assertEqual(selected.shape,(5,20));self.assertEqual(len(np.unique(selected)),100)
        self.assertTrue(np.isin(selected,v).all());self.assertFalse(np.isin(selected,t).any())
        np.testing.assert_array_equal(groups[selected],np.broadcast_to(np.arange(5)[:,None],(5,20)))
        self.assertTrue(np.all(np.diff(p[selected].mean(1))>0))
        self.assertEqual(sum(a['validation_counts']),len(v))

    def test_replay_independent_of_global_rng_and_no_target_selection_argument(self):
        q,t,v=self.bank();a=quality_plan(q,t,v,20,96001)
        np.random.seed(33);np.random.normal(size=55)
        b=quality_plan(q,t,v,20,96001)
        np.testing.assert_array_equal(a[0],b[0])
        self.assertNotIn('target',str(inspect.signature(quality_plan)))
        self.assertNotIn('incumbent',str(inspect.signature(decode_probe)))
        changed=q.copy();changed[v]+=1000
        with self.assertRaises(InsufficientGroupError) as e:quality_plan(changed,t,v,20,96001)
        self.assertEqual(e.exception.audit['quality_cutpoints'],a[3]['quality_cutpoints'])

    def test_short_group_and_ties_stop_without_replacement(self):
        q,t,v=self.bank()
        with self.assertRaises(InsufficientGroupError) as e:quality_plan(q,t,v,101,1)
        self.assertEqual(e.exception.audit['validation_counts'],[101,100,99,100,100])
        with self.assertRaises(InsufficientGroupError) as e:quality_plan(np.zeros(30),np.arange(20),np.arange(20,30),1,1)
        self.assertEqual(e.exception.audit['validation_counts'],[0,0,0,0,10])
        with self.assertRaises(ValueError):quality_plan(q,t,np.r_[v,t[0]],20,1)

    def test_original_inputs_and_scalar_rank_on_same_ids(self):
        q,t,v=self.bank();plan,_,p,_=quality_plan(q,t,v,20,96001)
        z=np.random.default_rng(1).normal(size=(len(q),8));rank=np.zeros_like(z);rank[:,0]=2*p-1
        c=generation_conditions(z,rank,plan)
        np.testing.assert_array_equal(c['source'],c['within_rank_shuffle'])
        np.testing.assert_array_equal(c['source'],z[plan])
        np.testing.assert_allclose(c['rank_only'][...,0],2*p[plan]-1)
        self.assertFalse(c['rank_only'][...,1:].any())

    def test_spearman_ties_direction_and_constant_handling(self):
        np.testing.assert_array_equal(average_ranks([1,2,2,4]),[1,2.5,2.5,4])
        self.assertAlmostEqual(spearman([1,2,2,4],[4,3,2,1])['rho'],-np.sqrt(.9))
        self.assertAlmostEqual(spearman([1,2,3],[1,2,3])['rho'],1.)
        self.assertIsNone(spearman([1,1,1],[2,3,4])['rho'])
        with self.assertRaises(ValueError):spearman([1,2],[1])

    def test_round_minima_incumbent_strict_rate_and_invalid_denominator(self):
        x=np.array([[9,10,11],[8,12,13],[7,14,15],[6,16,17],[5,18,19]],dtype=float)
        a=dict(lengths=x,route_counts=np.full_like(x,3),valid=np.ones_like(x,dtype=bool))
        r=summarize_mode(a,9.)
        self.assertEqual(r['round_best_lengths'],[9,8,7,6,5]);self.assertEqual(r['best_of_k_mean'],7.)
        self.assertAlmostEqual(r['best_of_k_std_across_sampling_rounds'],np.sqrt(2.5))
        self.assertEqual(r['better_than_adaptation_count'],4);self.assertAlmostEqual(r['better_than_adaptation_rate'],4/15)
        a['valid'][0,0]=False;a['lengths'][0,0]=np.nan
        r=summarize_mode(a,9.)
        self.assertEqual(r['round_best_lengths'][0],10.)
        self.assertAlmostEqual(r['valid_rate'],14/15);self.assertEqual(r['better_than_adaptation_count'],4)
        self.assertEqual(r['best_final_including_adaptation'],5.)

    def test_spearman_uses_mean_of_five_per_latent_and_aggregates_seed_std_separately(self):
        p=np.arange(100,dtype=float).reshape(5,20)
        x=np.stack([100-p+i for i in range(5)],axis=1)
        sampled=dict(lengths=x,route_counts=np.ones_like(x),valid=np.ones_like(x,dtype=bool))
        greedy={k:v[:,:1] for k,v in sampled.items()}
        r,mean=report_probe(greedy,sampled,p,9.)
        np.testing.assert_array_equal(mean,102-p)
        self.assertAlmostEqual(r['spearman']['rho'],-1.)
        self.assertEqual(r['spearman']['latent_count'],100)
        self.assertEqual(r['groups'][0]['sampled']['attempted_candidates'],100)
        records=[]
        for seed,offset in [(1088,0),(1089,1),(1090,2)]:
            record=copy.deepcopy(r)
            record['groups'][0]['sampled']['mean_length']+=offset
            records.append(dict(training_seed=seed,arms={arm:record for arm in ('source','within_rank_shuffle','rank_only')}))
        agg=aggregate_seeds(records)['source']['groups'][0]['sampled']['mean_length']
        self.assertAlmostEqual(agg['std_between_training_seeds'],1.)
        self.assertEqual(agg['training_seed_count'],3)

    def test_full_group_budget_feasible_rng_replay_and_frozen_state(self):
        torch.manual_seed(1)
        problem=CVRPProblem(dimension=3,capacity=5,seed=9)
        instance=(torch.tensor(problem.coordinates.copy(),dtype=torch.float32)[None],
                  torch.tensor(problem.demands.copy())[None],torch.tensor([5]))
        model,bridge=CVRPVAE(8,4),torch.nn.Linear(6,4)
        model.requires_grad_(False);bridge.requires_grad_(False)
        z=np.random.default_rng(1).normal(size=(5,20,6)).astype(np.float32)
        seeds=sampling_seeds(97001,1088)
        before=[state_digest(model),state_digest(bridge)]
        a=decode_probe(model,bridge,instance,z,seeds,problem,True)
        b=decode_probe(model,bridge,instance,z,seeds,problem,True)
        np.testing.assert_array_equal(a['padded'],b['padded'])
        self.assertEqual(a['lengths'].shape,(5,5,20));self.assertTrue(a['valid'].all())
        self.assertEqual(before,[state_digest(model),state_digest(bridge)])
        self.assertEqual(len(np.unique(seeds)),25)
        self.assertFalse(np.array_equal(seeds,sampling_seeds(97001,1089)))

    def test_invalid_decode_is_retained_and_not_repaired(self):
        problem=CVRPProblem(dimension=2,capacity=5,seed=1)
        class Broken(torch.nn.Module):
            def decode(self,coords,demands,cap,z,sample=False):
                return torch.full((len(z),5),-1,dtype=torch.long),torch.full((len(z),),4),None
        instance=(torch.tensor(problem.coordinates.copy(),dtype=torch.float32)[None],
                  torch.tensor(problem.demands.copy())[None],torch.tensor([5]))
        arrays=decode_probe(Broken(),torch.nn.Linear(2,2),instance,np.zeros((5,20,2)),sampling_seeds(1,2),problem,True)
        self.assertEqual(arrays['valid'].size,500);self.assertFalse(arrays['valid'].any())
        r,_=report_probe({k:v[:,:1] for k,v in arrays.items()},arrays,np.arange(100).reshape(5,20),9.)
        self.assertIsNone(r['spearman']['rho']);self.assertEqual(r['sampled_all_groups']['valid_rate'],0.)
        self.assertEqual(r['best_final_greedy_and_sampled_union'],9.)
        json.dumps(r,allow_nan=False)

    def test_checkpoint_rejects_wrong_arm_seed_epoch_or_nll(self):
        parent=dict(config=dict(seed=1088,latent_dim=4),input_sha256={'source':'abc'},code_sha256={'code':'hash'},
            seeds=[dict(training_seed=1088,initial_state_sha256={'bridge':'same'},batch_order_sha256='same',
                arms={'source':dict(best_epoch=30,best_validation_nll=1.2)})])
        payload=dict(arm='source',config=parent['config'],source_dim=8,target_dim=4,input_sha256=parent['input_sha256'],
            code_sha256=parent['code_sha256'],initial_state_sha256={'bridge':'same'},batch_order_sha256='same',epoch=30,validation_nll=1.2)
        validate_checkpoint(payload,parent,1088,'source',8)
        for key,value in [('arm','rank_only'),('epoch',29),('validation_nll',1.4),('config',dict(seed=1089,latent_dim=4))]:
            changed=dict(payload);changed[key]=value
            with self.assertRaises(ValueError):validate_checkpoint(changed,parent,1088,'source',8)

    def test_state_preserves_prior_protocol_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'PROJECT_STATE.md'
            path.write_text('# Prior completed within-rank result\nkeep this\n')
            update_state(path,'pending probe');update_state(path,'completed probe')
            result=path.read_text()
            self.assertIn('keep this',result);self.assertNotIn('pending probe',result)
            self.assertEqual(result.count('<!-- source-quality-probe:start -->'),1)


if __name__=='__main__':
    unittest.main()
