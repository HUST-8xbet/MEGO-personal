import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.cvrp_within_rank import (within_rank_permutation, percentile_from_train,
    generation_plan, generation_conditions, summarize_generation, generate_rounds,
    state_digest, batch_order_digest, derived_seed, summarize_seeds, ARMS)
from src.cvrp_mapping import prepare_pairs
from src.experiment_cvrp_within_rank import protocol_config, update_state
from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE


class WithinRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def bank(self):
        rng = np.random.default_rng(9)
        # Non-overlapping partitions interleaved across the quality range.
        q = np.linspace(-20, -1, 1200)
        z = rng.standard_normal((1200, 8)).astype(np.float32)
        train = np.flatnonzero(np.arange(1200) % 6 != 0)
        valid = np.flatnonzero(np.arange(1200) % 6 == 0)
        return z, q, train, valid

    def test_no_cross_bin_or_partition_and_exact_bank_preservation(self):
        z,q,train,valid = self.bank()
        output,report,arrays = within_rank_permutation(z,q,train,valid,73001,1088)
        permutation,groups = arrays['permutation'],arrays['group_ids']
        for ids in (train,valid):
            np.testing.assert_array_equal(np.sort(permutation[ids]),ids)
            np.testing.assert_array_equal(groups[ids],groups[permutation[ids]])
        np.testing.assert_array_equal(output,z[permutation])
        self.assertEqual(report['group_count'],10)
        self.assertGreater(report['train']['moved_fraction'],.8)
        self.assertLessEqual(report['train']['max_abs_percentile_change_pp'],10.1)
        self.assertAlmostEqual(report['train']['mean_signed_percentile_change_pp'],0.,places=10)
        self.assertAlmostEqual(report['train']['mean_abs_percentile_change_pp'],
            np.abs(arrays['percentile_after'][train]-arrays['percentile_before'][train]).mean())

    def test_train_boundaries_and_rng_independent_of_validation_values(self):
        z,q,train,valid = self.bank()
        _,a,pa = within_rank_permutation(z,q,train,valid,73001,1088)
        changed = q.copy(); changed[valid] = np.linspace(-100,100,len(valid))
        _,b,pb = within_rank_permutation(z,changed,train,valid,73001,1088)
        self.assertEqual(a['quality_cutpoints'],b['quality_cutpoints'])
        np.testing.assert_array_equal(pa['permutation'][train],pb['permutation'][train])
        self.assertNotEqual(a['seed_sequences']['train'],a['seed_sequences']['validation'])
        # Separate validation stream: edits to train group cardinalities cannot consume val RNG.
        self.assertEqual(a['seed_sequences']['validation'],[73001,1088,1])

    def test_fixed_seed_reproducibility_and_new_shuffle_stream(self):
        z,q,train,valid=self.bank()
        a=within_rank_permutation(z,q,train,valid,73001,1088)
        b=within_rank_permutation(z,q,train,valid,73001,1088)
        c=within_rank_permutation(z,q,train,valid,73002,1088)
        np.testing.assert_array_equal(a[2]['permutation'],b[2]['permutation'])
        self.assertEqual(a[1],b[1])
        self.assertFalse(np.array_equal(a[2]['permutation'],c[2]['permutation']))

    def test_ties_empty_bins_singletons_and_actual_change_rates(self):
        q=np.zeros(20); z=np.zeros((20,4))
        _,r,a=within_rank_permutation(z,q,np.arange(16),np.arange(16,20),7,8)
        self.assertEqual(r['train']['changed_latent_fraction'],0.)
        self.assertEqual(r['train']['max_abs_percentile_change_pp'],0.)
        self.assertGreater(r['train']['moved_fraction'],0.)
        self.assertEqual(sum(row['count']==0 for row in r['train']['groups']),9)
        np.testing.assert_array_equal(a['percentile_before'],50.)
        np.testing.assert_array_equal(percentile_from_train([-1,0,1],q[:16]),[0,.5,1])

    def test_generation_plan_common_original_source_and_no_label_input(self):
        pool=np.arange(100,510)
        plan=generation_plan(pool,10,41,94001)
        self.assertEqual(plan.shape,(10,41))
        np.testing.assert_array_equal(np.sort(plan.ravel()),pool)
        np.testing.assert_array_equal(plan,generation_plan(pool,10,41,94001))
        z=np.random.default_rng(9).normal(size=(510,8));rank=np.zeros_like(z)
        x=generation_conditions(z,rank,plan)
        np.testing.assert_array_equal(x['source'],x['within_rank_shuffle'])
        np.testing.assert_array_equal(x['source'],z[plan])
        self.assertNotIn('quality',str(inspect.signature(generation_plan)))
        self.assertNotIn('target',str(inspect.signature(generate_rounds)))
        self.assertNotIn('adaptation',str(inspect.signature(generate_rounds)))
        with self.assertRaises(ValueError):generation_plan(np.arange(409),10,41,1)

    def test_generation_ids_unchanged_when_target_quality_order_changes(self):
        p=CVRPProblem(demands=[0,1,2,1,2,1],capacity=3,seed=8)
        solutions=p.sample_solutions(64,seed=9);tq=-p.evaluate_batch(solutions)
        rng=np.random.default_rng(10);z=rng.normal(size=(64,8));q=rng.normal(size=64)
        a=prepare_pairs(p,z,q,solutions,tq,1088,32)
        b=prepare_pairs(p,z,q,solutions,-tq,1088,32)
        pa=generation_plan(a[4],2,1,94001);pb=generation_plan(b[4],2,1,94001)
        np.testing.assert_array_equal(np.array(a[5]['source_order'])[pa],np.array(b[5]['source_order'])[pb])
        np.testing.assert_array_equal(a[0]['source'][pa],b[0]['source'][pb])
        self.assertFalse(np.array_equal(a[5]['target_order'],b[5]['target_order']))

    def test_best_of_k_sample_std_and_adaptation_incumbent(self):
        lengths=np.array([[9,4,7],[8,2,3]],dtype=float);counts=np.full((2,3),3)
        report=summarize_generation(lengths,counts,1.5)
        self.assertEqual(report['round_best_lengths'],[4.,2.])
        self.assertEqual(report['best_of_k_mean'],3.)
        self.assertAlmostEqual(report['best_of_k_std'],np.sqrt(2))
        self.assertEqual(report['best_of_all'],2.)
        self.assertEqual(report['best_final_including_adaptation'],1.5)
        self.assertEqual(report['objective_queries'],6)
        self.assertEqual(summarize_generation(lengths,counts,5.)['best_final_including_adaptation'],2.)

    def test_generation_budget_rng_replay_and_feasibility(self):
        torch.manual_seed(2)
        p=CVRPProblem(dimension=5,capacity=3,seed=4)
        class Counted:
            def __init__(self,problem):self.problem,self.calls=problem,0
            def unpack_solutions(self,*args):return self.problem.unpack_solutions(*args)
            def evaluate_batch(self,solutions):
                self.calls+=len(solutions)
                return self.problem.evaluate_batch(solutions)
        counted=Counted(p)
        instance=(torch.tensor(p.coordinates.copy(),dtype=torch.float32)[None],
                  torch.tensor(p.demands.copy())[None],torch.tensor([3]))
        model,bridge=CVRPVAE(16,4),torch.nn.Linear(8,4)
        conditions=np.random.default_rng(4).normal(size=(2,3,8)).astype(np.float32)
        a,arrays=generate_rounds(model,bridge,instance,conditions,counted,95001,1088,True)
        b,replayed=generate_rounds(model,bridge,instance,conditions,counted,95001,1088,True)
        self.assertEqual(counted.calls,12)
        self.assertEqual(a['objective_queries'],6)
        np.testing.assert_array_equal(arrays['padded'],replayed['padded'])
        self.assertEqual(a['action_seeds_by_round'],b['action_seeds_by_round'])
        self.assertEqual(a['valid_rate'],1.)
        self.assertNotEqual(derived_seed(95001,1088,0),derived_seed(95001,1088,1))

    def test_paired_initial_weights_batch_order_and_seed_variation(self):
        torch.manual_seed(1088);template=CVRPVAE(16,4);bridge=torch.nn.Linear(8,4)
        for _ in ARMS:
            self.assertEqual(state_digest(copy.deepcopy(template)),state_digest(template))
            self.assertEqual(state_digest(copy.deepcopy(bridge)),state_digest(bridge))
        self.assertEqual(batch_order_digest(np.arange(51),30,1088),batch_order_digest(np.arange(51),30,1088))
        self.assertNotEqual(batch_order_digest(np.arange(51),30,1088),batch_order_digest(np.arange(51),30,1089))
        torch.manual_seed(1089);self.assertNotEqual(state_digest(CVRPVAE(16,4)),state_digest(template))

    def test_config_full_protocol_vs_explicit_smoke_and_state_update(self):
        root=Path(__file__).resolve().parents[1]
        spec=json.loads((root/'configs/cvrp_within_rank.json').read_text())
        baseline=json.loads((root/'configs/cvrp_mapping_smoke.json').read_text())
        c=protocol_config(spec,baseline,False);s=protocol_config(spec,baseline,True)
        self.assertEqual((len(c['training_seeds']),c['sampling_rounds'],c['candidates_per_round'],c['epochs']),(3,10,41,30))
        self.assertEqual(len(s['training_seeds']),3)
        self.assertIn('SMOKE',s['execution_mode'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'PROJECT_STATE.md';path.write_text('# Existing decisions\n')
            update_state(path,'pending');update_state(path,'completed')
            text=path.read_text()
            self.assertIn('# Existing decisions',text);self.assertNotIn('pending',text)
            self.assertEqual(text.count('within-rank-protocol:start'),1)

    def test_overlapping_partitions_rejected(self):
        z,q,train,valid=self.bank()
        with self.assertRaises(ValueError):within_rank_permutation(z,q,train,np.r_[valid,train[0]],1,2)


if __name__=='__main__':unittest.main()
