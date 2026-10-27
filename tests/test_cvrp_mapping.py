import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem
from src.regression.cvrp_vae import CVRPVAE
from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.cvrp_mapping import prepare_pairs, train_mapping, evaluate_mapping
from src.experiment_cvrp_mapping import load_source, sha256


class CVRPMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        self.problem = CVRPProblem(demands=[0, 1, 2, 1, 2, 1], capacity=3, seed=10)
        raw = self.problem.sample_solutions(64, seed=11)
        self.solutions = raw + [s[::-1] for s in raw]
        self.quality = -self.problem.evaluate_batch(self.solutions)
        rng = np.random.default_rng(12)
        self.source = rng.standard_normal((64, 8)).astype(np.float32)
        self.source[:, -1] = 7.
        self.source_quality = rng.integers(0, 5, size=64).astype(float)
        self.args = (self.problem, self.source, self.source_quality, self.solutions, self.quality, 1088, 32)
        self.instance = (torch.tensor(self.problem.coordinates.copy(), dtype=torch.float32)[None],
                         torch.tensor(self.problem.demands.copy())[None], torch.tensor([3]))

    def test_split_before_rank_pairing_no_equivalent_solution_leakage(self):
        conditions, padded, sizes, train, valid, meta = prepare_pairs(*self.args)
        self.assertFalse(set(map(tuple, padded[train])) & set(map(tuple, padded[valid])))
        src, tgt = np.array(meta['source_order']), np.array(meta['target_order'])
        self.assertFalse(set(src[train]) & set(src[valid]))
        for ids in (train, valid):
            self.assertTrue(np.all(np.diff(self.source_quality[src[ids]]) <= 0))
            self.assertTrue(np.all(np.diff(self.quality[tgt[ids]]) <= 0))
        self.assertTrue(meta['source_sorted_within_each_split'])
        self.assertTrue(meta['target_sorted_within_each_split'])
        self.problem.unpack_solutions(padded, sizes)

    def test_control_banks_train_only_stats_uniform_shuffle_and_tied_rank(self):
        conditions, _, _, train, valid, meta = prepare_pairs(*self.args)
        np.testing.assert_allclose(conditions['source'][train].mean(0), 0., atol=1e-6)
        self.assertEqual(meta['active_source_dimensions'], 7)
        self.assertTrue(np.all(conditions['noise_train'][:, -1] == 0))
        permutation = np.array(meta['shuffled_order'])
        for ids in (train, valid):
            np.testing.assert_array_equal(np.sort(permutation[ids]), ids)
            self.assertEqual(sorted(map(tuple, conditions['source'][ids])),
                             sorted(map(tuple, conditions['shuffled_train'][ids])))
        self.assertGreater(len(np.unique((permutation[train]-train) % len(train))), 1)
        np.testing.assert_array_equal(conditions['rank_only'][:, 1:], 0.)
        order = meta['source_order']
        for a in range(len(order)):
            for b in range(len(order)):
                if self.source_quality[order[a]] == self.source_quality[order[b]]:
                    self.assertEqual(conditions['rank_only'][a, 0], conditions['rank_only'][b, 0])

    def test_mapping_train_updates_bridge_decoder_not_unused_encoder_head(self):
        conditions, padded, sizes, train, valid, _ = prepare_pairs(*self.args)
        model, bridge = CVRPVAE(16, 4), torch.nn.Linear(8, 4)
        old_model, old_bridge = copy.deepcopy(model.state_dict()), copy.deepcopy(bridge.state_dict())
        config = dict(epochs=2, batch_size=8, learning_rate=.01, grad_clip=1., seed=4)
        history, best = train_mapping(model, bridge, self.instance, conditions['source'], padded, sizes, train, valid, config)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(best['valid_nll']))
        self.assertTrue(any(not torch.equal(v, old_bridge[k]) for k, v in bridge.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, old_model[k]) for k, v in model.state_dict().items() if k.startswith('decoder')))
        for key, value in model.state_dict().items():
            if key.startswith(('route_encoder', 'posterior', 'score_predictor')):
                torch.testing.assert_close(value, old_model[key])

    def test_generation_feasibility_sampling_and_equal_budgets(self):
        conditions, padded, sizes, _, valid, _ = prepare_pairs(*self.args)
        model = CVRPVAE(16, 4)
        bridge = torch.nn.Linear(8, 4)
        for arm in ('source', 'noise_train', 'no_source'):
            reports = []
            for sample in (False, True):
                report, output, true_sizes, costs = evaluate_mapping(model, None if arm == 'no_source' else bridge,
                    self.instance, conditions[arm][valid], padded[valid], sizes[valid], self.problem, 2, sample)
                self.assertEqual(report['valid_rate'], 1.)
                self.assertEqual(report['objective_queries'], len(valid))
                np.testing.assert_array_equal(self.problem.evaluate_batch(self.problem.unpack_solutions(output, true_sizes)), costs)
                reports.append(report)
            if arm == 'no_source':
                self.assertEqual(reports[0]['unique_solutions'], 1)
                self.assertGreater(reports[1]['unique_solutions'], 1)

    def test_real_source_loader_deduplication_posterior_mean_and_hash_guard(self):
        p = TSPProblem(dimension=5, seed=3)
        tours = p.sample_tours(64, seed=4)
        model = TSPVAE(16, 8).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, checkpoint = root/'data.npz', root/'model.pt'
            np.savez(data, coordinates=p.coordinates, tours=tours, qualities=-p.evaluate_batch(tours))
            torch.save(dict(config=dict(hidden_dim=16, latent_dim=8, data_sha256=sha256(data)), model=model.state_dict()), checkpoint)
            before = sha256(checkpoint)
            latent, quality, original, coords, calls = load_source(checkpoint, data, torch.device('cpu'), 7)
            canonical = canonicalize_tours(tours)[original]
            with torch.no_grad():
                expected = model.encode(torch.tensor(p.coordinates.copy(), dtype=torch.float32)[None].expand(len(original), -1, -1),
                                         torch.tensor(canonical))[0]
            torch.testing.assert_close(torch.tensor(latent), expected)
            np.testing.assert_allclose(quality, -p.evaluate_batch(canonical))
            np.testing.assert_array_equal(coords, p.coordinates)
            self.assertEqual(calls, 64)
            self.assertLess(len(original), 64)
            self.assertEqual(sha256(checkpoint), before)
            with data.open('ab') as f:
                f.write(b'changed')
            with self.assertRaises(ValueError):
                load_source(checkpoint, data, torch.device('cpu'), 7)


if __name__ == '__main__':
    unittest.main()
