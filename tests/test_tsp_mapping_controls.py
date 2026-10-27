import copy
import unittest

import numpy as np
import torch

from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE
from src.tsp_mapping_controls import prepare_controls, train_control, evaluate_control


class MappingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9)
        self.problem = TSPProblem(dimension=6, seed=10)
        tours = self.problem.sample_tours(64, seed=11)
        self.tours = np.concatenate([tours, np.roll(tours, 2, 1), tours[:, ::-1]])
        self.quality = -self.problem.evaluate_batch(self.tours)
        rng = np.random.default_rng(12)
        self.source = rng.normal(size=(64, 8)).astype(np.float32)
        self.source[:, -1] = 7.
        self.source_quality = rng.integers(0, 5, size=64).astype(float)
        self.args = (self.source, self.source_quality, self.tours, self.quality, 1088, 32)

    def test_split_before_rank_matching_without_cycle_leakage(self):
        conditions, tours, train, valid, meta = prepare_controls(*self.args)
        self.assertFalse(set(map(tuple, tours[train])) & set(map(tuple, tours[valid])))
        self.assertFalse(set(np.array(meta['source_order'])[train]) & set(np.array(meta['source_order'])[valid]))
        for ids in (train, valid):
            sq = self.source_quality[np.array(meta['source_order'])[ids]]
            tq = self.quality[np.array(meta['target_order'])[ids]]
            self.assertTrue(np.all(np.diff(sq) <= 0))
            self.assertTrue(np.all(np.diff(tq) <= 0))
        again = prepare_controls(*self.args)
        np.testing.assert_array_equal(tours, again[1])
        for name in conditions:
            np.testing.assert_array_equal(conditions[name], again[0][name])

    def test_conditioning_stats_noise_and_shuffle(self):
        conditions, _, train, valid, meta = prepare_controls(*self.args)
        np.testing.assert_allclose(conditions['source'][train].mean(0), 0., atol=1e-6)
        self.assertEqual(meta['active_source_dimensions'], 7)
        self.assertTrue(np.all(conditions['noise_train'][:, -1] == 0))
        self.assertFalse(np.array_equal(conditions['source'], conditions['noise_train']))
        for ids in (train, valid):
            self.assertEqual(sorted(map(tuple, conditions['source'][ids])),
                             sorted(map(tuple, conditions['shuffled_train'][ids])))
        np.testing.assert_array_equal(conditions['rank_only'][:, 1:], 0.)
        source_ids = meta['source_order']
        for a in range(len(source_ids)):
            for b in range(len(source_ids)):
                if self.source_quality[source_ids[a]] == self.source_quality[source_ids[b]]:
                    self.assertEqual(conditions['rank_only'][a, 0], conditions['rank_only'][b, 0])

    def test_training_updates_decoder_and_bridge_not_gru_or_score(self):
        conditions, tours, train, valid, _ = prepare_controls(*self.args)
        model, bridge = TSPVAE(16, 4), torch.nn.Linear(8, 4)
        old, old_bridge = copy.deepcopy(model.state_dict()), copy.deepcopy(bridge.state_dict())
        coords = torch.tensor(self.problem.coordinates.copy(), dtype=torch.float32)[None]
        config = dict(epochs=2, batch_size=8, seed=4, learning_rate=.01, grad_clip=1.)
        history, best = train_control(model, bridge, coords, conditions['source'], tours, train, valid, config)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(best['valid_nll']))
        self.assertTrue(any(not torch.equal(value, old_bridge[key]) for key, value in bridge.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, old[key]) for key, value in model.state_dict().items()
                            if key.startswith('decoder')))
        for key, value in model.state_dict().items():
            if key.startswith(('tour_encoder', 'posterior', 'score_predictor')):
                torch.testing.assert_close(value, old[key])

    def test_no_source_sampling_has_diversity_and_equal_query_count(self):
        conditions, tours, _, valid, _ = prepare_controls(*self.args)
        model = TSPVAE(16, 4)
        coords = torch.tensor(self.problem.coordinates.copy(), dtype=torch.float32)[None]
        greedy, _ = evaluate_control(model, None, coords, conditions['no_source'][valid],
                                     tours[valid], self.problem, 2, False)
        sampled, output = evaluate_control(model, None, coords, conditions['no_source'][valid],
                                           tours[valid], self.problem, 2, True)
        self.assertEqual(greedy['unique_cycles'], 1)
        self.assertGreater(sampled['unique_cycles'], 1)
        self.assertEqual(sampled['objective_queries'], greedy['objective_queries'])
        self.assertEqual(sampled['valid_rate'], 1.)
        self.problem.evaluate_batch(output)

    def test_invalid_bank_rejected(self):
        bad = self.source.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_controls(bad, *self.args[1:])
        with self.assertRaises(ValueError):
            prepare_controls(*self.args[:-1], 2)

    def test_shuffled_assignment_is_not_only_a_cyclic_rank_shift(self):
        _, _, train, valid, meta = prepare_controls(*self.args)
        permutation = np.array(meta['shuffled_order'])
        for ids in (train, valid):
            np.testing.assert_array_equal(np.sort(permutation[ids]), ids)
        offsets = (permutation[train] - train) % len(train)
        self.assertGreater(len(np.unique(offsets)), 1)
        self.assertIn('uniform_random_permutation', meta['shuffled'])


if __name__ == '__main__':
    unittest.main()
