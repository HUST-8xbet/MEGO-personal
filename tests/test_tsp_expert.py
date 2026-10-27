import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.experiment_tsp_expert import load_data


class TSPExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.problem = TSPProblem(seed=4)
        self.tours = torch.tensor(canonicalize_tours(self.problem.sample_tours(8, seed=5)))
        self.coords = torch.tensor(self.problem.coordinates.copy(), dtype=torch.float32)[None].expand(8, -1, -1)
        self.model = TSPVAE(hidden_dim=32, latent_dim=8)
        quality = -self.problem.evaluate_batch(self.tours.numpy())
        self.scores = torch.tensor((quality - quality.mean()) / quality.std(), dtype=torch.float32)

    def test_canonical_rotation_reversal_cost(self):
        tour = self.tours[0].numpy()
        variants = np.array([np.roll(t, k) for t in (tour, tour[::-1]) for k in range(20)])
        canonical = canonicalize_tours(variants)
        np.testing.assert_array_equal(canonical, np.tile(tour, (40, 1)))
        np.testing.assert_allclose(self.problem.evaluate_batch(canonical), self.problem.evaluate_batch(variants))
        for bad in (np.zeros((1, 20), dtype=int), np.arange(20)[None].astype(float)):
            with self.assertRaises(ValueError):
                canonicalize_tours(bad)

    def test_shapes_and_finite_gradients_all_components(self):
        output = self.model(self.coords, self.tours)
        self.assertEqual(output['logits'].shape, (8, 19, 20))
        self.assertEqual(output['mu'].shape, (8, 8))
        self.assertEqual(output['score'].shape, (8,))
        loss = self.model.loss(output, self.tours, self.scores)
        self.assertTrue(all(torch.isfinite(v) for v in loss.values()))
        loss['loss'].backward()
        for name, component in (('embedding', self.model.city_embedding), ('encoder', self.model.tour_encoder),
                                ('posterior', self.model.posterior), ('decoder', self.model.decoder_cell),
                                ('pointer', self.model.pointer), ('predictor', self.model.score_predictor)):
            gradients = [p.grad for p in component.parameters()]
            with self.subTest(component=name):
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_teacher_masks_and_no_future_leakage(self):
        z = torch.zeros(8, 8)
        _, logits = self.model.decode(self.coords, z, teacher_tours=self.tours)
        for step in range(19):
            visited = self.tours[:, :step + 1]
            self.assertTrue(torch.isneginf(logits[:, step].gather(1, visited)).all())
            self.assertTrue(torch.isfinite(logits[:, step].gather(1, self.tours[:, step+1:step+2])).all())
        altered = self.tours.clone()
        altered[:, 6:] = altered[:, 6:].flip(1)
        _, second = self.model.decode(self.coords, z, teacher_tours=altered)
        torch.testing.assert_close(logits[:, :6], second[:, :6])

    def test_free_greedy_and_sampled_decoding_always_valid(self):
        for n in (4, 20):
            coordinates = torch.rand(12, n, 2)
            for sample in (False, True):
                tours, _ = self.model.decode(coordinates, torch.randn(12, 8), sample=sample)
                torch.testing.assert_close(tours.sort(dim=1).values, torch.arange(n).expand(12, n))
                self.assertTrue(torch.all(tours[:, 0] == 0))
                for coords, tour in zip(coordinates.numpy(), tours.numpy()):
                    self.assertTrue(np.isfinite(TSPProblem(coordinates=coords).evaluate(tour)))

    def test_latent_has_decoder_gradient_and_changes_distribution(self):
        z = torch.randn(8, 8, requires_grad=True)
        _, logits = self.model.decode(self.coords, z, teacher_tours=self.tours)
        loss = torch.nn.functional.cross_entropy(logits[:, 0], self.tours[:, 1])
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0.)
        _, other = self.model.decode(self.coords, torch.zeros_like(z), teacher_tours=self.tours)
        self.assertGreater((logits.softmax(-1) - other.softmax(-1)).abs().max().item(), 1e-7)

    def test_checkpoint_round_trip(self):
        self.model.eval()
        expected = self.model(self.coords, self.tours, stochastic=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expert.pt'
            torch.save(self.model.state_dict(), path)
            other = TSPVAE(32, 8).eval()
            other.load_state_dict(torch.load(path, weights_only=True))
            actual = other(self.coords, self.tours, stochastic=False)
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name])

    def test_small_batch_overfit_reduces_reconstruction_loss(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        def loss():
            return self.model.loss(self.model(self.coords, self.tours, stochastic=False), self.tours,
                                   self.scores, beta=0., score_weight=0.)['nll']
        before = loss().item()
        for _ in range(80):
            optimizer.zero_grad()
            current = loss()
            current.backward()
            optimizer.step()
        after = loss().item()
        self.assertLess(after, before * 0.8)

    def test_split_has_no_equivalent_cycle_leakage_and_train_only_scaling(self):
        base = self.tours.numpy()
        duplicates = np.concatenate([base, np.roll(base, 3, axis=1), base[:, ::-1]])
        costs = self.problem.evaluate_batch(duplicates)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.npz'
            np.savez(path, coordinates=self.problem.coordinates, tours=duplicates, lengths=costs, qualities=-costs)
            _, tours, scores, lengths, train, valid, stats = load_data(path, 17)
        self.assertEqual(len(train) + len(valid), 8)
        self.assertFalse(set(map(tuple, tours[train])) & set(map(tuple, tours[valid])))
        self.assertAlmostEqual(stats['mean'], float((-lengths[train]).mean()))
        self.assertAlmostEqual(float(scores[train].mean()), 0., places=6)

    def test_invalid_model_inputs(self):
        for tour in (self.tours.float(), self.tours[:, :-1], self.tours.roll(1, 1)):
            with self.assertRaises(ValueError):
                self.model.encode(self.coords, tour)
        with self.assertRaises(ValueError):
            self.model.decode(self.coords, torch.zeros(8, 9))
        bad_coordinates = self.coords.clone()
        bad_coordinates[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.model.encode(bad_coordinates, self.tours)


if __name__ == '__main__':
    unittest.main()
