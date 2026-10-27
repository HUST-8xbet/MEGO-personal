import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions
from src.experiment_cvrp_expert import load_data


class CVRPExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.problem = CVRPProblem(demands=[0, 1, 2, 1, 2, 1], capacity=3, seed=4)
        solutions = canonicalize_solutions(self.problem, self.problem.sample_solutions(8, seed=5))
        padded, sizes = self.problem.pack_solutions(solutions)
        self.sequences, self.sizes = torch.tensor(padded), torch.tensor(sizes)
        self.coords = torch.tensor(self.problem.coordinates.copy(), dtype=torch.float32)[None].expand(8, -1, -1)
        self.demands = torch.tensor(self.problem.demands.copy())[None].expand(8, -1)
        self.capacity = torch.full((8,), self.problem.capacity, dtype=torch.long)
        self.model = CVRPVAE(32, 8)
        quality = -self.problem.evaluate_batch(solutions)
        self.quality = torch.tensor((quality - quality.mean()) / quality.std(), dtype=torch.float32)

    def test_canonicalization_vehicle_order_reversal_and_cost(self):
        p = CVRPProblem(coordinates=[[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]],
                        demands=[0, 2, 2, 2, 2], capacity=4)
        variants = [[0, 1, 2, 0, 3, 4, 0], [0, 4, 3, 0, 2, 1, 0], [0, 3, 4, 0, 1, 2, 0]]
        normalized = canonicalize_solutions(p, variants)
        for sequence in normalized:
            np.testing.assert_array_equal(sequence, variants[0])
        np.testing.assert_array_equal(p.evaluate_batch(normalized), p.evaluate_batch(variants))

    def test_shapes_finite_losses_and_gradients(self):
        output = self.model(self.coords, self.demands, self.capacity, self.sequences, self.sizes)
        self.assertEqual(output['logits'].shape, (8, 10, 6))
        self.assertEqual(output['mu'].shape, (8, 8))
        losses = self.model.loss(output, self.sequences, self.quality)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses['loss'].backward()
        for name in ('city_embedding', 'route_encoder', 'posterior', 'decoder_init',
                     'decoder_cell', 'query', 'pointer', 'score_predictor'):
            with self.subTest(module=name):
                gradients = [p.grad for p in getattr(self.model, name).parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_prefix_masks_visited_capacity_depot_and_completion(self):
        z = torch.zeros(8, 8)
        decoded, sizes, logits = self.model.decode(self.coords, self.demands, self.capacity, z,
                                                  self.sequences, self.sizes)
        torch.testing.assert_close(decoded, self.sequences)
        torch.testing.assert_close(sizes, self.sizes)
        for row, sequence in enumerate(self.sequences.numpy()):
            visited, remaining, current = set(), self.problem.capacity, 0
            for step in range(1, len(sequence)):
                if step >= self.sizes[row].item():
                    self.assertTrue(torch.isfinite(logits[row, step-1, 0]))
                    self.assertTrue(torch.isneginf(logits[row, step-1, 1:]).all())
                    continue
                for customer in range(1, 6):
                    forbidden = customer in visited or self.problem.demands[customer] > remaining
                    self.assertEqual(bool(torch.isneginf(logits[row, step-1, customer])), forbidden)
                self.assertEqual(bool(torch.isneginf(logits[row, step-1, 0])), current == 0)
                chosen = int(sequence[step])
                self.assertTrue(torch.isfinite(logits[row, step-1, chosen]))
                if chosen:
                    visited.add(chosen)
                    remaining -= int(self.problem.demands[chosen])
                else:
                    remaining = self.problem.capacity
                current = chosen

    def test_padding_loss_is_ignored_and_has_zero_gradient(self):
        output = self.model(self.coords, self.demands, self.capacity, self.sequences, self.sizes, stochastic=False)
        logits = output['logits'].detach().clone().requires_grad_(True)
        loss = self.model.reconstruction_nll(logits, self.sequences)
        padding = self.sequences[:, 1:] == -1
        self.assertTrue(padding.any())
        altered = logits.detach().clone()
        altered[padding] = torch.randn_like(altered[padding]) * 100
        torch.testing.assert_close(loss, self.model.reconstruction_nll(altered, self.sequences))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[padding] == 0))

    def test_no_future_teacher_leakage_with_fixed_latent(self):
        p = CVRPProblem(coordinates=[[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]],
                        demands=[0, 2, 2, 2, 2], capacity=4)
        sequences, sizes = p.pack_solutions([[0, 1, 2, 0, 3, 4, 0], [0, 1, 2, 0, 4, 3, 0]])
        coordinates = torch.tensor(p.coordinates.copy(), dtype=torch.float32)[None].expand(2, -1, -1)
        demands = torch.tensor(p.demands.copy())[None].expand(2, -1)
        _, _, logits = self.model.decode(coordinates, demands, torch.tensor([4, 4]), torch.zeros(2, 8),
                                         torch.tensor(sequences), torch.tensor(sizes))
        torch.testing.assert_close(logits[0, :4], logits[1, :4])

    def test_free_greedy_and_sampled_solutions_valid_across_instances(self):
        for n, capacity in ((1, 3), (5, 1), (20, 30)):
            problems = [CVRPProblem(dimension=n, capacity=capacity, seed=100+i) for i in range(12)]
            coords = torch.tensor(np.stack([p.coordinates for p in problems]), dtype=torch.float32)
            demands = torch.tensor(np.stack([p.demands for p in problems]))
            cap = torch.full((12,), capacity, dtype=torch.long)
            for sample in (False, True):
                output, sizes, _ = self.model.decode(coords, demands, cap, torch.randn(12, 8), sample=sample)
                for p, row, size in zip(problems, output.numpy(), sizes.numpy()):
                    solution = p.unpack_solutions(row[None], np.array([size]))[0]
                    self.assertTrue(np.isfinite(p.evaluate(solution)))
                    self.assertEqual(int(solution[-1]), 0)
                    self.assertTrue(np.all(p.route_loads(solution) <= capacity))
                    if capacity == 1:
                        self.assertEqual(int(size), 2*n+1)

    def test_latent_changes_probabilities_and_has_gradient(self):
        z = torch.randn(8, 8, requires_grad=True)
        logits = self.model.decode(self.coords, self.demands, self.capacity, z, self.sequences, self.sizes)[2]
        loss = self.model.reconstruction_nll(logits, self.sequences)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0.)
        zero = self.model.decode(self.coords, self.demands, self.capacity, torch.zeros_like(z),
                                 self.sequences, self.sizes)[2]
        self.assertGreater((logits.softmax(-1) - zero.softmax(-1)).abs().max().item(), 1e-7)

    def test_checkpoint_round_trip(self):
        self.model.eval()
        expected = self.model(self.coords, self.demands, self.capacity, self.sequences, self.sizes, stochastic=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'expert.pt'
            torch.save(self.model.state_dict(), path)
            other = CVRPVAE(32, 8).eval()
            other.load_state_dict(torch.load(path, weights_only=True))
            actual = other(self.coords, self.demands, self.capacity, self.sequences, self.sizes, stochastic=False)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])

    def test_small_batch_reconstruction_learnability(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        def loss():
            output = self.model(self.coords, self.demands, self.capacity, self.sequences, self.sizes, stochastic=False)
            return self.model.loss(output, self.sequences, self.quality, beta=0., score_weight=0.)['nll']
        before = loss().item()
        for _ in range(80):
            optimizer.zero_grad()
            value = loss()
            value.backward()
            optimizer.step()
        self.assertLess(loss().item(), before * .85)

    def test_split_deduplication_and_training_only_scaling(self):
        raw = self.problem.sample_solutions(64, seed=13)
        variants = raw + [row[::-1].copy() for row in raw]
        padded, sizes = self.problem.pack_solutions(variants)
        costs = self.problem.evaluate_batch(variants)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.npz'
            np.savez(path, coordinates=self.problem.coordinates, demands=self.problem.demands,
                     capacity=self.problem.capacity, routes_padded=padded, sequence_lengths=sizes,
                     lengths=costs, qualities=-costs)
            _, canonical, _, scores, _, train, valid, stats = load_data(path, 17)
        self.assertFalse(set(map(tuple, canonical[train])) & set(map(tuple, canonical[valid])))
        self.assertAlmostEqual(stats['mean'], float((-costs[train]).mean()))
        self.assertAlmostEqual(float(scores[train].mean()), 0., places=6)
        self.assertLess(len(train) + len(valid), len(variants))

    def test_invalid_teacher_and_instance_rejected(self):
        bad = self.sequences.clone()
        bad[0, 1] = bad[0, 2]
        with self.assertRaises(ValueError):
            self.model(self.coords, self.demands, self.capacity, bad, self.sizes)
        with self.assertRaises(ValueError):
            self.model.decode(self.coords, self.demands.float(), self.capacity, torch.zeros(8, 8))
        with self.assertRaises(ValueError):
            self.model.decode(self.coords, self.demands, self.capacity, torch.zeros(8, 9))
        p = CVRPProblem(coordinates=[[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]],
                        demands=[0, 2, 2, 2, 2], capacity=4)
        # Structural permutation is valid, but third consecutive customer overloads capacity.
        padded = torch.tensor([[0, 1, 2, 3, 4, 0, -1, -1, -1]])
        with self.assertRaises(ValueError):
            self.model.decode(torch.tensor(p.coordinates.copy(), dtype=torch.float32)[None],
                              torch.tensor(p.demands.copy())[None], torch.tensor([4]), torch.zeros(1, 8),
                              padded, torch.tensor([6]))


if __name__ == '__main__':
    unittest.main()
