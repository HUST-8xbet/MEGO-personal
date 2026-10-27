import unittest

import numpy as np
import torch

from src.problem.cvrp_problem import CVRPProblem
from src.tsp_cvrp_coupling import (
    average_total_variation,
    cvrp_relation_matrix,
    prepare_coupling,
    relation_cost,
    sample_target_indices,
    solve_coupling,
    tsp_relation_matrix,
)
from src.regression.cvrp_vae import CVRPVAE
from src.tsp_cvrp_coupling import train_coupled


class CouplingTests(unittest.TestCase):
    def setUp(self):
        self.problem = CVRPProblem(
            coordinates=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]]),
            demands=np.array([0, 1, 1, 1, 1]), capacity=4)
        self.tours = np.array([[0, 1, 2, 3], [0, 3, 2, 1], [0, 1, 3, 2]], dtype=np.int64)
        self.solutions = [
            np.array([0, 1, 2, 0, 3, 4, 0]),
            np.array([0, 2, 3, 0, 1, 4, 0]),
            np.array([0, 1, 3, 0, 2, 4, 0]),
        ]

    def test_tsp_closure_and_representation_invariance(self):
        reversed_tours = self.tours[:, ::-1]
        np.testing.assert_allclose(tsp_relation_matrix(self.tours), tsp_relation_matrix(reversed_tours))
        self.assertEqual(tsp_relation_matrix(np.array([[0, 1, 2]])).shape, (1, 1))

    def test_cvrp_route_edges_ignore_padding_and_reject_no_padding_effect(self):
        padded, sizes = self.problem.pack_solutions(self.solutions)
        relation_a = cvrp_relation_matrix(self.solutions)
        relation_b = cvrp_relation_matrix(padded)
        np.testing.assert_allclose(relation_a, relation_b)
        self.assertTrue(np.allclose(np.diag(relation_a), 0.))

    def test_sinkhorn_marginals_and_relation_coupling(self):
        sq = np.array([-.1, -.2, -.3, -.4])
        tq = np.array([-.11, -.21, -.31, -.41])
        sr = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
        tr = sr[::-1, ::-1].copy()
        quality, audit_q = solve_coupling(sq, tq, epsilon=.05)
        relation, audit_r = solve_coupling(sq, tq, sr, tr, alpha=.5, epsilon=.05, max_iter=5)
        np.testing.assert_allclose(quality.sum(1), .25, atol=1e-6)
        np.testing.assert_allclose(quality.sum(0), .25, atol=1e-6)
        np.testing.assert_allclose(relation.sum(1), .25, atol=1e-6)
        np.testing.assert_allclose(relation.sum(0), .25, atol=1e-6)
        self.assertTrue(audit_q['finite'] and audit_r['finite'])
        self.assertGreaterEqual(audit_r['relation_cost'], 0.)

    def test_matched_coefficients_and_solver_traces_are_explicit(self):
        sq = np.array([-.1, -.2, -.3, -.4])
        tq = np.array([-.11, -.21, -.31, -.41])
        sr = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
        tr = sr[::-1, ::-1].copy()
        quality, quality_audit = solve_coupling(
            sq, tq, epsilon=.05, max_iter=5, sinkhorn_max_iter=5,
            quality_weight=.5, relation_weight=0.)
        relation, relation_audit = solve_coupling(
            sq, tq, sr, tr, alpha=.5, epsilon=.05, max_iter=5, sinkhorn_max_iter=5,
            quality_weight=.5, relation_weight=.5)
        self.assertEqual(quality_audit['quality_weight'], .5)
        self.assertEqual(relation_audit['quality_weight'], .5)
        self.assertEqual(relation_audit['relation_weight'], .5)
        self.assertEqual(quality_audit['solver_dtype'], 'float64')
        self.assertGreaterEqual(len(quality_audit['objective_trace']), 1)
        self.assertGreaterEqual(len(relation_audit['objective_trace']), 1)
        self.assertEqual(len(relation_audit['coupling_change_trace']), relation_audit['relation_iterations'])
        self.assertIn('relative_marginal_residual', relation_audit)
        np.testing.assert_allclose(quality.sum(1), .25, atol=1e-4)
        np.testing.assert_allclose(relation.sum(0), .25, atol=1e-4)

    def test_quality_weight_and_entropy_rescaling_are_equivalent(self):
        sq = np.array([-.1, -.2, -.3, -.4])
        tq = np.array([-.11, -.21, -.31, -.41])
        half, half_audit = solve_coupling(
            sq, tq, epsilon=.05, quality_weight=.5, relation_weight=0.,
            sinkhorn_max_iter=200)
        full, full_audit = solve_coupling(
            sq, tq, epsilon=.1, quality_weight=1., relation_weight=0.,
            sinkhorn_max_iter=200)
        np.testing.assert_allclose(half, full, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(half.sum(1), full.sum(1), atol=1e-10)
        np.testing.assert_allclose(half.sum(0), full.sum(0), atol=1e-10)
        self.assertAlmostEqual(half_audit['objective'], full_audit['objective'] * .5, places=10)
        self.assertAlmostEqual(half_audit['quality_weight'], .5)

    def test_tv_and_relation_cost_are_evaluated_on_both_couplings(self):
        sq = np.array([-.1, -.2, -.3, -.4])
        tq = np.array([-.11, -.21, -.31, -.41])
        sr = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
        tr = sr[::-1, ::-1].copy()
        quality, _ = solve_coupling(sq, tq, epsilon=.05, quality_weight=.5,
                                    relation_weight=0., sinkhorn_max_iter=100)
        relation, _ = solve_coupling(sq, tq, sr, tr, epsilon=.05,
                                     quality_weight=.5, relation_weight=.5,
                                     max_iter=10, sinkhorn_max_iter=100)
        tv = average_total_variation(quality, relation)
        self.assertGreaterEqual(tv, 0.)
        self.assertLessEqual(tv, 1.)
        self.assertAlmostEqual(average_total_variation(quality, quality), 0.)
        self.assertGreaterEqual(relation_cost(sr, tr, quality), 0.)
        self.assertGreaterEqual(relation_cost(sr, tr, relation), 0.)

    def test_split_before_coupling_and_reproducible_sampling(self):
        source = np.arange(27, dtype=np.float32).reshape(9, 3)
        sq = -np.arange(9, dtype=float)
        tq = -np.arange(9, dtype=float)
        conditions, plans, pairing = prepare_coupling(
            source, sq, self.tours[[0, 1, 2, 0, 1, 2, 0, 1, 2]], tq, self.solutions * 3,
            seed=1088, max_pairs=8, alpha=.5, epsilon=.1)
        self.assertEqual(conditions.shape, (8, 3))
        self.assertEqual(set(pairing['train_indices']) & set(pairing['valid_indices']), set())
        a = sample_target_indices(plans['train'], 'quality_coupling', np.arange(6), np.random.default_rng(7))
        b = sample_target_indices(plans['train'], 'quality_coupling', np.arange(6), np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        for name in ('quality', 'relation'):
            np.testing.assert_allclose(plans['train'][name].sum(1), 1 / 6, atol=1e-6)

    def test_validation_sampling_uses_validation_pool_not_train_pool(self):
        source = np.arange(27, dtype=np.float32).reshape(9, 3)
        quality = -np.arange(9, dtype=float)
        conditions, plans, pairing = prepare_coupling(
            source, quality, self.tours[[0, 1, 2, 0, 1, 2, 0, 1, 2]], quality,
            self.solutions * 3, seed=1088, max_pairs=8, alpha=.5, epsilon=.1)
        train_order = np.asarray(pairing['train_target_order'])
        valid_order = np.asarray(pairing['valid_target_order'])
        self.assertTrue(set(train_order).isdisjoint(valid_order))
        train_padded, train_sizes = self.problem.pack_solutions([self.solutions[i % 3] for i in range(len(train_order))])
        valid_padded, valid_sizes = self.problem.pack_solutions([self.solutions[i % 3] for i in range(len(valid_order))])
        model = CVRPVAE(8, 3)
        bridge = torch.nn.Linear(3, 3)
        instance = (torch.tensor(self.problem.coordinates, dtype=torch.float32)[None],
                    torch.tensor(self.problem.demands, dtype=torch.long)[None],
                    torch.tensor([self.problem.capacity], dtype=torch.long))
        config = dict(epochs=1, batch_size=4, learning_rate=.001, grad_clip=1., seed=3,
                      pair_seed=4, validation_pair_seed=5)
        _, _, sampling = train_coupled(
            model, bridge, instance, conditions, train_padded, train_sizes, valid_padded, valid_sizes,
            np.asarray(pairing['train_target_order']), np.asarray(pairing['valid_target_order']),
            plans['train'], plans['valid'], 'quality_coupling', np.arange(6), np.arange(6, 8), config)
        self.assertTrue(np.all(sampling['validation_target_choices'] < len(valid_order)))
        self.assertEqual(len(sampling['train_target_choices']), 1)


if __name__ == '__main__':
    unittest.main()
