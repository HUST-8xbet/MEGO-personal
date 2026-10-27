import unittest

import numpy as np

from src.behavioral_profile_probe import (
    _cycle_equivalent, _moves_cvrp, _moves_tsp, _relocate, _two_opt,
)
from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem


class BehavioralOperatorTests(unittest.TestCase):
    def test_tsp_full_reverse_is_cycle_equivalent(self):
        tour = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        self.assertTrue(_cycle_equivalent(tour, tour[::-1]))
        problem = TSPProblem(coordinates=np.arange(10, dtype=float).reshape(5, 2))
        moves, _, metadata = _moves_tsp(tour, 4, np.random.default_rng(7), problem)
        self.assertTrue(moves)
        self.assertEqual(len(moves), len(metadata))
        self.assertTrue(all(not _cycle_equivalent(tour, move) for move in moves))

    def test_cvrp_relocate_deletes_single_customer_source_route(self):
        problem = CVRPProblem(coordinates=np.zeros((4, 2)), demands=np.array([0, 1, 1, 1]), capacity=3)
        original = np.array([0, 1, 0, 2, 3, 0], dtype=np.int64)
        candidate = _relocate(original, 0, 1, 1, 1, problem)
        np.testing.assert_array_equal(candidate, [0, 1, 2, 3, 0])
        problem.validate_solution(candidate)

    def test_cvrp_intra_route_relocate_changes_order(self):
        problem = CVRPProblem(coordinates=np.zeros((4, 2)), demands=np.array([0, 1, 1, 1]), capacity=3)
        original = np.array([0, 1, 2, 3, 0], dtype=np.int64)
        candidate = _relocate(original, 0, 2, 0, 1, problem)
        np.testing.assert_array_equal(candidate, [0, 2, 1, 3, 0])
        problem.validate_solution(candidate)

    def test_cvrp_sampling_records_customer_then_valid_position(self):
        problem = CVRPProblem(coordinates=np.zeros((4, 2)), demands=np.array([0, 1, 1, 1]), capacity=3)
        original = np.array([0, 1, 0, 2, 3, 0], dtype=np.int64)
        moves, trials, metadata = _moves_cvrp(original, 4, np.random.default_rng(3), problem)
        self.assertEqual(len(moves), len(metadata))
        self.assertLessEqual(trials, 128)
        self.assertTrue(all(m['sampling'] == 'customer_uniform_then_valid_insert_position' for m in metadata))
        for move in moves:
            problem.validate_solution(move)


if __name__ == '__main__':
    unittest.main()
