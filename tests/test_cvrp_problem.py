import unittest

import numpy as np

from src.problem.cvrp_problem import CVRPProblem


class CVRPProblemTests(unittest.TestCase):
    def setUp(self):
        self.problem = CVRPProblem(coordinates=[[0, 0], [1, 0], [2, 0], [0, 1], [0, 2]],
                                   demands=[0, 2, 2, 2, 2], capacity=4)
        self.sequence = np.array([0, 1, 2, 0, 3, 4, 0], dtype=np.int64)

    def test_known_geometry_and_return_edges(self):
        # Each route 0->near->far->0 has cost 1+1+2 = 4.
        self.assertEqual(self.problem.dimension, 4)
        self.assertAlmostEqual(self.problem.evaluate(self.sequence), 8.)
        self.assertAlmostEqual(self.problem.quality(self.sequence), -8.)
        np.testing.assert_array_equal(self.problem.route_loads(self.sequence), [4, 4])
        np.testing.assert_array_equal(self.problem.split_routes(self.sequence)[0], [0, 1, 2, 0])

    def test_route_order_and_reversal_symmetry(self):
        for sequence in ([0, 3, 4, 0, 1, 2, 0], [0, 2, 1, 0, 4, 3, 0],
                         [0, 4, 3, 0, 1, 2, 0]):
            self.assertAlmostEqual(self.problem.evaluate(sequence), 8.)
        np.testing.assert_array_equal(self.problem.distances, self.problem.distances.T)
        np.testing.assert_array_equal(np.diag(self.problem.distances), np.zeros(5))

    def test_missing_duplicate_and_out_of_range_customers_rejected(self):
        for bad in ([0, 1, 2, 0, 3, 0], [0, 1, 2, 0, 3, 3, 0],
                    [0, 1, 2, 0, 3, 5, 0], [0, 1, 2, 0, 3, -1, 0],
                    [0, 1, 2, 0, 3, 4, 1, 0]):
            with self.subTest(sequence=bad), self.assertRaises(ValueError):
                self.problem.evaluate(bad)

    def test_depot_empty_routes_and_noninteger_sequences_rejected(self):
        for bad in ([1, 2, 0, 3, 4, 0], [0, 1, 2, 0, 3, 4],
                    [0, 0, 1, 2, 0, 3, 4, 0], [0, 1, 2, 0, 0, 3, 4, 0],
                    self.sequence.astype(float), self.sequence[None], [],
                    ['0', '1', '2', '0', '3', '4', '0']):
            with self.subTest(sequence=bad), self.assertRaises(ValueError):
                self.problem.evaluate(bad)

    def test_capacity_boundary_and_overflow(self):
        self.problem.validate_solution(self.sequence)  # exactly capacity is feasible
        with self.assertRaises(ValueError):
            self.problem.evaluate([0, 1, 2, 3, 0, 4, 0])
        maximal = int(np.iinfo(np.int64).max)
        problem = CVRPProblem(coordinates=[[0, 0], [1, 0], [2, 0]],
                              demands=[0, maximal, maximal], capacity=maximal)
        problem.validate_solution([0, 1, 0, 2, 0])
        with self.assertRaises(ValueError):
            problem.evaluate([0, 1, 2, 0])

    def test_invalid_instance_inputs(self):
        for capacity in (0, -1, 2.5, True, np.iinfo(np.int64).max + 1):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                CVRPProblem(capacity=capacity)
        for demands in ([1, 2], [0, 0], [0, 31], [0, -1], [0., 1.], [[0, 1]], [False, True]):
            with self.subTest(demands=demands), self.assertRaises(ValueError):
                CVRPProblem(demands=demands)
        for dimension in (0, -1, 1.5, True):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                CVRPProblem(dimension=dimension)
        for coordinates in ([[0, 0]], [[0, 0, 0], [1, 1, 1]],
                            [[0, 0], [np.nan, 1]], [[0, 0], [np.inf, 1]]):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                CVRPProblem(coordinates=coordinates)
        with self.assertRaises(ValueError):
            CVRPProblem(dimension=20, coordinates=[[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            CVRPProblem(coordinates=[[0, 0], [1, 1]], demands=[0, 1, 1])

    def test_default_cvrp20_sampling_and_seed_reproducibility(self):
        problem = CVRPProblem(seed=1088)
        self.assertEqual(problem.coordinates.shape, (21, 2))
        self.assertEqual(problem.demands.shape, (21,))
        self.assertEqual(problem.capacity, 30)
        self.assertTrue(np.all((problem.demands[1:] >= 1) & (problem.demands[1:] <= 9)))
        other = CVRPProblem(seed=1088)
        np.testing.assert_array_equal(problem.coordinates, other.coordinates)
        np.testing.assert_array_equal(problem.demands, other.demands)
        solutions = problem.sample_solutions(2048, seed=1089)
        repeat = problem.sample_solutions(2048, seed=1089)
        different = problem.sample_solutions(2048, seed=1090)
        for first, second in zip(solutions, repeat):
            np.testing.assert_array_equal(first, second)
            loads = problem.route_loads(first)
            self.assertTrue(np.all(loads <= problem.capacity))
            self.assertEqual(int(loads.sum()), int(problem.demands.sum()))
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(solutions, different)))
        self.assertTrue(np.isfinite(problem.evaluate_batch(solutions)).all())

    def test_early_return_extremes_and_small_capacity(self):
        for probability in (0., 1.):
            solutions = self.problem.sample_solutions(16, seed=3, early_return_probability=probability)
            for sequence in solutions:
                self.problem.validate_solution(sequence)
                if probability == 1.:
                    self.assertEqual(len(sequence), self.problem.max_sequence_length)
        problem = CVRPProblem(dimension=20, capacity=1, seed=4)
        for sequence in problem.sample_solutions(4, seed=5):
            self.assertEqual(len(problem.split_routes(sequence)), 20)

    def test_independent_coordinate_based_evaluator(self):
        problem = CVRPProblem(seed=17)
        for sequence in problem.sample_solutions(32, seed=18):
            reference = sum(np.linalg.norm(problem.coordinates[int(a)] - problem.coordinates[int(b)])
                            for a, b in zip(sequence[:-1], sequence[1:]))
            self.assertAlmostEqual(problem.evaluate(sequence), reference)

    def test_pack_unpack_round_trip_and_empty_batch(self):
        solutions = self.problem.sample_solutions(16, seed=5)
        padded, sizes = self.problem.pack_solutions(solutions)
        self.assertEqual(padded.shape, (16, 9))
        self.assertEqual(padded.dtype, np.int64)
        unpacked = self.problem.unpack_solutions(padded, sizes)
        for before, after in zip(solutions, unpacked):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(self.problem.evaluate_batch(solutions), self.problem.evaluate_batch(unpacked))
        empty, sizes = self.problem.pack_solutions([])
        self.assertEqual(empty.shape, (0, 9))
        self.assertEqual(self.problem.unpack_solutions(empty, sizes), [])
        self.assertEqual(self.problem.evaluate_batch([]).shape, (0,))

    def test_bad_padding_and_lengths_rejected(self):
        padded, sizes = self.problem.pack_solutions([self.sequence])
        bad_padding = padded.copy()
        bad_padding[0, -1] = 0
        for matrix, lengths in ((bad_padding, sizes), (padded, [2]), (padded, [10]),
                                (padded, sizes.astype(float)), (padded.astype(float), sizes),
                                (padded[:, :-1], sizes)):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                self.problem.unpack_solutions(matrix, lengths)
        with self.assertRaises(ValueError):
            self.problem.evaluate(padded[0])  # padding cannot enter the evaluator

    def test_single_customer_and_coincident_coordinates(self):
        problem = CVRPProblem(coordinates=[[0, 0], [1, 0]], demands=[0, 3], capacity=3)
        self.assertAlmostEqual(problem.evaluate([0, 1, 0]), 2.)
        problem = CVRPProblem(coordinates=np.zeros((3, 2)), demands=[0, 1, 1], capacity=1)
        self.assertEqual(problem.evaluate([0, 1, 0, 2, 0]), 0.)

    def test_local_rng_and_input_copy(self):
        np.random.seed(7)
        expected = np.random.random(5)
        np.random.seed(7)
        CVRPProblem(seed=8).sample_solutions(3, seed=9)
        np.testing.assert_array_equal(np.random.random(5), expected)
        coordinates = np.array([[0., 0.], [1., 0.]])
        demands = np.array([0, 2])
        problem = CVRPProblem(coordinates=coordinates, demands=demands, capacity=2)
        coordinates[:] = 99
        demands[:] = 99
        self.assertAlmostEqual(problem.evaluate([0, 1, 0]), 2.)
        np.testing.assert_array_equal(problem.demands, [0, 2])

    def test_bad_sampler_arguments(self):
        self.assertEqual(self.problem.sample_solutions(0), [])
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.problem.sample_solutions(count)
        for probability in (-.1, 1.1, np.nan, np.inf, True, '0.2'):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                self.problem.sample_solutions(1, early_return_probability=probability)


if __name__ == '__main__':
    unittest.main()
