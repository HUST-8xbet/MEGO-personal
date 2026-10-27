import unittest

import numpy as np

from src.problem.tsp_problem import TSPProblem


class TSPProblemTests(unittest.TestCase):
    def setUp(self):
        self.square = TSPProblem(coordinates=[[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_square_closing_edge_and_quality(self):
        self.assertAlmostEqual(self.square.evaluate([0, 1, 2, 3]), 4.0)
        self.assertAlmostEqual(self.square.quality([0, 1, 2, 3]), -4.0)
        self.assertAlmostEqual(self.square.evaluate([0, 2, 1, 3]), 2 + 2 * np.sqrt(2))
        self.assertGreater(self.square.quality([0, 1, 2, 3]), self.square.quality([0, 2, 1, 3]))

    def test_rotation_and_reversal(self):
        problem = TSPProblem(seed=7)
        tour = problem.sample_tours(1, seed=8)[0]
        for candidate in (tour, tour[::-1]):
            for shift in range(problem.dimension):
                self.assertAlmostEqual(problem.evaluate(np.roll(candidate, shift)), problem.evaluate(tour))

    def test_invalid_tours_rejected(self):
        for bad in ([0, 1, 1, 3], [0, 1, 2, 4], [-1, 1, 2, 3],
                    [0, 1, 2], [0, 1, 2, 3, 0], [[0, 1, 2, 3]],
                    [0., 1., 2., 3.], [False, True, True, True],
                    [0, 1, 2, np.nan], ["0", "1", "2", "3"]):
            with self.subTest(tour=bad), self.assertRaises(ValueError):
                self.square.evaluate(bad)

    def test_instance_validation(self):
        for coords in ([], [[0, 0]], [[0, 0, 0]] * 3,
                       [[0, 0], [1, 1], [np.inf, 0]], [[0, 0], [1, 1], [np.nan, 0]]):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                TSPProblem(coordinates=coords)
        for dim in (2, 0, -1, 3.5, True):
            with self.subTest(dimension=dim), self.assertRaises(ValueError):
                TSPProblem(dimension=dim)
        with self.assertRaises(ValueError):
            TSPProblem(dimension=20, coordinates=self.square.coordinates)

    def test_sampling_and_reproducibility(self):
        problem = TSPProblem(seed=1088)
        self.assertEqual(problem.coordinates.shape, (20, 2))
        self.assertTrue(((problem.coordinates >= 0) & (problem.coordinates < 1)).all())
        np.testing.assert_array_equal(problem.coordinates, TSPProblem(seed=1088).coordinates)
        self.assertFalse(np.array_equal(problem.coordinates, TSPProblem(seed=1089).coordinates))
        tours = problem.sample_tours(2048, seed=1089)
        self.assertEqual(tours.shape, (2048, 20))
        self.assertEqual(tours.dtype, np.int64)
        np.testing.assert_array_equal(np.sort(tours, axis=1), np.tile(np.arange(20), (2048, 1)))
        np.testing.assert_array_equal(tours, problem.sample_tours(2048, seed=1089))
        self.assertFalse(np.array_equal(tours, problem.sample_tours(2048, seed=1090)))

    def test_empty_and_invalid_sample_sizes(self):
        self.assertEqual(self.square.sample_tours(0).shape, (0, 4))
        self.assertEqual(self.square.evaluate_batch(self.square.sample_tours(0)).shape, (0,))
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.square.sample_tours(count)

    def test_batch_and_independent_distance_calculation(self):
        problem = TSPProblem(seed=9)
        tours = problem.sample_tours(16, seed=10)
        costs = problem.evaluate_batch(tours)
        for tour, cost in zip(tours, costs):
            points = problem.coordinates[tour]
            reference = sum(np.linalg.norm(points[i] - points[(i + 1) % 20]) for i in range(20))
            self.assertAlmostEqual(cost, reference)
        for bad in (tours[0], tours.astype(float), [[0] * 20]):
            with self.assertRaises(ValueError):
                problem.evaluate_batch(bad)

    def test_symmetric_distance_and_coordinate_copy(self):
        coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
        problem = TSPProblem(coordinates=coords)
        coords[:] = 99
        self.assertAlmostEqual(problem.evaluate([0, 1, 2]), 2 + np.sqrt(2))
        np.testing.assert_array_equal(problem.distances, problem.distances.T)
        np.testing.assert_array_equal(np.diag(problem.distances), np.zeros(3))

    def test_coincident_cities_are_allowed(self):
        problem = TSPProblem(coordinates=np.zeros((20, 2)))
        self.assertEqual(problem.evaluate(np.arange(20)), 0.)

    def test_local_rng_does_not_change_global_rng(self):
        np.random.seed(42)
        expected = np.random.random(5)
        np.random.seed(42)
        TSPProblem(seed=1).sample_tours(5, seed=2)
        np.testing.assert_array_equal(np.random.random(5), expected)


if __name__ == "__main__":
    unittest.main()
