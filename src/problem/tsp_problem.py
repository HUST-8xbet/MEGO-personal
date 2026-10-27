"""Euclidean symmetric TSP with permutation solutions (no binary encoding)."""
from typing import Optional

import numpy as np

from src.problem.base_problem import BaseProblem


class TSPProblem(BaseProblem):
    """One instance. evaluate() is a cost to MINIMIZE; quality() is -cost.

    Random instances use independent uniform coordinates in [0, 1)^2.
    Tours are integer permutations of 0..dimension-1, without repeating the
    starting city at the end. Distances are unrounded Euclidean distances.
    Rotation/reversal equivalents are deliberately retained in this milestone.
    """

    objective_sense = "minimize"
    solution_representation = "permutation"

    def __init__(self, dimension: Optional[int] = None, coordinates=None,
                 seed: Optional[int] = None):
        if dimension is not None and (
            isinstance(dimension, (bool, np.bool_))
            or not isinstance(dimension, (int, np.integer)) or dimension < 3
        ):
            raise ValueError("dimension must be an integer >= 3")
        if coordinates is None:
            dimension = 20 if dimension is None else int(dimension)
            coords = np.random.default_rng(seed).random((dimension, 2))
        else:
            raw = np.asarray(coordinates)
            if raw.dtype.kind not in "iuf":
                raise ValueError("coordinates must be real numbers")
            coords = np.array(raw, dtype=np.float64, copy=True)
            if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
                raise ValueError("coordinates must have shape (n, 2), n >= 3")
            if dimension is not None and dimension != len(coords):
                raise ValueError("dimension disagrees with coordinates")
            dimension = len(coords)
        if not np.isfinite(coords).all():
            raise ValueError("coordinates must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = coords[:, None, :] - coords[None, :, :]
            distances = np.hypot(delta[..., 0], delta[..., 1])
        if not np.isfinite(distances).all():
            raise ValueError("pairwise distances must be finite")
        super().__init__(dimension=int(dimension))
        coords.setflags(write=False)
        distances.setflags(write=False)
        self.coordinates = coords
        self.distances = distances

    def validate_tour(self, solution):
        tour = np.asarray(solution)
        if tour.shape != (self.dimension,) or tour.dtype.kind not in "iu":
            raise ValueError("tour must be a 1D integer permutation of length n")
        if not np.array_equal(np.sort(tour), np.arange(self.dimension)):
            raise ValueError("tour must contain each city index 0..n-1 exactly once")
        return tour.astype(np.int64, copy=False)

    def evaluate(self, solution) -> float:
        tour = self.validate_tour(solution)
        # roll includes the last-city -> first-city closing edge.
        length = float(self.distances[tour, np.roll(tour, -1)].sum())
        if not np.isfinite(length):
            raise ValueError("tour length must be finite")
        return length

    def quality(self, solution) -> float:
        """Higher-is-better adapter for future MEGO quality ranking."""
        return -self.evaluate(solution)

    def sample_tours(self, sample_num: int, seed: Optional[int] = None):
        """IID uniform permutations, with replacement, using a local RNG.

        Duplicates and symmetry-equivalent tours are allowed; no rejection
        loop or implicit canonicalization is applied.
        """
        if (isinstance(sample_num, (bool, np.bool_))
                or not isinstance(sample_num, (int, np.integer)) or sample_num < 0):
            raise ValueError("sample_num must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        tours = np.empty((sample_num, self.dimension), dtype=np.int64)
        for i in range(sample_num):
            tours[i] = rng.permutation(self.dimension)
        return tours

    def evaluate_batch(self, tours):
        tours = np.asarray(tours)
        if tours.ndim != 2 or tours.shape[1] != self.dimension or tours.dtype.kind not in "iu":
            raise ValueError("tours must be an integer array of shape (batch, n)")
        return np.array([self.evaluate(tour) for tour in tours], dtype=np.float64)
