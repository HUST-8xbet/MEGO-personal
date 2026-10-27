"""Euclidean CVRP with integer demands and depot-separated route sequences."""
from typing import Optional

import numpy as np

from src.problem.base_problem import BaseProblem


def _integer(value, minimum, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum or value > np.iinfo(np.int64).max):
        raise ValueError(name + ' must be an integer >= ' + str(minimum) + ' fitting int64')
    return int(value)


class CVRPProblem(BaseProblem):
    """Homogeneous capacity, unlimited vehicles, no split delivery.

    dimension counts CUSTOMERS; there are dimension+1 vertices, depot index 0.
    A solution is [0, customers..., 0, customers..., 0]. Every customer is
    visited exactly once. Empty routes / adjacent depot tokens are forbidden.
    Coordinates are unrounded Euclidean; demands are unnormalized integers.
    evaluate returns total distance to minimize; quality returns its negative.
    Vehicle count is not penalized or minimized lexicographically.
    """

    objective_sense = 'minimize'
    solution_representation = 'depot_separated_routes'

    def __init__(self, dimension: Optional[int] = None, coordinates=None,
                 demands=None, capacity: int = 30, seed: Optional[int] = None):
        self.capacity = _integer(capacity, 1, 'capacity')
        if dimension is not None:
            dimension = _integer(dimension, 1, 'dimension')
        coords = None
        if coordinates is not None:
            raw = np.asarray(coordinates)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 2:
                raise ValueError('coordinates must be real with shape (customers+1, 2)')
            coords = np.array(raw, dtype=np.float64, copy=True)
            if dimension is not None and dimension != len(coords) - 1:
                raise ValueError('dimension disagrees with coordinates')
            dimension = len(coords) - 1
        demand_array = None
        if demands is not None:
            raw = np.asarray(demands)
            if raw.dtype.kind not in 'iu' or raw.ndim != 1 or len(raw) < 2:
                raise ValueError('demands must be a 1D integer vector including the depot')
            if raw[0] != 0 or np.any(raw[1:] < 1) or np.any(raw[1:] > self.capacity):
                raise ValueError('depot demand must be 0; customer demand must be in 1..capacity')
            if dimension is not None and dimension != len(raw) - 1:
                raise ValueError('dimension disagrees with demands')
            dimension = len(raw) - 1
            demand_array = np.array(raw, dtype=np.int64, copy=True)
        dimension = 20 if dimension is None else dimension
        rng = np.random.default_rng(seed)
        if coords is None:
            coords = rng.random((dimension + 1, 2))
        if demand_array is None:
            demand_array = np.r_[np.int64(0), rng.integers(1, min(9, self.capacity) + 1,
                                                         size=dimension, dtype=np.int64)]
        if not np.isfinite(coords).all():
            raise ValueError('coordinates must be finite')
        with np.errstate(over='ignore', invalid='ignore'):
            difference = coords[:, None, :] - coords[None, :, :]
            distances = np.hypot(difference[..., 0], difference[..., 1])
        if not np.isfinite(distances).all():
            raise ValueError('pairwise distances must be finite')
        super().__init__(dimension=dimension)
        self.coordinates, self.demands, self.distances = coords, demand_array, distances
        for array in (self.coordinates, self.demands, self.distances):
            array.setflags(write=False)

    @property
    def max_sequence_length(self):
        # All customers served by separate vehicles: 0,c1,0,c2,0,...,cn,0.
        return 2 * self.dimension + 1

    def validate_solution(self, solution):
        sequence = np.asarray(solution)
        if sequence.ndim != 1 or sequence.dtype.kind not in 'iu':
            raise ValueError('solution must be a 1D integer depot-separated sequence')
        if not self.dimension + 2 <= len(sequence) <= self.max_sequence_length:
            raise ValueError('solution has an invalid length')
        if sequence[0] != 0 or sequence[-1] != 0:
            raise ValueError('solution must start and end at the depot')
        if np.any(sequence < 0) or np.any(sequence > self.dimension):
            raise ValueError('customer index out of range; remove padding before evaluation')
        if np.any((sequence[:-1] == 0) & (sequence[1:] == 0)):
            raise ValueError('empty routes / adjacent depot tokens are forbidden')
        customers = sequence[sequence != 0]
        if not np.array_equal(np.sort(customers), np.arange(1, self.dimension + 1)):
            raise ValueError('each customer must appear exactly once; split delivery is forbidden')
        load = 0  # Python integer, preventing int64 overflow while adding demands.
        for vertex in sequence:
            if vertex == 0:
                load = 0
            else:
                load += int(self.demands[int(vertex)])
                if load > self.capacity:
                    raise ValueError('route load exceeds vehicle capacity')
        return sequence.astype(np.int64, copy=False)

    def evaluate(self, solution) -> float:
        sequence = self.validate_solution(solution)
        # Routes already contain depot endpoints; do not close with a new edge.
        length = float(self.distances[sequence[:-1], sequence[1:]].sum())
        if not np.isfinite(length):
            raise ValueError('solution length must be finite')
        return length

    def quality(self, solution) -> float:
        return -self.evaluate(solution)

    def evaluate_batch(self, solutions):
        return np.array([self.evaluate(solution) for solution in solutions], dtype=np.float64)

    def split_routes(self, solution):
        """Return individual arrays [0, customers..., 0], retaining depot ends."""
        sequence = self.validate_solution(solution)
        depot_positions = np.flatnonzero(sequence == 0)
        return [sequence[start:end + 1].copy()
                for start, end in zip(depot_positions[:-1], depot_positions[1:])]

    def route_loads(self, solution):
        return np.array([sum(int(self.demands[i]) for i in route)
                         for route in self.split_routes(solution)], dtype=np.int64)

    def sample_solutions(self, sample_num: int, seed: Optional[int] = None,
                         early_return_probability: float = 0.2):
        """Random permutations with feasible capacity splits and early returns.

        This is NOT uniform over CVRP solutions and uses no solver/objective.
        Early return probability 0 gives next-fit packing; 1 gives one customer
        per vehicle. Duplicates and route symmetry equivalents are allowed.
        """
        sample_num = _integer(sample_num, 0, 'sample_num')
        probability = early_return_probability
        if (isinstance(probability, (bool, np.bool_))
                or not isinstance(probability, (int, float, np.integer, np.floating))
                or not np.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError('early_return_probability must be a finite number in [0, 1]')
        rng = np.random.default_rng(seed)
        solutions = []
        for _ in range(sample_num):
            sequence, load = [0], 0
            for customer in rng.permutation(np.arange(1, self.dimension + 1)):
                demand = int(self.demands[customer])
                if load and (load + demand > self.capacity or rng.random() < probability):
                    sequence.append(0)
                    load = 0
                sequence.append(int(customer))
                load += demand
            sequence.append(0)
            solutions.append(np.array(sequence, dtype=np.int64))
        return solutions

    def pack_solutions(self, solutions):
        """Fixed width int64 arrays, with -1 padding and explicit true lengths."""
        sequences = [self.validate_solution(solution) for solution in solutions]
        padded = np.full((len(sequences), self.max_sequence_length), -1, dtype=np.int64)
        sizes = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
        for i, sequence in enumerate(sequences):
            padded[i, :len(sequence)] = sequence
        return padded, sizes

    def unpack_solutions(self, padded, sequence_lengths):
        padded, sizes = np.asarray(padded), np.asarray(sequence_lengths)
        if (padded.ndim != 2 or padded.shape[1] != self.max_sequence_length
                or padded.dtype.kind not in 'iu' or sizes.dtype.kind not in 'iu'
                or sizes.shape != (len(padded),)):
            raise ValueError('invalid padded solution / sequence_lengths shapes or types')
        if np.any(sizes < self.dimension + 2) or np.any(sizes > self.max_sequence_length):
            raise ValueError('invalid sequence length')
        solutions = []
        for row, size in zip(padded, sizes):
            size = int(size)
            if np.any(row[size:] != -1):
                raise ValueError('padding must be -1, not the depot index')
            solutions.append(self.validate_solution(row[:size]).copy())
        return solutions
