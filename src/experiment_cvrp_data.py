"""CVRP milestone D-A: one instance, feasible samples, evaluator and NPZ checks."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.problem.cvrp_problem import CVRPProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, default=20, help='number of customers, excluding depot')
    parser.add_argument('--capacity', type=int, default=30)
    parser.add_argument('--sample-num', type=int, default=2048)
    parser.add_argument('--instance-seed', type=int, default=1088)
    parser.add_argument('--solution-seed', type=int, default=1089)
    parser.add_argument('--early-return-probability', type=float, default=0.2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.sample_num < 1:
        parser.error('--sample-num must be >= 1')
    problem = CVRPProblem(dimension=args.dimension, capacity=args.capacity, seed=args.instance_seed)
    solutions = problem.sample_solutions(args.sample_num, seed=args.solution_seed,
                                        early_return_probability=args.early_return_probability)
    padded, sizes = problem.pack_solutions(solutions)
    lengths = problem.evaluate_batch(solutions)
    route_counts = np.array([np.count_nonzero(sequence == 0) - 1 for sequence in solutions], dtype=np.int64)
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    metadata = dict(format_version=1, problem='euclidean_cvrp', dimension=problem.dimension,
                    dimension_definition='customers_excluding_depot', depot_index=0,
                    capacity=problem.capacity, instance_seed=args.instance_seed,
                    solution_seed=args.solution_seed, sample_num=args.sample_num,
                    early_return_probability=args.early_return_probability,
                    coordinate_distribution='iid_uniform_[0,1)',
                    demand_distribution='iid_uniform_integers_1_to_min(9,capacity); depot=0',
                    objective_sense='minimize', quality_definition='-length',
                    distance='euclidean_unrounded', split_delivery=False,
                    vehicle_count='unlimited_identical; no_vehicle_penalty',
                    representation='depot_separated_sequence', padding_value=-1,
                    sampling='random_customer_permutation_with_capacity_and_early_return_splits; nonuniform',
                    canonicalized=False,
                    base_commit=commit, numpy_version=np.__version__,
                    source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ('src/problem/cvrp_problem.py', 'src/experiment_cvrp_data.py')},
                    data_generation_objective_queries=args.sample_num,
                    reload_verification_objective_queries=args.sample_num)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, coordinates=problem.coordinates, demands=problem.demands,
                            capacity=np.array(problem.capacity, dtype=np.int64), routes_padded=padded,
                            sequence_lengths=sizes, lengths=lengths, qualities=-lengths,
                            route_counts=route_counts,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    with np.load(args.output, allow_pickle=False) as saved:
        restored = CVRPProblem(coordinates=saved['coordinates'], demands=saved['demands'],
                               capacity=int(saved['capacity'].item()))
        unpacked = restored.unpack_solutions(saved['routes_padded'], saved['sequence_lengths'])
        for original, loaded in zip(solutions, unpacked):
            np.testing.assert_array_equal(original, loaded)
        np.testing.assert_allclose(restored.evaluate_batch(unpacked), saved['lengths'])
        np.testing.assert_array_equal(saved['qualities'], -saved['lengths'])
        np.testing.assert_array_equal(saved['route_counts'],
                                      [len(restored.split_routes(sequence)) for sequence in unpacked])
        if json.loads(saved['metadata'].item()) != metadata:
            raise RuntimeError('metadata changed during save/reload')
    print('CVRP DATA CHECK PASSED')
    print('customers:', problem.dimension, 'depot: 0 capacity:', problem.capacity)
    print('coordinates:', problem.coordinates.shape)
    print('demands:', problem.demands.shape, str(problem.demands.dtype))
    print('routes_padded:', padded.shape, str(padded.dtype))
    print('sequence_lengths:', sizes.shape, str(sizes.dtype))
    print('feasible_rate: 1.0')
    print('objective: minimize total route length; quality = -length')
    print('saved:', args.output)


if __name__ == '__main__':
    main()
