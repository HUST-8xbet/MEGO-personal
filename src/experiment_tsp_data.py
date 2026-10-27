"""Milestone A only: create one instance and sampled tours; no training."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.problem.tsp_problem import TSPProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--sample-num", type=int, default=2048)
    parser.add_argument("--instance-seed", type=int, default=1088)
    parser.add_argument("--tour-seed", type=int, default=1089)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_num < 1:
        parser.error("--sample-num must be >= 1")
    problem = TSPProblem(dimension=args.dimension, seed=args.instance_seed)
    tours = problem.sample_tours(args.sample_num, seed=args.tour_seed)
    lengths = problem.evaluate_batch(tours)
    metadata = dict(format_version=1, problem="euclidean_symmetric_tsp",
                    dimension=problem.dimension, sample_num=args.sample_num,
                    instance_seed=args.instance_seed, tour_seed=args.tour_seed,
                    objective_sense="minimize", quality_definition="-length",
                    coordinate_distribution="iid_uniform_[0,1)",
                    distance="euclidean_unrounded", sampling="iid_with_replacement",
                    canonicalized=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite earlier experiment data.
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, coordinates=problem.coordinates, tours=tours,
                            lengths=lengths, qualities=-lengths,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
    with np.load(args.output, allow_pickle=False) as saved:
        restored = TSPProblem(coordinates=saved["coordinates"])
        np.testing.assert_array_equal(saved["tours"], tours)
        np.testing.assert_allclose(restored.evaluate_batch(saved["tours"]), saved["lengths"])
        np.testing.assert_array_equal(saved["qualities"], -saved["lengths"])
        assert json.loads(saved["metadata"].item()) == metadata
    print("TSP DATA CHECK PASSED")
    print("coordinates:", problem.coordinates.shape)
    print("tours:", tours.shape, str(tours.dtype))
    print("lengths:", lengths.shape, str(lengths.dtype))
    print("objective: minimize length; quality = -length")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
