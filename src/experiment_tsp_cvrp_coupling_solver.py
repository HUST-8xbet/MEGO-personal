"""Coupling-only diagnostic: matched quality OT versus quality+relation GW."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem
from src.regression.cvrp_vae import canonicalize_solutions
from src.regression.tsp_vae import canonicalize_tours
from src.tsp_cvrp_coupling import (
    average_total_variation,
    cvrp_relation_matrix,
    prepare_coupling,
    relation_cost,
    solve_coupling,
    tsp_relation_matrix,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_banks(source_path, target_path):
    with np.load(source_path, allow_pickle=False) as saved:
        source_problem = TSPProblem(coordinates=saved["coordinates"])
        source_tours = canonicalize_tours(saved["tours"].copy())
        source_quality = saved["qualities"].copy()
    _, source_keep = np.unique(source_tours, axis=0, return_index=True)
    source_keep = np.sort(source_keep)
    source_tours, source_quality = source_tours[source_keep], source_quality[source_keep]
    with np.load(target_path, allow_pickle=False) as saved:
        target_problem = CVRPProblem(coordinates=saved["coordinates"], demands=saved["demands"],
                                     capacity=int(saved["capacity"].item()))
        target_solutions = target_problem.unpack_solutions(saved["routes_padded"], saved["sequence_lengths"])
        target_quality = saved["qualities"].copy()
    canonical = canonicalize_solutions(target_problem, target_solutions)
    seen, keep = set(), []
    for i, solution in enumerate(canonical):
        key = tuple(solution)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    target_solutions = [canonical[i] for i in keep]
    target_quality = target_quality[keep]
    np.testing.assert_allclose(-source_problem.evaluate_batch(source_tours), source_quality, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(-target_problem.evaluate_batch(target_solutions), target_quality, rtol=1e-7, atol=1e-9)
    # Latents are not needed for a coupling-only diagnostic; use a finite
    # placeholder solely because prepare_coupling also prepares train scaling.
    source_placeholder = np.zeros((len(source_tours), 1), dtype=np.float32)
    return source_placeholder, source_quality, source_tours, target_quality, target_solutions


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=root / "data/tsp_milestone_a/tsp20_seed1088.npz")
    parser.add_argument("--target-data", type=Path, default=root / "data/cvrp_milestone_a/cvrp20_seed2088.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=2048)
    parser.add_argument("--split", choices=("train", "valid", "both"), default="train")
    parser.add_argument("--long-iterations", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output directory")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source, source_quality, source_tours, target_quality, target_solutions = load_banks(args.source_data, args.target_data)
    _, plans, pairing = prepare_coupling(
        source, source_quality, source_tours, target_quality, target_solutions,
        seed=1088, max_pairs=args.max_pairs, alpha=.5, epsilon=.05, device=args.device)
    splits = ("train", "valid") if args.split == "both" else (args.split,)
    results, matrices = [], {}
    for split in splits:
        plan = plans[split]
        sr = tsp_relation_matrix(source_tours[plan["source_global_ids"]])
        tr = cvrp_relation_matrix([target_solutions[i] for i in plan["target_global_ids"]])
        for arm in ("quality_matched", "quality_relation_matched"):
            for iterations in (50, args.long_iterations):
                start = time.perf_counter()
                if arm == "quality_matched":
                    gamma, audit = solve_coupling(
                        source_quality[plan["source_global_ids"]], target_quality[plan["target_global_ids"]],
                        alpha=.5, epsilon=.05, max_iter=iterations, sinkhorn_max_iter=iterations,
                        quality_weight=.5, relation_weight=0., device=args.device)
                else:
                    gamma, audit = solve_coupling(
                        source_quality[plan["source_global_ids"]], target_quality[plan["target_global_ids"]], sr, tr,
                        alpha=.5, epsilon=.05, max_iter=iterations, sinkhorn_max_iter=iterations,
                        quality_weight=.5, relation_weight=.5, device=args.device)
                actual_relation = relation_cost(sr, tr, gamma)
                audit["relation_cost_in_objective"] = audit["relation_cost"]
                audit["relation_cost"] = actual_relation
                audit["relation_cost_evaluated_on_this_coupling"] = True
                elapsed = time.perf_counter() - start
                key = f"{split}_{arm}_{iterations}"
                matrices[key] = gamma
                results.append(dict(split=split, arm=arm, iterations=iterations,
                                    seconds=elapsed, audit=audit))
    for split in splits:
        for iterations in (50, args.long_iterations):
            q = matrices[f"{split}_quality_matched_{iterations}"]
            qr = matrices[f"{split}_quality_relation_matched_{iterations}"]
            results.append(dict(split=split, arm="quality_vs_relation_distance", iterations=iterations,
                                tv_avg=average_total_variation(q, qr),
                                quality_relation_coupling_max_abs=float(np.max(np.abs(q - qr))),
                                quality_coupling_relation_cost=relation_cost(
                                    tsp_relation_matrix(source_tours[plans[split]["source_global_ids"]]),
                                    cvrp_relation_matrix([target_solutions[i] for i in plans[split]["target_global_ids"]]), q),
                                quality_relation_coupling_relation_cost=relation_cost(
                                    tsp_relation_matrix(source_tours[plans[split]["source_global_ids"]]),
                                    cvrp_relation_matrix([target_solutions[i] for i in plans[split]["target_global_ids"]]), qr)))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "couplings.npz", **matrices)
    (args.output / "pairing.json").write_text(json.dumps(pairing, indent=2, allow_nan=False) + "\n")
    summary = dict(protocol="coupling_only; float64; matched_coefficients; no_decoder_training",
                   alpha=.5, epsilon=.05, split_seed=1088, requested_split=args.split,
                   source_data_sha256=sha256(args.source_data), target_data_sha256=sha256(args.target_data),
                   device=args.device, dtype="float64", iterations=[50, args.long_iterations],
                   results=results,
                   comparison="quality_matched=0.5Q+epsilonE; quality_relation_matched=0.5Q+0.5R+epsilonE",
                   note="R is evaluated on both final couplings, including quality-only; no decoder or generated quality is used.")
    (args.output / "solver_diagnostics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("COUPLING-ONLY DIAGNOSTIC COMPLETED")
    print("diagnostics:", args.output / "solver_diagnostics.json")


if __name__ == "__main__":
    main()
