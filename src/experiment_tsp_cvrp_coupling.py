"""TSP -> CVRP coupling experiment: rank, quality OT and quality+relation GW."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.problem.cvrp_problem import CVRPProblem
from src.problem.tsp_problem import TSPProblem
from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions
from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.tsp_cvrp_coupling import ARMS, prepare_coupling, train_coupled


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_checksum(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def load_source(checkpoint, data_path, device, batch_size):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    model = TSPVAE(config["hidden_dim"], config["latent_dim"]).to(device).eval()
    model.load_state_dict(payload["model"])
    with np.load(data_path, allow_pickle=False) as saved:
        problem = TSPProblem(coordinates=saved["coordinates"])
        tours, qualities = saved["tours"].copy(), saved["qualities"].copy()
    canonical = canonicalize_tours(tours)
    _, unique = np.unique(canonical, axis=0, return_index=True)
    unique = np.sort(unique)
    coordinates = torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None]
    latent = []
    for start in range(0, len(unique), batch_size):
        ids = unique[start:start + batch_size]
        mu, _ = model.encode(coordinates.expand(len(ids), -1, -1),
                             torch.tensor(canonical[ids], dtype=torch.long, device=device))
        latent.append(mu.cpu().numpy())
    # Loading/re-encoding the source bank is not an objective evaluation.
    return np.concatenate(latent), qualities[unique], canonical[unique], problem.coordinates, 0


def _quality_and_unique(problem, data_path):
    with np.load(data_path, allow_pickle=False) as saved:
        problem = CVRPProblem(coordinates=saved["coordinates"], demands=saved["demands"],
                              capacity=int(saved["capacity"].item()))
        solutions = problem.unpack_solutions(saved["routes_padded"], saved["sequence_lengths"])
        qualities = saved["qualities"].copy()
    canonical = canonicalize_solutions(problem, solutions)
    seen, keep = set(), []
    for i, solution in enumerate(canonical):
        key = tuple(solution)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    unique = [canonical[i] for i in keep]
    quality = qualities[keep]
    np.testing.assert_allclose(-problem.evaluate_batch(unique), quality, rtol=1e-7, atol=1e-9)
    return problem, unique, quality


@torch.no_grad()
def evaluate_budget(model, bridge, instance, conditions, problem, seed, rounds, samples_per_round):
    model.eval()
    if bridge is not None:
        bridge.eval()
    device = instance[0].device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    all_sampled, sampled_solutions, per_round = [], [], []
    sampled_padded, sampled_sizes, sampled_ids = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        ids = rng.choice(len(conditions), size=samples_per_round, replace=len(conditions) < samples_per_round)
        x = torch.tensor(conditions[ids], dtype=torch.float32, device=device)
        z = bridge(x) if bridge is not None else torch.zeros(len(ids), model.latent_dim, device=device)
        coords, demands, capacity = (t.expand(len(ids), *t.shape[1:]) for t in instance)
        padded, sizes, _ = model.decode(coords, demands, capacity, z, sample=True)
        solutions = problem.unpack_solutions(padded.cpu().numpy(), sizes.cpu().numpy())
        costs = problem.evaluate_batch(solutions)
        per_round.append(dict(mean=float(costs.mean()), best=float(costs.min()), std=float(costs.std()),
                              route_count=float(np.mean([len(problem.split_routes(s)) for s in solutions])),
                              unique=len({tuple(s) for s in solutions}), valid_rate=1.0))
        all_sampled.extend(costs.tolist())
        sampled_solutions.extend(solutions)
        sampled_padded.append(padded.cpu().numpy())
        sampled_sizes.append(sizes.cpu().numpy())
        sampled_ids.append(ids.copy())
    ids = np.arange(len(conditions))
    x = torch.tensor(conditions[ids], dtype=torch.float32, device=device)
    z = bridge(x) if bridge is not None else torch.zeros(len(ids), model.latent_dim, device=device)
    coords, demands, capacity = (t.expand(len(ids), *t.shape[1:]) for t in instance)
    padded, sizes, _ = model.decode(coords, demands, capacity, z, sample=False)
    greedy_solutions = problem.unpack_solutions(padded.cpu().numpy(), sizes.cpu().numpy())
    greedy = problem.evaluate_batch(greedy_solutions)
    round_best = np.asarray([item["best"] for item in per_round])
    sampled_unique = len({tuple(s) for s in canonicalize_solutions(problem, sampled_solutions)})
    greedy_unique = len({tuple(s) for s in canonicalize_solutions(problem, greedy_solutions)})
    adaptation = 9.399715696516127
    union_best = float(min(np.min(all_sampled), np.min(greedy)))
    return dict(sampled=dict(mean=float(np.mean(all_sampled)), std=float(np.std(all_sampled)),
                             best_of_410=float(np.min(all_sampled)), best_of_41_mean=float(round_best.mean()),
                             best_of_41_std=float(round_best.std(ddof=1)) if len(round_best) > 1 else 0., rounds=per_round,
                             unique=sampled_unique, valid_rate=1.0,
                             mean_route_count=float(np.mean([len(problem.split_routes(s)) for s in sampled_solutions])),
                             objective_queries=len(all_sampled)),
                greedy=dict(mean=float(greedy.mean()), std=float(greedy.std()), best_of_410=float(greedy.min()),
                            route_count=float(np.mean([len(problem.split_routes(s)) for s in greedy_solutions])),
                            unique=greedy_unique, valid_rate=1.0,
                            objective_queries=len(greedy)),
                adaptation_best=float(adaptation), union_best_820=union_best,
                final_best_including_adaptation=float(min(adaptation, union_best)),
                generation_seed=int(seed), generation_source_ids=[ids.tolist() for ids in sampled_ids],
                generated=dict(sampled_padded=np.concatenate(sampled_padded),
                               sampled_sizes=np.concatenate(sampled_sizes),
                               greedy_padded=padded.cpu().numpy(), greedy_sizes=sizes.cpu().numpy(),
                               greedy_source_ids=np.arange(len(conditions), dtype=np.int64)))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=root / "logs_tsp/tsp20_expert_smoke_01/best_model.pt")
    parser.add_argument("--source-data", type=Path, default=root / "data/tsp_milestone_a/tsp20_seed1088.npz")
    parser.add_argument("--target-data", type=Path, default=root / "data/cvrp_milestone_a/cvrp20_seed2088.npz")
    parser.add_argument("--config", type=Path, default=root / "configs/tsp_cvrp_coupling.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="override max_pairs/epochs/evaluation for a small wiring run")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(max_pairs=min(int(config["max_pairs"]), 32), epochs=2, evaluation_rounds=2, samples_per_round=7)
    if args.output.exists():
        raise FileExistsError("choose a new output directory")
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    source, source_quality, source_tours, source_coordinates, source_calls = load_source(
        args.source_checkpoint, args.source_data, device, config["batch_size"])
    problem, target_solutions, target_quality = _quality_and_unique(None, args.target_data)
    target_padded, target_sizes = problem.pack_solutions(target_solutions)
    instance = (torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None],
                torch.tensor(problem.demands.copy(), dtype=torch.long, device=device)[None],
                torch.tensor([problem.capacity], dtype=torch.long, device=device))
    args.output.mkdir(parents=True)
    results = []
    fixed_split_seed = int(config["split_seed"])
    for training_seed in config["training_seeds"]:
        local_config = dict(config, seed=int(training_seed), pair_seed=int(training_seed) + 100,
                            validation_pair_seed=int(training_seed) + 200)
        t0 = time.perf_counter()
        conditions, plans, pairing = prepare_coupling(
            source, source_quality, source_tours, target_quality, target_solutions,
            fixed_split_seed, local_config["max_pairs"], local_config["alpha"], local_config["epsilon"],
            device=str(device))
        coupling_seconds = time.perf_counter() - t0
        train_target_order = np.asarray(pairing["train_target_order"])
        valid_target_order = np.asarray(pairing["valid_target_order"])
        train_padded, train_sizes = target_padded[train_target_order], target_sizes[train_target_order]
        valid_padded, valid_sizes = target_padded[valid_target_order], target_sizes[valid_target_order]
        train, valid = np.asarray(pairing["train_indices"]), np.asarray(pairing["valid_indices"])
        seed_dir = args.output / ("seed_" + str(training_seed))
        seed_dir.mkdir()
        np.savez_compressed(seed_dir / "coupling.npz", **{
            "train_quality": plans["train"]["quality"], "train_relation": plans["train"]["relation"],
            "valid_quality": plans["valid"]["quality"], "valid_relation": plans["valid"]["relation"]})
        (seed_dir / "conditioning.json").write_text(json.dumps(pairing, indent=2, allow_nan=False) + "\n")
        torch.manual_seed(int(training_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(training_seed))
        template = CVRPVAE(local_config["hidden_dim"], local_config["latent_dim"]).to(device)
        bridge_template = torch.nn.Linear(source.shape[1], template.latent_dim).to(device)
        initialization_checksum = hashlib.sha256(
            state_checksum(template.state_dict()).encode() + state_checksum(bridge_template.state_dict()).encode()).hexdigest()
        seed_results = {"training_seed": int(training_seed), "split_seed": fixed_split_seed,
                        "coupling_seconds": coupling_seconds, "initialization_checksum": initialization_checksum, "arms": {}}
        for arm in ARMS:
            model, bridge = copy.deepcopy(template), copy.deepcopy(bridge_template)
            train_start = time.perf_counter()
            history, best, sampling = train_coupled(
                model, bridge, instance, conditions, train_padded, train_sizes, valid_padded, valid_sizes,
                train_target_order, valid_target_order,
                plans["train"], plans["valid"], arm, train, valid, local_config)
            train_seconds = time.perf_counter() - train_start
            generation_start = time.perf_counter()
            report = evaluate_budget(model, bridge, instance, conditions[valid], problem,
                                     int(training_seed) + 5000, local_config["evaluation_rounds"], local_config["samples_per_round"])
            generation_seconds = time.perf_counter() - generation_start
            generated = report.pop("generated")
            report.update(best_epoch=best["epoch"], validation_nll=best["valid_nll"],
                          train_target_choice_hash=sampling["train_target_choice_hash"],
                          validation_target_choice_hash=sampling["validation_target_choice_hash"],
                          batch_order_hash=sampling["batch_order_hash"],
                          training_seconds=train_seconds, generation_seconds=generation_seconds,
                          source_ids=pairing["valid_source_order"],
                          train_target_pool_ids=train_target_order.tolist(),
                          validation_target_pool_ids=valid_target_order.tolist(),
                          evaluation_objective_calls=report["sampled"]["objective_queries"] + report["greedy"]["objective_queries"])
            seed_results["arms"][arm] = report
            (seed_dir / (arm + "_pairing.json")).write_text(json.dumps(dict(
                train_target_pool_ids=train_target_order.tolist(),
                validation_target_pool_ids=valid_target_order.tolist(),
                train_target_global_choices=sampling["train_target_global_choices"].tolist(),
                validation_target_global_choices=sampling["validation_target_global_choices"].tolist(),
                membership_audit=sampling["target_membership_audit"]), indent=2) + "\n")
            np.savez_compressed(seed_dir / (arm + "_pairing.npz"),
                                train_target_global_choices=sampling["train_target_global_choices"],
                                validation_target_global_choices=sampling["validation_target_global_choices"],
                                train_target_pool_ids=train_target_order,
                                validation_target_pool_ids=valid_target_order)
            np.savez_compressed(seed_dir / (arm + "_generated.npz"), **generated)
            torch.save(dict(model=best["model"], bridge=best["bridge"], arm=arm,
                            training_seed=training_seed, config=local_config), seed_dir / (arm + ".pt"))
            (seed_dir / (arm + "_history.json")).write_text(json.dumps(history, indent=2) + "\n")
        seed_results["pair_count"] = len(train_padded) + len(valid_padded)
        seed_results["train_count"], seed_results["valid_count"] = len(train), len(valid)
        seed_results["split_checksum"] = pairing["split_checksum"]
        seed_results["resolved_config"] = local_config
        seed_results["audit"] = pairing["audits"]
        (seed_dir / "diagnostics.json").write_text(json.dumps(seed_results, indent=2, allow_nan=False) + "\n")
        results.append(seed_results)
    across_seed = {}
    for arm in ARMS:
        rows = [item["arms"][arm] for item in results]
        def stats(path):
            values = np.asarray([path(row) for row in rows], dtype=float)
            return dict(mean=float(values.mean()), std=float(values.std(ddof=1)), values=values.tolist())
        across_seed[arm] = dict(
            validation_nll=stats(lambda row: row["validation_nll"]),
            sampled_mean_length=stats(lambda row: row["sampled"]["mean"]),
            sampled_best_of_41_mean=stats(lambda row: row["sampled"]["best_of_41_mean"]),
            sampled_best_of_410=stats(lambda row: row["sampled"]["best_of_410"]),
            greedy_mean_length=stats(lambda row: row["greedy"]["mean"]),
            greedy_best_of_410=stats(lambda row: row["greedy"]["best_of_410"]),
            final_best_including_adaptation=stats(lambda row: row["final_best_including_adaptation"]),
        )
    summary = dict(protocol="TSP_CVRP_fixed_coupling; split_before_coupling; quality=-length",
                   arms=list(ARMS), config=config, results=results,
                   across_training_seed_summary=across_seed,
                   source_calls_current_run=source_calls,
                   source_imported_bank_count=len(source), adaptation_labels_shared=True,
                   adaptation_objective_verification_calls="unknown",
                   coupling_and_relation_cost="reported per seed/arm; not objective calls",
                   source_data_sha256=sha256(args.source_data), target_data_sha256=sha256(args.target_data),
                   source_checkpoint_sha256=sha256(args.source_checkpoint),
                   adaptation_best_reference=9.399715696516127,
                   note="Relation is 1-Jaccard of undirected edge sets; multiplicity and vehicle load are not represented. Validation coupling is independent and fixed before checkpoint selection. Historical costs not observed by this invocation remain unknown.")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("TSP -> CVRP COUPLING EXPERIMENT COMPLETED")
    print("summary:", args.output / "summary.json")


if __name__ == "__main__":
    main()
