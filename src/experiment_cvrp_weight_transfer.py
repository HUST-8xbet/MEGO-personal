"""Baseline pilot: partial TSPVAE initialization for direct CVRPVAE training.

The transfer is intentionally conservative.  It copies only hidden-state
recurrent parameters whose tensor shapes and role are shared by the two GRU
implementations.  TSP latent values are never used as CVRP conditioning.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.problem.cvrp_problem import CVRPProblem
from src.regression.cvrp_vae import CVRPVAE, canonicalize_solutions


BRANCHES = ("scratch", "tsp_weight_init")
TRANSFER_MAP = {
    "route_encoder.weight_hh_l0": "tour_encoder.weight_hh_l0",
    "route_encoder.bias_hh_l0": "tour_encoder.bias_hh_l0",
    "decoder_cell.weight_hh": "decoder_cell.weight_hh",
    "decoder_cell.bias_hh": "decoder_cell.bias_hh",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(snapshot):
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def rng_digest(snapshot) -> str:
    h = hashlib.sha256()
    h.update(repr(snapshot["python"]).encode())
    np_state = snapshot["numpy"]
    h.update(str(np_state[0]).encode())
    h.update(np.asarray(np_state[1], dtype=np.uint32).tobytes())
    h.update(str(np_state[2:]).encode())
    h.update(snapshot["torch_cpu"].numpy().tobytes())
    for state in snapshot["torch_cuda"]:
        h.update(state.numpy().tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def isolated_generation_rng(seed, device):
    """Run generation with a private/restored CPU+CUDA RNG state."""
    previous = rng_snapshot()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        restore_rng(previous)


def load_target(data_path: Path, pairing_path: Path, split_seed: int):
    with np.load(data_path, allow_pickle=False) as data:
        problem = CVRPProblem(coordinates=data["coordinates"], demands=data["demands"],
                              capacity=int(data["capacity"].item()))
        raw_solutions = problem.unpack_solutions(data["routes_padded"], data["sequence_lengths"])
        stored_lengths = data["lengths"].copy()
        qualities = data["qualities"].copy()
    checked_lengths = problem.evaluate_batch(raw_solutions)
    np.testing.assert_allclose(checked_lengths, stored_lengths, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(qualities, -checked_lengths, rtol=1e-7, atol=1e-9)
    pairing = json.loads(pairing_path.read_text())
    train_ids = np.asarray(pairing["train_target_global_ids"], dtype=np.int64)
    valid_ids = np.asarray(pairing["validation_target_global_ids"], dtype=np.int64)
    if set(train_ids.tolist()) & set(valid_ids.tolist()):
        raise AssertionError("target train/validation ID leakage")
    if sorted(np.concatenate([train_ids, valid_ids]).tolist()) != list(range(len(raw_solutions))):
        raise ValueError("pairing does not cover target rows exactly")
    canonical = canonicalize_solutions(problem, raw_solutions)
    padded, sizes = problem.pack_solutions(canonical)
    train_quality = qualities[train_ids]
    mean = float(train_quality.mean())
    raw_std = float(train_quality.std())
    scale = raw_std if raw_std > 1e-8 else 1.0
    standardized = ((qualities - mean) / scale).astype(np.float32)
    return (problem, padded, sizes, standardized, checked_lengths, qualities,
            train_ids, valid_ids, dict(mean=mean, scale=scale, raw_std=raw_std))


def run_epoch(model, loader, instance, device, config, optimizer=None):
    model.train(optimizer is not None)
    nll_sum = kl_sum = mse_sum = 0.0
    token_count = count = 0
    for sequences, sizes, quality in loader:
        sequences, sizes, quality = sequences.to(device), sizes.to(device), quality.to(device)
        batch = len(sequences)
        coords, demands, capacity = (x.expand(batch, *x.shape[1:]) for x in instance)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(coords, demands, capacity, sequences, sizes,
                           stochastic=optimizer is not None)
            losses = model.loss(output, sequences, quality, config["beta"], config["score_weight"])
            if not all(torch.isfinite(v).item() for v in losses.values()):
                raise RuntimeError("nonfinite CVRP loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("nonfinite gradient")
                optimizer.step()
        actions = int((sizes - 1).sum().item())
        nll_sum += losses["nll"].item() * actions
        kl_sum += losses["kl"].item() * batch
        mse_sum += losses["score_mse"].item() * batch
        token_count += actions
        count += batch
    nll, kl, mse = nll_sum / token_count, kl_sum / count, mse_sum / count
    return {"loss": nll + config["beta"] * kl + config["score_weight"] * mse,
            "nll": nll, "kl": kl, "score_mse": mse,
            "real_action_count": token_count}


def copy_transfer(source_state, target_model):
    target_state = target_model.state_dict()
    copied = []
    for target_key, source_key in TRANSFER_MAP.items():
        if target_key not in target_state or source_key not in source_state:
            raise KeyError(f"missing transfer key: {source_key} -> {target_key}")
        if tuple(target_state[target_key].shape) != tuple(source_state[source_key].shape):
            raise ValueError(f"shape mismatch: {source_key} -> {target_key}")
        target_state[target_key].copy_(source_state[source_key])
        copied.append(target_key)
    target_model.load_state_dict(target_state)
    return copied


def initialization_audit(template, transfer_model, source_state):
    template_state, transfer_state = template.state_dict(), transfer_model.state_dict()
    copied, unchanged = [], []
    for key in template_state:
        if key in TRANSFER_MAP:
            source_key = TRANSFER_MAP[key]
            if tensor_hash(transfer_state[key]) != tensor_hash(source_state[source_key]):
                raise AssertionError(f"copied parameter differs from source: {key}")
            copied.append(key)
        else:
            if tensor_hash(template_state[key]) != tensor_hash(transfer_state[key]):
                raise AssertionError(f"non-copied parameter differs: {key}")
            unchanged.append(key)
    copied_count = sum(template_state[k].numel() for k in copied)
    total_count = sum(v.numel() for v in template_state.values())
    return {"copied_keys": copied, "unchanged_keys": unchanged,
            "copied_parameter_count": int(copied_count),
            "total_parameter_count": int(total_count),
            "copied_fraction": float(copied_count / total_count),
            "template_state_hash": state_hash(template),
            "transfer_state_hash": state_hash(transfer_model),
            "per_key_source_hash": {k: tensor_hash(source_state[v]) for k, v in TRANSFER_MAP.items()},
            "per_key_transfer_hash": {k: tensor_hash(transfer_state[k]) for k in TRANSFER_MAP},
            "noncopied_equal_template": True}


@torch.no_grad()
def generate(model, instance, problem, latents, action_seeds, rounds, candidates, device):
    model.eval()
    greedy_rows, sampled_rows = [], []
    greedy_sizes, sampled_sizes = [], []
    greedy_lengths, sampled_lengths = [], []
    with isolated_generation_rng(int(action_seeds[0]), device):
        for round_id in range(rounds):
            z = latents[round_id * candidates:(round_id + 1) * candidates].to(device)
            coords, demands, capacity = (x.expand(candidates, *x.shape[1:]) for x in instance)
            g, gs, _ = model.decode(coords, demands, capacity, z, sample=False)
            torch.manual_seed(int(action_seeds[round_id]))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(action_seeds[round_id]))
            s, ss, _ = model.decode(coords, demands, capacity, z, sample=True)
            g_np, gs_np = g.cpu().numpy(), gs.cpu().numpy()
            s_np, ss_np = s.cpu().numpy(), ss.cpu().numpy()
            g_sol = problem.unpack_solutions(g_np, gs_np)
            s_sol = problem.unpack_solutions(s_np, ss_np)
            greedy_rows.append(g_np); sampled_rows.append(s_np)
            greedy_sizes.append(gs_np); sampled_sizes.append(ss_np)
            greedy_lengths.append(problem.evaluate_batch(g_sol))
            sampled_lengths.append(problem.evaluate_batch(s_sol))
    def pack(rows, sizes, lengths):
        return {"padded": np.concatenate(rows), "sequence_lengths": np.concatenate(sizes),
                "lengths": np.concatenate(lengths)}
    return pack(greedy_rows, greedy_sizes, greedy_lengths), pack(sampled_rows, sampled_sizes, sampled_lengths)


def metrics(name, arrays, incumbent, rounds, candidates):
    lengths = np.asarray(arrays["lengths"], dtype=np.float64).reshape(rounds, candidates)
    best_round = lengths.min(axis=1)
    best_all = float(lengths.min())
    flat = int(np.argmin(lengths))
    assert np.isclose(best_all, float(best_round.min()), rtol=0, atol=1e-12)
    result = {"name": name, "candidate_count": int(lengths.size),
              "mean_length": float(lengths.mean()),
              "round_best_lengths": best_round.tolist(),
              "best_of_41_mean": float(best_round.mean()),
              "best_of_41_std_sample": float(best_round.std(ddof=1)),
              "best_of_410": best_all,
              "best_length": best_all,
              "best_final_including_adaptation": min(float(incumbent), best_all),
              "best_candidate_id": flat,
              "best_candidate_objective": best_all}
    assert np.isclose(result["best_length"], result["best_of_410"], rtol=0, atol=1e-12)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-checkpoint", type=Path, default=root / "logs_tsp/tsp20_expert_smoke_01/best_model.pt")
    p.add_argument("--source-data", type=Path, default=root / "data/tsp_milestone_a/tsp20_seed1088.npz")
    p.add_argument("--target-data", type=Path, default=root / "data/cvrp_milestone_a/cvrp20_seed2088.npz")
    p.add_argument("--target-pairing", type=Path, default=root / "logs_cvrp/tsp_cvrp_residual_v2/pairing.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output directory")
    config = dict(hidden_dim=64, latent_dim=16, epochs=30, batch_size=128,
                  learning_rate=0.001, beta=0.001, score_weight=1.0,
                  grad_clip=1.0, cpu_threads=1, training_seeds=[1088, 1089, 1090],
                  generation_seed=96001, generation_rounds=10, candidates_per_round=41,
                  split_seed=1088, source_pretraining="historical_only",
                  primary_metric="sampled_mean_best_of_41")
    if args.smoke_test:
        config.update(epochs=1, training_seeds=[1088], generation_rounds=2,
                      candidates_per_round=3, smoke_test=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_payload = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    source_state = source_payload["model"] if "model" in source_payload else source_payload
    (problem, padded, sizes, standardized, checked_lengths, raw_quality,
     train_ids, valid_ids, quality_norm) = load_target(args.target_data, args.target_pairing, config["split_seed"])
    instance = (torch.tensor(problem.coordinates.copy(), dtype=torch.float32, device=device)[None],
                torch.tensor(problem.demands.copy(), dtype=torch.long, device=device)[None],
                torch.tensor([problem.capacity], dtype=torch.long, device=device))
    args.output.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    input_hashes = {"source_checkpoint": sha256(args.source_checkpoint),
                    "source_data": sha256(args.source_data),
                    "target_data": sha256(args.target_data),
                    "target_pairing": sha256(args.target_pairing)}
    config.update(device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
                  base_commit=commit, input_sha256=input_hashes, quality_normalization=quality_norm,
                  train_target_global_ids=train_ids.tolist(), validation_target_global_ids=valid_ids.tolist(),
                  target_train_validation_disjoint=True,
                  run_code_sha256={name: sha256(root / name) for name in (
                      "src/experiment_cvrp_weight_transfer.py",
                      "src/regression/cvrp_vae.py",
                      "src/problem/cvrp_problem.py")})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(args.output / "split.npz", train_target_global_ids=train_ids,
                        validation_target_global_ids=valid_ids)
    # This is one shared target-data verification, recorded separately from generation.
    verification_queries = int(len(checked_lengths))
    incumbent_pos = int(np.argmin(checked_lengths))
    incumbent = float(checked_lengths[incumbent_pos])
    np.savez_compressed(args.output / "incumbent.npz", padded=padded[incumbent_pos],
                        sequence_length=sizes[incumbent_pos], length=incumbent,
                        target_global_id=incumbent_pos)
    seed_reports = []
    total_generation_queries = 0
    for seed in config["training_seeds"]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed)
        template = CVRPVAE(config["hidden_dim"], config["latent_dim"]).to(device)
        transfer = copy.deepcopy(template)
        copied_keys = copy_transfer(source_state, transfer)
        init_audit = initialization_audit(template, transfer, source_state)
        training_rng_state = rng_snapshot()
        init_audit["training_rng_state_digest"] = rng_digest(training_rng_state)
        seed_dir = args.output / f"seed_{seed}"; seed_dir.mkdir()
        (seed_dir / "initialization.json").write_text(json.dumps(init_audit, indent=2) + "\n")
        branch_reports = {}
        # The same DataLoader seed gives the same batch order for both branches.
        def loader(ids, shuffle):
            ds = TensorDataset(torch.tensor(padded[ids]), torch.tensor(sizes[ids]), torch.tensor(standardized[ids]))
            return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
        for branch, model in (("scratch", template), ("tsp_weight_init", transfer)):
            restore_rng(training_rng_state)
            training_rng_before_digest = rng_digest(rng_snapshot())
            train_loader, valid_loader = loader(train_ids, True), loader(valid_ids, False)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            history = []
            initial_valid = run_epoch(model, valid_loader, instance, device, config)
            best_value, best_epoch, best_state = float("inf"), None, None
            train_start = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                train_metrics = run_epoch(model, train_loader, instance, device, config, optimizer)
                valid_metrics = run_epoch(model, valid_loader, instance, device, config)
                history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})
                if valid_metrics["loss"] < best_value:
                    best_value, best_epoch = valid_metrics["loss"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            rng_after_training = rng_snapshot()
            rng_after_training_digest = rng_digest(rng_after_training)
            generation_seed = config["generation_seed"] + seed
            latent_generator = torch.Generator(device=device).manual_seed(generation_seed)
            total = config["generation_rounds"] * config["candidates_per_round"]
            latents = torch.randn(total, config["latent_dim"], generator=latent_generator, device=device)
            action_seeds = [generation_seed + 1000 + r for r in range(config["generation_rounds"])]
            generation_start = time.perf_counter()
            greedy, sampled = generate(model, instance, problem, latents, action_seeds,
                                       config["generation_rounds"], config["candidates_per_round"], device)
            generation_seconds = time.perf_counter() - generation_start
            rng_after_generation_digest = rng_digest(rng_snapshot())
            greedy_m = metrics("greedy", greedy, incumbent, config["generation_rounds"], config["candidates_per_round"])
            sampled_m = metrics("sampled", sampled, incumbent, config["generation_rounds"], config["candidates_per_round"])
            union_lengths = np.minimum(greedy["lengths"], sampled["lengths"])
            union_best = float(min(greedy["lengths"].min(), sampled["lengths"].min()))
            assert np.isclose(union_best, float(min(greedy_m["best_of_410"], sampled_m["best_of_410"])), rtol=0, atol=1e-12)
            branch_dir = seed_dir / branch; branch_dir.mkdir()
            np.savez_compressed(branch_dir / "generated.npz",
                                greedy_padded=greedy["padded"], greedy_sequence_lengths=greedy["sequence_lengths"],
                                greedy_lengths=greedy["lengths"], sampled_padded=sampled["padded"],
                                sampled_sequence_lengths=sampled["sequence_lengths"], sampled_lengths=sampled["lengths"],
                                latents=latents.cpu().numpy(), action_seeds=np.asarray(action_seeds, dtype=np.int64),
                                latent_ids=np.arange(total, dtype=np.int64))
            best_id = int(np.argmin(sampled["lengths"]))
            np.savez_compressed(branch_dir / "best_sampled.npz",
                                padded=sampled["padded"][best_id], sequence_length=sampled["sequence_lengths"][best_id],
                                length=sampled["lengths"][best_id], latent_id=best_id)
            torch.save({"model": best_state, "branch": branch, "training_seed": seed,
                        "best_epoch": best_epoch, "validation_loss": best_value,
                        "config": config, "source_checkpoint_sha256": input_hashes["source_checkpoint"],
                        "copied_keys": (copied_keys if branch == "tsp_weight_init" else []),
                        "run_code_sha256": config["run_code_sha256"],
                        "training_rng_state_digest": training_rng_before_digest}, branch_dir / "best_model.pt")
            (branch_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            branch_reports[branch] = {"initial_validation": initial_valid, "best_epoch": best_epoch,
                                     "best_validation": history[best_epoch - 1]["valid"],
                                     "training_seconds": float(time.perf_counter() - train_start),
                                     "generation_seconds": float(generation_seconds),
                                     "training_rng_state_digest": training_rng_before_digest,
                                     "rng_after_training_digest": rng_after_training_digest,
                                     "rng_after_generation_digest": rng_after_generation_digest,
                                     "generation_rng_isolated": (rng_after_training_digest == rng_after_generation_digest),
                                     "greedy": greedy_m, "sampled": sampled_m,
                                     "union_greedy_sampled": {"candidate_count": int(2 * total),
                                                               "best_generated_length": union_best,
                                                               "best_final_including_adaptation": min(incumbent, union_best)},
                                     "generation_queries": int(2 * total),
                                     "latent_seed": generation_seed,
                                     "action_seeds": action_seeds,
                                     "incumbent_length": incumbent}
            total_generation_queries += 2 * total
        seed_reports.append({"training_seed": seed, "initialization": init_audit, "branches": branch_reports})
    across = {}
    for metric in ("best_of_41_mean", "best_of_410", "best_final_including_adaptation"):
        across[metric] = {}
        for branch in BRANCHES:
            values = np.asarray([s["branches"][branch]["sampled"][metric] for s in seed_reports])
            across[metric][branch] = {"mean": float(values.mean()),
                                      "sample_std_ddof1": (float(values.std(ddof=1)) if len(values) > 1 else None),
                                      "values": values.tolist()}
    report = {"status": "SMOKE_COMPLETED" if config.get("smoke_test") else "PILOT_COMPLETED",
              "protocol": "direct_CVRPVAE_scratch_vs_partial_TSP_weight_init",
              "no_latent_conditioning": True, "no_mapping_or_bridge": True,
              "primary_metric": config["primary_metric"],
              "copy_map": TRANSFER_MAP, "source_pretraining_cost_is_historical": True,
              "verification_target_queries": verification_queries,
              "generation_target_queries": total_generation_queries,
              "new_target_objective_queries_current_run": verification_queries + total_generation_queries,
              "seed_reports": seed_reports, "across_training_seeds": across,
              "incumbent": {"target_global_id": incumbent_pos, "length": incumbent},
              "input_sha256": input_hashes, "base_commit": commit}
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "costs.json").write_text(json.dumps({"source_pretraining_historical": True,
        "target_verification_queries": verification_queries,
        "generation_queries": total_generation_queries,
        "target_objective_queries_current_run": verification_queries + total_generation_queries,
        "new_objective_calls_by_audit": 0}, indent=2) + "\n")
    lines = ["# TSP → CVRP partial weight initialization pilot", "", report["status"], "",
             "Primary metric: **mean best-of-41 sampled** (pre-registered).", "",
             "Copied only hidden-state recurrent parameters; all other target parameters start from the same cloned template.", "",
             "| Seed | Branch | Best-of-41 mean | Best-of-410 | Final incl. incumbent | Best epoch |", "|---:|---|---:|---:|---:|---:|"]
    for s in seed_reports:
        for branch in BRANCHES:
            r = s["branches"][branch]
            lines.append(f"| {s['training_seed']} | {branch} | {r['sampled']['best_of_41_mean']:.9f} | {r['sampled']['best_of_410']:.9f} | {r['sampled']['best_final_including_adaptation']:.9f} | {r['best_epoch']} |")
    lines += ["", "## Across training seeds", "", "```json", json.dumps(across, indent=2), "```", "", "All generation metrics are recomputed from saved candidate arrays; no preliminary decode is used."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("PILOT COMPLETED" if not config.get("smoke_test") else "SMOKE COMPLETED")


if __name__ == "__main__":
    main()
