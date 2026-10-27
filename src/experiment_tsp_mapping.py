"""Milestone C: rank-matched Max-Cut latent mapping into a TSP decoder."""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.problem.tsp_problem import TSPProblem
from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.tsp_mapping import (LatentBridge, load_maxcut_latent,
                              rank_matched_pairs, set_decoder_trainable)


def load_tsp_data(path):
    with np.load(path, allow_pickle=False) as saved:
        problem = TSPProblem(coordinates=saved["coordinates"])
        tours = canonicalize_tours(saved["tours"])
        qualities = saved["qualities"].astype(np.float64, copy=True)
    if qualities.shape != (len(tours),):
        raise ValueError("TSP quality shape does not match tours")
    lengths = problem.evaluate_batch(tours)
    np.testing.assert_allclose(qualities, -lengths, rtol=1e-7, atol=1e-9)
    return problem, tours, qualities


def load_tsp_expert(checkpoint, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["config"]
    model = TSPVAE(config["hidden_dim"], config["latent_dim"]).to(device)
    model.load_state_dict(payload["model"])
    return model, config


def _decoder_parameters(model):
    return set_decoder_trainable(model)


def train_decoder(model, bridge, coordinates, source_latent, tours, train_ids,
                  valid_ids, config, device, seed, use_source):
    """Fine-tune the existing TSP decoder using rank-matched source latents."""
    parameters = _decoder_parameters(model)
    if bridge is not None:
        parameters += list(bridge.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    source_tensor = torch.as_tensor(source_latent, dtype=torch.float32, device=device)
    tour_tensor = torch.as_tensor(tours, dtype=torch.long, device=device)
    coords = coordinates.expand(len(tours), -1, -1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history, best_loss, best_state = [], float("inf"), None

    def run(ids, update):
        model.train(update)
        if bridge is not None:
            bridge.train(update)
        order = ids.copy()
        if update:
            order = order[torch.randperm(len(order), generator=generator).numpy()]
        totals, count = 0.0, 0
        for start in range(0, len(order), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            batch_tours = tour_tensor[batch]
            if use_source:
                z = bridge(source_tensor[batch])
            else:
                z = torch.zeros(len(batch), model.latent_dim, device=device)
            _, logits = model.decode(coords[batch], z, teacher_tours=batch_tours)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   batch_tours[:, 1:].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite mapping loss")
            if update:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("nonfinite mapping gradient")
                optimizer.step()
            totals += loss.item() * len(batch)
            count += len(batch)
        return totals / count

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run(train_ids, True)
        valid_loss = run(valid_ids, False)
        history.append(dict(epoch=epoch, train_nll=train_loss, valid_nll=valid_loss))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = dict(model=copy.deepcopy(model.state_dict()),
                              bridge=copy.deepcopy(bridge.state_dict()) if bridge is not None else None,
                              epoch=epoch, valid_nll=valid_loss)
    model.load_state_dict(best_state["model"])
    if bridge is not None:
        bridge.load_state_dict(best_state["bridge"])
    return history, best_state


@torch.no_grad()
def evaluate_condition(model, bridge, coordinates, source_latent, tours, problem,
                       device, condition, seed):
    model.eval()
    if bridge is not None:
        bridge.eval()
    latent = torch.as_tensor(source_latent, dtype=torch.float32, device=device)
    if condition == "shuffled":
        latent = latent.roll(1, 0)
    elif condition == "random":
        latent = torch.randn(latent.shape, generator=torch.Generator(device=device).manual_seed(seed),
                             device=device)
    if bridge is None or condition == "no_source":
        z = torch.zeros(len(tours), model.latent_dim, device=device)
    else:
        z = bridge(latent)
    coords = coordinates.expand(len(tours), -1, -1)
    generated, logits = model.decode(coords, z)
    generated_np = generated.cpu().numpy()
    costs = problem.evaluate_batch(generated_np)
    canonical = canonicalize_tours(generated_np)
    target = canonicalize_tours(tours)
    teacher_logits = model.decode(coords, z, teacher_tours=torch.as_tensor(target, device=device))[1]
    nll = F.cross_entropy(teacher_logits.reshape(-1, problem.dimension),
                          torch.as_tensor(target[:, 1:], device=device).reshape(-1)).item()
    return dict(condition=condition, valid_rate=1.0, teacher_forced_nll=nll,
                mean_length=float(costs.mean()), best_length=float(costs.min()),
                mean_quality=float((-costs).mean()),
                exact_cycle_rate=float(np.mean(np.all(canonical == target, axis=1))),
                unique_cycles=int(len(np.unique(canonical, axis=0)))), generated_np, len(costs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source-checkpoint", type=Path,
                        default=root / "logs_smoke_20261020_224736/MaxCutProblem-0-SurrogateVAE/best_model.pt")
    parser.add_argument("--source-config", type=Path,
                        default=root / "logs_smoke_20261020_224736/MaxCutProblem-0-SurrogateVAE/HyperParam.yaml")
    parser.add_argument("--source-x", type=Path,
                        default=root / "data/smoke_20261020_224736/train/max_cut_problem_0_30/x.npy")
    parser.add_argument("--source-y", type=Path,
                        default=root / "data/smoke_20261020_224736/train/max_cut_problem_0_30/y.npy")
    parser.add_argument("--target-data", type=Path,
                        default=root / "data/tsp_milestone_a/tsp20_seed1088.npz")
    parser.add_argument("--target-checkpoint", type=Path,
                        default=root / "logs_tsp/tsp20_expert_smoke_01/best_model.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/tsp_mapping_smoke.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.output.exists():
        raise FileExistsError("output already exists; choose a new directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    source_config = __import__("yaml").safe_load(args.source_config.read_text())
    source_latent, source_quality, source_model_config = load_maxcut_latent(
        args.source_checkpoint, source_config, args.source_x, args.source_y, device)
    problem, target_tours, target_quality = load_tsp_data(args.target_data)
    pairs = rank_matched_pairs(source_latent, source_quality, target_tours, target_quality,
                               config.get("max_pairs"))
    order_check = dict(source_quality_descending=bool(np.all(np.diff(pairs.source_quality) <= 0)),
                       target_quality_descending=bool(np.all(np.diff(pairs.target_quality) <= 0)),
                       source_best=float(pairs.source_quality[0]), target_best=float(pairs.target_quality[0]),
                       source_worst=float(pairs.source_quality[-1]), target_worst=float(pairs.target_quality[-1]),
                       target_quality_definition="-length; higher is better")
    if not order_check["source_quality_descending"] or not order_check["target_quality_descending"]:
        raise RuntimeError("quality pairing is not descending")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs.source_latent))
    cut = max(1, int(0.8 * len(indices)))
    train_ids, valid_ids = indices[:cut], indices[cut:]
    coordinates = torch.tensor(problem.coordinates, dtype=torch.float32, device=device)[None]
    target_template, target_config = load_tsp_expert(args.target_checkpoint, device)
    source_model = source_model_config
    source_dim, target_dim = pairs.source_latent.shape[1], target_template.latent_dim

    source_model_target = copy.deepcopy(target_template)
    source_bridge = LatentBridge(source_dim, target_dim).to(device)
    history, best = train_decoder(source_model_target, source_bridge, coordinates,
                                  pairs.source_latent, pairs.target_tours, train_ids, valid_ids,
                                  config, device, seed, use_source=True)
    no_source_model = copy.deepcopy(target_template)
    no_source_history, no_source_best = train_decoder(no_source_model, None, coordinates,
                                                      pairs.source_latent, pairs.target_tours,
                                                      train_ids, valid_ids, config, device, seed,
                                                      use_source=False)

    conditions = {}
    generated = {}
    eval_queries = 0
    for condition in ("aligned", "shuffled", "random"):
        report, tours, queries = evaluate_condition(source_model_target, source_bridge, coordinates,
                                                    pairs.source_latent[valid_ids], pairs.target_tours[valid_ids],
                                                    problem, device, condition, seed + 1)
        conditions[condition] = report
        generated[condition] = tours
        eval_queries += queries
    report, tours, queries = evaluate_condition(no_source_model, None, coordinates,
                                                pairs.source_latent[valid_ids], pairs.target_tours[valid_ids],
                                                problem, device, "no_source", seed + 2)
    conditions["no_source"] = report
    generated["no_source"] = tours
    eval_queries += queries

    args.output.mkdir(parents=True)
    checkpoint = dict(model=source_model_target.state_dict(), bridge=source_bridge.state_dict(),
                      source_latent_dim=source_dim, target_latent_dim=target_dim,
                      source_checkpoint=str(args.source_checkpoint), target_checkpoint=str(args.target_checkpoint),
                      quality_pairing=order_check, best=best)
    torch.save(checkpoint, args.output / "mapping_checkpoint.pt")
    torch.save(dict(model=no_source_model.state_dict(), best=no_source_best),
               args.output / "no_source_checkpoint.pt")
    costs = dict(source_pretraining_objective_queries="not available in checkpoint",
                 target_data_objective_queries=len(target_tours),
                 target_adaptation_training_objective_queries=0,
                 target_adaptation_verification_queries=len(target_tours),
                 target_evaluation_objective_queries=eval_queries,
                 target_total_recorded_queries=len(target_tours) + eval_queries,
                 note="Training consumes stored target qualities; no new target objective is queried per gradient step.")
    metadata = dict(status="mapping_functional_run_completed", source_quality="Max-Cut score; maximize",
                    target_quality="TSP -length; maximize", pair_count=len(pairs.source_latent),
                    train_count=len(train_ids), valid_count=len(valid_ids), seed=seed,
                    source_latent_dim=source_dim, target_latent_dim=target_dim,
                    quality_pairing=order_check, conditions=conditions, target_costs=costs,
                    source_checkpoint_sha256=hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest(),
                    target_checkpoint_sha256=hashlib.sha256(args.target_checkpoint.read_bytes()).hexdigest())
    (args.output / "history.json").write_text(json.dumps({"source": history, "no_source": no_source_history}, indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "costs.json").write_text(json.dumps(costs, indent=2) + "\n")
    np.savez_compressed(args.output / "split.npz", train_indices=train_ids, valid_indices=valid_ids,
                        source_order=pairs.source_order, target_order=pairs.target_order)
    np.savez_compressed(args.output / "generated.npz", coordinates=problem.coordinates,
                        valid_target_tours=pairs.target_tours[valid_ids], **generated)
    print("TSP MAPPING FUNCTIONAL RUN COMPLETED")
    print("pair_count:", len(pairs.source_latent))
    print("aligned_mean_length:", conditions["aligned"]["mean_length"])
    print("no_source_mean_length:", conditions["no_source"]["mean_length"])
    print("checkpoint:", args.output / "mapping_checkpoint.pt")
    print("diagnostics:", args.output / "diagnostics.json")


if __name__ == "__main__":
    main()
