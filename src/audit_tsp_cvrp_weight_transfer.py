"""Audit whether a TSPVAE checkpoint has valid direct CVRPVAE transfers.

This is deliberately a compatibility audit.  It does not instantiate a
training run, alter a checkpoint, reshape tensors, or call an objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"] if "model" in payload else payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path,
                        default=root / "logs_tsp/tsp20_expert_smoke_01/best_model.pt")
    parser.add_argument("--target-checkpoint", type=Path,
                        default=root / "logs_cvrp/cvrp20_expert_smoke_01/best_model.pt")
    parser.add_argument("--source-data", type=Path,
                        default=root / "data/tsp_milestone_a/tsp20_seed1088.npz")
    parser.add_argument("--target-data", type=Path,
                        default=root / "data/cvrp_milestone_a/cvrp20_seed2088.npz")
    parser.add_argument("--target-pairing", type=Path,
                        default=root / "logs_cvrp/tsp_cvrp_residual_v2/pairing.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new output directory")

    source = checkpoint_state(args.source_checkpoint)
    target = checkpoint_state(args.target_checkpoint)
    same_shape = {key: tuple(source[key].shape) == tuple(target[key].shape)
                  for key in source.keys() & target.keys()}
    semantic = {
        "city_embedding.0": ("reject", "TSP consumes (x,y); CVRP consumes (x,y,demand/capacity,depot-indicator); input feature meaning differs"),
        "city_embedding.2": ("reject", "same shape but transforms embeddings produced from incompatible four-vs-two feature inputs"),
        "tour_encoder/route_encoder": ("reject", "TSP encodes a permutation tour; CVRP encodes a padded multi-route sequence with depot tokens"),
        "posterior": ("reject", "same shape, but posterior inputs are different route/tour representations; shape equality is insufficient"),
        "decoder_init": ("reject", "same shape, but initializes decoders from incompatible instance representations"),
        "decoder_cell": ("reject", "CVRP has an additional remaining-load feature and the checkpoint shapes differ"),
        "key": ("reject", "same shape but keys operate on incompatible TSP/CVRP city embeddings and masks"),
        "query": ("reject", "CVRP query has an additional load feature and the checkpoint shapes differ"),
        "pointer": ("reject", "same shape but action space/mask semantics differ: permutation versus capacity-constrained routes"),
        "score_predictor": ("reject", "explicitly prohibited: source and target objective definitions/normalizations differ"),
    }
    rows = []
    for source_key in source:
        target_key = source_key
        if source_key.startswith("tour_encoder."):
            target_key = source_key.replace("tour_encoder.", "route_encoder.")
        source_shape = list(source[source_key].shape)
        target_shape = list(target[target_key].shape) if target_key in target else None
        group = source_key.rsplit(".", 1)[0]
        if group.startswith("city_embedding"):
            decision, reason = semantic["city_embedding.0" if group == "city_embedding.0" else "city_embedding.2"]
        elif group.startswith("tour_encoder"):
            decision, reason = semantic["tour_encoder/route_encoder"]
        elif group.startswith("score_predictor"):
            decision, reason = semantic["score_predictor"]
        else:
            decision, reason = semantic.get(group, ("reject", "not an explicitly approved same-meaning module"))
        if target_shape != source_shape:
            reason += "; shape mismatch" if target_shape is not None else "; target key absent"
        rows.append({"source_key": source_key, "target_key": target_key,
                     "source_shape": source_shape, "target_shape": target_shape,
                     "same_shape": target_shape == source_shape,
                     "decision": decision, "reason": reason})

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    pairing = json.loads(args.target_pairing.read_text())
    report = {
        "status": "FULL_RUN_BLOCKED_NO_VALID_DIRECT_MODULE_TRANSFER",
        "objective_calls": 0,
        "training_started": False,
        "base_commit": commit,
        "inputs": {
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_checkpoint_sha256": sha256(args.source_checkpoint),
            "source_data": str(args.source_data.resolve()),
            "source_data_sha256": sha256(args.source_data),
            "target_reference_checkpoint": str(args.target_checkpoint.resolve()),
            "target_reference_checkpoint_sha256": sha256(args.target_checkpoint),
            "target_data": str(args.target_data.resolve()),
            "target_data_sha256": sha256(args.target_data),
            "target_pairing": str(args.target_pairing.resolve()),
            "target_pairing_sha256": sha256(args.target_pairing),
        },
        "source_checkpoint_config": torch.load(args.source_checkpoint, map_location="cpu", weights_only=False).get("config", {}),
        "target_reference_config": torch.load(args.target_checkpoint, map_location="cpu", weights_only=False).get("config", {}),
        "target_split_reference": {
            "pairing_split": pairing.get("split"),
            "train_target_count": len(pairing.get("train_target_global_ids", [])),
            "validation_target_count": len(pairing.get("validation_target_global_ids", [])),
            "target_partitions_disjoint": pairing.get("target_partitions_disjoint"),
        },
        "compatibility_rows": rows,
        "approved_copy_keys": [],
        "rejected_copy_keys": [row["source_key"] for row in rows],
        "conclusion": "No meaningful direct module transfer is valid under the requested semantics. Do not run scratch/transfer comparison until a minimal architectural adapter or an explicitly approved shared representation is specified.",
        "smallest_change_proposal": "Define and implement one shared, semantically identical input representation/module between TSP and CVRP, then rerun this audit. Do not use padding, truncation, reshape, strict=False omission, or score-head transfer as a substitute.",
    }
    args.output.mkdir(parents=True)
    (args.output / "compatibility_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# TSP → CVRP weight-transfer compatibility audit", "",
        "Status: **FULL RUN BLOCKED — no valid direct module transfer**", "",
        "The source TSPVAE and target CVRPVAE have different input/action semantics. Same tensor shape alone was not accepted. No checkpoint was modified and no objective/training call was made.", "",
        "## Summary", "",
        "- Approved copy keys: **0**.",
        "- Rejected copy keys: **{}**.".format(len(rows)),
        "- TSP source: `tsp20_seed1088.npz`, checkpoint `tsp20_expert_smoke_01/best_model.pt`.",
        "- CVRP target: `cvrp20_seed2088.npz`; the `seed1088` CVRP instance was not used.",
        "- Existing target pairing reference: 1638 train / 410 validation, disjoint by target ID.", "",
        "## Compatibility table", "",
        "| Source key | Target key | Shapes | Direct copy | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| `{}` | `{}` | `{}` → `{}` | No | {} |".format(row["source_key"], row["target_key"], row["source_shape"], row["target_shape"], row["reason"]))
    lines += ["", "## Decision", "", report["conclusion"], "", "Smallest proposed next change: " + report["smallest_change_proposal"], "", "Machine-readable hashes and all rows are in `compatibility_audit.json`."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
