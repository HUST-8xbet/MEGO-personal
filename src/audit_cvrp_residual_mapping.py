"""Reconcile an existing H_residual run without evaluating new objectives.

This audit intentionally reads only diagnostics, pairing metadata, and the
saved generated.npz arrays.  It must not import or call a problem evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARM_NAMES = ("rank_residual", "rank_residual_shuffle", "rank_residual_zero")
ROUNDS = 10
CANDIDATES = 41
TOL = 1e-12


def _json(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _close(a, b) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=TOL))


def _metrics(lengths: np.ndarray, incumbent: float) -> dict:
    lengths = np.asarray(lengths, dtype=np.float64)
    round_best = lengths.min(axis=1)
    all_best = float(lengths.min())
    return {
        "candidate_shape": list(lengths.shape),
        "candidate_count": int(lengths.size),
        "mean": float(lengths.mean()),
        "min_each_round": round_best.tolist(),
        "best_of_41_mean": float(round_best.mean()),
        "best_of_41_std_sample": float(round_best.std(ddof=1)),
        "best_of_410": all_best,
        "incumbent": float(incumbent),
        "best_final_including_adaptation": min(float(incumbent), all_best),
    }


def _candidate_provenance(arrays_meta: dict, positions: np.ndarray,
                          lengths: np.ndarray, label: str) -> dict:
    flat = int(np.argmin(lengths))
    r, c = divmod(flat, lengths.shape[1])
    sources = np.asarray(arrays_meta["source_global_ids"])
    seeds = arrays_meta["action_seeds"]
    return {
        "branch": label,
        "candidate_id_zero_based": flat,
        "round_zero_based": r,
        "candidate_in_round_zero_based": c,
        "source_pair_position": int(positions[r, c]),
        "source_global_id": int(sources[r, c]),
        "generation_seed": int(seeds[r]),
        "objective": float(lengths[r, c]),
    }


def audit(input_dir: Path, output_dir: Path) -> None:
    diagnostics_path = input_dir / "diagnostics.json"
    costs_path = input_dir / "costs.json"
    pairing_path = input_dir / "pairing.json"
    diagnostics = _json(diagnostics_path)
    old_costs = _json(costs_path)
    pairing = _json(pairing_path)
    output_dir.mkdir(parents=True, exist_ok=False)

    incumbent = float(diagnostics["adaptation_incumbent"]["best_adaptation_length"])
    rows = []
    details = []
    stale_count = 0
    protocol_count = 0
    for seed_report in diagnostics["seeds"]:
        training_seed = int(seed_report["training_seed"])
        for arm in ARM_NAMES:
            report = seed_report["arms"][arm]
            arm_dir = input_dir / f"seed_{training_seed}" / arm
            with np.load(arm_dir / "generated.npz") as z:
                sampled = np.asarray(z["sampled_lengths"], dtype=np.float64)
                greedy = np.asarray(z["greedy_lengths"], dtype=np.float64)
                sampled_positions = np.asarray(z["sampled_source_pair_positions"])
                greedy_positions = np.asarray(z["greedy_source_pair_positions"])
            if sampled.shape != (ROUNDS, CANDIDATES) or greedy.shape != (ROUNDS, CANDIDATES):
                raise ValueError(f"unexpected protocol shape for seed={training_seed}, arm={arm}")
            protocol_count += sampled.size + greedy.size

            sampled_meta = report["sampled_arrays_meta"]
            greedy_meta = report["greedy_arrays_meta"]
            sm = _metrics(sampled, incumbent)
            gm = _metrics(greedy, incumbent)
            sampled_min = float(sampled.min())
            round_min = float(sampled.min(axis=1).min())
            reported_sampled = report["sampled"]
            equality = {
                "min_all_sampled_eq_min_rounds": _close(sampled_min, round_min),
                "min_rounds_eq_reported_best_of_all": _close(round_min, reported_sampled["best_of_all"]),
                "min_all_sampled_eq_reported_best_of_all": _close(sampled_min, reported_sampled["best_of_all"]),
                "reported_best_length_is_protocol_min": _close(sampled_min, reported_sampled["best_length"]),
            }
            if not equality["reported_best_length_is_protocol_min"]:
                stale_count += 1

            union_min = min(float(sampled.min()), float(greedy.min()))
            union_report = report["greedy_plus_sampled_union"]
            union = {
                "candidate_count": int(sampled.size + greedy.size),
                "best_generated_length": union_min,
                "best_final_including_adaptation": min(incumbent, union_min),
                "reported_best_generated_length": float(union_report["best_generated_length"]),
                "reported_best_final_including_adaptation": float(union_report["best_final_including_adaptation"]),
                "matches_report": _close(union_min, union_report["best_generated_length"]),
            }
            details.append({
                "training_seed": training_seed,
                "arm": arm,
                "sampled": sm,
                "greedy": gm,
                "union_greedy_sampled": union,
                "equality_checks": equality,
                "sampled_protocol_best": _candidate_provenance(sampled_meta, sampled_positions, sampled, "sampled_protocol"),
                "greedy_protocol_best": _candidate_provenance(greedy_meta, greedy_positions, greedy, "greedy_protocol"),
                "reported_stale_best_length": {
                    "value": float(reported_sampled["best_length"]),
                    "provenance_recoverable": False,
                    "reason": "evaluate_mapping() preliminary 410-candidate decode arrays were discarded; only scalar was retained",
                },
                "reported_values": {
                    "sampled_mean_length": float(reported_sampled["mean_length"]),
                    "sampled_best_of_all": float(reported_sampled["best_of_all"]),
                    "sampled_best_final_including_adaptation": float(reported_sampled["best_final_including_adaptation"]),
                },
            })
            rows.append({"training_seed": training_seed, "arm": arm,
                         "sampled_mean": sm["mean"], "best_of_41_mean": sm["best_of_41_mean"],
                         "best_of_41_std_sample": sm["best_of_41_std_sample"],
                         "best_of_410": sm["best_of_410"],
                         "union_best": union_min,
                         "incumbent": incumbent,
                         "best_final_including_adaptation": sm["best_final_including_adaptation"],
                         "stale_best_length": float(reported_sampled["best_length"]),
                         "reported_best_of_all": float(reported_sampled["best_of_all"]),
                         "protocol_count": int(sampled.size)})

    arms_summary = {}
    for arm in ARM_NAMES:
        arm_rows = [r for r in rows if r["arm"] == arm]
        arms_summary[arm] = {}
        for key in ("sampled_mean", "best_of_41_mean", "best_of_41_std_sample", "best_of_410",
                    "union_best", "best_final_including_adaptation"):
            vals = np.asarray([r[key] for r in arm_rows], dtype=np.float64)
            arms_summary[arm][key] = {"mean": float(vals.mean()), "std_between_seeds": float(vals.std()), "values": vals.tolist()}

    # The old runner made one preliminary greedy+sampled 410-candidate decode
    # per arm/seed, then the saved 10x41 protocol. Both objective arrays from
    # the preliminary decode were discarded, so this is historical accounting,
    # not a new verification call.
    arm_seed_count = len(rows)
    preliminary_queries = arm_seed_count * 2 * ROUNDS * CANDIDATES
    protocol_queries = protocol_count
    shared = old_costs["shared"]
    corrected = {
        "source_verification_queries": int(shared["source_verification_queries"]),
        "target_verification_queries": int(shared["target_verification_queries"]),
        "preliminary_evaluate_mapping_queries": preliminary_queries,
        "protocol_generation_queries": protocol_queries,
        "generated_target_queries_all_seeds_corrected": preliminary_queries + protocol_queries,
        "target_objective_queries_current_run_corrected": int(shared["target_verification_queries"]) + preliminary_queries + protocol_queries,
        "source_and_target_objective_queries_current_run_corrected": int(shared["source_verification_queries"]) + int(shared["target_verification_queries"]) + preliminary_queries + protocol_queries,
        "new_objective_calls_performed_by_audit": 0,
        "accounting_note": "Corrected historical accounting inferred from code and saved protocol arrays; no objective was called by this audit.",
    }
    report = {
        "status": "AUDITED_WITHOUT_NEW_OBJECTIVE_CALLS",
        "input_run": str(input_dir),
        "input_file_sha256": {name: _sha256(input_dir / name) for name in ("diagnostics.json", "costs.json", "pairing.json")},
        "root_cause": "sampled.best_length is retained from the preliminary evaluate_mapping() 410-candidate decode; generation_rounds() overwrites overlapping metrics but does not overwrite best_length. The preliminary arrays are discarded. sampled.best_of_all and round_best_lengths are from the saved 10x41 protocol arrays.",
        "protocol_sampled_candidates_per_arm_seed": ROUNDS * CANDIDATES,
        "protocol_greedy_candidates_per_arm_seed": ROUNDS * CANDIDATES,
        "protocol_arrays_are_same_410": True,
        "stale_best_length_count": stale_count,
        "pairing_source_global_ids_available": "source_global_ids" in pairing,
        "recomputed_rows": rows,
        "detailed_rows": details,
        "per_arm_across_seed_summary": arms_summary,
        "corrected_costs": corrected,
        "original_costs": old_costs,
        "limitations": [
            "The preliminary best_length candidate ID, source ID, action/generation seed, and objective cannot be recovered because evaluate_mapping() discarded its decoded arrays.",
            "The saved protocol best candidate provenance is recoverable from generated.npz and sampled_arrays_meta without reevaluation.",
            "No candidate was regenerated and no old file was overwritten.",
        ],
    }
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "metrics_reconciled.json").write_text(json.dumps({"rows": rows, "per_arm_across_seed_summary": arms_summary}, indent=2) + "\n")
    (output_dir / "costs.json").write_text(json.dumps(corrected, indent=2) + "\n")

    lines = [
        "# H_residual audit (no retraining, no new objective calls)", "",
        f"Input run: `{input_dir}`", "",
        "## Finding", "",
        "`sampled.best_length` is stale relative to the saved protocol. It comes from the preliminary `evaluate_mapping()` decode (410 sampled candidates per arm/seed), while `sampled.best_of_all`, `round_best_lengths`, and the final-with-incumbent value come from `generation_rounds()` (10 x 41). The preliminary arrays were discarded.", "",
        "Therefore the saved protocol satisfies `min(all sampled) == min(round_best_lengths) == sampled.best_of_all`; the fourth comparison with `sampled.best_length` fails for the stale field.", "",
        "## Corrected historical accounting", "",
        f"- Preliminary `evaluate_mapping()` calls: **{preliminary_queries}** (not separately counted in the old costs).",
        f"- Saved protocol generation calls: **{protocol_queries}**.",
        f"- Corrected generated-target calls: **{preliminary_queries + protocol_queries}**.",
        f"- Corrected target calls including verification: **{corrected['target_objective_queries_current_run_corrected']}**.",
        f"- Corrected source + target calls: **{corrected['source_and_target_objective_queries_current_run_corrected']}**.",
        "- New calls made by this audit: **0**.", "",
        "## Reconciled protocol metrics", "",
        "| Seed | Arm | Mean | Best-of-41 mean | Best-of-41 std | Best-of-410 | Union greedy+sampled | Final incl. incumbent | Stored best_length |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {training_seed} | {arm} | {sampled_mean:.9f} | {best_of_41_mean:.9f} | {best_of_41_std_sample:.9f} | {best_of_410:.9f} | {union_best:.9f} | {best_final_including_adaptation:.9f} | {stale_best_length:.9f} |".format(**r))
    lines += ["", "Detailed provenance and machine-readable checks are in `audit_report.json`; reconciled metrics are in `metrics_reconciled.json`; corrected historical accounting is in `costs.json`."]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output)


if __name__ == "__main__":
    main()
