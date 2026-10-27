# Within-rank shuffle — three-branch, three-seed diagnostic

Implementation of the requested experiment on the EXISTING TSP→CVRP data.
No new architecture, objective, problem family, target pretraining or data
instance is introduced. Prior baselines remain historical and are NOT reused
as results of this experiment. Novelty remains undecided.

## Files and dependencies

Five new code/protocol files:

- src/cvrp_within_rank.py: train-fit quality bins, audited stratified shuffle,
  target-label-free source plan/decode, round and seed aggregation.
- src/experiment_cvrp_within_rank.py: one invocation for all 9 training jobs.
- configs/cvrp_within_rank.json: locked data/baseline hashes, seeds and budgets.
- tests/test_cvrp_within_rank.py: 11 new tests.
- docs/CVRP_WITHIN_RANK.md: this guide.

The separate PROJECT_STATE.md in the package is the updated project snapshot.
The patch does NOT overwrite an existing PROJECT_STATE.md on WSL; the runner
updates/appends its protocol marker after successful production completion.
Existing CVRPVAE/decoder/bridge type, train_mapping(), optimizer and checkpoint
selection are reused without modifying their source. All three arms have the
same linear bridge width, initialization, target adaptation data and schedule.

## Fixed production protocol

| Item | Setting |
| --- | --- |
| Arms | source, within_rank_shuffle, rank_only |
| Training seeds | 1088,1089,1090 |
| Pairing/data split seed | 1088, fixed across all seeds/arms |
| Epochs/batch/Adam LR/grad_clip | 30 /128 /.001 /1 |
| Hidden/target latent/bridge | 64 /16 /16→16 |
| Target | Original CVRP20 seed2088 NPZ, capacity30 |
| Quality bins | 10 decile groups, fitted on SOURCE TRAIN ONLY |
| Shuffle RNG | SeedSequence [73001,training_seed,partition_number] |
| Source selection RNG | 94001, fixed across all training seeds/arms |
| Sampling action RNG | SeedSequence [95001,training_seed,round_number] |
| Candidate selection | Uniform permutation of source validation IDs, no target labels |
| Rounds /K | 10 /41, disjoint source IDs across rounds; covers410 |
| Per arm/seed objective calls | 410 greedy +410 sampled =820 |

Data split/pairing matches the previous prepare_pairs(seed1088) protocol; source
standardization and scalar rank use only source training rows. Each training
seed resets initialization immediately before creating the common decoder and
bridge templates. All arms clone these weights. The previous seed1088 result
need not be numerically identical because the old runner let source-model
construction consume its global initialization RNG; this experiment reruns ALL
three branches consistently. Do not pool old runs with this one.

## Shuffle construction and audit

Use actual higher-is-better source quality (-TSP distance). Fit raw quality
quantiles .1,.2,...,.9 on source TRAIN. Assign both partitions via searchsorted
with side=right using the same nine cutpoints. Equivalent qualities stay in the
same bin; tied boundaries can yield empty bins. Within each bin/partition draw
a uniform permutation, fixed for the entire run. Train and validation have
independent RNG streams; validation never determines train boundaries or
consumes train RNG draws. Empty/singleton bins are left unchanged. No forced
derangement is imposed or resampled to improve the moved fraction.

Report train and validation audits independently:

- Train quality cutpoints and requested percentile boundaries0..100.
- Midpoint empirical source-train CDF before/after assignment, in percentile
  POINTS (0..100), not a relative percent difference.
- Mean/max/p95 absolute drift, mean signed drift and RMSE.
- moved_count/fraction (donor row ID differs); changed_latent_fraction (actual
  normalized condition differs); counts/movement/drift for every group.
- Saved arrays: permutation, group IDs, original/assigned quality and percentile.

A permutation preserves each group's latent multiset and per-partition marginal
distribution, but allows nonzero within-bin quality/rank drift. This is a
coarse-quality diagnostic; any effect is not automatically pure structural
information beyond exact scalar rank. Ties and fixed points are disclosed.

## Validation versus generation (important distinction)

Validation reconstruction uses the arm's intended condition: true source,
independently within-bin-shuffled validation source, or original scalar rank.
Checkpoint selection remains the MINIMUM real-action-weighted validation NLL
returned by the existing train_mapping(). Padding is ignored exactly as before.
An independent NLL recheck of the selected checkpoint is asserted. Generated
costs never select a checkpoint, epoch, hyperparameter, source policy or seed.

At generation, the shuffled-training model receives ORIGINAL source conditions
at the SAME source IDs as the source model. Rank-only receives those IDs' original
train-reference percentile values. Do not feed the validation permutation into
generation. Selection takes only source pool IDs, rounds/K and a fixed RNG seed;
no target quality or target solution is accepted. Decode/evaluation takes no
adaptation incumbent or target labels. Adaptation-incumbent metrics are added
only AFTER generation. CVRP coordinates/demands/capacity are required inputs.

The410 source validation IDs are shuffled once and partitioned into10 distinct
sets of41. This source plan is shared across arms AND training seeds. Each
round draws stochastic decoder actions with a distinct seed, shared across arms
within a training seed. Greedy uses exactly the same410 IDs and is reported with
its own410-call budget. Repeated generated solutions are retained and count as
objective calls. No repair, deduplication/resampling or early stopping by cost.

## Metrics / adaptation incumbent

Per seed and arm, both modes report mean length, average route count, validity,
10 round minima, mean/std(best-of-41), and best-of-410. Standard deviation over
10 rounds uses ddof=1. Sampling metrics are the requested primary round metrics;
greedy round metrics are also saved for audit.

Best adaptation includes all2048 selected support labels: train1638 plus
validation410, because validation labels are already used to pick checkpoints.
Train-only, validation-only and all-loaded-data minima are recorded separately.
For each mode, best_final_including_adaptation=min(best_support,best_generated).
The union of greedy+sampling is reported separately as an820-call result; it
must not be confused with the410-call sampling result. The actual incumbent
solution is saved once in adaptation_incumbent.npz.

Across three training seeds, diagnostics retains every seed and reports mean/std
(ddof1) of per-seed quantities. It does not treat30 round minima as30 training
seeds. Paired source-minus-control deltas are provided, with negative length
differences favoring source. There is no benchmark/significance/novelty claim.

## Run on the user's WSL

Extract MEGO-CVRP-Within-Rank.zip. Substitute the actual patch path.
Use the SAME existing source checkpoint and data NPZs; do not regenerate data.

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-cvrp-within-rank.patch
git apply /absolute/path/mego-cvrp-within-rank.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_cvrp_within_rank.py' -v
.venv/bin/python -m src.experiment_cvrp_within_rank \
  --device cuda:0 --output logs_cvrp/within_rank_shuffle_v1
```

Tests should say Ran11 tests ... OK. The runner prints9 branch summaries, then:

```text
WITHIN-RANK SHUFFLE DIAGNOSTIC COMPLETED
diagnostics: logs_cvrp/within_rank_shuffle_v1/diagnostics.json
summary: logs_cvrp/within_rank_shuffle_v1/summary.md
```

Default inputs are the same source/target paths used by the completed CVRP run.
Paths can be overridden if files moved, but byte hashes must still match. The
runner checks recorded hashes of source checkpoint, source NPZ, target NPZ,
baseline config and baseline code, and detects modifications while running.
A mismatch stops before training; inspect it rather than replacing the data
or weakening the guard. No dependency reinstall, new expert pretraining or
model changes are needed. Existing output directories are refused; no resume
or old-checkpoint-reuse option is provided. This runs9 models×30epochs.

The completed GPU command updates/appends the bounded protocol section in
PROJECT_STATE.md. Other sections are preserved. --project-state can point to
another local project-state path. The updated snapshot supplied with the ZIP
records implementation status; it does NOT claim the nine GPU runs have occurred.

## Saved layout

Root output:

- config.json: resolved protocol and all configured seeds/hashes.
- pairing.json: existing split/rank pairing and normalization parameters.
- shared_plan.npz: train/validation rows, original source/target IDs, common
  generation source IDs (10×41), original latent/rank conditions.
- adaptation_incumbent.npz: common known best support solution and its cost.
- diagnostics.json: all9 results, bin audits, seed aggregates/paired deltas,
  initial-weight/batch-order hashes, code/data/commit and runtime metadata.
- costs.json: shared adaptation/data provenance and individual arm/seed costs.
- summary.md: per-seed comparison, best-of-K statistics and adaptation-aware best.
- project_state_update.md: status/results section also written to project state.

Per seed:

- shuffle_audit.json/npz: fit boundaries, movement/drift, actual donor mapping.
- diagnostics.json: all3 arm results for this seed.

Per seed/arm:

- config.json, history.json, diagnostics.json, costs.json.
- mapping_checkpoint.pt, selected only by validation NLL.
- generated.npz: every greedy/sampled padded sequence, true length, route count,
  objective value, and source original IDs aligned round-by-round.

Send root diagnostics.json, costs.json and summary.md first. Per-arm history
and shuffle_audit files remain available for follow-up inspection.

## Budget accounting

Shared once for the FULL experiment:2048 imported adaptation labels,2048 target
verification calls,2048 source verification calls. Existing label-creation/
reload/source-pretraining costs are provenance, not zero and not charged9 times.
There are no new objective calls per gradient or validation-NLL step.

Per arm/seed:410 greedy +410 sampling =820 target calls. Nine arm/seeds use
7380 target generation calls; current target total=2048+7380=9428;
source+target current total=11476. No random-reference generation is added.
Adaptation incumbent selection and writing generated outputs reuse known costs.
Training+validation wall time and generation/evaluation time are recorded per
arm; initialization/logging overhead is not a claimed speedup comparison.

## Checks actually run by the assistant

11 new tests passed; all66 existing/new Python tests passed on CPU. Full pipeline
smoke exercised ALL3seeds×ALL3arms, but only1epoch/64pairs/2rounds×K3. It used a
local CPU-trained TSP checkpoint, not the user's unavailable GPU checkpoint.
Smoke explicitly disables production input-hash checks, retains baseline code
checks, marks every result SMOKE_TEST_ONLY and never updates production state.
These numbers are NOT results of the requested diagnostic.

No access to /home/minh1234/MEGO, no CUDA device run, no full9×30epoch experiment
with the exact user inputs has been performed by the assistant. The production
command above is ready for WSL. No new scientific result is claimed before it.

Tests cover train-only cutpoints, independent partition RNGs, no cross-bin/split
moves, exact latent multisets, fixed shuffle seeds, ties/empty bins, actual moved
fractions and percentile drift, target-label-free/common source generation,
source IDs invariant to changing target-quality order, repeated RNG replay,
feasible generation/query counts, K-minima/std/incumbent arithmetic, paired
initial weights and batch order, full-versus-smoke protocol, and state updates.
