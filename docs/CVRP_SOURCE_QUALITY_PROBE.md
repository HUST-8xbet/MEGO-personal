# TSP -> CVRP: fixed-checkpoint source-quality diagnostic

Question: do better source solutions produce better target solutions, and is
the relationship stronger/useful beyond scalar-rank conditioning? This is a
descriptive probe on the current single CVRP20 instance. No training, checkpoint
selection, temperature tuning, architecture/loss changes or group selection
based on target results occurs. All five groups are evaluated and retained.

## Files changed

Five NEW files only; existing architecture/training/decoder files stay intact:

- `src/cvrp_quality_probe.py`: source-only plan, frozen decode, metrics/Spearman.
- `src/experiment_cvrp_quality_probe.py`: reuse all nine existing checkpoints.
- `configs/cvrp_quality_probe.json`: locked protocol and parent/data checksums.
- `tests/test_cvrp_quality_probe.py`: 11 functional checks.
- `docs/CVRP_SOURCE_QUALITY_PROBE.md`: this guide.

Requires the COMPLETE existing `logs_cvrp/within_rank_shuffle_v1` directory:
root diagnostics, shared_plan, pairing, adaptation_incumbent and each of the
nine arm checkpoint/history files plus three shuffle_audit.npz files. No source
expert loading/retraining is needed: reuse its original normalized latent bank.
The original TSP/CVRP data NPZs must still be present, with recorded checksums.

## Run on WSL

Extract the ZIP and substitute the actual patch path:

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-cvrp-source-quality.patch
git apply /absolute/path/mego-cvrp-source-quality.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_cvrp_quality_probe.py' -v
.venv/bin/python -m src.experiment_cvrp_quality_probe \
  --run-dir logs_cvrp/within_rank_shuffle_v1 \
  --device cuda:0 --output logs_cvrp/source_quality_quintile_v1
```

Expected: 11 tests OK; the runner prints five validation pool counts, selected
count20, nine arm/seed lines, then `SOURCE_QUALITY_PROBE_COMPLETED`.
No expected numerical relationship or sign is imposed on the results.
If paths moved, `--source-data`/`--target-data` can be overridden; the same byte
hashes must match. A checksum mismatch stops; inspect it rather than weakening
the guard. Missing files are reported explicitly. Existing output paths are
refused. Checkpoint files and every read-only dependency are hashed and checked
again after generation. State-dict digests are unchanged before/after inference.

Production is pinned to the completed 3-seed within-rank diagnostics uploaded
for review, including its byte checksum. Checkpoint arm/config/input/code/epoch,
initialization/batch-order metadata and minimum-NLL history are checked against
that report. No best generated result selects a checkpoint or configuration.

## Source plan and fairness

- Quality is `-TSP length`; larger quality/percentile means better source.
- Fit linear quality quantiles20/40/60/80% on **source train only**. Apply the
  same boundaries to source validation with `searchsorted(side='right')`.
  Ties stay together; validation groups need not have equal sizes.
- Report train and validation pool counts. If ANY validation group has <20
  rows, print the counts and STOP BEFORE generation. No replacement, bin merge,
  smaller group, or alternative selection is performed automatically.
- Q1 worst -> Q5 best. Select20 distinct validation IDs uniformly without
  replacement in each group using SeedSequence `[96001,group_index]`.
  The resulting100 IDs/order remain identical across arms AND training seeds.
- Train-reference midpoint empirical CDF percentile is saved in0..100 points.
  The rank-only decoder input retains the original `2*CDF-1` scalar with the
  other input dimensions zero. Both source and within-rank-shuffle models get
  the corresponding ORIGINAL source latent, not a shuffled validation latent.
- On each group, sample five batches of20. Action seed is derived from
  `[97001,training_seed,group_index,round_index]`, identical across arms.
  Greedy decodes each selected latent once, using the first round seed (greedy
  itself is deterministic). Original categorical decoder logits/masks apply;
  no temperature argument or changed decoding rule is introduced.
- Source selection accepts source data only. Target file reads geometry,
  demands and capacity only, not adaptation routes or target quality arrays.
  A separate incumbent file is verified once and used only in postprocessing.

## Metrics and saved outputs

Per group x arm x training seed (45 cells): mean greedy/sample length, mean
route count, validity, five round best-of-20 lengths, their mean and sample
std(ddof1), strict improvement counts/rates over the EXACT original adaptation
incumbent9.399715696516127 (display9.399716), generated best and final best with
adaptation. Report greedy(20 calls), sampling(100) and union(120) separately.
The incumbent stays fixed; prior generated best9.328974 is not substituted for
the requested adaptation incumbent. No early stopping by target cost.

Spearman is reported separately for every arm/training seed over100 selected
latents: source percentile versus EACH latent's mean target length over its
five samples. Ties use average ranks. Negative is the desired direction.
This is a balanced stratified validation selection, not a population-weighted
estimate over all410 validation sources, and not500 independent observations.
Constant inputs yield null with a reason; no iid p-value is reported. Paired
source-minus-control group mean differences and descriptive rho differences
are saved. A negative rho by itself cannot establish benefit beyond rank-only.

All invalid candidates are retained, flagged, counted in the attempted budget
and treated as no improvement; no repair/retry. Length/route means use valid
candidates and disclose denominators. An all-invalid round has null best-of-K
statistics; Spearman is undefined if any selected latent has invalid samples,
so the runner never silently drops inconvenient observations.

Output root:

- `config.json`, `selection_audit.json`, `source_plan.npz` (all boundaries,
  partitions, group IDs, selected source IDs, quality, percentile and conditions).
- `diagnostics.json`: all45 cells, five round minima each, nine Spearman values,
  checkpoint/dependency hashes, paired deltas, aggregates across training seeds.
- `group_results.csv`:45 rows, including greedy/sample route counts, validity,
  incumbent rates, generated/final bests and sampling-round std.
- `summary.md`: per-group results, per-seed Spearman and separate seed aggregates.
- `costs.json`: shared verification and each arm/seed generation cost/time.
- `project_state_update.md`: production result section; after success the runner
  also appends/replaces only its own bounded section in `PROJECT_STATE.md`,
  preserving prior protocol/results. Override location with `--project-state`.
- `seed_<seed>/<arm>/generated.npz`, diagnostics/costs: every raw padded route,
  true sequence size, length, route count, validity, source ID/group alignment,
  percentile, per-latent five-sample mean and sampling action seeds.

Within-seed round std (five draws) and between-training-seed std (three runs)
are named separately; seed aggregates never treat15 sampling rounds as15 seeds.
No results are pooled with the earlier410-candidate diagnostic.

## NEW costs

Per arm/seed:5 groups x(20 greedy+20x5 sampling)=600 new candidates.
All nine:900 greedy+4500 sampling=5400 new objective attempts, plus ONE shared
incumbent verification:5401 target attempts total if production completes.
No new adaptation data, optimizer epochs, validation-NLL evaluations, source
objective calls, or full2048-row target verification pass. Historical training,
adaptation labels and earlier generation are recorded as prior costs, not zero
and not charged again. Invalid attempts/repeated solutions still consume budget.

## Assistant verification versus actual experiment

Tests cover full20x5 budget/feasibility, RNG replay, train-only quintiles,
original source/rank inputs, shortage halt, tied percentile Spearman, per-latent
sample means, invalid outputs, strict incumbent rates, separate seed std,
checkpoint metadata rejection, frozen parameters and state preservation.
Local smoke reuses all nine existing CPU SMOKE checkpoints with only ONE
latent/quintile x TWO rounds. It is explicitly marked SMOKE_ONLY_NOT_EXPERIMENT,
still validates the actual smoke checkpoint/data/code provenance, and never
updates production PROJECT_STATE. No GPU experiment has been run by the
assistant; the user's nine production checkpoints remain on WSL.
