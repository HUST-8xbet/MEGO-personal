# MEGO D-C — TSP -> CVRP mapping with trained controls

## Expert CVRP result received from WSL

The 30-epoch CVRP expert completed successfully; best total validation loss at
epoch 30. All reported generated solutions satisfy evaluator constraints.
Validation NLL decreased 2.04800 -> 1.83800; standardized quality MSE decreased
1.15983 -> 0.21761. On 128 diagnostic solutions, quality Pearson is 0.89486;
MAE 0.43360 vs training-mean baseline 1.05792. Aligned/shuffled/zero teacher NLL:
1.85219 / 1.89565 / 1.88394. Full canonical reconstruction remains 0%.
Aligned sampling mean distance 14.35422 vs random feasible reference 14.20714.
Prior greedy/sampling mean distances 15.61995 / 14.63312. Thus the expert works
functionally and learns a quality signal, but optimization benefit is unproven.
This does not require additional expert epochs before the technical mapping.

## Scope and new files

MEGO remains the framework; novelty is undecided. This patch adds five files:

- src/cvrp_mapping.py: rank pairs, training, evaluation and capacity-safe decode.
- src/experiment_cvrp_mapping.py: TSP source loading and controlled runner.
- configs/cvrp_mapping_smoke.json: default single-seed 30-epoch configuration.
- tests/test_cvrp_mapping.py: five mapping tests.
- docs/CVRP_MAPPING.md: this review and guide.

Existing source files/checkpoints/logs are unchanged. Requires the previously
installed TSP expert, CVRP D-A/D-B modules and tsp_mapping_controls.py. It reuses
that module's ARMS list and decoder-parameter selection, not its TSP evaluator.

TSP20 source has 20 vertices; CVRP20 target has 20 customers PLUS the depot.
A learned linear bridge maps the actual source latent width (normally 16) to
the CVRP decoder width (16). Source expert posterior means are frozen. The
mapping trains the CVRP city embedding, decoder and bridge, with masked
teacher-forced real-action NLL only; no target encoder/score/KL training is used.
The city embedding is shared with the unused target encoder, so its complete
function is not claimed to remain fixed. CVRP feasibility uses the unchanged
D-B integer capacity and depot masks; no repair or extra search is used.

## Source/target separation and target decoder cost

Source: existing TSP expert logs_tsp/tsp20_expert_smoke_01/best_model.pt and its
original data/tsp_milestone_a/tsp20_seed1088.npz. The loader checks the training
data hash recorded in that checkpoint, verifies stored source quality by the
TSP evaluator, canonicalizes/deduplicates source cycles and batches encoding.
It uses posterior MEANS, freezes all source parameters and never trains source.

New default target: cvrp20_seed2088.npz (instance seed 2088, solution seed 2089).
Using the same coordinate RNG seed for TSP20 and CVRP20 would make their first
20 coordinate rows identical, even though CVRP has one extra vertex. A different
seed avoids that incidental shared geometry in this diagnostic. Exact source
vertex overlap is recorded in diagnostics; this is not a statistical guarantee
of independent problem distributions or a held-out-family benchmark.

All target decoders are initialized FRESH, sharing exactly the same initial
weights and per-arm initial bridge weights. The existing CVRP expert trained
on seed 1088 is not used as target pretraining. Its architecture/feasibility has
been verified; its weights are deliberately excluded from this comparison.
Therefore only GENERATE the new target data; do NOT train another CVRP expert
on seed 2088 before this mapping. The necessary target decoder adaptation cost
is part of this run and is not hidden as free target-expert training.

## Five independently trained arms

| Arm | Training AND evaluation condition |
| --- | --- |
| source | Actual source latent, standardized by source training rows |
| shuffled_train | Fixed UNIFORM random permutation within each partition |
| noise_train | Independent Gaussian bank in standardized active dimensions |
| rank_only | Only source-quality percentile, tied scores receive same scalar |
| no_source | Zero target latent, no bridge |

Every arm uses the same CVRP solutions/split, decoder initialization, optimizer,
epoch count and batch order. First four arms have the same input/bridge sizes;
no-source has no bridge. Noise is a same-width stochastic control but does not
match full source covariance. Noise and shuffled conditions are present during
training, not only injected at evaluation. Source quality is -TSP length and
target quality is -CVRP total length: both are sorted descending, not compared
in raw units. Rank-only's empirical percentile reference uses source TRAINING
qualities only. Target canonical equivalence removes route reversals and vehicle
order before splitting; source cycles are deduplicated as well. Both partitions
are assigned first and rank-matched separately; no target cycle crosses splits.

Validation NLL selects best checkpoint per arm. It averages REAL target actions,
including real depot returns; -1 padding is excluded. Validation target labels
form paired reconstruction diagnostics; generation uses source conditions and
the target instance, not validation quality labels. Validation is not a final
test set. Each trained arm generates greedy AND sampled solutions using the
same number of candidates/evaluator calls per mode. Compare within a mode;
common sampling seeds couple action randomness and are not independent trials.

This is TSP -> CVRP routing/constraint transfer. It is not by itself transfer
between unrelated combinatorial families, unlike the earlier Max-Cut -> TSP
functional connection. Neither connection alone establishes useful transfer.

## Apply, generate independent target, run on WSL

Extract MEGO-CVRP-Mapping.zip. Substitute the actual patch location.

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-cvrp-mapping.patch
git apply /absolute/path/mego-cvrp-mapping.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_cvrp*.py' -v
.venv/bin/python -m src.experiment_cvrp_data \
  --dimension 20 --capacity 30 --sample-num 2048 \
  --instance-seed 2088 --solution-seed 2089 \
  --output data/cvrp_milestone_a/cvrp20_seed2088.npz
.venv/bin/python -m src.experiment_cvrp_mapping \
  --output logs_cvrp/mapping_tsp_to_cvrp_01 \
  --device cuda:0
```

Expected test summary: Ran 30 tests ... OK (14 data + 11 expert + 5 mapping).
The data generator should print CVRP DATA CHECK PASSED with (2048,41) routes.
The runner prints five branch summaries, then:

```text
CVRP MAPPING FUNCTIONAL RUN COMPLETED
pair_count: 2048
diagnostics: logs_cvrp/mapping_tsp_to_cvrp_01/diagnostics.json
```

Do not expect any particular branch to win. There are five models x30 epochs;
this takes longer than the single expert run. CPU: --device cpu. A short wiring
run can add --epochs 2 --max-pairs 64 and choose a different output directory.
--arms source (or another subset) allows a targeted run if debugging; the default
trains all five. Query counts adjust to selected arms. Output files/directories
are not overwritten, and there is no resume option. If new target data already
exist, use that file rather than regenerating it at the same path.

If source files were moved, pass --source-checkpoint and --source-data. If the
source-data hash check fails, use the exact original TSP NPZ the checkpoint was
trained on; do not silently substitute a new source instance or regenerated ZIP.
--target-data can select another CVRP instance, with geometry overlap recorded.

After the full run, send diagnostics.json, history.json and costs.json. That
is the point to summarize TSP and CVRP results together for Research & Decisions
and Experiments & Results, separating functional completion from evidence of
source utility. No extra source-expert optimization is a prerequisite here.

## Artifacts and budgets

Per-arm .pt files contain best decoder/bridge state, dimensions/config/arm.
conditioning.json/npz retain transforms, ranks, permutations and every condition.
split.npz stores paired rows, original source/target row identities and split.
generated.npz stores both decoding modes, sequence lengths and already-evaluated
costs. diagnostics.json records hashes, commit, device/versions, geometry
comparison, scopes and objective counts; history.json stores real-token-weighted
NLL curves; costs.json separates source/target and current/prior calls.

For default 2048 rows each, 410 validation solutions and five arms:

- Source verification: 2048 source objective calls in this invocation.
- Target verification: 2048 target objective calls.
- Target evaluation: 5*2*410 + 410 random-reference calls = 4510.
- Target current-run total: 6558; all-objective current-run total: 8606.
- Gradient steps and saving generated arrays: zero objective calls.

The new target data generator separately uses 2048 generation and 2048 reload
calls; these are copied from its metadata into prior-data provenance fields.
Source expert pretraining and earlier CVRP experiments are not silently counted
as zero or added multiple times. Stored labels are used by all arms, but their
creation is not repeated per gradient step. No wall-clock speedup or query-efficiency
claim is made from this small functional run.

## Local verification boundary

CPU PyTorch 2.5.1+cpu: all 30 CVRP tests and all 55 project tests passed. A
2-epoch / 64-pair CLI run exercised all five branches using an actually trained
local TSP expert, with source/target file validation, capacity-safe generation,
checkpoint writes and cost accounting. This is a wiring test, not the full
30-epoch/2048-pair comparison. No assistant-side CUDA run or user-checkpoint
execution is claimed. Python 3.8 syntax was checked; the user's PyTorch 2.0.1
CUDA runtime needs the commands above.

Tests cover source posterior-mean loading/hash guard and deduplication,
source/target disjoint splits and descending pairing, train-only normalization,
uniform shuffling (not cyclic shift), tied source ranks, bridge/decoder gradients,
frozen unused encoder/score parameters, integer capacity/service feasibility,
sampling diversity and equal evaluation budgets. Exact full reconstruction is
a diagnostic, not a gate that must be optimized indefinitely before mapping.
