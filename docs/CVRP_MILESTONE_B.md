# MEGO CVRP20 — D-B: instance-conditioned expert

D-A passed on the user's WSL: all 14 data/evaluator tests, 2048 feasible
solutions and NPZ save/reload verification. This patch adds the first CVRP
encoder-latent-decoder expert. MEGO remains the base; novelty is undecided.

## Files added (existing source unchanged)

- src/regression/cvrp_vae.py: CVRPVAE and solution canonicalization.
- src/experiment_cvrp_expert.py: single-instance train/validation runner.
- configs/cvrp_expert_smoke.json: 30 epochs, hidden 64, latent 16, batch 128.
- tests/test_cvrp_expert.py: 11 expert tests.
- docs/CVRP_MILESTONE_B.md: this guide.

Depends on CVRP D-A, including CVRPProblem and its padding codec. No TSP model,
legacy binary model, mapping, or previous control code is modified. Does not
require retraining TSP or Max-Cut. No CVRP transfer/mapping is included yet.

## Architecture and feasibility

City features are x, y, demand/capacity and a depot indicator. The input retains
integer demands and capacity; only network features are normalized. An MLP
embeds cities, and a packed GRU reads route tokens, including real depot returns
but excluding -1 padding. Its final state plus the pooled instance embedding
produce posterior mean/log variance. A score head predicts standardized quality
(-length) from latent; its final layer is linear.

The autoregressive GRU pointer decoder sees current city, latent and remaining
capacity fraction. Its exact state tracks visited customers and remaining
INTEGER capacity. At every step:

- Mask visited customers and any customer whose demand exceeds remaining load.
- Forbid depot at depot, preventing empty vehicle routes.
- Allow a depot return from a customer, resetting remaining capacity.
- Once all customers are served, return to depot and finish.
- Completed batch rows emit -1 padding. A safe dummy depot distribution keeps
  ignored logits finite at their valid dummy entry, avoiding all-masked NaNs.

All customers have demand <= capacity, as enforced by D-A. Every departure from
the depot must serve an unvisited customer, so completion takes at most 2*n
actions after the initial depot. The implementation raises if this bound is
violated; it does not truncate, repair or silently drop customers. Validity is
then verified by the independent D-A evaluator/unpack_solutions, including every
capacity and service constraint. Greedy and categorical-sampling modes use the
same feasibility mask. Solutions support variable route counts and true lengths.

## Data, symmetry, loss and split

Canonicalization reverses each vehicle route if that gives a lexicographically
smaller customer sequence, then sorts vehicle routes lexicographically. This
removes direction/vehicle-order representation duplicates before a seeded
80/20 split. It does not merge different vehicle partitions, relabel customers
or implement a symmetry-invariant neural network. Original row indices are
saved in split.npz. Quality normalization mean/std uses training solutions only.

Loss = real-action NLL + 0.001*KL + quality MSE. NLL averages over actual target
actions, INCLUDING each real return to depot and the final depot, and excludes
all -1 targets. KL sums across latent dimensions then averages across solutions.
Epoch reporting aggregates NLL by real-token count, and KL/score error by sample
count. Best checkpoint is chosen by validation TOTAL LOSS. Validation uses
posterior means, while training samples the posterior.

This is one CVRP instance with held-out solutions, not held-out instances or an
optimization benchmark. It learns from feasible random solutions, not optimal
solver labels. Reconstruction/score/latent diagnostics must be read separately
from functional completion. Exact reconstruction is not a mandatory threshold
for moving to the technical mapping step.

## Apply and train on WSL

Extract MEGO-CVRP20-Milestone-B.zip. Replace the patch path below. No dependency
reinstall is needed in the existing PyTorch environment.

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-cvrp20-milestone-b.patch
git apply /absolute/path/mego-cvrp20-milestone-b.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_cvrp*.py' -v
.venv/bin/python -m src.experiment_cvrp_expert \
  --data data/cvrp_milestone_a/cvrp20_seed1088.npz \
  --output logs_cvrp/cvrp20_expert_smoke_01 \
  --device cuda:0
```

A silent git apply --check with exit status 0 means it passed. Inspect conflicts
rather than forcing. Existing output directories are refused; use a new output
name for another run. CPU option: --device cpu. A shorter wiring-only run can
add --epochs 2 --max-samples 256 and use a different output directory.

Expected test summary: Ran 25 tests ... OK (14 data + 11 expert tests).
Training prints INITIAL and epoch lines. Exact numerical losses can vary across
runtime/device. Expected final lines:

```text
CVRP EXPERT FUNCTIONAL RUN COMPLETED
best_epoch: <integer from 1 to 30>
prior_greedy_valid_rate: 1.0
prior_sampled_valid_rate: 1.0
checkpoint: logs_cvrp/cvrp20_expert_smoke_01/best_model.pt
diagnostics: logs_cvrp/cvrp20_expert_smoke_01/diagnostics.json
```

Send diagnostics.json and history.json from this new directory for review.
No need to upload the checkpoint unless a later diagnostic requires it.

## Output contract and interpretation

- best_model.pt: best weights, model config, quality scaling, epoch, validation.
- config.json: resolved config, seeds, device/versions, source/data hashes, base
  commit, sample counts, capacity, split convention and normalization.
- split.npz: original dataset indices after canonical deduplication and split.
- history.json: train/validation loss components and actual target token counts.
- diagnostics.json: best-model reconstruction, score, latent and generation.
- generated.npz: coordinates/demands/capacity, and for each case/mode its padded
  output, sequence lengths and already-evaluated objective lengths.

Diagnostics use up to 128 validation solutions. aligned, shuffled (uniformly
permuted posterior means), and zero are INFERENCE PERTURBATIONS of the same
trained expert. They are not independently trained mapping controls. Their
teacher-forced NLL and fixed-prefix probability changes probe latent use.
On one fixed instance, shuffled greedy latents give the same output multiset
as aligned; aggregate lengths are not a test of correct pairing. Compare paired
reconstruction and teacher-prefix NLL instead. Chance permutation fixed points
are allowed. Prior uses N(0,I) latents in the target expert's latent space.

Each of the four cases is decoded both greedily and by sampling. Reports include
validity, mean/best objective, mean number of vehicles, route-count MAE,
canonical exact-solution rate, distinct solutions, and undirected edge-MULTISET
F1 (single-customer routes traverse the depot edge twice, so counts matter).
Vehicle count is descriptive, not an additional objective. Compare greedy with
greedy and sampled with sampled. Sampling uses a common action RNG seed.

Score error is reported in distance units, with a training-mean baseline and
Pearson correlation (null when target/predictions are constant). The random
reference uses the nonuniform D-A sampler with its default early-return rule.
This is a functional diagnostic rather than a fair paper-level solver comparison.
Prior latent distributions can differ from learned posterior aggregates.

## Objective accounting and validation scope

Input revalidation costs one evaluator call per stored solution. Gradient
training uses stored labels, with zero new objective calls. Four latent cases
x two generation modes x diagnostic count, plus one random-reference batch,
use 9*diagnostic_count evaluator calls. Saving generated.npz reuses costs
already computed and adds none. For 2048 input and 128 diagnostic solutions:
2048 + 9*128 = 3200 calls in this expert invocation. Prior data generation,
D-A reload checks, source pretraining and previous experiments remain outside
this current-run total. Counts include repeats, not only distinct solutions.

Validation selects the checkpoint; this is not a final test set. No mapping or
transfer claim should be inferred from a successful run. Solver quality and
cross-instance/seed evidence are later experimental work.

## Verification boundary

CPU PyTorch 2.5.1+cpu / NumPy 2.3.5: all 25 CVRP tests passed. A full default
2048-solution, 30-epoch CPU run was exercised through generation/evaluator and
save checks. Existing TSP/control tests are also checked separately. Python 3.8
syntax is checked. The user's PyTorch 2.0.1+cu117 / RTX 4050 still needs the WSL
commands above; no assistant-side CUDA run is claimed.

Tests cover variable-length padding NLL and zero gradients on ignored slots,
prefix capacity/visited/depot masks, complete termination, no future teacher
leakage with fixed latent, overloaded/invalid teacher rejection, packed encoder
and decoder gradients, latent sensitivity, checkpoint round-trip, canonical
split disjointness/train-only quality scaling, greedy/sampled feasibility across
instances and capacity extremes, and a tiny-batch learnability test. The latter
uses beta=0 and score_weight=0 for 80 updates; it is not a convergence benchmark.

## Observed CPU example (one run, not acceptance thresholds)

On the default 2048-solution input, best total validation loss was at epoch 28:

- Validation real-action NLL: 2.04800 initially -> 1.83172.
- Standardized quality MSE: 1.15983 -> 0.21099.
- On 128 diagnostic validation solutions: quality MAE 0.42485 distance units,
  training-mean MAE 1.05792, Pearson 0.90046.
- Aligned / shuffled / zero teacher-forced NLL: 1.84469 / 1.89701 / 1.87727.
- All generated solutions valid in both modes for every latent case.
- Canonical exact-solution reconstruction: 0%. Aligned greedy edge-multiset
  F1: 0.29127. Full structural reconstruction remains weak.
- Current-run objective calls: 3200.

All 50 tests passed when run across CVRP, TSP and the existing mapping controls.
Reference CPU diagnostics/history/config are bundled; no training data or model
checkpoint is bundled. Exact values on the user's CUDA runtime can differ.
