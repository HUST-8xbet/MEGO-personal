# MEGO TSP20 — Milestone B: expert functional implementation

Base inspected: MetaronWang/MEGO, commit
63785c662dd5a2d8ea91ec8cc1e7552d1f4ce01d, plus Milestone A.

Milestone A passed on WSL according to the user's confirmation. This milestone
adds the TSP expert locally; WSL/GPU execution still needs to be verified.
MEGO remains the framework base. This architecture is an implementation
candidate, not the claimed research novelty. Decoder mapping is not included.

## Files added

- src/regression/tsp_vae.py: permutation VAE and tour canonicalization.
- src/experiment_tsp_expert.py: train/validation loop, checkpoint, diagnostics.
- configs/tsp_expert_smoke.json: small run configuration.
- tests/test_tsp_expert.py: nine expert functional tests.
- docs/TSP_MILESTONE_B.md: this guide.

The patch contains these five new files only and depends on the files installed
in Milestone A. Existing binary model/training scripts are untouched.

## Architecture and conventions

Input: one instance's coordinates (n, 2) and a permutation tour (n,).
Coordinate MLP embeds cities; GRU reads city embeddings in tour order. Its final
state and pooled instance embedding produce posterior mean/log variance.
A latent-conditioned GRU pointer decoder starts at city 0 and selects an
unvisited city at every step. The latent is used both at initialization and
at each decoding step. Prior and posterior generation require no repair.
A separate head predicts standardized quality from latent; its last layer
is linear so standardized negative values remain possible.

Loss = mean teacher-forced token cross entropy + 0.001 * KL + quality MSE.
KL sums over latent dimensions; NLL averages over n-1 predictions, including
the final forced city. Teacher forcing sees only the visited prefix. Validation
uses posterior mean instead of sampled latent. Best checkpoint is selected by
validation total loss, not generation quality.

Canonicalization rotates the tour to city 0, then chooses direction so the
second city index is less than the last. It removes equivalent cycle duplicates
before the seeded 80/20 split. Quality is -length; mean and standard deviation
are fitted on training tours only. The checkpoint stores this scaling. A
constant-quality dataset uses scale 1, with raw_std recorded.

This is a single-instance expert with held-out tours, not held-out instances.
Coordinates condition the architecture, but this run does not show cross-instance
generalization. Canonicalization is a deterministic indexing convention, not
a permutation- or symmetry-invariant neural architecture. Greedy generated
outputs start at city 0 but need not have canonical orientation; reconstruction
metrics canonicalize their output before comparing cycles.

## Apply on the user's WSL

Extract MEGO-TSP20-Milestone-B.zip and set the path below to the extracted patch.
No clone or dependency reinstall is required. This uses existing PyTorch,
NumPy and networkx and does not import the binary Lightning training path.

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-tsp20-milestone-b.patch
git apply /absolute/path/mego-tsp20-milestone-b.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_tsp*.py' -v
.venv/bin/python -m src.experiment_tsp_expert \
  --data data/tsp_milestone_a/tsp20_seed1088.npz \
  --output logs_tsp/tsp20_expert_smoke_01 \
  --device cuda:0
```

If git apply --check reports a conflict, stop and inspect the named files.
No output from that command means its check passed. Existing output directories
are refused. Use a new output name for another run; no resume flag is provided.
CPU alternative: --device cpu. For a shorter wiring check, add --epochs 2
--max-samples 256 and use another output directory; that run is not a quality
result. The default is 30 epochs, batch 128, hidden 64, latent 16, seed 1088.

Expected tests:

```text
Ran 19 tests in ...s

OK
```

Training prints INITIAL and 30 epoch lines. Exact losses depend on runtime.
Expected final lines:

```text
TSP EXPERT FUNCTIONAL RUN COMPLETED
best_epoch: <integer from 1 to 30>
prior_greedy_valid_rate: 1.0
checkpoint: logs_tsp/tsp20_expert_smoke_01/best_model.pt
diagnostics: logs_tsp/tsp20_expert_smoke_01/diagnostics.json
```

For review, print diagnostics after the run:

```bash
.venv/bin/python -m json.tool logs_tsp/tsp20_expert_smoke_01/diagnostics.json
```

## Outputs and interpretation

- best_model.pt: weights, model/run config, quality scaling and validation loss.
- config.json: actual config, source/data SHA256, base commit, versions, split
  convention, sample counts, device and quality scaling.
- split.npz: original dataset row indices for train and validation.
- history.json: per-epoch train/validation loss, NLL, KL and quality MSE.
- diagnostics.json: best-model reconstruction, quality and latent diagnostics.
- generated.npz: prior-greedy tours, costs, coordinates and diagnostic indices.

Diagnostics use up to 128 validation tours. aligned/shuffled/zero compare
posterior mean, cyclically reassigned posterior means, and zero latent while
holding instance and teacher prefix fixed. Compare their NLL, canonical exact
cycle rate, undirected edge overlap and greedy cycle changes. Probability
changes and nonzero latent gradients show a computational path; they do not
establish that useful solution information has been learned.

score reports quality MAE in length units, training-mean baseline MAE, and
Pearson correlation (null for constant predictions/targets). prior_greedy reports
validity, mean/best length and number of distinct cycles. random_reference uses
the same number of randomly sampled tours. These small diagnostics are not an
optimization benchmark, a transfer result, or a pass/fail quality threshold.
Validity is computed by sending generated tours through the Milestone A
validator/evaluator; malformed tours raise errors.

Objective calls are counted explicitly: every input tour is revalidated;
diagnostics evaluate three latent cases, prior generation and random reference;
prior tours are evaluated once more when writing generated.npz. No new labels
are queried during gradient training. Data generation cost precedes this run.

A completed run proves execution, finite losses and valid generated tours. Weak
reconstruction, score correlation or latent dependence must be reported rather
than hidden behind that completion message. Review diagnostics before deciding
whether to train longer, adjust the expert or start mapping.

## Verification environment and limits

The assistant checked all 19 tests on CPU with PyTorch 2.5.1+cpu and NumPy 2.3.5.
Python 3.8 syntax was checked. A full default 2048-tour / 30-epoch CPU run was
also exercised. The user's PyTorch 2.0.1+cu117 and RTX 4050 still require the WSL
commands above; no CUDA run is claimed here. Exact floating-point values need
not match across PyTorch versions/devices.

The small-batch reconstruction test uses deterministic posterior means, beta=0
and score_weight=0 for 80 updates; it checks learnability and gradients, not VAE
convergence. Other tests check masks, no future-tour leakage in teacher forcing,
valid greedy and sampled generation, latent sensitivity, canonical symmetry,
checkpoint round-trip, no equivalent-cycle split leakage and invalid inputs.

## Observed CPU smoke output (one run; not expected numeric thresholds)

2048 tours, 1638 training / 410 validation, 30 epochs, best epoch 30:

- Validation teacher-forced NLL: 2.07057 initially -> 1.98531.
- Validation standardized quality MSE: 0.98897 -> 0.06830.
- On 128 diagnostic validation tours: quality MAE 0.19397 length units,
  versus 0.76362 for the training-mean baseline; Pearson 0.97535.
- Aligned / shuffled / zero teacher-forced NLL: 1.98492 / 2.10636 / 2.07105.
- Prior-greedy validity 100%; 127 distinct cycles from 128 latent samples.
- Aligned exact-cycle reconstruction: 0%; edge overlap: 0.12148.

The quality head and latent path learn a signal in this instance, but full tour
reconstruction is still poor. This is not a trained expert quality baseline for
research. Revisit capacity/training/representation after inspecting WSL output.
For this single instance, shuffling latent vectors permutes the set of greedy
outputs: aggregate aligned/shuffled length statistics are necessarily equal;
use paired reconstruction and fixed-prefix NLL for that diagnostic instead.

The bundled reference diagnostics are from the full CPU run. A final two-epoch,
256-sample run additionally checked the new objective-query bookkeeping fields.
No model weights or training data are included in this delivery.
