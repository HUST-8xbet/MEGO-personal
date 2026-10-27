# C.1 — Audit and independently trained Max-Cut -> TSP controls

This is a new diagnostic protocol. It preserves the original C mapping code and
outputs, and does not claim to reproduce or improve their measured quality.
MEGO is the selected framework. Novelty remains undecided.

## Audit of the supplied mapping code and artifacts

Confirmed by reading experiment_tsp_mapping.py and tsp_mapping.py, and by
recomputing costs, validity and cycle multiplicity from supplied generated.npz:

- Higher-is-better pairing is correct: Max-Cut score and TSP negative length.
- Legacy Max-Cut posterior means feed a learned linear 120 -> 16 bridge.
- Bridge and TSP decoder are trained. GRU encoder, posterior layers and quality
  head parameters are frozen. City embedding is shared and remains trainable;
  therefore the complete original encoder function is not frozen. That encoder
  is not used to form conditions during this mapping run.
- no_source IS independently trained from the same target checkpoint, with the
  same split, number of epochs, optimizer settings and batch-order seed. The
  earlier concern that this might only be a zero-latent inference ablation is
  resolved. One greedy cycle is expected for a fixed instance/condition.
- shuffled only rolls latent rows during EVALUATION, holding the trained source
  model fixed. It exactly permutes the same generated cycles. Equal aggregate
  costs are expected, not evidence of a broken bridge or useless source.
- random is N(0,I) in the raw source latent space at EVALUATION only. It is not
  a model trained with random conditioning. Raw source mean/std/covariance were
  not measured, so distribution comparability has not been established.
- Best source checkpoint is epoch 15, validation NLL 2.0555145. No-source's best
  is epoch 30, NLL 2.0526101. Reconstruction NLL does not directly optimize tour
  length; these values do not establish a transfer benefit.
- 410 generated tours per condition are valid. Mean lengths source/shuffled/
  random/no_source: 11.3706698 / 11.3706698 / 9.1832305 / 12.5483234. Unique cycles:
  353 / 353 / 399 / 1. Uploaded outputs reproduce the reported statistics.
- The target checkpoint default is the Milestone B expert; its default dataset
  is the same TSP NPZ used for mapping. A newly randomized mapping split does not
  ensure validation tours were unseen during target-expert pretraining. Actual
  overlap would require inspecting that expert's saved split/provenance.
- The old script canonicalizes tours but does not remove equivalent duplicates
  before splitting. Supplied validation has 410 distinct cycles, but its full
  train/validation overlap cannot be verified without split.npz and input data.
- 3688 = 2048 verification + 1640 evaluation calls in that invocation. Existing
  labels were loaded, not generated there. 5736 includes an assumed 2048 earlier
  label-generation calls. Neither number includes all prior expert run costs.

This is a successful functional mapping run. No transfer-benefit conclusion is
supported yet. The original single-seed random-ablation advantage does not by
itself establish negative transfer either.

## New files only

- src/tsp_mapping_controls.py: disjoint splits, conditioning banks, training/eval.
- src/experiment_tsp_mapping_controls.py: new runner and provenance/cost outputs.
- configs/tsp_mapping_controls.json: default 30-epoch configuration.
- tests/test_tsp_mapping_controls.py: five tests (24 total with A/B).
- docs/TSP_MAPPING_CONTROLS.md: this review and instructions.

The patch adds these five files. It does not modify uploaded mapping files,
Milestone A/B source, or existing checkpoints. It needs the original
src/tsp_mapping.py only to load the real legacy Max-Cut expert.

## Training conditions

All five arms use identical target tours/split, initial decoder weights,
optimizer schedule and batch order. The first four use identical initial linear
bridge weights and source-dimensional input width:

| Arm | Conditioning during both training and evaluation |
| --- | --- |
| source | Actual source posterior means, standardized using training rows |
| shuffled_train | Same source bank uniformly permuted within each partition before training |
| noise_train | Independent Gaussian bank in standardized active dimensions |
| rank_only | Only source-quality percentile; tied qualities share one value |
| no_source | Zero target latent; no bridge |

Noise matches standardized marginal mean/variance, not full covariance.
rank_only percentiles use only source training qualities as their reference.
Shuffling is a fixed uniform random permutation, not resampled per epoch. The
original C.1 cyclic-shift implementation is superseded by C.1.1. Source latent rows and canonical
undirected target cycles are split first; train/validation are rank-matched
separately. Equal source/target lengths are not assumed. Counts are capped by
max_pairs after random bank selection, avoiding deterministic best-only slicing.

Default initialization is a SHARED FRESH target decoder, excluding prior target
expert exposure. This isolates the source signal during target adaptation; it
is a different experiment from the original warm-start mapping. Optional
--target-checkpoint enables a shared warm start and explicitly marks validation
pretraining exposure as uncertified. Do not compare fresh-vs-warm scores as a
source effect. A reusable target decoder still needs its own cost accounting.

Each trained arm is evaluated in both greedy and sampled decoding, with equal
candidate/evaluator counts within each decoding mode. Sampling uses a common
action RNG seed. Compare greedy with greedy and sampled with sampled. No-source
has fewer parameters because it has no bridge; noise_train is the same-width,
same-bridge stochastic control for source. No generated tour is repaired or
selected using extra objective calls.

Validation selects best checkpoint per arm; it is not a final test set. The
validation rank pairing uses validation target qualities ONLY to construct the
paired reconstruction diagnostic. Generation consumes source conditions and
coordinates, not validation target qualities. This is a one-instance diagnostic;
a later protocol needs multiple sources/instances/seeds and separate final tests.

## Apply and run on WSL

Extract MEGO-TSP-Mapping-Controls.zip. Substitute the actual patch path below.
Keep the previous logs and use a new output directory. No dependencies need to
be reinstalled in the existing WSL environment.

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-tsp-mapping-controls.patch
git apply /absolute/path/mego-tsp-mapping-controls.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_tsp*.py' -v
.venv/bin/python -m src.experiment_tsp_mapping_controls \
  --output logs_tsp/mapping_controls_01 \
  --device cuda:0
```

Default source checkpoint/config/x/y paths match the supplied old script, and
--target-data defaults to data/tsp_milestone_a/tsp20_seed1088.npz. If these files
were moved, set --source-checkpoint, --source-config, --source-x, --source-y and
--target-data explicitly. By default do NOT pass --target-checkpoint.

A wiring-only shorter run can use --epochs 2 and a different output directory.
The full command trains five models for 30 epochs each, so it takes longer than
the original two-model script. CPU option: --device cpu.

Expected test summary after C.1.1: Ran 25 tests ... OK.
Expected training: five arm lines followed by

```text
TSP MAPPING CONTROLS FUNCTIONAL RUN COMPLETED
diagnostics: logs_tsp/mapping_controls_01/diagnostics.json
```

Do not expect particular numeric lengths or assume source will win.

## Outputs and cost boundaries

Per-arm .pt files contain best decoder/bridge state, dimensions and configuration.
conditioning.npz/json retain all conditioning banks, transforms and pairing.
split.npz records paired indices and original source/target row identities.
generated.npz contains both decoding modes and a random-tour reference.
history.json stores all training curves; diagnostics.json records comparisons,
file hashes, runtime versions, device, initialization and scope. costs.json
separates imported labels, current verification/evaluation and an estimate that
assumes one prior evaluator call per imported label. Original source/target
pretraining and earlier runs remain unknown and are not silently set to zero.

For unchanged default 2048-row input / 410-row validation: evaluation uses
5 arms * 2 decoding modes * 410 + 410 random tours = 4510 calls; verification
adds 2048, making 6558 calls in this new invocation. An additional assumed
2048 original label-generation calls gives 8606, excluding previous expert
and mapping runs. Repeated evaluations count as calls even when tours repeat.

## Verification actually performed

CPU PyTorch 2.5.1: 24/24 A/B/C.1 tests passed. A two-epoch CLI fixture exercised
all five branches, greedy/sampled generation, checkpoint writes and query counts.
That fixture used a synthetic source bank; it is a software test, NOT a Max-Cut
transfer result. No real source checkpoint, CUDA or full 30-epoch C.1 comparison
has run on the assistant's machine. Real Max-Cut loading reuses the uploaded
legacy loader already exercised in the user's original mapping run.

Tests cover cycle/source-row split disjointness, sorted pairing, train-only
normalization, tied-quality scalar controls, equal-bank shuffling, noise,
trainable/frozen components, finite optimization, and no-source sampling.
