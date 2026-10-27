# TSP → CVRP coupling prototype

This prototype compares three fixed pairing mechanisms:

- `rank_pairing`: existing descending-quality pairing;
- `quality_coupling`: entropic optimal transport using squared percentile gaps;
- `quality_relation_coupling`: quality OT plus a matrix-form GW relation term.

Quality is `-length`, so larger is better. Percentiles are fitted separately
inside each source/target split. The source TSP and target CVRP vertices are
never compared directly. The structural relation is `1 - Jaccard` over sets of
undirected edges. TSP includes the closing edge; CVRP includes depot edges and
removes `-1` padding. Repeated edges are not counted with multiplicity and
vehicle load is not represented by this relation.

The quality cost is `(r_source - r_target)^2`. Both coupling solvers use uniform
marginals. Quality-only uses entropic Sinkhorn. Quality+relation uses an
iterative GW majorization update whose cross term is evaluated as matrix
multiplication; no four-dimensional tensor is constructed. Every run records
iterations, marginal errors, entropy, effective target count, quality cost and
relation cost.

Splits are made before coupling. Train and validation couplings are solved
independently. Couplings remain fixed; train target rows are sampled with a
separate RNG stream per epoch, while validation target rows are sampled once
with a dedicated seed. All arms use the same decoder initialization, optimizer,
gradient steps and source conditions within a training seed. The source expert
is frozen; only the bridge and CVRP decoder are updated.

The large relation products and Sinkhorn/GW iterations have a Torch backend.
When `--device cuda:0` is selected, relation matrices, transport plans and the
matrix-form GW updates remain on the GPU; the final audit is copied back to
NumPy only for JSON/NPZ serialization. CPU remains the fallback. The incidence
construction is vectorized before transfer, so it does not run an O(n^2) Python
pair loop.

The full protocol is intentionally explicit and can still be expensive for
2048 rows because GW remains cubic in the number of support rows. A wiring
smoke run is available:

```bash
cd /home/minh1234/MEGO
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m src.experiment_tsp_cvrp_coupling \
  --output logs_cvrp/tsp_cvrp_coupling_smoke \
  --device cuda:0 --smoke
```

The full configuration uses training seeds 1088, 1089 and 1090, 30 epochs,
`alpha=0.5`, `epsilon=0.05`, and 10 rounds of 41 sampled solutions plus 410
greedy solutions per arm/seed. It stores coupling matrices, diagnostics,
checkpoints, histories, costs and a summary. The resulting experiment is a
prototype test of the structural-coupling hypothesis, not a novelty claim.

## Validation protocol correction

The corrected runner keeps separate train and validation coupling plans and
target pools. Validation choices are sampled once with an independent seed and
then reused for every checkpoint comparison; they are never indexed from the
train plan. The split is fixed with `split_seed=1088` across training seeds
1088, 1089 and 1090. Diagnostics include split, initialization, target-choice
and batch-order hashes. The pre-correction output
`logs_cvrp/tsp_cvrp_coupling_full/` is retained as audit only and must be
excluded from the main result table because its validation pairing was invalid.

Before any decoder rerun, use the coupling-only audit:

```bash
.venv/bin/python -m src.experiment_tsp_cvrp_coupling_solver \
  --output logs_cvrp/tsp_cvrp_coupling_solver_v1 \
  --device cuda:0 --split train --long-iterations 200
```

It compares `0.5Q + epsilon E` with `0.5Q + 0.5R + epsilon E` in float64 and
stores objective/coupling traces. Decoder results from an earlier run with
unmatched quality weighting must not be used to choose a method.

The solver receives the weighted quality cost itself. Therefore the quality
kernel, trace and audit all refer to the same objective; `0.5Q+0.05E` is
numerically equivalent to `Q+0.1E` under the same uniform marginals. The loss
path additionally checks the final global target IDs against the active pool
before calling the CVRP decoder.

The solver-only audit also reports `TV_avg` between the two conditional target
distributions and evaluates relation cost on the quality-only coupling;
`relation_cost=0` is never interpreted as evidence of zero structural cost.
The decoder runner writes per-arm `*_pairing.json` files with the actual global
target IDs selected for each train/validation loss batch.
