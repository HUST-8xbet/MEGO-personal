# MEGO CVRP20 — D-A: instances, feasible sampler, evaluator

This milestone starts CVRP after the TSP functional pipeline completed.
MEGO is the selected framework. Research novelty and transfer benefit remain
unconfirmed. The corrected TSP control run does not require more training
before CVRP data/evaluation work.

## Repository inspection and files added

Inspected BaseProblem, TSPProblem, experiment_tsp_data.py and current tests.
The repository uses dimension/evaluate, but the legacy binary samplers and
surrogate training path cannot consume depot-separated routes. CVRP therefore
has its own data generator and has not been registered in binary training.

Four NEW files only:

- src/problem/cvrp_problem.py: CVRP instance, validation, evaluation, sampling,
  route inspection and padding codec.
- src/experiment_cvrp_data.py: CLI generation, NPZ storage and reload checks.
- tests/test_cvrp_problem.py: 14 functional tests.
- docs/CVRP_MILESTONE_A.md: this guide and state update.

No existing TSP, binary expert, mapping or control file is modified.

## Exact problem definition

- CVRP20 means 20 customers PLUS one depot, 21 vertices in total.
- Coordinates: float64, shape (21, 2); default IID uniform in [0,1)^2.
  Vertex 0 is the depot and is sampled from the same square.
- Customer indices: 1..20. Depot demand 0; customer demands are positive
  unnormalized int64 values, by default independent uniform integers 1..9.
- Homogeneous vehicle capacity: 30 by default, in the same units as demands.
  Each customer demand must be <= capacity; otherwise construction is rejected.
  For capacity < 9 the random generator samples demands 1..capacity.
- Unlimited identical vehicles; no fixed fleet size, vehicle penalty, time
  windows or split delivery. Every customer is served once, entirely by one
  vehicle. Each route departs from and returns to the depot. Route load <= Q.
- Objective: minimize total UNROUNDED Euclidean distance over all routes.
  evaluate() returns this distance; quality() returns -distance.
- This is not a TSPLIB/CVRPLIB rounded-distance parser or a benchmark claim.

Integer-only demands/capacity are deliberate in this initial implementation.
Fractional/normalized values are rejected. Future neural inputs can normalize
these stored integer demands by capacity without changing evaluator units.

## Solution representation and validation

One solution is a 1D integer sequence with depot separators:

```python
[0, 3, 7, 0, 2, 5, 0]  # example for a different customer set
```

The sequence starts and ends at 0. Repeated 0 separates vehicles; adjacent
zeros are rejected as empty routes. Customer indices must be in range and each
customer must occur exactly once. Each route is checked independently against
capacity. Missing/repeated/out-of-range customers, floats, missing depot ends,
empty routes, and overloaded routes raise ValueError. No silent repair is used.
The evaluator sums consecutive edges, including every return-to-depot edge.
Since depot endpoints are explicit it does not add another closing edge.

The sampler draws a random permutation of customers, then starts a new vehicle
when the next demand would exceed capacity or with early-return probability
0.2 at a nonempty route. This generates feasible solutions without querying
an objective or using a solver. It is NONUNIFORM over feasible solutions.
Probability 0 gives next-fit capacity splitting; 1 yields one customer per route.
Duplicates and equivalents under route reversal/vehicle ordering are retained.
Instance and solution seeds are independent local RNGs.

## Saved NPZ schema (allow_pickle=False)

| Key | Default shape / type | Meaning |
| --- | --- | --- |
| coordinates | (21, 2), float64 | Depot then customer positions |
| demands | (21,), int64 | Depot demand 0, customer demands |
| capacity | scalar, int64 | Vehicle capacity |
| routes_padded | (2048, 41), int64 | Depot-separated sequence; -1 padding |
| sequence_lengths | (2048,), int64 | Number of actual tokens, including depot tokens |
| lengths | (2048,), float64 | Total route distance (NOT sequence length) |
| qualities | (2048,), float64 | Negative distance |
| route_counts | (2048,), int64 | Number of nonempty vehicle routes |
| metadata | scalar Unicode JSON | Seeds, assumptions, commit, source hashes, versions |

The maximum token count is 2*n+1, achieved by n single-customer routes. Padding
uses -1, never 0. Slice by sequence_lengths or call unpack_solutions before
evaluation. The evaluator rejects padding tokens. pack_solutions and
unpack_solutions validate feasibility and padding, and support empty batches.
No pickle/object arrays are needed for variable route lengths.

## Apply and run on WSL

Extract MEGO-CVRP20-Milestone-A.zip. Substitute its actual WSL patch path:

```bash
cd /home/minh1234/MEGO
git apply --check /absolute/path/mego-cvrp20-milestone-a.patch
git apply /absolute/path/mego-cvrp20-milestone-a.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_cvrp_problem.py' -v
.venv/bin/python -m src.experiment_cvrp_data \
  --dimension 20 --capacity 30 --sample-num 2048 \
  --instance-seed 1088 --solution-seed 1089 \
  --early-return-probability 0.2 \
  --output data/cvrp_milestone_a/cvrp20_seed1088.npz
```

git apply --check should exit 0 with no output. If there is a conflict, inspect
it rather than forcing. No dependency reinstall or GPU is needed. Uses existing
NumPy/networkx requirements; no PyTorch/Lightning import is required here.
Existing output files are refused. Use a new filename for another run.

Expected test summary:

```text
Ran 14 tests in ...s

OK
```

Expected default data output:

```text
CVRP DATA CHECK PASSED
customers: 20 depot: 0 capacity: 30
coordinates: (21, 2)
demands: (21,) int64
routes_padded: (2048, 41) int64
sequence_lengths: (2048,) int64
feasible_rate: 1.0
objective: minimize total route length; quality = -length
saved: data/cvrp_milestone_a/cvrp20_seed1088.npz
```

The CLI rebuilds the problem after loading its file, validates every sequence,
recomputes costs, verifies quality signs, route counts and metadata. The two
passes use 2048 label-generation + 2048 reload-verification objective calls;
metadata records both. Sampling itself makes no objective calls. These 2048
solutions are functional data, not optimal labels or a fixed benchmark split.

## Verification performed locally

14/14 CVRP tests passed. Combined with 10 existing TSPProblem tests, 24/24 passed.
The default 2048-solution CVRP20 generation/reload check passed on CPU.
Python 3.8 syntax was checked. No command was executed on the user's WSL.

Tests cover known geometry (two routes of length 4, total 8), individual closing
edges, route reorder/reversal invariance, independent coordinate-based costs,
capacity equality and overload, integer-overflow-safe loads, bad instance and
solution inputs, 2048 seeded feasible samples, early-return extremes, small
capacity, padding, empty batches, one customer, coincident positions, input-copy
isolation and local RNG reproducibility.

## Current milestone boundary

D-A is implemented and locally verified; WSL confirmation is pending. No CVRP
expert, capacity mask in a neural decoder, training loss or TSP->CVRP mapping is
included. Next is the CVRP expert after the WSL commands above pass, with visited
customer and remaining-capacity masks, depot-return actions and padded-sequence
loss masks. The new evaluator remains the feasibility authority. No TSP expert
or Max-Cut retraining is a prerequisite for this data milestone.
