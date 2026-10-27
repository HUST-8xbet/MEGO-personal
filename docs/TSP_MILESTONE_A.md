# MEGO TSP20 — Milestone A

Inspected upstream: https://github.com/MetaronWang/MEGO
Base commit: 63785c662dd5a2d8ea91ec8cc1e7552d1f4ce01d

## Repository findings and new files

BaseProblem defines dimension/evaluate. src/experiment_problem.py samples binary
vectors and casts them to float32. src/regression/dataset.py and the existing
surrogate VAE use a binary/vector path. This patch adds a separate permutation
path; it does not register TSP in binary training or mapping.

Four new files; no existing source file modified:

- src/problem/tsp_problem.py: BaseProblem subclass; validated coordinates/tours,
  closed-tour evaluator, quality adapter, sampler and batch cost.
- src/experiment_tsp_data.py: save one NPZ and verify its contents after reload.
- tests/test_tsp_problem.py: 10 functional tests.
- docs/TSP_MILESTONE_A.md: this guide.

## Data contract

Default instance: float64 coordinates (20, 2), uniform in [0,1)^2.
Tour: int64 permutation 0..19, without repeating the starting city at the end.
Cost includes the last-to-first edge. Distances are unrounded Euclidean,
not TSPLIB integer weights. evaluate() returns length to minimize; quality()
returns negative length to maximize. Saved arrays: coordinates, tours, lengths,
qualities, metadata (JSON string). No normalization or pickle is used.

Sampler draws IID permutations with replacement, using a local seeded RNG.
Duplicates and rotation/reversal equivalents are allowed; no canonicalization
or rejection loop. Instance and tour seeds are separate. Coincident city
coordinates are allowed. Do not pass these data to the original binary scripts.

## Apply on WSL

Extract the ZIP. Replace /absolute/path below with its actual WSL path.

```bash
cd /home/minh1234/MEGO
git rev-parse HEAD
git status --short
git apply --check /absolute/path/mego-tsp20-milestone-a.patch
git apply /absolute/path/mego-tsp20-milestone-a.patch
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_tsp_problem.py' -v
.venv/bin/python -m src.experiment_tsp_data \
  --dimension 20 --sample-num 2048 \
  --instance-seed 1088 --tour-seed 1089 \
  --output data/tsp_milestone_a/tsp20_seed1088.npz
```

The apply check should exit 0 with no output. If there is a conflict, stop and
inspect the conflicting files. Do not force. Public upstream does not contain
the user's local smoke scripts; this patch adds files only.

Expected test summary:

```text
Ran 10 tests in ...s

OK
```

Expected data output:

```text
TSP DATA CHECK PASSED
coordinates: (20, 2)
tours: (2048, 20) int64
lengths: (2048,) float64
objective: minimize length; quality = -length
saved: data/tsp_milestone_a/tsp20_seed1088.npz
```

Existing output files are refused. Use a new filename for another run.
No GPU is needed; imports use NumPy/networkx already required by upstream.

## Validation and milestone boundary

On the assistant checkout: all 10 tests passed; default 2048-tour CLI passed
save/reload and evaluator verification. Tests cover known square perimeter and
crossing costs, rotations/reversals, independent coordinate-based costs,
symmetry, invalid tours/instances, seeded reproducibility, local RNG isolation,
batch/empty inputs and coincident cities.

No command has run on the user's WSL. No TSP expert, mapping or quality
benchmark is included. Confirm these commands on WSL before Milestone B.
MEGO is the selected base; research novelty remains undecided.
