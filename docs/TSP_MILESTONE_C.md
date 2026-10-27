# MEGO TSP20 — Milestone C: quality-ranked Max-Cut → TSP mapping

This milestone tests whether a latent from the existing Max-Cut expert can
condition the TSP decoder after pairing source and target samples by quality.
It is a functional transfer diagnostic, not a cross-family research result.

## Quality contract

- Max-Cut uses its stored score `y`; larger is better.
- TSP stores `quality = -length`; larger is better.
- Pairing sorts both arrays in descending quality and matches equal rank.
- Raw score units are never compared across problem families.

## Run

```bash
cd /home/minh1234/MEGO
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m unittest discover -s tests -p 'test_tsp*.py' -v
.venv/bin/python -m src.experiment_tsp_mapping \
  --output logs_tsp/mapping_maxcut_to_tsp_01 \
  --device cuda:0
```

The default paths use the existing Max-Cut checkpoint/data, the Milestone A
TSP data and the Milestone B TSP expert checkpoint. Use a new output directory;
the experiment refuses to overwrite an existing one. `--epochs 2` is available
for a wiring check, but is not a quality result.

## Conditions

The target TSP architecture is unchanged. Its encoder and score head are
frozen. The existing TSP decoder is initialized from the Milestone B
checkpoint and fine-tuned together with a linear 120→16 latent bridge:

- `aligned`: Max-Cut posterior means paired by descending quality.
- `shuffled`: the same trained bridge receives a cyclically shuffled source.
- `random`: the bridge receives random source-dimensional latent values.
- `no_source`: a separately trained target decoder receives zero latent and no
  source expert input.

The source Max-Cut encoder is frozen. The target decoder receives teacher tours
and is trained with the existing TSP token cross-entropy objective. No epoch or
TSP architecture increase is introduced.

## Recorded evidence

- `diagnostics.json`: quality ordering, NLL, valid tours, mean/best target
  length, and aligned/shuffled/random/no-source comparisons.
- `mapping_checkpoint.pt`: target decoder and source-to-target bridge.
- `no_source_checkpoint.pt`: decoder-only control.
- `costs.json`: source pretraining cost label, target data/verification cost,
  adaptation training cost (zero new queries because labels are stored), and
  generated-tour evaluation cost separately.
- `history.json`, `split.npz`, `generated.npz`: training and reproducibility
  artifacts.

`valid_rate` is checked by the TSP evaluator. Gradient finiteness is checked
for the bridge and decoder during training; checkpoint state is saved after the
best validation NLL. A lower generated length is better, while higher quality
is better. The result does not establish useful transfer without comparing the
four conditions and accounting for target adaptation cost.
