# C.1.1 — Correct the training-shuffle control

The `shuffled_train` control now uses a fixed uniform random permutation within
each train/validation split. The previous cyclic shift could preserve a
systematic rank relationship and was not a proper random-pairing null control.

The bridge, decoder, optimizer, losses, epoch count, and original configuration
are unchanged. The CLI now supports selecting arms explicitly:

```bash
.venv/bin/python -m src.experiment_tsp_mapping_controls \
  --arms shuffled_train \
  --output logs_tsp/mapping_controls_c11_shuffled \
  --device cuda:0
```

The default still trains all five arms. Use `--arms` to rerun only selected
controls; duplicate arm names are rejected. Existing runs and checkpoints are
preserved.
