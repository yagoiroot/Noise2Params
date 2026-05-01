# Model Checkpoints

**Trained model weights (`.ckpt` files) are NOT shipped with this code
release.**  Users must regenerate them from the public dataset using
`scripts/03_train_cnn.py` (see `configs/training_published.yaml` for the
exact flags used for each published variant).

## Expected local layout

Once trained, Lightning writes checkpoints under `noise2image/lightning_logs/`:

```
noise2image/
└── lightning_logs/
    └── precomputed_training/
        ├── version_0/
        │   └── checkpoints/
        │       ├── last.ckpt
        │       └── epoch=N-step=M.ckpt  (top-3 by validation loss)
        ├── version_1/
        ...
```

The `last.ckpt` is what the inference scripts (`04_infer.py`) load by
default.  You may point `--ckpt` at any of the `epoch=*.ckpt` files to
evaluate an earlier snapshot.

## Three published variants

| Variant | Training set | LR | Epochs | Pretrained checkpoint |
|---|---|---|---|---|
| Saddle-synthetic CNN | `data_5e6_5` | 2e-5 | 40 | (retrain) |
| Gaussian-synthetic CNN | `data_5e6_6` | 2e-5 | 40 | (retrain) |
| Mixed (real + synth fine-tune) | `data_real_5e6_2` from a warm-started `.ckpt` | 5e-5 | 40 | (retrain) |

Validation in all cases uses `validation_5e6_5`.

## Loading a checkpoint

```python
from noise2image.train import Model
from noise2params.noise_image import load_compiled_checkpoint

model = Model(dim=64, in_channels=2, lr=5e-5, vanilla_unet=False)
model = load_compiled_checkpoint(model, 'path/to/last.ckpt', strict=True)
model.eval()
```

`load_compiled_checkpoint` handles the `_orig_mod.` key prefix that
`torch.compile` inserts; a plain `model.load_state_dict(torch.load(...)["state_dict"])`
will fail with a missing-key error if the checkpoint was saved from a
compiled model.
