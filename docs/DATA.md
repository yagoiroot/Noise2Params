# Public Dataset ↔ Code Mapping

The Noise2Params public dataset is distributed separately from this code
repository.  This document maps each item in the dataset to the function
that produces (or consumes) it in :mod:`noise2params`.

## Dataset layout

```
Noise2Params Dataset/
├── indist_images_full/        (.png base images, Unsplash-derived, training set)
├── ood_DIV2K_images/          (.png base images, DIV2K-derived, val/test)
│
├── indist_image_train_rec/    (.parquet recordings of indist_images_full, training)
├── ood_images_val_rec/        (.parquet recordings of ood_DIV2K_images, validation)
├── ood_images_test_rec/       (.parquet recordings of ood_DIV2K_images, test)
│
├── data_5e6_5/                (synthetic training pairs — saddle-point model)
│     ├── synth_events/events_all.npy       shape (N, 2, 360, 640) float32
│     ├── synth_images/images_all.npy       shape (N, 360, 640)    float32 in [0,1]
│     └── synth_events/sample_names.txt
├── data_5e6_6/                (synthetic training pairs — Gaussian model)
├── data_real_5e6_2/           (real training pairs — in-dist recordings)
├── validation_5e6_5/          (real validation pairs — OOD recordings)
│
├── 19-38-49_to_00-17-48_recording_data.csv   (training-set mapping)
├── 12-18-17_to_12-25-48_recording_data.csv   (validation-set mapping)
├── 12-08-40_to_12-36-01_recording_data.csv   (test-set mapping)
│
├── Extreme Outliers.txt       (always-on pixel block-list)
└── Hot Pixels Column 30.txt   (default-bias hot-pixel block-list)
```

## Recording files

Each `.parquet` in the recording folders is an array of rows with four
columns: `x`, `y`, `polarity`, `timestamp`.  Polarity 0 is a negative
event, polarity 1 is a positive event.  Timestamps are in microseconds.
None of the shipped `.parquet` files have outliers pre-removed; the
outlier filtering is applied at load-time by
:func:`noise2params.reading.recording_reader` when the caller passes
`remove_outliers=True` (all pipeline entry points do).

## Recording ↔ base-image mapping CSVs

Each of the three CSV files at the dataset root has columns:

| Column           | Meaning |
|------------------|---------|
| `recording_name` | Stem of the `.parquet` file (no extension) |
| `time ID`        | Wall-clock timestamp of the recording (HH-MM-SS) |
| ` image`         | Base-image stem (note the literal leading space in the column name — preserved from the internal logs) |
| `intensity`      | Mean illuminance measured during the recording, in lux |
| `dead_time`      | EC refractory period, in microseconds (79 for all shipped data) |
| `biases`         | Bias setting label (`None` means Prophesee EVK4 factory defaults) |

These CSVs drive
:func:`noise2params.noise_image.real_data_base_maker` (for dataset
construction) and
:func:`noise2params.noise_image.group_infer_recording` (for batch
inference).

## Dataset → code mapping

| Dataset item | Produced by | Consumed by | Stage |
|---|---|---|---|
| `indist_images_full/` | Human-curated (adapted from Unsplash via the original Noise2Image paper). | :func:`synthetic_data_base_maker(source_folder=...)` <br> :func:`real_data_base_maker(val_pair_folder=...)` | Input to synth/real gen |
| `ood_DIV2K_images/` | Human-curated (adapted from DIV2K via the original Noise2Image paper). | :func:`real_data_base_maker(val_pair_folder=...)` <br> Base images for :func:`group_sim_metrics_2` scoring | Input to OOD real-pair gen; scoring reference |
| `indist_image_train_rec/` | Prophesee EC recordings converted by `metavision_file_to_parquet_v2` (not included here; see the Metavision SDK). | :func:`real_data_base_maker` (via the training CSV) | Real training recordings |
| `ood_images_val_rec/` | Same Prophesee pipeline, on OOD image subset. | :func:`real_data_base_maker` (via validation CSV) | Real validation recordings |
| `ood_images_test_rec/` | Same. | :func:`group_infer_recording` (via test CSV) | Real test recordings |
| `19-38-49_to_00-17-48_recording_data.csv` | Manual log generated at recording time. | :func:`real_data_base_maker(image_rec_mapping=...)` | Training-set mapping |
| `12-18-17_to_12-25-48_recording_data.csv` | Same. | :func:`real_data_base_maker(image_rec_mapping=...)` | Validation-set mapping |
| `12-08-40_to_12-36-01_recording_data.csv` | Same. | :func:`group_infer_recording(file_list_csv=...)` | Test-set mapping |
| `data_5e6_5/` | :func:`synthetic_data_base_maker(model='saddle', ...)` on `indist_images_full/` at 5 s integration. | `03_train_cnn.py --data-folder ./data/data_5e6_5` | Saddle-point synthetic training pairs |
| `data_5e6_6/` | :func:`synthetic_data_base_maker(model='Gaussian', ...)` on `indist_images_full/`. | `03_train_cnn.py --data-folder ./data/data_5e6_6` | Gaussian synthetic training pairs |
| `data_real_5e6_2/` | :func:`real_data_base_maker` on `indist_images_full/` × `indist_image_train_rec/`. | `03_train_cnn.py --data-folder ./data/data_real_5e6_2` | Real training pairs |
| `validation_5e6_5/` | :func:`real_data_base_maker` on `ood_DIV2K_images/` × `ood_images_val_rec/`. | `03_train_cnn.py --val-data-folders ./data/validation_5e6_5` | Real validation pairs |
| `Extreme Outliers.txt` | Manual. | :func:`noise2params.outlier_masks.load_default_pixel_list` (in turn called by :func:`recording_reader` when `remove_outliers=True`) | Outlier pixel filter |
| `Hot Pixels Column 30.txt` | Manual. | Same as above. | Default-bias hot-pixel filter |

## No Poisson-model synthetic dataset

The paper's exact-Poisson probability model is implemented in
:func:`noise2params.prob_models.pos_event_prob_vec_numba_2` and is
accessible via `event_noise_image_modeler_core(model='Poisson')` and
`synthetic_data_base_maker(... noise_modeler_kwargs={'model':'Poisson'})`.
However, **no Poisson-model synthetic dataset is shipped** with the
paper.  Based on synthetic noise images, we expect the Poisson-trained CNN's performance to be
marginal over the saddle-point-trained CNN while the Poisson probability
computation is less numerically stable, so Poisson was omitted from the
deep-learning comparison.  The code path is retained purely for
reproducibility; users who want to regenerate a Poisson-synthetic
dataset can do so with:

```bash
python scripts/02_build_synthetic_dataset.py --model Poisson ...
```

## Placing the dataset on disk

The scripts do not require any particular layout, but the shipped
training configs assume the dataset has been placed (or symlinked) into
a `data/` folder one level above this repository, e.g.:

```
<somewhere>/
├── Noise2Params Public Facing Code/       (this repo)
└── data/
    ├── data_5e6_5/
    ├── data_5e6_6/
    ├── data_real_5e6_2/
    ├── validation_5e6_5/
    ├── indist_images_full/
    ├── ood_DIV2K_images/
    ├── indist_image_train_rec/
    ├── ood_images_val_rec/
    ├── ood_images_test_rec/
    ├── Extreme Outliers.txt
    ├── Hot Pixels Column 30.txt
    └── *.csv
```

Because :func:`noise2params.utils.file_checker` walks the project root
for fuzzy name matches, bare filename stems (no extension, no folder
prefix) passed to most functions will resolve anywhere under this tree.
