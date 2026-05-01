# Noise2Params — Public Reconstruction Code

Public-facing subset of the code accompanying the paper

> O. Root, J. Mujo, M. Xu.  **Noise2Params: Unification and Parameter
> Determination from Noise via a Probabilistic Event Camera Model.**

This repository contains the code needed to **reproduce the deep-learning
noise-image reconstruction** portion of Noise2Params:

* Construct real (event-camera-recorded) noise-image / base-image training
  pairs from the shipped Prophesee `.parquet` recordings.
* Generate synthetic noise-image / base-image pairs under the paper's
  three probability models (saddle-point, Gaussian, exact Poisson).
* Train a U-Net + multi-head self-attention CNN (adapted from the original
  *Noise2Image* paper of Cao *et al.*) to reconstruct the base image from
  5 s-integrated event counts.
* Run inference and score reconstructions with PSNR, NCC, SSIM, MS-SSIM,
  3-SSIM, FSIM, VIF, GMSD, VSI, LPIPS, DreamSim, DISTS, and PieAPP.

The parameter-fitting / S-curve / probability-analysis portions of the
broader Noise2Params study are **not** included here; this release is
scoped to the image-reconstruction pipeline.

## Repository layout

```
Noise2Params Public Facing Code/
├── README.md                          (you are here)
├── LICENSE
├── CITATION.cff
├── requirements.txt / environment.yml (Python dependencies)
│
├── noise2params/                      (the analysis package)
├── noise2image/                       (training code, adapted from Noise2Image)
├── scripts/                           (five CLI wrappers, one per pipeline stage)
├── configs/                           (published parameter + training configs)
├── docs/                              (dataset map, pipeline diagram, symbol table)
├── examples/                          (short runnable demos)
└── checkpoints/                       (empty; populated by training)
```

See `docs/PIPELINE.md` for a diagrammatic view and `docs/PAPER_CROSSREF.md`
for a paper-symbol ↔ code-variable table.

## Setup and installation

This section walks through everything needed to go from a clean machine
to a working install of this code.  If you are already comfortable with
Python virtual environments and CUDA-enabled PyTorch, you can skim it.
The high-level steps are:

  0. Check the system requirements.
  1. Get the code.
  2. Get the dataset.
  3. Set up a Python environment.
  4. Install PyTorch with CUDA (training only).
  5. Install the rest of the dependencies.
  6. Apply the package-pin remedy if needed.
  7. Verify the install with a smoke test.

### 0. System requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Linux, macOS, or Windows | Linux, or Windows + WSL2 (Ubuntu 22.04) |
| Python | 3.11 | 3.11 |
| Disk for the dataset | ~30 GB | ~50 GB free |
| GPU (training only) | any CUDA-capable NVIDIA card with ≥ 8 GB VRAM | RTX 30/40-series with ≥ 12 GB VRAM |
| CPU-only inference | OK; slower than GPU | — |
| RAM | 16 GB | 32 GB+ for training |

You can run dataset construction and inference on CPU.  Training on CPU
is technically possible but impractically slow — plan on a CUDA GPU for
training.  See "Known issues and cautionary notes" below: WSL/Linux is
 recommended over native Windows for the training step.

### 1. Get the code

```bash
# If the repo is on git:
git clone <repository-url> "Noise2Params Public Facing Code"
cd "Noise2Params Public Facing Code"
# Or just unzip the release archive somewhere and `cd` into it.
```

All commands below assume your current working directory is the root of
this repository (the folder that contains this `README.md`).

### 2. Get the dataset

The dataset is distributed separately as a single archive.  Download
and extract it.  The extracted layout is documented in
`docs/DATA.md`; in summary:

```
Noise2Params Dataset/
├── indist_images_full/     ood_DIV2K_images/
├── indist_image_train_rec/ ood_images_val_rec/  ood_images_test_rec/
├── data_5e6_5/             (saddle-synth training pairs)
├── data_5e6_6/             (Gaussian-synth training pairs)
├── data_real_5e6_2/        (real training pairs)
├── validation_5e6_5/       (real validation pairs)
├── *.csv                   (three recording ↔ base-image mapping CSVs)
├── Extreme Outliers.txt    Hot Pixels Column 30.txt
└── README.txt              (per-folder description files inside each subfolder)
```

* **Recommended**: place `Noise2Params Dataset/` (or symlink it) as
  `./data/` inside this repository.  Most code paths default to that.
  ```bash
  # Linux/macOS:
  ln -s /path/to/Noise2Params\ Dataset ./data
  # Windows (admin Powershell):
  New-Item -ItemType SymbolicLink -Path .\data -Target "C:\path\to\Noise2Params Dataset"
  ```

### 3. Set up a Python environment

You need Python 3.11.  Pick **one** of the three options below.

#### Option A — `venv` (lightweight, no conda)

```bash
# Make sure you have python 3.11 specifically:
python3.11 --version

# Create and activate the venv:
python3.11 -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Upgrade pip just in case:
python -m pip install --upgrade pip
```

#### Option B — `conda` (one-shot install)

```bash
conda env create -f environment.yml
conda activate noise2params
```

Note that `environment.yml` deliberately leaves the CUDA-enabled
PyTorch wheel for you to install in step 4 — see below.

#### Option C — WSL2 (Windows users who plan to train)

If you are on Windows and intend to train models, install WSL2 with
Ubuntu 22.04 and do steps 3–7 inside WSL.  See
<https://learn.microsoft.com/windows/wsl/install>.  Inside the WSL
shell, install Python 3.11 (`sudo apt install python3.11
python3.11-venv`) and then follow Option A or B above from inside WSL.
This is the path that produced the published Noise2Params results;
see "Known issues and cautionary notes" #3 for why we recommend it.

### 4. Install PyTorch with CUDA

This is the trickiest step.  Pick the right wheel for your CUDA driver
version using the official selector at <https://pytorch.org/get-started/locally/>.
The general form is:

```bash
# Example for CUDA 12.1 — adjust the index URL to match your driver:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

To check the wheel actually sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If you only intend to run inference or dataset construction on CPU,
you can install the CPU-only wheel instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

This installs the analysis stack (numpy, numba, scipy, polars,
matplotlib, scikit-image, opencv, Pillow, h5py), the training stack
(lightning, torchmetrics, denoising-diffusion-pytorch, einops), and the
default image-similarity metric backend (`piq`).  The "deep learning"
metrics — LPIPS, DreamSim, DISTS, PieAPP — are commented out in
`requirements.txt` because each pulls in its own large pretrained
weights; uncomment the lines you want and re-run `pip install`.

### 6. Apply the package-pin remedy if needed

If you intend to train (i.e., run `scripts/03_train_cnn.py`) and you
get an error from `denoising_diffusion_pytorch`, `torchmetrics`, or a
flash-attention kernel on the first forward pass, run (inside your
training environment):

```bash
pip uninstall -y denoising-diffusion-pytorch
pip install denoising-diffusion-pytorch==1.10.7
pip install "torchmetrics<1.8.0"
```

This is the combination we found to be reliable across our development
machines.  See "Known issues and cautionary notes" #2 for the longer
discussion.  Users on different hardware generations may need
different pins; if the above does not fix things, look at the error
trace and consult the upstream issue trackers for those packages.

### 7. Verify the install

Quick smoke test — these should all run without error:

```bash
# (a) Imports work:
python -c "import noise2params; import noise2params.noise_image; import noise2params.prob_models; print('imports OK')"

# (b) The training script's --help works (validates that the heavy
#     training-environment imports also resolve):
python noise2image/train_synthetic_6.py --help

# (c) End-to-end functional test: generate a single synthetic noise
#     image from any greyscale .png/.jpg you have lying around.
#     The first call will pay numba's JIT-compilation cost (~5-10 s).
python examples/generate_one_synthetic_image.py path/to/any.png
```

If any of these fail, please consult the "Known issues and cautionary
notes" section before reporting a bug.

## Quickstart

After completing the setup above, with the dataset symlinked at
`./data/`:

```bash
# (1) Reproduce the saddle-point synthetic training set (~minutes to hours
#     depending on CPU; the numba-parallel saddle-point solver does
#     most of the work).  Skip this step if you are using the shipped
#     data_5e6_5/ — it is already built.
python scripts/02_build_synthetic_dataset.py \
    --source ./data/indist_images_full \
    --event-target ./data/data_5e6_5/synth_events \
    --image-target ./data/data_5e6_5/synth_images \
    --model saddle

# (2) Train the saddle-synthetic CNN (Linux / WSL recommended; see
#     caveats below).  ALWAYS pass --batch-size 2 — see caveat #1.
python scripts/03_train_cnn.py \
    --target wsl \
    --data-folder ./data/data_5e6_5 \
    --val-data-folders ./data/validation_5e6_5 \
    --num-epochs 40 --batch-size 2 --lr 2e-5

# (3) Reconstruct the test set with a trained checkpoint and score it
#     against the base images.  Substitute the actual path to the
#     checkpoint your training run produced for `<ckpt-path>`.
python scripts/04_infer.py \
    --ckpt <ckpt-path> \
    --mapping-csv "12-08-40_to_12-36-01_recording_data.csv" \
    --save-folder ./reconstructions/test_saddle \
    --integration-time 5 --rand-seg 5 4

python scripts/05_benchmark.py \
    --save-folder ./reconstructions/test_saddle \
    --name saddle_test --metrics all
```

`configs/training_published.yaml` collects the exact flags for each of
the three published CNN variants.

## Known issues and cautionary notes

### 1. Batch size > 2 causes extreme training slowdowns

On both of the machines we used to develop this code (desktop with an
RTX 5080, laptop with a 4060), a training batch size of 2 ran at the
expected throughput for the hardware.  At any batch size ≥ 3, a single
epoch stretched from ~30 minutes to ~12 hours on the desktop and to
multi-day durations on the laptop.  This is **not** an expected
behaviour, larger batch sizes should normally speed up training. We 
do not believe it is a hardware limitation as we reproduced it on two
distinct GPUs.  We suspect it is a latent bug in the U-Net +
attention architecture (possibly in the `denoising_diffusion_pytorch`
Attend module or its interaction with our FiLM-conditioning) but we
could not track it down.

**To reproduce the paper you must train with `--batch-size 2`.**  If
you encounter the slowdown with a different batch size, treat it as
a known issue rather than a hardware fault.

### 2. Training-environment package compatibility is fragile

Installing the training dependencies has been consistently painful.
The combination that works on our hardware is, approximately:

* Python 3.11
* CUDA-enabled PyTorch wheel matching your driver
* `lightning ~= 2.1.0`
* `torchmetrics < 1.8.0`
* `denoising-diffusion-pytorch == 1.10.7`
* `numpy == 1.26.4`, `numba == 0.59.1`, `scipy == 1.13.1` (shared with the
  analysis env; required by the numba JITs in `noise2params/prob_models.py`)

If you hit errors in attention layers, flash attention kernels, or
PSNR/torchmetric shape mismatches at first forward pass, try (inside the
training venv):

```bash
pip uninstall -y denoising-diffusion-pytorch
pip install denoising-diffusion-pytorch==1.10.7
pip install "torchmetrics<1.8.0"
```

We have observed that upgrading unrelated packages will occasionally
re-break this, and the fix above has to be re-applied.  Users on
different hardware generations may need different pins; we cannot
predict exactly what will work, only that this combination worked for us.

### 3. WSL / Linux strongly recommended for training

We developed and ran the final training on WSL.  The native-Windows
training launcher (`run_train_synthetic_6`) is included because we
wrote it first and it worked, but WSL was less fragile and lended (minor) 
performance benefits.  If you have a choice, use Linux or WSL.

### 4. `EventCountNormalization` is disabled

The training code originally included an
:class:`EventCountNormalization` transform that mean-centres and scales
the event counts by the expected events-per-second, inherited from the
*Noise2Image* paper's pipeline.  We found that **at the 5 s integration
time used in Noise2Params, this normalization hurt validation-set
performance** (possibly due to creating missalignments in the the 
low-intensity regime where our probability models make their most 
distinctive predictions).  The transform is retained in `noise2image/utils.py` 
and in the pipeline stages of `noise_image.py` as an identity 
pass-through, commented accordingly. Do not enable it without understanding 
what it does. It may be the case that normalizing the event counts improves
peformance when working with integration times of variable length (by preventing the 
model from learning too heavily from a fixed time) but we did not explore this.

## Naming conventions

A number of file, function, dataset-folder, and column names in this
release are **historical** — they carry suffixes or prefixes that do
not correspond to anything visible from this repository alone.  They
are retained as-is to minimise the diff between this public code and
the research code that produced the paper, and to keep the shipped
dataset's filenames readable against the mapping CSVs.  They are not 
of consequence here. Some examples:

* `noise2image/train_synthetic_6.py` — the active training entry point.
  The trailing `_6` does not imply that `train_synthetic_1.py` …
  `train_synthetic_5.py` live in this repository; they do not.  It
  likewise now also trains on real (non-synthetic) datasets despite
  the name.
* `data_5e6_5`, `data_5e6_6`, `data_real_5e6_2`, `validation_5e6_5` —
  dataset folder names.  `5e6` refers to the 5 × 10⁶ µs (= 5 s) event
  integration window; the trailing index is a dataset-version counter
  and has no meaning beyond distinguishing successive builds.  The
  only folders distributed with this release are the ones listed in
  `docs/DATA.md`.
* `approach='NB sampling'` in
  `noise2params.noise_image.event_noise_image_modeler_core` — the
  label is historical.  The active branch is a simple
  `Binomial(T, m_eff)` draw.
* Column `" image"` (with a literal leading space) in the mapping
  CSVs — preserved so the CSVs remain readable as shipped (this is 
  is a likely candidate for a future revision).
* `Hot Pixels Column 30.txt` in the dataset folder. Contains outlier
  pixel coordinates for known outlier pixels at default bias settings, 
  includes outlier of all types, not just hot pixels. "Column 30" 
  was employed by a preliminary organization scheme, it has no meaning 
  here.

These names (and others like them) may change in a future revision of
the public release.  For the current release, treat them just as
identifiers.

## Adapted from Noise2Image

The network architecture and the training loop are adapted (with
modifications) from the open-source code accompanying:

> Cao, R., Divekar, S., Nunez-Elizalde, J. K., Kim, K., Waller, L.,
> Olshausen, B. A., Yi, G.  **Noise2Image: Noise-Enabled Static Scene
> Recovery for Event Cameras.**

Specifically: `noise2image/models/unet_attention.py`, `resunet.py`,
`np_transforms.py`, `precomputed_dataset.py`, and the overall Lightning
training scaffold in `train.py`.  Our modifications live in
`train_synthetic_6.py` (entry point, multi-validation-dataloader
support, FiLM time-conditioning on a fixed 5 s integration, flash-attn
SDPA fallback for Windows) and in the non-learned data pipeline in
`noise2params/noise_image.py`.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
