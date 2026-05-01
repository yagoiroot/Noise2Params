# Noise2Params Reconstruction Pipeline

End-to-end dataflow for the deep-learning noise-image reconstruction
portion of Noise2Params.

```
(Unsplash / DIV2K source images)
            │
            │  scripts/prepare_dataset (optional; not needed for shipped data)
            ▼
(greyscale 16:9 base images: indist_images_full, ood_DIV2K_images)
            │
    ┌───────┴─────────────────────────────────┐
    │                                         │
    ▼                                         ▼
scripts/02_build_synthetic_dataset    (base images are displayed on an LED
(noise_image.synthetic_data_base_maker)  screen and recorded by the Prophesee
model ∈ {saddle, Gaussian, Poisson})     EC -> metavision_file_to_parquet_v2)
    │                                         │
    ▼                                         ▼
data_5e6_5/ (saddle)                   indist_image_train_rec/*.parquet
data_5e6_6/ (Gaussian)                 ood_images_val_rec/*.parquet
    │                                  ood_images_test_rec/*.parquet
    │                                         │
    │                            scripts/01_build_real_dataset
    │                            (noise_image.real_data_base_maker
    │                             takes rand_seg=(5,4), 2x2 pixel bin)
    │                                         │
    │                                         ▼
    │                            data_real_5e6_2/ (training real)
    │                            validation_5e6_5/ (validation real)
    │                                         │
    └─────────────┬───────────────────────────┘
                  │
                  ▼
scripts/03_train_cnn  -> noise2image/train_synthetic_6.py
(U-Net + multi-head self-attention, ~35.7 M params,
 FiLM time conditioning, 5 s integration time,
 batch_size = 2  [see caveat], mixed-precision fp16,
 early stopping on multi-metric val)
                  │
                  ▼
lightning_logs/.../checkpoints/last.ckpt
                  │
                  ▼
scripts/04_infer  (single recording / group via mapping CSV)
 -> noise_image.infer_recording / group_infer_recording
                  │
                  ▼
reconstructed .png files in ./reconstructions/...
                  │
                  ▼
scripts/05_benchmark  -> noise_image.group_sim_metrics_2
(PSNR, NCC, SSIM, MS-SSIM, 3-SSIM, FSIM, VIF, VSI, GMSD,
 LPIPS, DreamSim, DISTS, PieAPP)
                  │
                  ▼
<save-folder>/<name>.csv  (one row per predicted image)
```

## Stage-by-stage detail

### Stage A — base image preparation (optional)

Input: arbitrary `.jpg`/`.png` images.
Code: :func:`noise2params.noise_image.prepare_dataset`.
Action: convert to greyscale, crop to 16:9, save `_mod.png`.

Skip this stage if using the shipped `indist_images_full` /
`ood_DIV2K_images` sets — they are already processed.

### Stage B1 — synthetic noise-image generation

Input: a folder of greyscale base images.
Code: :func:`noise2params.noise_image.synthetic_data_base_maker` →
:func:`noise2params.noise_image.event_noise_image_modeler_core`.
Per-image actions (see :file:`configs/params_published.yaml` for
numerical values):

1. Resize to `(1280, 720)` (or keep full-size; random choice with `--crop-or-resize rand`).
2. Random horizontal flip (p=0.5), random vertical flip (p=0.5), center-crop to `(720, 1280)`.
3. Optional brightness augmentation (paper: disabled).
4. Map greyscale → illuminance via `I_lux = 2.15e-5 * I_gs**2.52 + 0.15`.
5. `λ = α · I_lux` with α=4.5.
6. `θ(λ) = c1 + c2√λ + c3λ` per pixel, per polarity.
7. Per-pixel `B` drawn from `TruncNorm(μ=0.15, σ=0.00651, [0, +∞))`.
8. Per-pixel `(P_pos, P_neg)` under the chosen probability model
   (saddle / Gaussian / Poisson).
9. Dead-time correction: `P_eff = P / (1 + P · R)` with R=79 µs.
10. Sample event counts: `Binomial(T, m_eff)` with T=5×10⁶ µs.
11. 2×2 block-average the counts and the base image → `(2, 360, 640)` events + `(360, 640)` image.
12. Save stacked arrays to `events_all.npy` / `images_all.npy`.

### Stage B2 — real noise-image compilation

Input: greyscale base images + matching Prophesee `.parquet` recordings.
Code: :func:`noise2params.noise_image.real_data_base_maker` →
:func:`noise2params.noise_image.event_image_compiler` →
:func:`noise2params.reading.counts_pixel_array`.
Per-pair actions:

1. Load base image, resize to `(1280, 720)`, 2×2 block-average, divide by 255.
2. Randomly select a 5 s segment of the recording (`rand_seg=(5,4)`).
3. Accumulate events into `(pos_counts, neg_counts)` arrays with outliers removed.
4. 2×2 block-average.
5. Stack as `(2, 360, 640)` events + `(360, 640)` image; save.

### Stage C — training

Input: a training `data_folder` (from B1 or B2) + validation
`val_data_folders`.
Code: :file:`noise2image/train_synthetic_6.py` (launched via
:func:`noise2params.utils.run_train_synthetic_6_wsl`).
Architecture: `noise2image/models/unet_attention.py` — U-Net with linear
attention at resolutions 1-3 and full dot-product attention at the
bottleneck.  FiLM time-conditioning takes the 5 s scalar through a
sinusoidal encoding and 2-layer MLP.  Base feature dim 64.
Training: MSE loss, Adam, **batch size 2** (see README caveat), up to
40–100 epochs with 5/7-epoch early stopping on a multi-metric validation
callback.

### Stage D — inference

Input: a single recording or a mapping CSV + a `.ckpt`.
Code: :func:`noise2params.noise_image.infer_recording` /
:func:`noise2params.noise_image.group_infer_recording`.
Per-recording actions:

1. Load events, filter outliers, extract a 5 s random segment.
2. Accumulate into `(pos, neg)` per-pixel counts.
3. 2×2 block-average → `(360, 640)` each.
4. Stack as `(H, W, 2)` and run through the CNN with
   `integration_time_s=5`.
5. Clamp output to `[0, 1]`, scale to `uint8`, save as `.png` (group
   mode) or display (single-file mode).

### Stage E — benchmarking

Input: folder of reconstructed `.png` files (from stage D), named
`{base_stem}_...png`.
Code: :func:`noise2params.noise_image.group_sim_metrics_2` →
:func:`noise2params.noise_image.compare_image_metrics`.
Action: for each predicted image, locate the base image by stem, compute
the requested similarity metrics, and write/merge the scores into a CSV.
