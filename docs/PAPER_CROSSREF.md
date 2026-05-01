# Paper Symbols ↔ Code Variables

This document cross-references the mathematical symbols in the
Noise2Params paper with their names in the code, the function where they
live, and (when applicable) the default numerical value used for the
published experiments.

Equation numbers in the **Paper loc.** column refer to equations in the
current paper draft at
`Manuscript Materials/_Draft__Event_Camera_Probabilities_*.pdf`
(and the LaTeX source at `Manuscript Materials/main.tex` /
`Manuscript Materials/supplemental_info.tex`).  Where equation numbers
are not yet stable, section names are given instead.

## Core probability-model parameters

| Symbol | Meaning | Code variable | Where | Default | Paper loc. |
|---|---|---|---|---|---|
| B | Log-contrast threshold | `B`, `B_array`, `B_pos_array`, `B_neg_array` | `noise_image.event_noise_image_modeler_core` | 0.15 | Sec. "Event-detection model"; thresholding condition Z₊ > 0 |
| σ_B | Per-pixel B truncated-normal std | `B_sigma` | same | 0.0065 | Sec. "Synthetic noise image generation" |
| α | Lux-to-photon conversion factor | `alpha` | same | 4.5 | Sec. "Lambda / illuminance mapping" |
| λ | Mean photon count per pixel | `lam_array` | same | `α · I` | Eq. for Z₊/Z₋ |
| I (lux) | Illuminance at the pixel | `data` (post `int_mapping`) | same | — | — |
| R | Refractory / dead time | `dead_time` | same | 79 µs | Sec. "Dead-time correction" |
| T | Integration window | `time_steps` | same | 5 × 10⁶ µs (5 s) | Sec. "Synthetic noise image generation" |

## Leakage model θ(λ)

Paper form:  θ(λ) = c₁ + c₂√λ + c₃λ    (3-parameter fit, used for publication)

| Symbol | Code | Location | Positive events | Negative events |
|---|---|---|---|---|
| c₁ | `params_pos[0]` / `params_neg[0]` | `prob_models.theta_model(..., fit_form=4)` | 18.9173... | 16.4175... |
| c₂ | `params_pos[1]` / `params_neg[1]` | same | 35.4904... | 37.4225... |
| c₃ | `params_pos[2]` / `params_neg[2]` | same | 0.4394... | 0.0676... |

Additive dark-count floor in event probability:

| Symbol | Code | Default |
|---|---|---|
| P⁰₊ (baseline positive prob.) | `vert_offset_pos` | 9.568 × 10⁻⁹ |
| P⁰₋ (baseline negative prob.) | `vert_offset_neg` | 3.179 × 10⁻⁸ |

## Probability distributions

| Paper | Function | Notes |
|---|---|---|
| Exact Poisson P(Z₊ > 0) | `prob_models.pos_event_prob_vec_numba_2` | Single-sum truncation; `n_max = n_0_max = 300` default |
| Gaussian approx. P(Z₊ > 0) | `prob_models.pos_event_prob_gaussian` | Valid λ ≳ 10 |
| Gaussian approx. P(Z₋ < 0) | `prob_models.neg_event_prob_gaussian` |  |
| Saddle-point P(Z₊ > 0)    | `prob_models.pos_prob_saddle_bracket_numba` (and `pos_prob_saddle_bracket` for the pure-NumPy forms) | Primary path |
| Saddle-point P(Z₋ < 0)    | `prob_models.neg_prob_saddle_bracket_numba` |  |

The cumulant-generating function κ(s) and its derivatives live in
`prob_models.kappa` / `kappa_p` / `kappa_pp`.

## Dead-time-corrected effective probabilities

Paper: `P_eff = (P₊ + P₋) / (1 + (P₊ + P₋) · R)`, with per-polarity
effective means `m_pos = P_eff · P₊/(P₊+P₋)` and `m_neg = P_eff - m_pos`.

| Code | Location |
|---|---|
| `tot_prob_array` | `event_noise_image_modeler_core` |
| `m_tot` | same |
| `m_pos`, `m_neg` | same |

## Image dimensions

| Paper | Code |
|---|---|
| EC resolution 1280 × 720 | `cur_data` shape in `synthetic_data_base_maker` |
| 2×2 spatially-binned 640 × 360 | `synth_counts_pos_binned` / `block_mean(arr, 2)` |
| `(N, 2, 360, 640)` training events | `events_all.npy` |
| `(N, 360, 640)` training images | `images_all.npy` |

## Noise2Image (upstream) identifiers

The training script imports :class:`noise2image.train.Model` — the
Lightning wrapper around the U-Net defined in
:file:`noise2image/models/unet_attention.py`.  This architecture is
adapted from Cao *et al.*, Noise2Image;  Our primary modifications are: 2-channel
(pos, neg) input, FiLM time-conditioning at a scalar 5 s, and the
multi-dataloader early-stopping callback
(`train.MultiDataloaderEarlyStopping`).

## Paper figures / tables to code locations

| Figure / table | Produced by |
|---|---|
| Three-way probability-distribution comparison (Poisson / saddle / Gaussian) | `prob_models.*`; the paper's plotter for this figure is not included in this public code |
| Synthetic vs. real noise-image panel | Call `event_noise_image_modeler_core` + `event_image_compiler` side-by-side; the paper's plotter for this figure is not included |
| CNN reconstruction metric comparisons | `group_sim_metrics_2` on folders of predicted images |
| Inference-time benchmark | Column `infer_time` returned from `predict_from_events(return_time=True)` / `infer_recording` |

If a figure or table is not listed here, it corresponds to analyses
outside the scope of this public release (specifically: parameter
fitting, S-curve / step-response comparisons, and their
figure-generation routines — not included in this public code).
