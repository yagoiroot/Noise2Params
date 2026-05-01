"""
noise2params.utils — generic helpers used throughout the reconstruction pipeline.

    * :func:`file_checker` — fuzzy file lookup under the project root.
      Heavily used: many other functions accept bare file stems rather
      than full paths.
    * :func:`text_to_data_format` / :func:`data_to_text_data_format` —
      serialise / deserialise arbitrary Python data structures to
      their ``repr()`` as text.  Used by
      :mod:`noise2params.outlier_masks` to read the shipped outlier lists.
    * :func:`sample_truncated`, :func:`sample_truncnorm`, :func:`sample_trunct` —
      inverse-CDF truncated-distribution sampling.  The Noise2Params paper
      uses only the ``'normal'`` variant (truncated normal on [0, +∞)
      for per-pixel B and θ sampling).  **Do not use anything other
      than ``dist='normal'`` to reproduce the paper.**
    * :func:`center_crop`, :func:`find_content_bounds`, :func:`crop_black_bars`
      — image cropping helpers used by the synthetic/real dataset
      builders and by :func:`compare_image_metrics`.
    * Image-similarity metrics:
      :func:`normalized_cross_correlation`, :func:`ms_ssim_numpy`,
      :func:`three_ssim`, :func:`fsim`, :func:`vif`, :func:`gmsd`,
      :func:`vsi`, :func:`lpips`, :func:`DreamSim`, :func:`dists`,
      :func:`pieapp`.  Each imports its backend (:mod:`piq`,
      :mod:`lpips`, :mod:`dreamsim`, :mod:`pyiqa`) lazily inside the
      function so users without those packages can still import the
      module.  Marked optional in :file:`requirements.txt`.
    * :func:`run_train_synthetic_6` — launches
      :file:`noise2image/train_synthetic_6.py` under a native-Windows
      Python interpreter.
    * :func:`run_train_synthetic_6_wsl` — launches the same script under
      a WSL bash shell with a Linux virtualenv.  **This is the path
      that produced the Noise2Params paper results.**  Linux/WSL is
      recommended; the native-Windows path is shipped because it was
      developed alongside.
    * :func:`convert_windows_path_to_wsl` — tiny helper for the WSL
      launcher.
"""

import os
import time
import subprocess
import math
from pathlib import Path
import difflib

import numpy as np
import scipy
import scipy.stats
import polars as pl
import matplotlib.colors as mcolors
from scipy.special import gammaln, logsumexp
import torch


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Text serialisation

def data_to_text_data_format(l, location):
    """Write a Python data structure to a text file as its ``repr()``.

    The inverse is :func:`text_to_data_format`.  Works for any data
    structure whose string representation round-trips through :func:`eval`.
    """
    with open(location, "w") as f:
        f.write(str(l))
        f.close()
    print(f'file {location} created successfuly')


def text_to_data_format(location):
    """Read a text file containing a Python literal and return the value.

    Uses a restricted :func:`eval` (no builtins; ``nan``/``NaN`` mapped to
    :data:`math.nan`).
    """
    with open(location) as f:
        text = f.read()
        f.close()
    output = eval(text, {"__builtins__": {}}, {"nan": math.nan, "NaN": math.nan})
    return output


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Truncated-distribution sampling

def sample_truncnorm(mu, sigma, size, a=0, b=1):
    """
    Draw ``size`` samples from N(mu, sigma^2) truncated to ``[a, b]``,
    using rejection sampling + Box–Muller for integer ``size``, or
    :func:`scipy.stats.truncnorm.rvs` when ``size`` is an array whose
    shape should be matched.
    """
    if type(size)==int:
        out = np.empty(size, dtype=np.float64)
        for i in range(size):
            while True:
                u1 = np.random.random()
                u2 = np.random.random()
                z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
                x = mu + sigma * z
                if a <= x <= b:
                    out[i] = x
                    break
        return out
    if type(size)==np.ndarray:
        size=np.shape(size)
        a = (a - mu) / sigma
        b = (b - mu) / sigma
        return scipy.stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)


def sample_trunct(df, loc, scale, size, a=0, b=1):
    """Student's t truncated to ``[a, b]`` via inverse-CDF.

    NOT used by the Noise2Params paper (which uses the truncated-normal
    branch of :func:`sample_truncated`); retained as a companion to that
    function.
    """
    if isinstance(size, np.ndarray):
        size = np.shape(size)

    a_std = (a - loc) / scale
    b_std = (b - loc) / scale

    cdf_a = scipy.stats.t.cdf(a_std, df)
    cdf_b = scipy.stats.t.cdf(b_std, df)

    u = np.random.uniform(cdf_a, cdf_b, size=size)
    samples_std = scipy.stats.t.ppf(u, df)
    samples = loc + scale * samples_std
    return samples


def sample_truncated(dist='normal', size=1, loc=0, scale=1, df=None, alpha=None,
                     lambda_param=None, match_variance=False, a=0, b=1, seed=None):
    """
    Draw samples from a truncated distribution using the inverse-CDF method.

    Supports ``'normal'``, ``'t'``, ``'skewnorm'``, ``'skewnorm same mean'``,
    and ``'emg'`` (exponentially modified Gaussian).  Only ``'normal'`` is
    used to reproduce the Noise2Params paper — per-pixel B and θ are drawn
    from truncated normals with the parameters reported in the paper.
    **Use only ``dist='normal'`` to reproduce the paper.**

    Parameters
    ----------
    dist : {'normal', 't', 'skewnorm', 'skewnorm same mean', 'emg'}
    size : int, tuple, or ndarray
        If int/tuple, the sample shape; if an ndarray, its shape is used.
    loc, scale : float
        Location / scale (mean / std for normal).
    df : float, optional
        Degrees of freedom (t-distribution only).
    alpha : float, optional
        Skewness (skewnorm only).
    lambda_param : float, optional
        Exponential rate (EMG only).
    match_variance : bool
        EMG only: solve for σ to match N(loc, scale²) variance.
    a, b : float
        Truncation bounds.
    seed : int, optional
        RNG seed.

    Returns
    -------
    ndarray
        Samples from the specified truncated distribution.
    """
    rng = np.random.default_rng(seed)

    if isinstance(size, np.ndarray):
        size = np.shape(size)

    a_std = (a - loc) / scale
    b_std = (b - loc) / scale

    if dist == 'normal':
        cdf_a = scipy.stats.norm.cdf(a_std)
        cdf_b = scipy.stats.norm.cdf(b_std)
        u = rng.uniform(cdf_a, cdf_b, size=size)
        samples_std = scipy.stats.norm.ppf(u)

    elif dist == 't':
        if df is None:
            raise ValueError("df (degrees of freedom) must be specified for t-distribution")
        cdf_a = scipy.stats.t.cdf(a_std, df)
        cdf_b = scipy.stats.t.cdf(b_std, df)
        u = rng.uniform(cdf_a, cdf_b, size=size)
        samples_std = scipy.stats.t.ppf(u, df)

    elif dist == 'skewnorm':
        if alpha is None:
            raise ValueError("alpha (skewness parameter) must be specified for skew normal distribution")
        cdf_a = scipy.stats.skewnorm.cdf(a_std, alpha)
        cdf_b = scipy.stats.skewnorm.cdf(b_std, alpha)
        u = rng.uniform(cdf_a, cdf_b, size=size)
        samples_std = scipy.stats.skewnorm.ppf(u, alpha)

    elif dist == 'skewnorm same mean':
        if alpha is None:
            raise ValueError("alpha must be specified for skew normal distribution")
        delta = alpha / np.sqrt(1 + alpha ** 2)
        mean_offset = scale * delta * np.sqrt(2 / np.pi)
        adjusted_loc = loc - mean_offset
        a_std = (a - adjusted_loc) / scale
        b_std = (b - adjusted_loc) / scale
        cdf_a = scipy.stats.skewnorm.cdf(a_std, alpha)
        cdf_b = scipy.stats.skewnorm.cdf(b_std, alpha)
        u = rng.uniform(cdf_a, cdf_b, size=size)
        samples_std = scipy.stats.skewnorm.ppf(u, alpha)
        return adjusted_loc + scale * samples_std

    elif dist == 'emg':
        if lambda_param is None:
            raise ValueError("lambda_param must be specified for EMG distribution")
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive")
        mu_emg = loc - 1 / lambda_param

        if match_variance:
            var_exp = 1 / lambda_param ** 2
            if scale ** 2 <= var_exp:
                raise ValueError(
                    f"Cannot match variance: scale² ({scale ** 2:.4f}) must exceed "
                    f"1/λ² ({var_exp:.4f}). Increase scale or lambda_param.")
            sigma_emg = np.sqrt(scale ** 2 - var_exp)
        else:
            sigma_emg = scale

        K = 1 / (lambda_param * sigma_emg)
        cdf_a = scipy.stats.exponnorm.cdf(a, K, loc=mu_emg, scale=sigma_emg)
        cdf_b = scipy.stats.exponnorm.cdf(b, K, loc=mu_emg, scale=sigma_emg)
        if cdf_b - cdf_a < 1e-10:
            raise ValueError("Truncation bounds exclude nearly all probability mass")
        u = rng.uniform(cdf_a, cdf_b, size=size)
        samples = scipy.stats.exponnorm.ppf(u, K, loc=mu_emg, scale=sigma_emg)
        return samples

    else:
        raise ValueError(f"Unknown distribution: {dist}. Use 'normal', 't', 'skewnorm', or 'emg'")

    samples = loc + scale * samples_std
    return samples


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image cropping

def center_crop(arr, crop_size):
    """Center-crop an ndarray to the given shape.  ``crop_size`` may be
    a 2- or 3-tuple (height, width[, depth])."""
    start_indices = [(arr.shape[i] - crop_size[i]) // 2 for i in range(len(crop_size))]
    end_indices = [start + size for start, size in zip(start_indices, crop_size)]
    slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
    return arr[slices]


def find_content_bounds(image: np.ndarray, threshold: int = 10):
    """Return the bounding box ``(top, bottom, left, right)`` of the
    non-black region of a grayscale image (rows/cols with no pixel
    above ``threshold`` are treated as black bars)."""
    row_has_content = np.max(image, axis=1) >= threshold
    col_has_content = np.max(image, axis=0) >= threshold
    rows = np.where(row_has_content)[0]
    cols = np.where(col_has_content)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, image.shape[0], 0, image.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def crop_black_bars(img1: np.ndarray, img2: np.ndarray, threshold: int = 10):
    """Detect black bars in ``img1`` and crop both images to the matching bounds."""
    top, bottom, left, right = find_content_bounds(img1, threshold)
    return img1[top:bottom, left:right], img2[top:bottom, left:right]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image-similarity metrics.  Each function imports its backend lazily.

def normalized_cross_correlation(img1, img2):
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.mean(img1_norm * img2_norm)


def ms_ssim_numpy(img1, img2, backend='piq'):
    """Multi-scale SSIM via :mod:`piq` (default) or :mod:`sewar`."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shapes differ: {img1.shape} vs {img2.shape}")
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    data_range = 1.0 if np.issubdtype(img1.dtype, np.floating) else 255.0

    if backend == 'piq':
        import piq
        img1_f = img1.astype(np.float32) / data_range
        img2_f = img2.astype(np.float32) / data_range
        if img1_f.ndim == 2:
            img1_tensor = torch.from_numpy(img1_f).unsqueeze(0).unsqueeze(0)
            img2_tensor = torch.from_numpy(img2_f).unsqueeze(0).unsqueeze(0)
        else:
            img1_tensor = torch.from_numpy(img1_f).permute(2, 0, 1).unsqueeze(0)
            img2_tensor = torch.from_numpy(img2_f).permute(2, 0, 1).unsqueeze(0)
        score = piq.multi_scale_ssim(img1_tensor, img2_tensor, data_range=1.0)
        return score.item()
    elif backend == 'sewar':
        from sewar.full_ref import msssim
        if img1.ndim == 2:
            img1_ = img1[..., None]; img2_ = img2[..., None]
        else:
            img1_ = img1; img2_ = img2
        return float(msssim(img1_, img2_, MAX=data_range))
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'sewar' or 'piq'.")


def three_ssim(ref, dist, edge_weight=0.5, tex_weight=0.25, smooth_weight=0.25,
               grad_percentiles=(33.0, 66.0)):
    """3-SSIM (Li & Bovik three-component SSIM).  Uses
    :mod:`skimage`'s SSIM and Sobel for the edge/texture/smooth partition."""
    from skimage.color import rgb2gray
    from skimage.filters import sobel
    from skimage.metrics import structural_similarity as ssim

    if ref.shape != dist.shape:
        raise ValueError(f"Shapes differ: {ref.shape} vs {dist.shape}")

    ref = np.asarray(ref); dist = np.asarray(dist)

    if ref.ndim == 3 and ref.shape[2] == 3:
        ref_gray = rgb2gray(ref); dist_gray = rgb2gray(dist)
    elif ref.ndim == 2:
        ref_gray = ref.astype(np.float64); dist_gray = dist.astype(np.float64)
    else:
        raise ValueError("Expected shape (H, W) or (H, W, 3)")

    ref_gray = ref_gray.astype(np.float64); dist_gray = dist_gray.astype(np.float64)
    data_range = ref_gray.max() - ref_gray.min()
    if data_range == 0:
        return float(np.allclose(ref_gray, dist_gray))

    ssim_mean, ssim_map = ssim(ref_gray, dist_gray, data_range=data_range,
                               gaussian_weights=True, sigma=1.5,
                               use_sample_covariance=False, full=True)
    grad_mag = sobel(ref_gray)
    p1, p2 = grad_percentiles
    t_smooth, t_edge = np.percentile(grad_mag, [p1, p2])

    smooth_mask = grad_mag <= t_smooth
    edge_mask = grad_mag >= t_edge
    texture_mask = (~smooth_mask) & (~edge_mask)

    def region_mean(mask):
        return float(ssim_map[mask].mean()) if np.any(mask) else 0.0

    s_edge = region_mean(edge_mask)
    s_tex = region_mean(texture_mask)
    s_smooth = region_mean(smooth_mask)

    return edge_weight * s_edge + tex_weight * s_tex + smooth_weight * s_smooth


def fsim(img1, img2):
    import piq
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1); img1 = np.repeat(img1, 3, axis=-1)
    if img2.ndim == 2:
        img2 = np.expand_dims(img2, axis=-1); img2 = np.repeat(img2, 3, axis=-1)
    if np.max(img1)>255:
        img1=img1*(1/np.max(img1))
    if np.max(img2)>255:
        img2 = img2 * (1 / np.max(img2))
    img1=np.clip(img1, 0, 255).astype('uint8')
    img2 = np.clip(img2, 0, 255).astype('uint8')
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return piq.fsim(img1_tensor, img2_tensor, data_range=1.0).item()


def _piq_2d(img1, img2, fn_name):
    """Shared preprocess for the :mod:`piq` 2D-grayscale metrics."""
    import piq
    if img1.shape != img2.shape:
        raise ValueError(f"Shapes differ: {img1.shape} vs {img2.shape}")
    img1 = np.asarray(img1); img2 = np.asarray(img2)
    data_range = 1.0 if np.issubdtype(img1.dtype, np.floating) else 255.0
    img1_f = img1.astype(np.float32) / data_range
    img2_f = img2.astype(np.float32) / data_range
    if img1_f.ndim == 2:
        img1_tensor = torch.from_numpy(img1_f).unsqueeze(0).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_f).unsqueeze(0).unsqueeze(0)
    else:
        img1_tensor = torch.from_numpy(img1_f).permute(2, 0, 1).unsqueeze(0)
        img2_tensor = torch.from_numpy(img2_f).permute(2, 0, 1).unsqueeze(0)
    fn = getattr(piq, fn_name)
    return fn(img1_tensor, img2_tensor, data_range=1.0).item()


def vif(img1, img2):  return _piq_2d(img1, img2, 'vif_p')
def gmsd(img1, img2): return _piq_2d(img1, img2, 'gmsd')
def vsi(img1, img2):  return _piq_2d(img1, img2, 'vsi')


def lpips(img1, img2):
    import lpips
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    return loss_fn(img1_tensor, img2_tensor).item()


def DreamSim(img1, img2):
    import dreamsim
    from PIL import Image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = dreamsim.dreamsim(pretrained=True, device=device)
    if isinstance(img1, np.ndarray):
        if img1.ndim == 2:
            img1 = np.stack([img1] * 3, axis=-1)
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        img1 = Image.fromarray(img1)
    if isinstance(img2, np.ndarray):
        if img2.ndim == 2:
            img2 = np.stack([img2] * 3, axis=-1)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
        img2 = Image.fromarray(img2)
    img1_tensor = preprocess(img1).to(device)
    img2_tensor = preprocess(img2).to(device)
    return model(img1_tensor, img2_tensor).item()


def dists(img1, img2):
    import piq
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if img1.ndim == 2: img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2: img2 = np.stack([img2] * 3, axis=-1)
    if img1.dtype == np.uint8 or img1.max() > 1.0:
        img1 = img1.astype(np.float32) / 255.0
    else:
        img1 = img1.astype(np.float32)
    if img2.dtype == np.uint8 or img2.max() > 1.0:
        img2 = img2.astype(np.float32) / 255.0
    else:
        img2 = img2.astype(np.float32)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    dists_metric = piq.DISTS(reduction='mean')
    return dists_metric(img1_tensor, img2_tensor).item()


def pieapp(img1, img2):
    import pyiqa
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('pieapp', device=device)
    if img1.ndim == 2: img1 = np.stack([img1] * 3, axis=-1)
    if img2.ndim == 2: img2 = np.stack([img2] * 3, axis=-1)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device)
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device)
    if img1_tensor.max() > 1.0: img1_tensor = img1_tensor / 255.0
    if img2_tensor.max() > 1.0: img2_tensor = img2_tensor / 255.0
    return metric(img1_tensor, img2_tensor).item()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training-job launchers.  These call the training script in a subprocess so
# that the training Python environment (which has strict version pins; see
# ``noise2image/requirements.txt`` and the repository-level README for the
# known-good pins, incl. denoising-diffusion-pytorch==1.10.7 and
# torchmetrics<1.8.0) can be kept separate from the analysis Python
# environment the rest of ``noise2params`` runs in.

# NOTE ON NAMING: ``train_synthetic_6.py`` is historical — it handles
# training on both synthetic and real datasets, despite the name.  See
# the "Naming conventions" section of the repository README.

def run_train_synthetic_6(data_folder,
                          script_dir=None, python_executable=None,
                          val_data_folders='./data/validation_5e6_5',
                          num_epochs='60', batch_size='2',
                          num_workers='0', prefetch_factor='1',
                          mixed_precision=True, fast_mode=True,
                          extra_args=None):
    """
    Launch :file:`noise2image/train_synthetic_6.py` under a native
    Python interpreter (typically a Windows conda env).

    See :func:`run_train_synthetic_6_wsl` for the WSL/Linux equivalent,
    which is the path that produced the Noise2Params paper results.

    Parameters
    ----------
    data_folder : str
        Folder containing ``synth_events/events_all.npy`` and
        ``synth_images/images_all.npy`` for training.
    script_dir : str, optional
        Directory containing ``train_synthetic_6.py``.  Defaults to the
        ``noise2image/`` directory that sits next to this package.
    python_executable : str, optional
        Absolute path to the Python interpreter that has the training
        dependencies installed.  Required if not already on PATH.
    val_data_folders : str or list of str
        Validation dataset folder(s) (same layout as ``data_folder``).
    num_epochs, batch_size, num_workers, prefetch_factor : str
        Lightning / DataLoader args.  **Use ``batch_size='2'`` to reproduce
        the paper.**  See the caveat in the repository-level README:
        batch_size > 2 has been observed to cause order-of-magnitude
        training slowdowns with the U-Net + attention architecture used
        here, on multiple machines; we believe this is a latent bug in
        the attention-layer wiring but have not resolved it.
    mixed_precision, fast_mode : bool
        Forwarded as ``--mixed_precision`` / ``--fast_mode`` flags.
    extra_args : list of str, optional
        Additional CLI flags appended verbatim.
    """
    if script_dir is None:
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'noise2image'))
    if python_executable is None:
        python_executable = 'python'

    script_path = os.path.join(script_dir, "train_synthetic_6.py")

    if isinstance(val_data_folders, str):
        val_data_folders = [val_data_folders]

    args = [
        "--data_folder", data_folder,
        '--val_data_folders', *val_data_folders,
        '--format', 'memmap',
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers),
        '--num_epochs', str(num_epochs),
        '--prefetch_factor', str(prefetch_factor),
    ]
    if fast_mode:
        args.append('--fast_mode')
    if mixed_precision:
        args.append('--mixed_precision')
    if extra_args:
        args.extend(extra_args)

    command = [python_executable, script_path] + args

    try:
        subprocess.run(command, cwd=script_dir, text=True)
        print("Script executed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def run_train_synthetic_6_wsl(data_folders='~/datasets/data_real_5e6_2',
                              val_data_folders='~/datasets/validation_5e6_5',
                              num_epochs='40', resume_from=None, lr='5e-5',
                              wsl_project_dir='~/projects/noise2image',
                              wsl_venv_python='~/.venvs/n2i/bin/python',
                              integration_time='5'):
    """
    Launch :file:`noise2image/train_synthetic_6.py` inside a WSL shell.

    This is the path that produced the published Noise2Params models.
    WSL / Linux is recommended due to fewer package-compatibility
    surprises (notably around CUDA-enabled PyTorch wheels, flash
    attention, and the `denoising-diffusion-pytorch` package's interplay
    with `torchmetrics`).

    Parameters
    ----------
    data_folders : str or list of str
        Training dataset folder(s) as WSL-side paths (e.g.
        ``~/datasets/data_5e6_5``).
    val_data_folders : str or list of str
        Validation dataset folder(s).
    num_epochs : str
    resume_from : str, optional
        Path to a ``.ckpt`` to resume training from.
    lr : str
        Initial learning rate.  ``5e-5`` for experimental-data training;
        ``2e-5`` for synthetic-only training — as reported in the paper.
    wsl_project_dir : str
        WSL-side path to the ``noise2image/`` directory.
    wsl_venv_python : str
        WSL-side path to the Python interpreter inside the training venv.
    integration_time : str
        Event integration time in seconds (5 for Noise2Params).
    """
    if isinstance(data_folders, str):
        data_folders = [data_folders]
    else:
        data_folders = list(data_folders)

    if isinstance(val_data_folders, str):
        val_data_folders = [val_data_folders]
    else:
        val_data_folders = list(val_data_folders)

    args = [
        "--data_folders", *data_folders,
        '--val_data_folders', *val_data_folders,
        '--format', 'memmap',
        '--batch_size', '2',
        '--num_workers', '1',
        '--num_epochs', num_epochs,
        '--lr', lr,
        '--mixed_precision',
        '--prefetch_factor', '1',
        '--integration_time', integration_time,
    ]

    if resume_from is not None:
        args.extend(['--resume_from', resume_from])

    wsl_command = (
        f"cd {wsl_project_dir} && "
        f"source {wsl_venv_python.replace('/bin/python', '/bin/activate')} && "
        f"{wsl_venv_python} train_synthetic_6.py {' '.join(args)}"
    )

    print(f"Executing in WSL: {wsl_command}\n")

    try:
        result = subprocess.run(["wsl", "bash", "-c", wsl_command],
                                text=True, check=True)
        print("\nScript executed successfully in WSL!")
        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running in WSL: {e}")
        raise
    except FileNotFoundError:
        print("ERROR: WSL not found. Is WSL installed and in your PATH?")
        raise


def convert_windows_path_to_wsl(windows_path):
    """Convert a Windows filesystem path to the equivalent ``/mnt/...``
    WSL path.  Leaves paths starting with ``~`` or ``/`` unchanged."""
    if windows_path.startswith(('~', '/')):
        return windows_path
    path = Path(windows_path)
    drive = path.drive.rstrip(':').lower()
    path_without_drive = str(path).replace(path.drive, '').replace('\\', '/')
    return f"/mnt/{drive}{path_without_drive}"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File-lookup helper

def file_checker(file_name, file_type):
    """
    Walk the project root and return the full path of the first file
    whose stem matches ``file_name`` and whose extension matches
    ``file_type``.

    The "project root" is the folder two levels above this file
    (``<Noise2Params Public Facing Code>/..``), matching how the internal
    codebase searches the overall "BPL Lab" directory tree.  If
    ``file_name`` is already an absolute path to an existing file, it is
    returned as-is.

    Parameters
    ----------
    file_name : str
        Full path OR bare stem (no extension) of the file.
    file_type : str or list of str
        Extension (``'csv'`` / ``'.csv'``) or list of extensions.

    Returns
    -------
    str
        Full path to the file.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.  Includes the closest :mod:`difflib`
        suggestion when one is available.
    """
    scrip_dir = os.path.abspath(__file__)
    # Project root: two levels above noise2params/utils.py
    # (<root>/noise2params/utils.py  ->  <root>/..)
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(scrip_dir)))

    if os.path.isfile(file_name):
        print(f"Looking at file '{file_name}'.")
        return file_name

    try:
        all_files = []
        for root, dirs, files in os.walk(proj_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
    except FileNotFoundError:
        raise FileNotFoundError(f"The directory {proj_dir} does not exist.")

    if type(file_type)==str:
        same_type_files = [file for file in all_files if file_type in str(file)]
    if type(file_type)==list:
        same_type_files = [file for file in all_files if any(ext in str(file) for ext in file_type)]

    file_found=False
    for file in same_type_files:
        curr_file_1=os.path.basename(file)
        curr_file_name, curr_file_extension = os.path.splitext(curr_file_1)
        if file_name==curr_file_name and 'tmp' not in curr_file_extension:
            file_found=True
            print(f"Looking at file '{file}'.")
            return file

    if file_found==False:
        same_type_files_base=[os.path.basename(file) for file in same_type_files]
        closest_matches = difflib.get_close_matches(file_name, same_type_files_base, n=1, cutoff=0.6)
        if closest_matches:
            suggestion = closest_matches[0]
            suggestion_name, suggestion_extension = os.path.splitext(suggestion)
            raise FileNotFoundError(
                f"The file '{file_name}' with file type {file_type} does not exist. "
                f"Did you mean '{suggestion_name}' with file type '{suggestion_extension}'?")
        else:
            raise FileNotFoundError(f"The file '{file_name}' does not exist and no similar files were found.")
