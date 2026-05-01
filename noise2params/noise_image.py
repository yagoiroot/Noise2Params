"""
noise2params.noise_image — noise-image pipeline.

Synthetic noise-image generation
--------------------------------
:func:`event_noise_image_modeler_core`
    Core generator.  Takes a base greyscale image and a probability
    model (``'Gaussian'`` / ``'saddle'`` / ``'Poisson'``) and returns
    synthetic positive/negative event-count arrays.
:func:`event_noise_image_modeler`
    Thin wrapper with default kwargs and optional display.

Real noise-image compilation from event recordings
--------------------------------------------------
:func:`event_image_compiler`
    Accumulates events from a Prophesee ``.parquet`` recording into a
    per-pixel count image.

Image-similarity metric panels
------------------------------
:func:`compare_image_metrics`, :func:`two_image_compare_to_reference`,
:func:`group_sim_metrics_2`.

Inference
---------
:func:`load_compiled_checkpoint`, :func:`predict_from_events`,
:func:`configure_deterministic_inference`, :func:`infer_recording`,
:func:`group_infer_recording`.

Dataset construction
--------------------
:func:`synthetic_data_base_maker`, :func:`real_data_base_maker`,
:func:`_save_as_memmap`, :func:`_save_as_hdf5`, :func:`prepare_dataset`,
:func:`block_mean`.

Notes on per-pixel B and θ sampling (IMPORTANT)
-----------------------------------------------
In the published paper, per-pixel B and θ variations are both drawn from
**truncated normal distributions** using the parameters given in the
paper's experimental section.  See
:func:`noise2params.utils.sample_truncated` — only ``dist='normal'`` is
relevant to reproducing the paper.
"""

import os
import time
import itertools

import PIL.ImageOps
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import polars as pl
import numpy as np, math
from scipy import optimize
import scipy
import cv2
import random

from noise2image.train import Model
import torch

from noise2params import utils as ut
from noise2params import reading as read
from noise2params import prob_models as pmdls


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Synthetic noise-image generation

def event_noise_image_modeler_core(input_image_location='noise2image_control_image',
                              time_steps=1e6, B=0.15, B_sigma=0.009, int_mapping='mine', dead_time=79,
                              params_pos=(18.917319991515015, 35.49036559887983, 0.4394373558876216),
                              params_neg=(16.41747084565865, 37.42249743893243, 0.06759986822465787),
                              vert_offset_pos=9.567536526687546e-09, vert_offset_neg=3.178722048504662e-08,
                              int_mapping_coef=[2.14990578e-05, 2.52106648, 0.15],
                              sample_theta=False, theta_sampler_sigma=0.001,
                              model='saddle', uniform_pixels=True, approach='poisson sampling', alpha=4.5,
                              plot_histogram=False, cap_or_norm='norm', combine_polarities=True,
                              plot=True, return_prob=False):
    """
    Generate a synthetic noise-image from a base greyscale image using the
    chosen probability model.

    Workflow (paper, "Synthetic Noise Image Generation" section):

    1.  Load the base image and convert to greyscale.
    2.  Map greyscale intensity I ∈ [0, 255] to illuminance-equivalent via
        ``int_mapping`` (``'mine'`` applies the polynomial
        ``int_mapping_coef[0] * I**int_mapping_coef[1] + int_mapping_coef[2]``
        fit against lab lux-meter measurements; ``'N2I'`` uses the
        Noise2Image paper's original calibration).
    3.  Compute mean photon count per pixel  λ = α · I.
    4.  Compute θ(λ) per pixel (via :func:`noise2params.prob_models.theta_model`,
        form 4 — the 3-parameter  θ = c₁ + c₂√λ + c₃λ  form).
    5.  Per-pixel B: if ``uniform_pixels=True``, constant.  Otherwise drawn
        from a truncated normal N(B, B_sigma²) restricted to [0, +∞) via
        :func:`noise2params.utils.sample_truncated` — this is the only
        sampling distribution used in the paper.
    6.  Compute per-pixel positive and negative event probabilities under
        the chosen ``model`` ∈ {'Gaussian', 'saddle', 'Poisson'}.
    7.  Apply dead-time correction P^eff = P / (1 + P · R) and split back
        into the per-polarity effective means.
    8.  Sample event counts over the integration window:
          * ``approach='poisson sampling'``: Poisson(m · T).
          * ``approach='NB sampling'``: Binomial(T, m) — used for the
            published synthetic datasets.  (The name is historical, from
            an earlier negative-binomial sweep; the active branch is
            binomial sampling.)
          * ``approach='expectations'``: return the expected counts m·T.
          * ``approach='raw'``: naive 0/1 Bernoulli trials per time step
            (retained for pedagogical reference).
    9.  Optionally sum the polarities and rescale to [0, 255].

    The returned array is the raw (or normalised) count-per-pixel map;
    dataset construction (see :func:`synthetic_data_base_maker`) does its
    own binning and normalisation on top.

    Parameters
    ----------
    input_image_location : str or ndarray
        Path/stem of the base image or a pre-loaded greyscale ndarray.
    time_steps : int
        Integration window length in microseconds.  Noise2Params uses 5 s
        → 5e6.
    B : float
        Log-contrast threshold (paper symbol B).
    B_sigma : float
        Per-pixel B truncated-normal standard deviation.
    int_mapping : {'mine', 'N2I'}
        Greyscale → illuminance mapping choice.
    dead_time : float
        Refractory period R in microseconds (Prophesee EVK4 default: 79).
    params_pos, params_neg : (c1, c2, c3)
        3-parameter θ(λ) coefficients for positive and negative events.
    vert_offset_pos, vert_offset_neg : float
        Additive probability offsets (dark-count floor).
    int_mapping_coef : list of float
        Coefficients for ``int_mapping='mine'``.
    sample_theta : bool
        If True, multiplicatively jitter θ per pixel by a truncated normal
        centered at 1 with std ``theta_sampler_sigma`` (published
        Noise2Params results: disabled).
    model : {'Gaussian', 'saddle', 'Poisson'}
        Probability model.  ``'saddle'`` and ``'Gaussian'`` are the two
        models for which synthetic datasets ship with the paper.
    uniform_pixels : bool
        Constant B per pixel, vs per-pixel truncated-normal B.
    approach : {'raw', 'expectations', 'poisson sampling', 'NB sampling'}
        Event-count sampling method.  ``'NB sampling'`` (binomial) is the
        default for dataset construction.
    alpha : float
        Lux-to-photon conversion factor α (paper parameter).  4.5 is the
        published value for the Prophesee EVK4 the dataset was recorded on.
    plot_histogram, plot, cap_or_norm, combine_polarities, return_prob
        Display / normalisation / return-shape toggles.

    Returns
    -------
    ndarray or tuple
        Depending on ``combine_polarities`` / ``return_prob``:
          * combined:   (H, W) counts array
          * separate:   (counts_pos, counts_neg)
          * +probs:     plus per-pixel (prob_pos, prob_neg, time_steps_array)
    """
    if int_mapping not in [None, 'mine', 'N2I']:
        raise ValueError("int_mapping must be None, 'mine', or 'N2I'")
    if model not in ['Gaussian','saddle', 'Poisson']:
        raise ValueError("model must be 'saddle' or 'Poisson'")
    if approach not in ['raw', 'expectations', 'poisson sampling', 'NB sampling']:
        raise ValueError("approach must be 'raw', 'expectations', 'poisson sampling', or 'NB sampling'")

    # Load the base image: either from disk or via a supplied ndarray.
    if type(input_image_location)==str:
        exts = Image.registered_extensions()
        supported_extensions = [ex for ex, f in exts.items() if f in Image.OPEN]
        # '.h5' is a PIL-registered extension by some installations; our event
        # recordings also use '.h5'.  Remove to avoid ambiguity with the
        # file_checker fuzzy search.
        if '.h5' in supported_extensions:
            supported_extensions.remove('.h5')

        input_image_location = ut.file_checker(input_image_location, supported_extensions)
        img = Image.open(input_image_location).convert('L')
        data = np.array(img)

    if type(input_image_location)==np.ndarray:
        data=input_image_location

    data=data.astype(np.float64)

    # Greyscale-to-illuminance mapping.  The 'mine' branch uses the
    # lab-calibrated polynomial coefficients in int_mapping_coef.
    if int_mapping=='mine':
        a=int_mapping_coef[0]
        b=int_mapping_coef[1]
        c=int_mapping_coef[2]
        data = a * (data ** b) + c
    if int_mapping=='N2I':
        # The original Noise2Image paper's grayscale-to-lux mapping.
        # The `n2i` helper module is not included in this public copy;
        # callers using ``int_mapping='N2I'`` must provide it via the
        # upstream Noise2Image code (noise2image.Noise2Image_code).
        import noise2image.Noise2Image_code as n2i  # lazy import
        data = n2i.grayscale_to_lux(data)

    lam_array = data * alpha
    # Intensity-dependent leakage theta(lambda) per pixel, per polarity.
    theta_array_pos = pmdls.theta_model(lam_array, 4, params_pos)
    theta_array_neg = pmdls.theta_model(lam_array, 4, params_neg)

    if sample_theta==True:
        # Multiplicative per-pixel jitter on theta.  Noise2Params publication:
        # disabled (sample_theta=False).
        theta_array_pos = theta_array_pos * np.abs(np.random.normal(1.0, theta_sampler_sigma, size=np.shape(theta_array_pos)))
        theta_array_neg = theta_array_neg * np.abs(np.random.normal(1.0, theta_sampler_sigma, size=np.shape(theta_array_neg)))

    # Per-pixel B array.  Published results: truncated normal (uniform_pixels=False).
    if uniform_pixels==True:
        B_array=data*0+B
        B_pos_array=B_array
        B_neg_array=B_array
    if uniform_pixels==False:
        # Truncated-normal sampling on [0, +inf).  This is the ONLY per-pixel
        # B sampling distribution used in the Noise2Params paper.
        B_pos_array = ut.sample_truncated(dist='normal', size=data,
                                          loc=B, scale=B_sigma,
                                          a=0, b=100)
        B_neg_array =  B_pos_array
        print(f'B mean: {np.mean(B_pos_array)}, B sigma={np.std(B_pos_array)}')

    # Per-pixel event probabilities under the chosen model.
    if model=='Gaussian':
        prob_array_pos = pmdls.pos_event_prob_gaussian(lam_array, B_pos_array, theta_array_pos) + vert_offset_pos
        prob_array_neg = pmdls.neg_event_prob_gaussian(lam_array, B_neg_array, theta_array_neg) + vert_offset_neg
    if model=='saddle':
        prob_array_pos = pmdls.pos_prob_saddle_bracket_numba(lam_array, B_pos_array, theta_array_pos) + vert_offset_pos
        prob_array_neg = pmdls.neg_prob_saddle_bracket_numba(lam_array, B_neg_array, theta_array_neg) + vert_offset_neg
    if model=='Poisson':
        prob_array_pos = pmdls.pos_event_prob_vec_numba_2(lam_array, B_pos_array, theta_array_pos) + vert_offset_pos
        prob_array_neg = pmdls.pos_event_prob_vec_numba_2(lam_array, B_neg_array, theta_array_neg) + vert_offset_neg
        # (For Noise2Params, the Poisson branch is symmetric in pos/neg since
        # the negative-event Poisson probability at default B ≈ 0.15 differs
        # by a small amount that is absorbed by the vert_offset terms.)

    # Dead-time correction.  paper Eq. for P^eff:
    #   P^eff = (P_pos + P_neg) / (1 + (P_pos+P_neg) * dead_time)
    # The per-polarity m is the total effective probability multiplied by
    # each polarity's share.
    time_steps_array = np.zeros_like(data)
    time_steps_array = np.where(time_steps_array == 0, time_steps, time_steps_array)
    tot_prob_array=prob_array_pos + prob_array_neg
    effective_time_steps_array = time_steps_array - dead_time * (time_steps_array * (prob_array_pos + prob_array_neg))
    m_tot=tot_prob_array/(1+tot_prob_array*dead_time)

    share = np.divide(prob_array_pos, tot_prob_array, out=np.zeros_like(prob_array_pos), where=(tot_prob_array > 0))
    m_pos = m_tot * share
    m_neg = m_tot - m_pos

    time_steps_array=time_steps_array.astype(np.int32)

    if approach=='raw':
        # Naive Bernoulli-per-time-step sampling.  Included only for
        # pedagogical reference.  Scales the probability by 6e3 to skip
        # trivial zero-probability time steps; NOT used for publication.
        prob_array_pos = prob_array_pos * 6e3
        prob_array_neg = prob_array_neg * 6e3
        prob_array_pos = np.where(prob_array_pos>=0.5, 0.5, prob_array_pos)
        prob_array_neg = np.where(prob_array_neg >= 0.5, 0.5, prob_array_neg)

        prob_arrays=[prob_array_pos, prob_array_neg]
        counts_arrays=[]
        for prob_array in prob_arrays:
            counts_array = np.zeros_like(lam_array)
            for idx, prob in np.ndenumerate(prob_array):
                x, y = idx
                counts=np.sum(np.random.choice(a=[0, 1], size=time_steps, p=[1-prob, prob]))
                counts_array[x][y]=counts
            counts_arrays.append(counts_array)
        counts_array_pos = counts_arrays[0]
        counts_array_neg = counts_arrays[1]

    if approach=='expectations':
        # Return the expected count m*T (no stochasticity).
        counts_array_pos = m_pos * time_steps_array
        counts_array_neg = m_neg * time_steps_array

    if approach=='poisson sampling':
        counts_array_pos = np.random.poisson(m_pos * time_steps_array).astype(np.float32)
        counts_array_neg = np.random.poisson(m_neg * time_steps_array).astype(np.float32)

    if approach=='NB sampling':
        # Despite the historical name, this branch is a simple Binomial(T, m)
        # draw — the sampling method used to construct the published
        # synthetic datasets.
        counts_array_pos = np.random.binomial(time_steps_array, m_pos).astype(np.float32)
        counts_array_neg = np.random.binomial(time_steps_array, m_neg).astype(np.float32)

    if combine_polarities==True:
        counts_array = counts_array_pos + counts_array_neg

        if cap_or_norm=='cap':
            counts_array = np.clip(counts_array, 0, 255)
        if cap_or_norm=='norm':
            mx = counts_array.max(initial=0)
            counts_array = 0 if mx == 0 else counts_array * (255.0 / mx)

        if plot==True:
            output_image=Image.fromarray(counts_array*10)
            if plot_histogram == False:
                output_image.show()
            if plot_histogram==True:
                step1 = counts_array
                new_min, new_max = 0, 100
                new_range = new_max - new_min
                old_min = np.min(step1); old_max = np.max(step1)
                old_range = old_max - old_min
                counts_array_scaled = (((step1 - old_min) * new_range) / old_range) + new_min

                plt.rcParams['mathtext.fontset'] = 'stix'
                plt.rcParams['axes.axisbelow'] = True
                fig = plt.figure(figsize=(3.5, 4), dpi=200)
                ax_img = fig.add_axes([0.05, 0.05, .9, .9])
                ax_img.imshow(output_image)
                ax_img.axis('off')

                ax_hist = fig.add_axes([0.14, 0.07, 0.23, 0.29])
                ax_hist.hist(np.ravel(counts_array_scaled), bins=20)
                ax_hist.set_yscale('log')
                ax_hist.set_xlim([0,100])
                ax_hist.set_yticks([1e1, 1e3, 1e5])
                ax_hist.tick_params(axis='both', which='both', labelsize=8, direction='out')
                plt.show()

        if return_prob == False:
            return counts_array
        if return_prob==True:
            return counts_array, prob_array_pos, prob_array_neg, time_steps_array

    if combine_polarities==False:
        if cap_or_norm=='cap':
            counts_array_pos = np.where(counts_array_pos >=255, 255, counts_array_pos)
            counts_array_neg = np.where(counts_array_neg >= 255, 255, counts_array_neg)
        if cap_or_norm=='norm':
            scale_factor = 255 / np.max(counts_array_pos + counts_array_neg)
            counts_array_pos = counts_array_pos * scale_factor
            counts_array_neg = counts_array_neg * scale_factor
        if plot==True:
            Image.fromarray(counts_array_pos).show()
            Image.fromarray(counts_array_neg).show()
        if return_prob==False:
            return counts_array_pos, counts_array_neg
        if return_prob==True:
            return counts_array_pos, counts_array_neg, prob_array_pos, prob_array_neg, time_steps_array


def event_noise_image_modeler(input_image_location='noise2image_control_image',
                              noise_modeler_kwargs=None, source='here', Plot=True):
    """Thin wrapper around :func:`event_noise_image_modeler_core` with
    default kwargs matching the published Noise2Params pipeline.  Optionally
    displays the result via matplotlib.

    ``source='here'`` uses the core above; ``source='owen_event_model'``
    is an internal legacy path that is not included in this public copy.
    """
    default_noise_modeler_params = {
        'time_steps': 1 * 1e6,
        'B': 0.15,
        'B_sigma': 0.009,
        'int_mapping': 'mine',
        'dead_time': 79,
        'model': 'saddle',
        'uniform_pixels': False,
        'approach': 'poisson sampling',
        'alpha': 4.5,
        'plot_histogram': False,
        'cap_or_norm': 'norm',
        'combine_polarities': True,
        'plot': False,
    }
    noise_modeler_params = {**default_noise_modeler_params, **(noise_modeler_kwargs or {})}

    if source=='here':
        synth_counts = event_noise_image_modeler_core(input_image_location=input_image_location, **noise_modeler_params)
    else:
        raise ValueError("Only source='here' is supported; the "
                         "'owen_event_model' alternative generator is not "
                         "included in this public code.")

    if Plot==True:
        synth_counts_disp = np.rot90(synth_counts*10, axes=(1,0))
        COLOR = 'white'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        dpi=200
        W,H = np.shape(synth_counts_disp)
        plt.figure(facecolor='black', dpi=dpi, constrained_layout=True,
                   figsize=(H/dpi, W/dpi))
        plt.imshow(synth_counts_disp, cmap='grey', vmin=0, vmax=255)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Real noise-image compilation from event recordings

def event_image_compiler(input_file, plot_type='image', polarity=None, file_type='parquet',
                         lazy=True, cap_or_norm='norm', plot=True, display_factor=10,
                         remove_from_start=None, remove_from_end=None, from_start=None,
                         remove_outliers=True, rand_seg=None, seg=None, rotate=False,
                         center_crop_size=None):
    """
    Build a 2D event-count image from an event recording.

    Forwards slicing / outlier-removal options to
    :func:`noise2params.reading.counts_pixel_array`; optionally center-crops,
    caps or normalises the counts, and displays the result.

    Parameters
    ----------
    input_file : str
        Path/stem of the event recording file.
    plot_type : {'image', 'heatmap', 'bargraph'}
        Display mode.  Only ``'image'`` is supported; ``'heatmap'`` and
        ``'bargraph'`` require plotting helpers that are not included
        in this public code and will raise :class:`NotImplementedError`.
    polarity : {None, 0, 1, 'both'}
        0 → negative only, 1 → positive only, ``'both'`` → return the pair
        (pos, neg).
    file_type, lazy, remove_from_start, remove_from_end, from_start, rand_seg, seg, remove_outliers
        Forwarded to :func:`noise2params.reading.counts_pixel_array`.
    cap_or_norm : {'cap', 'norm', None}
        Clipping / rescaling mode (``'norm'`` rescales so max=255).
    display_factor : int
        Multiplicative scale for on-screen display.
    rotate : bool
        If True, rotate 90° for display.
    center_crop_size : (H, W), optional
        Center crop before return.

    Returns
    -------
    ndarray or (ndarray, ndarray)
        (H, W) count array, or the tuple (pos, neg).
    """
    if polarity in [0,1, None]:
        event_counts_array = read.counts_pixel_array(input_file, polarity, file_type, lazy=lazy,
                                              remove_from_start=remove_from_start, remove_from_end=remove_from_end,
                                              from_start=from_start, rand_seg=rand_seg, seg=seg,
                                              remove_outliers=remove_outliers)
    if polarity=='both':
        counts_pos, counts_neg = read.counts_pixel_array(input_file, polarity, file_type, lazy=lazy,
                                                     remove_from_start=remove_from_start,
                                                     remove_from_end=remove_from_end, seg=seg,
                                                     from_start=from_start, rand_seg=rand_seg,
                                                     remove_outliers=remove_outliers)
        event_counts_array = counts_pos + counts_neg

    if center_crop_size != None:
        if type(center_crop_size) == tuple:
            event_counts_array = ut.center_crop(event_counts_array, center_crop_size)
        else:
            raise ValueError('center_crop_size must be a tuple (H,W) or (H,W,D) to crop to')

    if cap_or_norm == 'norm':
        mx = event_counts_array.max(initial=0)
        event_counts_array = 0 if mx == 0 else event_counts_array * (255.0 / mx)
        if polarity=='both':
            scale_factor = 255 / np.max(counts_pos + counts_neg)
            counts_pos = counts_pos * scale_factor
            counts_neg = counts_neg * scale_factor
    if cap_or_norm == 'cap':
        event_counts_array = np.where(event_counts_array >= 255, 255, event_counts_array)

    if plot==True:
        if rotate==True:
            event_counts_array=np.rot90(event_counts_array, axes=(1,0))
        if plot_type=='image':
            event_counts_array=event_counts_array*display_factor
            h, w = event_counts_array.shape
            fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(event_counts_array, cmap='grey', vmin=0, vmax=255)
            ax.axis('off')
            plt.show()
        if plot_type in ('heatmap', 'bargraph'):
            raise NotImplementedError(
                "plot_type in {'heatmap', 'bargraph'} is not available in "
                "this public code.")

    if polarity in [0,1, None]:
        return event_counts_array
    if polarity=='both':
        return counts_pos, counts_neg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Image-similarity metric panels

def compare_image_metrics(img1, img2, metrics='all', remove_black_bars=True):
    """
    Compute per-pair image-similarity metrics between two images.

    Parameters
    ----------
    img1, img2 : ndarray or str
        Two 2D grayscale images, or paths/stems resolved via file_checker.
        ``img1`` is treated as the ground truth when resizing is needed.
    metrics : {'classic', 'modern', 'all', 'deep learning'} or list
        Metric set selector.  'classic' = {PSNR, NCC, SSIM};
        'modern' = {MS-SSIM, 3-SSIM, FSIM, VIF, GMSD, VSI};
        'deep learning' = {LPIPS, DreamSim, DISTS, PieAPP}; 'all' = all of the above.
    remove_black_bars : bool
        Whether to crop black bars (common around 16:9 images that have
        been letter-boxed in the base set).

    Returns
    -------
    list
        Metric values in the order of the (sorted, fixed) ``allowed_metrics``
        list.  Compare columns against
        ``['PSNR', 'NCC', 'SSIM', 'MS-SSIM', '3-SSIM', 'FSIM', 'VIF', 'VSI', 'GMSD', 'LPIPS', 'PieAPP', 'DISTS', 'DreamSim']``.
    """
    allowed_metrics=['PSNR', 'NCC', 'SSIM', 'MS-SSIM','3-SSIM', 'FSIM', 'VIF', 'VSI', 'GMSD', 'LPIPS', 'PieAPP','DISTS', 'DreamSim']
    err="`metrics` must be a string in ['classic', 'modern', 'all', 'deep learning'] or a list of " + str(allowed_metrics)
    if isinstance(metrics, str):
        if metrics not in ['classic', 'modern', 'all', 'deep learning']:
            raise ValueError(err)
    elif isinstance(metrics, list):
        if not set(metrics).issubset(set(allowed_metrics)):
            raise ValueError(err)
    else:
        raise ValueError(err)

    if type(metrics)==str:
        if metrics=='classic':
            metric_list=['PSNR', 'NCC', 'SSIM']
        if metrics=='modern':
            metric_list=['MS-SSIM','3-SSIM', 'FSIM', 'VIF', 'GMSD', 'VSI']
        if metrics=='deep learning':
            metric_list=['LPIPS', 'DreamSim', 'DISTS', 'PieAPP']
        if metrics=='all':
            metric_list=allowed_metrics
    if type(metrics)==list:
        metric_list=metrics
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import piq

    exts = Image.registered_extensions()
    supported_extensions = [ex for ex, f in exts.items() if f in Image.OPEN]
    if '.h5' in supported_extensions:
        supported_extensions.remove('.h5')
    if type(img2) == str:
        input_image_location = ut.file_checker(img2, supported_extensions)
        img2 = Image.open(input_image_location).convert('L')
        img2 = np.array(img2)
    if type(img1)== str:
        input_image_location = ut.file_checker(img1, supported_extensions)
        img1 = Image.open(input_image_location).convert('L')
        if np.shape(img1) != np.shape(img2):
            img1 = img1.resize(size=(np.shape(img2)[1], np.shape(img2)[0]),
                               resample=Image.Resampling.LANCZOS)
        img1 = np.array(img1)

    if remove_black_bars==True:
        img1, img2 = ut.crop_black_bars(img1, img2, threshold=2)

    if np.max(img1) > 255:
        img1 = img1 * (1 / np.max(img1))
    if np.max(img2) > 255:
        img2 = img2 * (1 / np.max(img2))

    output=[]
    if 'PSNR' in metric_list:
        output.append(psnr(img1, img2, data_range=img1.max() - img1.min()))
    if 'NCC' in metric_list:
        output.append(ut.normalized_cross_correlation(img1, img2))
    if 'SSIM' in metric_list:
        output.append(ssim(img1, img2, data_range=img1.max() - img1.min()))
    if 'MS-SSIM' in metric_list:
        output.append(ut.ms_ssim_numpy(img1, img2))
    if '3-SSIM' in metric_list:
        output.append(ut.three_ssim(img1, img2))
    if 'FSIM' in metric_list:
        output.append(ut.fsim(img1, img2))
    if 'VIF' in metric_list:
        output.append(ut.vif(img1, img2))
    if 'VSI' in metric_list:
        output.append(ut.vsi(img1, img2))
    if 'GMSD' in metric_list:
        output.append(ut.gmsd(img1, img2))
    if 'LPIPS' in metric_list:
        output.append(ut.lpips(img1, img2))
    if 'DreamSim' in metric_list:
        output.append(ut.DreamSim(img1, img2))
    if 'DISTS' in metric_list:
        output.append(ut.dists(img1, img2))
    if 'PieAPP' in metric_list:
        output.append(ut.pieapp(img1, img2))
    return output


def two_image_compare_to_reference(ref, img1, img2):
    """Plot two candidate images side-by-side with a reference image and
    print the similarity metrics under each.  Figure-generation helper."""
    exts = Image.registered_extensions()
    supported_extensions = [ex for ex, f in exts.items() if f in Image.OPEN]
    if '.h5' in supported_extensions:
        supported_extensions.remove('.h5')
    if type(img1) == str:
        img1 = np.array(Image.open(ut.file_checker(img1, supported_extensions)).convert('L'))
    if type(img2) == str:
        img2 = np.array(Image.open(ut.file_checker(img2, supported_extensions)).convert('L'))
    if np.shape(img1) != np.shape(img2):
        raise ValueError('img1 and img2 must be of the same size')
    if type(ref) == str:
        ref_img = Image.open(ut.file_checker(ref, supported_extensions)).convert('L')
        if np.shape(ref_img) != np.shape(img2):
            ref_img = ref_img.resize(size=(np.shape(img2)[1], np.shape(img2)[0]),
                                     resample=Image.Resampling.LANCZOS)
        ref = np.array(ref_img)
    if np.shape(ref)!=np.shape(img1):
        raise ValueError('ref must be the same size as img1 and img2.')

    s1 = compare_image_metrics(img1=ref, img2=img1, metrics='all')
    s2 = compare_image_metrics(img1=ref, img2=img2, metrics='all')
    s1_str=f'PSNR: {s1[0]:.2f}, NCC: {s1[1]:.2f}, SSIM: {s1[2]:.2f},\n MS-SSIM: {s1[3]:.2f}, LPIPS: {s1[5]:.2f}, DreamSim: {s1[6]:.2f}'
    s2_str=f'PSNR: {s2[0]:.2f}, NCC: {s2[1]:.2f}, SSIM: {s2[2]:.2f},\n MS-SSIM: {s2[3]:.2f}, LPIPS: {s2[5]:.2f}, DreamSim: {s2[6]:.2f}'

    dpi = 200
    W, H = np.shape(img2)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(dpi=dpi, constrained_layout=True, figsize=(H * 3.1 / dpi, W * 1.3 / dpi))
    plt.subplot(1,3,1); plt.imshow(img1, cmap='grey'); plt.xticks([]); plt.yticks([]); plt.box(False)
    plt.title('Image 1'); plt.xlabel(s1_str, fontsize=7)
    plt.subplot(1,3,2); plt.imshow(ref, cmap='grey'); plt.xticks([]); plt.yticks([]); plt.box(False)
    plt.title('Reference')
    plt.subplot(1,3,3); plt.imshow(img2, cmap='grey'); plt.xticks([]); plt.yticks([]); plt.box(False)
    plt.title('Image 2'); plt.xlabel(s2_str, fontsize=7)
    plt.show()


def group_sim_metrics_2(save_folder, name_str='', metrics='all', recompute=False):
    """
    Evaluate similarity metrics for every predicted-image / base-image pair in
    ``save_folder`` and write/update a CSV of scores.

    The predicted images in ``save_folder`` are matched to base images by the
    filename convention ``{base_image_stem}_{anything}.png``.  The CSV is
    resumable: existing metric columns are preserved unless ``recompute`` is
    True (or a list of metric names to force-recompute).

    Parameters
    ----------
    save_folder : str
        Folder of predicted ``.png`` files (each named ``{base}_*.png``).
    name_str : str
        CSV filename stem (``{name_str}.csv`` under ``save_folder``).
    metrics : str or list
        Metric selection.  See :func:`compare_image_metrics`.
    recompute : bool or list of str
        Force recomputation of these (or all, if True) metrics.
    """
    allowed_metrics = ['PSNR', 'NCC', 'SSIM', 'MS-SSIM', '3-SSIM', 'FSIM', 'VIF', 'VSI', 'GMSD', 'LPIPS', 'DISTS', 'PieAPP',
                       'DreamSim']
    err = ("`metrics` must be a string in ['classic', 'modern', 'all', 'deep learning'] "
           "or a list of " + str(allowed_metrics))

    if isinstance(metrics, str):
        if metrics not in ['classic', 'modern', 'all', 'deep learning']:
            raise ValueError(err)
        metric_list = {
            'classic': ['PSNR', 'NCC', 'SSIM'],
            'modern': ['MS-SSIM', '3-SSIM', 'FSIM', 'VIF', 'GMSD', 'VSI'],
            'deep learning': ['LPIPS', 'DreamSim', 'DISTS', 'PieAPP'],
            'all': allowed_metrics,
        }[metrics]
    elif isinstance(metrics, list):
        if not set(metrics).issubset(set(allowed_metrics)):
            raise ValueError(err)
        metric_list = metrics
    else:
        raise ValueError(err)

    if isinstance(recompute, bool):
        recompute_set = set(metric_list) if recompute else set()
    elif isinstance(recompute, list):
        invalid_recompute = set(recompute) - set(allowed_metrics)
        if invalid_recompute:
            raise ValueError(f"Invalid metrics in recompute: {invalid_recompute}")
        recompute_set = set(recompute)
    else:
        raise ValueError("`recompute` must be a bool or a list of metric names")

    saved_metrics_file = os.path.join(save_folder, name_str + '.csv')

    if os.path.exists(saved_metrics_file):
        existing_df = pl.read_csv(saved_metrics_file, schema_overrides={"base_image": pl.String})
        existing_metrics = [col for col in existing_df.columns if col in allowed_metrics]
        metrics_to_compute = [m for m in metric_list if m not in existing_metrics or m in recompute_set]
        if not metrics_to_compute:
            print(f"All requested metrics already exist in {saved_metrics_file}")
            return existing_df
        metrics_to_replace = [m for m in metrics_to_compute if m in existing_metrics]
        if metrics_to_replace:
            print(f"Recomputing existing metrics: {metrics_to_replace}")
        base_images = existing_df['base_image'].to_list()
        pred_images = existing_df['pred_image'].to_list()
    else:
        existing_df = None
        metrics_to_compute = metric_list
        metrics_to_replace = []
        pred_images = [x.replace('.png', '') for x in os.listdir(save_folder) if x.endswith('.png')]
        base_images = [x.split('_')[0] for x in pred_images]
        print(f"Discovered {len(pred_images)} images")
        print(f"Computing metrics: {metrics_to_compute}")

    metrics_scores_list = []
    for i, pred_image in enumerate(pred_images):
        print(f'Computing metrics for base: {base_images[i]}, pred: {pred_image} ({i + 1}/{len(pred_images)})')
        scores = compare_image_metrics(base_images[i], pred_image, metrics=metrics_to_compute)
        metrics_scores_list.append([base_images[i], pred_image] + list(scores))

    new_headers = ['base_image', 'pred_image'] + metrics_to_compute
    new_df = pl.DataFrame(metrics_scores_list, schema=new_headers, orient='row')

    if existing_df is not None:
        if metrics_to_replace:
            existing_df = existing_df.drop(metrics_to_replace)
        final_df = existing_df.join(
            new_df.select(['base_image', 'pred_image'] + metrics_to_compute),
            on=['base_image', 'pred_image'], how='left')
    else:
        final_df = new_df

    final_columns = ['base_image', 'pred_image'] + [m for m in allowed_metrics if m in final_df.columns]
    final_df = final_df.select(final_columns)
    final_df.write_csv(saved_metrics_file)
    print(f"Saved to {saved_metrics_file}")
    return final_df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CNN inference

def load_compiled_checkpoint(model, checkpoint_path, strict=True):
    """Load a Lightning ``.ckpt`` that may have been saved after a
    :func:`torch.compile` wrap (which inserts a ``_orig_mod`` prefix in
    keys).  Strips the prefix automatically before calling
    :meth:`model.load_state_dict`.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt["state_dict"]
    has_orig_mod = any(k.startswith("model._orig_mod.") for k in state_dict.keys())
    if has_orig_mod:
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model._orig_mod."):
                new_key = key.replace("model._orig_mod.", "model.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=strict)
    return model


def predict_from_events(event_counts, ckpt_path, integration_time_s=1.0,
                        in_channels=2, vanilla_unet=False, use_gpu=True, return_time=False):
    """
    Run a trained reconstruction CNN on a single event-count array.

    Parameters
    ----------
    event_counts : ndarray
        Shape (H, W, C) — raw event counts, C usually 2 (pos, neg).
    ckpt_path : str
        Path to a Lightning ``.ckpt``.
    integration_time_s : float
        Event-integration window used for this ``event_counts``.  Fed into
        the FiLM time-conditioning of the U-Net.  Noise2Params: 5.0.
    in_channels : int
        Number of input channels (2 for pos/neg).
    vanilla_unet : bool
        If True, instantiate the simpler vanilla U-Net; otherwise the
        attention U-Net used in the paper.
    use_gpu : bool
        Move model to CUDA when available.
    return_time : bool
        Also return wall-clock inference time (post-warmup).

    Returns
    -------
    ndarray of uint8 shape (H, W), optionally (img, infer_time).
    """
    assert event_counts.ndim == 3, f"Expected (H, W, C) input, got shape {event_counts.shape}"
    assert event_counts.shape[2] == in_channels, \
        f"Event channels ({event_counts.shape[2]}) don't match in_channels ({in_channels})"

    # In the published pipeline EventCountNormalization is a no-op (kept
    # here as a vestigial code path for clarity; see the note in the
    # repository README about why normalization is not applied at the
    # integration times used in Noise2Params).
    H, W, C = event_counts.shape
    dummy_image = np.zeros((H, W, 1), dtype=np.float32)
    combined = np.concatenate([event_counts.astype(np.float32), dummy_image], axis=-1)
    combined_normalized = combined  # EventCountNormalization no-op
    events_normalized = combined_normalized[..., :-1]

    events_tensor = torch.from_numpy(events_normalized.transpose(2, 0, 1)).unsqueeze(0)
    integration_time_tensor = torch.tensor(integration_time_s, dtype=torch.float32).unsqueeze(0)

    model = Model(dim=64, in_channels=in_channels, lr=5e-5, vanilla_unet=vanilla_unet)
    model = load_compiled_checkpoint(model, ckpt_path, strict=True)
    model.eval()

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    events_tensor = events_tensor.to(device)
    integration_time_tensor = integration_time_tensor.to(device)

    # Warmup (triggers compilation) + timed pass.
    with torch.no_grad():
        _ = model(events_tensor, integration_time_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        prediction = model(events_tensor, integration_time_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    infer_time = time.time() - start

    pred_img = prediction[0, 0].clamp(0, 1).cpu().numpy()
    print(f'pred_img, max: {np.max(pred_img)}, min: {np.min(pred_img)}')
    pred_uint8 = (pred_img * 255).astype(np.uint8)

    if return_time:
        return pred_uint8, infer_time
    else:
        return pred_uint8


def configure_deterministic_inference():
    """Configure PyTorch for deterministic cross-platform inference.

    Disables TF32 and the cuDNN benchmark heuristic, enables the
    deterministic algorithm set, and sets ``CUBLAS_WORKSPACE_CONFIG``
    for CUDA ≥ 10.2.  Call once before inference if byte-identical
    results are required across runs.
    """
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def infer_recording(rec_file, CKPT, from_start=None, remove_from_start=None,
                    remove_from_end=None, file_type='parquet',
                    remove_outliers=True, rand_seg=None, integration_time_s=1,
                    center_crop_size=None, plot=True):
    """
    Load a single event recording, accumulate events into pos/neg count
    arrays (with a 2×2 spatial bin to match the training-time input shape
    of 360×640), and run the given CNN checkpoint on the result.

    Parameters
    ----------
    rec_file : str
        Path or stem of a ``.parquet`` (or ``.h5``/``.csv``) recording.
    CKPT : str
        Path to the Lightning ``.ckpt`` to run.  **Required** — there is
        no built-in default.
    from_start, remove_from_start, remove_from_end, rand_seg
        Mutually exclusive time-slicing options (see
        :func:`noise2params.reading.recording_reader`).
    file_type : {'parquet', 'csv', 'h5'}
    remove_outliers : bool
    integration_time_s : float
        Value fed into the model's time-conditioning head.
    center_crop_size : tuple, optional
        Crop before inference; if omitted, the counts are 2×2 block-averaged.
    plot : bool
        Display the reconstructed image.

    Returns
    -------
    PIL.Image.Image
    """
    pos, neg = read.counts_pixel_array(rec_file, polarity='both', remove_from_end=remove_from_end,
                                       remove_from_start=remove_from_start, rand_seg=rand_seg,
                                       from_start=from_start, file_type=file_type, remove_outliers=remove_outliers)

    if center_crop_size != None:
        if type(center_crop_size) == tuple:
            pos = ut.center_crop(pos, center_crop_size)
            neg = ut.center_crop(neg, center_crop_size)
        else:
            raise ValueError('center_crop_size must be a tuple (H,W) or (H,W,D) to crop to')

    if center_crop_size == None:
        pos = block_mean(pos, 2)
        neg = block_mean(neg, 2)

    counts = np.stack([pos, neg], axis=-1)
    image_array, infer_time = predict_from_events(counts, CKPT, vanilla_unet=False,
                                                  integration_time_s=integration_time_s,
                                                  return_time=True)
    print(f'inference time: {infer_time}')
    img = Image.fromarray(image_array)
    if plot:
        img.show()
    return img


def group_infer_recording(file_list_csv, save_folder, checkpoint_path,
                          name_str='', from_start=None,
                          remove_from_start=None, remove_from_end=None,
                          file_type='parquet', remove_outliers=True,
                          rand_seg=(5, 4), integration_time_s=5):
    """
    Run :func:`infer_recording` on every recording listed in a recording /
    base-image mapping CSV and save the reconstructions as ``.png`` under
    ``save_folder``.

    The mapping CSV is the format shipped with the public dataset
    (see :file:`docs/DATA.md`): columns ``recording_name``, ``time ID``,
    ``image`` (note the leading space: the column is literally named
    ``" image"`` in the shipped files — this is a historical artifact
    retained for compatibility), ``intensity``, ``dead_time``, ``biases``.

    Output filenames are ``{image}{name_str}.png``.
    """
    df = pl.read_csv(ut.file_checker(file_list_csv, file_type='csv'))
    rec_name_list = list(np.ravel(df.select(pl.col('recording_name')).to_numpy()))
    rec_image_list = list(np.ravel(df.select(pl.col(' image')).to_numpy()))
    rec_image_list_save = [x.replace(' ', '') + name_str + '.png' for x in rec_image_list]

    for i, rec_file in enumerate(rec_name_list):
        print(f'Generating reconstruction from {rec_file}, recording {i+1} of {len(rec_name_list)}')
        img = infer_recording(rec_file=rec_file, CKPT=checkpoint_path, from_start=from_start,
                              remove_from_start=remove_from_start,
                              remove_from_end=remove_from_end, file_type=file_type,
                              remove_outliers=remove_outliers,
                              rand_seg=rand_seg, integration_time_s=integration_time_s, plot=False)
        img.save(os.path.join(save_folder, rec_image_list_save[i]))
        print(f'Successfully saved {rec_image_list_save[i]}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataset construction

def block_mean(arr, bin_sz):
    """Return ``arr`` spatially down-sampled by ``bin_sz``-sized block averaging."""
    if bin_sz == 1:
        return arr
    H, W = arr.shape
    H2, W2 = H // bin_sz, W // bin_sz
    arr = arr[:H2 * bin_sz, :W2 * bin_sz]
    return arr.reshape(H2, bin_sz, W2, bin_sz).mean(axis=(1, 3))


def synthetic_data_base_maker(source_folder, event_target_folder, image_target_folder,
                              num_variations=2, noise_modeler_kwargs=None, save_format='memmap',
                              apply_brightness_aug=False, crop_or_resize='rand',
                              just_probs=False, cap_or_norm='norm',
                              Plot=True):
    """
    Generate and save a synthetic-noise-image / base-image pair database for
    training.

    This matches the exact pipeline that produced the published
    ``data_5e6_5`` (saddle) and ``data_5e6_6`` (Gaussian) datasets.  Images
    are loaded, optionally resized to 1280×720, random-flipped, center-
    cropped, optionally brightness-augmented, then the synthetic event
    image is generated with :func:`event_noise_image_modeler_core`, 2×2
    block-averaged, and saved.

    Defaults match the published pipeline:

        time_steps: 5 s (5 × 1e6 µs)
        pixel_bin: 2
        B: 0.15,  B_sigma: 0.00651
        alpha: 4.5
        params_pos: (18.917319991515015, 35.49036559887983, 0.4394373558876216)
        params_neg: (16.41747084565865, 37.42249743893243, 0.06759986822465787)
        vert_offset_pos: 9.567536526687546e-09
        vert_offset_neg: 3.178722048504662e-08
        int_mapping_coef: [2.14990578e-05, 2.52106648, 0.15]
        approach: 'NB sampling'  (binomial; see event_noise_image_modeler_core)
        model: 'Poisson'          (override with 'saddle' or 'Gaussian')
        combine_polarities: False

    See :file:`configs/params_published.yaml` for the exact values and
    :file:`scripts/02_build_synthetic_dataset.py` for the CLI wrapper.

    Parameters
    ----------
    source_folder : str
        Folder of base .png/.jpg images (e.g. ``indist_images_full``).
    event_target_folder, image_target_folder : str
        Output folders for event and image arrays.
    num_variations : int
        Random augmentations per source image (published: 2).
    noise_modeler_kwargs : dict, optional
        Override any of the :func:`event_noise_image_modeler_core` kwargs
        (notably ``model='saddle'`` / ``'Gaussian'`` / ``'Poisson'``).
    save_format : {'memmap', 'hdf5', 'npy'}
        Output format.  ``'memmap'`` produces the ``events_all.npy`` +
        ``images_all.npy`` files that the precomputed_dataset loader
        expects.
    apply_brightness_aug : bool
        Optional per-pair brightness multiplier in [0.7, 1.3].
    crop_or_resize : {'rand', 'crop', 'resize'}
        How to deal with source images that are larger than 1280×720.
    just_probs : bool
        If True, store probability arrays instead of sampled counts (used
        for the ``--on_fly_sampling`` training branch).
    cap_or_norm, Plot : pass-through flags.
    """
    default_noise_modeler_params = {
        'time_steps': 5 * (1e6),
        'pixel_bin': 2,
        'B': 0.15,
        'B_sigma': 0.00651,
        'int_mapping': 'mine',
        'dead_time': 79.0,
        'uniform_pixels': False,
        'approach': 'NB sampling',
        'alpha': 4.5,
        'params_pos': [18.917319991515015, 35.49036559887983, 0.4394373558876216],
        'vert_offset_pos': 9.567536526687546e-09,
        'params_neg': [16.41747084565865, 37.42249743893243, 0.06759986822465787],
        'vert_offset_neg': 3.178722048504662e-08,
        'int_mapping_coef': [2.14990578e-05, 2.52106648, 0.15],
        'model': 'Poisson',
        'cap_or_norm': cap_or_norm,
        'combine_polarities': False,
        'just_probs': just_probs,
        'plot': False,
        'sample_theta': False,
    }
    noise_modeler_params = {**default_noise_modeler_params, **(noise_modeler_kwargs or {})}

    os.makedirs(event_target_folder, exist_ok=True)
    os.makedirs(image_target_folder, exist_ok=True)

    # Published pipeline: input_size=(720, 1280) before pixel_bin=2 -> (360, 640).
    crop_size_before_binning = (720, 1280)

    all_images = []
    for root, dirs, images in os.walk(source_folder):
        for image_file in images:
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, image_file))

    print(f"Processing {len(all_images)} images with {num_variations} variations each")
    print(f"Save format: {save_format}")
    print(f"Crop size (before binning): {crop_size_before_binning}")
    print(f"Pixel bin: {noise_modeler_params['pixel_bin']}")
    print(f"Time steps: {noise_modeler_params['time_steps']:.0e}")
    print(f"Brightness augmentation: {apply_brightness_aug}")

    all_events_list = []
    all_images_list = []
    all_names_list = []

    for n, image_file in enumerate(all_images):
        num_images = len(all_images)
        head, tail = os.path.split(image_file)
        file_name_str, ext = os.path.splitext(tail)
        print(f'Building synthetic images from {file_name_str}, image {n + 1} out of {num_images}')

        img = Image.open(image_file).convert('L')
        data_full_size = np.array(img)
        img = img.resize(size=(1280, 720), resample=Image.Resampling.LANCZOS)
        data = np.array(img)

        for i in range(num_variations):
            if crop_or_resize == 'rand':
                if np.random.rand() < 0.5:
                    cur_data = np.copy(data_full_size)
                else:
                    cur_data = np.copy(data)
            elif crop_or_resize == 'crop':
                cur_data = np.copy(data_full_size)
            elif crop_or_resize == 'resize':
                cur_data = np.copy(data)

            # Flips before crop / event generation.
            if np.random.rand() < 0.5:
                cur_data = np.flip(cur_data, axis=1)
            if np.random.rand() < 0.5:
                cur_data = np.flip(cur_data, axis=0)

            # Centre-crop (no-op if already 720x1280).
            H, W = cur_data.shape
            target_h, target_w = crop_size_before_binning
            start_h = (H - target_h) // 2
            start_w = (W - target_w) // 2
            cur_data = cur_data[start_h:start_h + target_h, start_w:start_w + target_w]

            if apply_brightness_aug and np.random.rand() < (1/2):
                mult = np.random.uniform(0.7, 1.3)
                cur_data = np.clip(cur_data * mult, 0, 255).astype(np.uint8)
            else:
                cur_data = cur_data.astype(np.uint8)

            cur_data = np.ascontiguousarray(cur_data)

            synth_counts_pos, synth_counts_neg = event_noise_image_modeler_core(
                input_image_location=cur_data,
                time_steps=int(noise_modeler_params['time_steps']),
                B=noise_modeler_params['B'],
                B_sigma=noise_modeler_params['B_sigma'],
                int_mapping=noise_modeler_params['int_mapping'],
                dead_time=noise_modeler_params['dead_time'],
                uniform_pixels=noise_modeler_params['uniform_pixels'],
                approach=noise_modeler_params['approach'],
                alpha=noise_modeler_params['alpha'],
                cap_or_norm=noise_modeler_params['cap_or_norm'],
                model=noise_modeler_params['model'],
                plot=noise_modeler_params['plot'],
                combine_polarities=noise_modeler_params['combine_polarities'],
                return_prob=False,
                params_pos=noise_modeler_params['params_pos'],
                params_neg=noise_modeler_params['params_neg'],
                vert_offset_pos=noise_modeler_params['vert_offset_pos'],
                vert_offset_neg=noise_modeler_params['vert_offset_neg'],
                int_mapping_coef=noise_modeler_params['int_mapping_coef'],
                sample_theta=noise_modeler_params['sample_theta'],
            )

            if just_probs==False:
                if Plot == True and n<3:
                    synth_counts = (synth_counts_pos + synth_counts_neg) * 10
                    synth_counts = np.rot90(synth_counts, axes=(1, 0))
                    COLOR = 'white'
                    mpl.rcParams['text.color'] = COLOR
                    mpl.rcParams['axes.labelcolor'] = COLOR
                    mpl.rcParams['xtick.color'] = COLOR
                    mpl.rcParams['ytick.color'] = COLOR
                    dpi = 200
                    W_, H_ = np.shape(synth_counts)
                    plt.figure(facecolor='black', dpi=dpi, constrained_layout=True,
                               figsize=(H_ / dpi, W_ / dpi))
                    plt.imshow(synth_counts, cmap='hot', vmin=0, vmax=255)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.show()

                pixel_bin = noise_modeler_params['pixel_bin']
                if pixel_bin > 1:
                    synth_counts_pos_binned = block_mean(synth_counts_pos.astype(np.float32), pixel_bin)
                    synth_counts_neg_binned = block_mean(synth_counts_neg.astype(np.float32), pixel_bin)
                    cur_data_binned = block_mean(cur_data.astype(np.float32), pixel_bin)
                else:
                    synth_counts_pos_binned = synth_counts_pos.astype(np.float32)
                    synth_counts_neg_binned = synth_counts_neg.astype(np.float32)
                    cur_data_binned = cur_data.astype(np.float32)

                events = np.stack([synth_counts_pos_binned, synth_counts_neg_binned], axis=0).astype(np.float32)

                # No EventCountNormalization (identity).  See the project
                # README for why normalisation is disabled in this pipeline.
                events_hwc = events.transpose(1, 2, 0)
                image_normalized = cur_data_binned / 255.0
                combined = np.concatenate([events_hwc, image_normalized[..., np.newaxis]], axis=-1)
                combined_normalized = combined
                events_normalized = combined_normalized[..., :-1].transpose(2, 0, 1)
                image_final = combined_normalized[..., -1]
                image_uint8 = np.clip(cur_data_binned, 0, 255).astype(np.uint8)

            if just_probs==True:
                events_normalized = np.stack([synth_counts_pos, synth_counts_neg], axis=0).astype(np.float32)
                image_final = cur_data / 255.0

            sample_name = f"{file_name_str}_{i}"

            if save_format == 'npy':
                base_image = Image.fromarray(image_uint8)
                cur_event_path = os.path.join(event_target_folder, sample_name)
                cur_img_path = os.path.join(image_target_folder, f"{sample_name}.png")
                base_image.save(cur_img_path)
                np.save(cur_event_path, events_normalized)

                if (n * num_variations + i) % 100 == 0:
                    print(f'  Saved: {sample_name}')
            else:
                all_events_list.append(events_normalized)
                all_images_list.append(image_final)
                all_names_list.append(sample_name)

    if save_format == 'memmap':
        print("\nSaving as memory-mapped arrays...")
        _save_as_memmap(all_events_list, all_images_list, all_names_list,
                        event_target_folder, image_target_folder)
    elif save_format == 'hdf5':
        print("\nSaving as HDF5 file...")
        _save_as_hdf5(all_events_list, all_images_list, all_names_list,
                      os.path.dirname(event_target_folder))

    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE!")
    print("=" * 60)
    print(f"Total samples: {len(all_images) * num_variations}")
    print(f"Format: {save_format}")
    print(f"Time steps: {noise_modeler_params['time_steps']:.0e}")
    print("=" * 60)


def real_data_base_maker(image_rec_mapping, val_pair_folder, event_target_folder,
                         image_target_folder, integration_time_s=5,
                         use_full_time=False, rand_seg=(5, 4), num_variations=1):
    """
    Build a database of real (EC-recorded) noise-image / base-image pairs.

    For every ``(recording, base-image)`` pair in ``image_rec_mapping`` (a
    CSV matching the format of the shipped recording-data CSVs: columns
    ``recording_name, time ID, image, intensity, dead_time, biases``), the
    event counts are accumulated over a random ``integration_time_s``-long
    segment of the recording, 2×2 block-averaged, and paired with the
    block-averaged greyscale base image.  Pairs are saved as
    :func:`numpy.memmap`-compatible ``.npy`` arrays.

    Parameters
    ----------
    image_rec_mapping : str
        Path to the mapping CSV.  If ``None``, falls back to scanning
        ``val_pair_folder`` for paired files with matching stems — NOT
        the path used for the published dataset; use the mapping CSV.
    val_pair_folder : str
        Folder where the base-image ``.png`` files live.  For the shipped
        dataset this is ``ood_DIV2K_images`` (for OOD val/test) or
        ``indist_images_full`` (for in-distribution training).
    event_target_folder, image_target_folder : str
        Output folders.
    integration_time_s : int
        Integration window length in seconds (must be integer; published: 5).
    use_full_time : bool
        If True, tile the recording into contiguous ``integration_time_s``
        segments (produces multiple pairs per recording).  Mutually exclusive
        with ``rand_seg``.
    rand_seg : tuple, optional
        ``(length_s, start_offset_s, end_offset_s)``.  Random-segment mode.
    num_variations : int
        Number of random segments per recording when ``rand_seg`` is set.
    """
    if int(integration_time_s) != integration_time_s:
        raise ValueError('integration_time_s must be an integer value')
    if use_full_time and rand_seg is not None:
        raise ValueError('Cannot use use_full_time and rand_seg together, only one can be specified.')

    os.makedirs(event_target_folder, exist_ok=True)
    os.makedirs(image_target_folder, exist_ok=True)

    if image_rec_mapping is None:
        # Legacy fallback: scan val_pair_folder for paired files by stem match.
        all_val_pair_names = []
        for root, dirs, images in os.walk(val_pair_folder):
            for image_file in images:
                if image_file.lower().endswith('.png'):
                    stem = os.path.splitext(image_file)[0]
                    all_val_pair_names.append((stem, stem))
    elif isinstance(image_rec_mapping, str):
        map_df = pl.read_csv(ut.file_checker(image_rec_mapping, file_type='csv'))
        # Column ' image' has a literal leading space in the shipped CSVs.
        map_df = map_df.with_columns(
            pl.col(" image").str.replace_all(" ", "").alias(" image"))
        rec_names = np.ravel(map_df.select(pl.col('recording_name')).to_numpy())
        image_names = np.ravel(map_df.select(pl.col(' image')).to_numpy())
        all_val_pair_names = list(zip(list(image_names), list(rec_names)))
    else:
        raise TypeError('image_rec_mapping must be a CSV path or None')

    print(f'Creating database of real event recordings and greyscale image pairs.')
    print(f"Processing {len(all_val_pair_names)} images")

    all_events_list = []
    all_images_list = []
    all_names_list = []

    for n_outer, val_pair in enumerate(all_val_pair_names):
        print(f'Building base image-event image pairs from {val_pair}, '
              f'{n_outer}/{len(all_val_pair_names)}')
        img_name = val_pair[0]
        rec_name = val_pair[1]
        img_path = ut.file_checker(file_name=img_name, file_type=['png', 'jpg'])
        img = Image.open(img_path).convert('L')
        img = img.resize(size=(1280, 720), resample=Image.Resampling.LANCZOS)
        img_arr = np.array(img)
        img_arr = block_mean(img_arr, 2)
        img_arr = img_arr / 255
        print(f'Image size: {np.shape(img_arr)}')

        for n in range(num_variations):
            if rand_seg is not None:
                counts_pos, counts_neg = event_image_compiler(rec_name, cap_or_norm=None, plot=False,
                                                              rand_seg=rand_seg, polarity='both')
                counts_pos = block_mean(counts_pos, 2)
                counts_neg = block_mean(counts_neg, 2)
                events = np.stack([counts_pos, counts_neg], axis=0).astype(np.float32)
                events_hwc = events.transpose(1, 2, 0)
                combined = np.concatenate([events_hwc, img_arr[..., np.newaxis]], axis=-1)
                combined_normalized = combined  # no EventCountNormalization
                events_normalized = combined_normalized[..., :-1].transpose(2, 0, 1)
                image_final = combined_normalized[..., -1]

                all_events_list.append(events_normalized)
                all_images_list.append(image_final)
                all_names_list.append(img_name)
                print(f'Processed pair {img_name}, variation {n+1} of {num_variations}')

        if use_full_time:
            df = read.recording_reader(rec_name, lazy=True)
            start_time = df.select(pl.col("timestamp").first()).collect().item()
            end_time = df.select(pl.col("timestamp").last()).collect().item()
            start_time_s = start_time / 1e6
            end_time_s = end_time / 1e6
            rec_length = end_time - start_time
            rec_length_full_s = int(np.floor(rec_length / 1e6))
            print(start_time_s, end_time_s, rec_length_full_s)

            for i in range(int(rec_length_full_s / integration_time_s)):
                print(f'pair {i+1} out of {int(rec_length_full_s/integration_time_s)}')
                seg = (i*integration_time_s + start_time_s, (i+1)*integration_time_s + start_time_s)
                counts_pos, counts_neg = event_image_compiler(rec_name, cap_or_norm='norm',
                                                              plot=False, seg=seg, polarity='both')
                counts_pos = block_mean(counts_pos, 2)
                counts_neg = block_mean(counts_neg, 2)
                events = np.stack([counts_pos, counts_neg], axis=0).astype(np.float32)
                events_hwc = events.transpose(1, 2, 0)
                combined = np.concatenate([events_hwc, img_arr[..., np.newaxis]], axis=-1)
                combined_normalized = combined
                events_normalized = combined_normalized[..., :-1].transpose(2, 0, 1)
                image_final = combined_normalized[..., -1]

                all_events_list.append(events_normalized)
                all_images_list.append(image_final)
                all_names_list.append(img_name + '_' + str(i))
                print(f'Processed pair {img_name + "_" + str(i)}')

    _save_as_memmap(all_events_list, all_images_list, all_names_list,
                    event_target_folder, image_target_folder)
    print("\nSaving as memory-mapped arrays...")
    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE!")
    print("=" * 60)
    print(f"Total samples: {len(all_names_list)}")
    print("=" * 60)


def _save_as_memmap(events_list, images_list, names_list, event_folder, image_folder):
    """Stack and save lists of (events, image) arrays as ``events_all.npy`` +
    ``images_all.npy`` files (which are memory-mappable by :class:`numpy.memmap`)."""
    n_samples = len(events_list)
    print(f"Stacking {n_samples} samples...")
    events_all = np.stack(events_list, axis=0)
    images_all = np.stack(images_list, axis=0)
    print(f"Events shape: {events_all.shape}, dtype: {events_all.dtype}")
    print(f"Images shape: {images_all.shape}, dtype: {images_all.dtype}")

    events_path = os.path.join(event_folder, 'events_all.npy')
    images_path = os.path.join(image_folder, 'images_all.npy')
    print(f"Saving events to {events_path}")
    np.save(events_path, events_all)
    print(f"Saving images to {images_path}")
    np.save(images_path, images_all)

    names_path = os.path.join(event_folder, 'sample_names.txt')
    with open(names_path, 'w') as f:
        f.write('\n'.join(names_list))
    print(f"  Memory-mapped files created successfully!")
    print(f"   Size: Events={events_all.nbytes / 1e9:.2f}GB, "
          f"Images={images_all.nbytes / 1e9:.2f}GB")


def _save_as_hdf5(events_list, images_list, names_list, output_folder):
    """Save stacked events/images/names as a compressed HDF5 file."""
    import h5py
    n_samples = len(events_list)
    h5_path = os.path.join(output_folder, 'synthetic_dataset.h5')
    print(f"Creating HDF5 file: {h5_path}")
    events_all = np.stack(events_list, axis=0)
    images_all = np.stack(images_list, axis=0)
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('events', data=events_all, compression='gzip', compression_opts=4)
        f.create_dataset('images', data=images_all, compression='gzip', compression_opts=4)
        dt = h5py.special_dtype(vlen=str)
        names_ds = f.create_dataset('sample_names', (n_samples,), dtype=dt)
        names_ds[:] = names_list
    print(f"  HDF5 file created successfully!")
    print(f"   Samples: {n_samples}")


def prepare_dataset(source_folder, save_folder):
    """
    Crop-to-16:9 + greyscale-convert every image in ``source_folder`` and
    save under ``save_folder``.  This is a pre-recording preparation step —
    it produces the base images that are displayed on an LED screen for
    EC-noise-image capture.  **Not required** for working with the shipped
    public dataset (its base images are already processed); included for
    completeness if you want to record new recordings on your own set.
    """
    all_images = []
    for root, dirs, images in os.walk(source_folder):
        for image_file in images:
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, image_file))

    os.makedirs(save_folder, exist_ok=True)

    for n, image_file in enumerate(all_images):
        num_images = len(all_images)
        head, tail = os.path.split(image_file)
        file_name_str, ext = os.path.splitext(tail)
        print(f'Processing image: {file_name_str}, {n + 1} out of {num_images}')

        img = Image.open(image_file).convert('L')
        print(img.size)
        width = img.size[0]
        height = img.size[1]
        if width >= height:
            factor = np.floor(width / 16)
            new_width = int(factor * 16)
            new_height = int(factor * 9)
            new_img = PIL.ImageOps.fit(img, size=(new_width, new_height))
        if width < height:
            # Assume vertical source: rotate to horizontal and crop.
            img = img.rotate(-90)
            width = img.size[0]
            height = img.size[1]
            factor = np.floor(width / 16)
            new_width = int(factor * 16)
            new_height = int(factor * 9)
            new_img = PIL.ImageOps.fit(img, size=(new_width, new_height))

        new_file_name = file_name_str + '_mod.png'
        new_file_path = os.path.join(save_folder, new_file_name)
        new_img.save(new_file_path)
        print(f'Saved image: {new_file_name}')
