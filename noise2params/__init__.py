"""
noise2params — reconstruction pipeline for the Noise2Params paper.

This package contains the code needed to reproduce the deep-learning
noise-image reconstruction portion of:

    O. Root, J. Mujo, M. Xu.
    "Noise2Params: Unification and Parameter Determination from Noise via a
    Probabilistic Event Camera Model."

Submodules
----------
matlab_recreations
    Saddle-point / Gaussian reference implementations, exposed via
    ``from matlab_recreations import *`` (provides ``sqrt``, ``pi``, ``exp``
    from :mod:`math` plus a few standalone comparisons).
prob_models
    The Poisson / Gaussian / saddle-point probability models used by the
    noise-image generator.  See the paper's "Event-Detection Probability"
    section.
reading
    Event-recording I/O.  Reads Prophesee Metavision ``.parquet`` (or legacy
    ``.csv``) event files into :mod:`polars` dataframes and histograms them
    per-pixel.
utils
    Generic helpers: file lookup, truncated-distribution sampling, image
    cropping, image-similarity metrics, training-job launchers for WSL and
    native-Windows workflows.
noise_image
    The core reconstruction-pipeline module: synthetic noise-image
    generation, real noise-image compilation from recordings, CNN inference,
    dataset construction, and benchmark-metric evaluation.
outlier_masks
    A small helper to load the shipped ``Extreme Outliers.txt`` and
    ``Hot Pixels Column 30.txt`` files into coordinate lists used by the
    outlier-removal path of :func:`noise2params.reading.recording_reader`.

Conventional import aliases
---------------------------
Throughout the codebase the following aliases are used::

    from noise2params import utils as ut
    from noise2params import reading as read
    from noise2params import prob_models as pmdls
    from noise2params import noise_image as nimg
"""
