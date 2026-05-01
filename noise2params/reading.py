"""
noise2params.reading — event-recording I/O.

    * :func:`recording_reader` — loads a event camera recording ``.parquet``
      (or legacy ``.csv``/``.h5``) event file into a :mod:`polars`
      DataFrame/LazyFrame.  Provides time-slicing, random-segment
      extraction, and optional outlier-pixel removal.
    * :func:`counts_pixel_array` — accumulates events into a 2D per-pixel
      count histogram.  The hot loop is offloaded to
      :func:`accumulate_counts_numba`.
    * :func:`accumulate_counts_numba` — ``@njit`` 2D histogram kernel.
    * :func:`remove_pixels_df` — drops rows whose ``(x, y)`` matches a
      supplied pixel coordinate list.  Used by
      :func:`recording_reader` when ``remove_outliers=True``.
    * :func:`crop_recording` — spatial crop of a recording dataframe.
    * :func:`read_hf_recording` — HDF5 recording reader (kept for ``.h5``
      support; the public dataset ships only ``.parquet``).

Outlier removal
---------------
The outlier pixel list is loaded from two plain-text files shipped with
the public dataset:
    * ``Extreme Outliers.txt`` — pixels that are consistently extreme
      outliers across all bias settings.
    * ``Hot Pixels Column 30.txt`` — per-bias-setting outlier pixel list for
      the default bias settings that the shipped recordings were made at.

These paths are resolved via :func:`noise2params.outlier_masks.load_default_pixel_list`,
which searches the current project directory with
:func:`noise2params.utils.file_checker`.  To point at custom outlier
lists, pass ``pixel_list`` explicitly to :func:`recording_reader` via
the ``remove_all_but`` / private kwargs, or edit the call site in
:func:`recording_reader` below.
"""

import os
import time
from numba import jit, njit, prange
import polars as pl
import difflib
from datetime import datetime
import numpy as np
from scipy.stats import poisson, norm

from noise2params import utils as ut
from noise2params.outlier_masks import load_default_pixel_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Event-file reading

def recording_reader(input_file, file_type='parquet', lazy=False, remove_outliers=False,
                     remove_from_start=None, remove_from_end=None, from_start=None,
                     return_outlier_count=False, rand_seg=None, seg=None,
                     remove_all_but=None, crop_params=None):
    """
    Load and process an event recording file.

    The ``recording_reader`` function is designed to load and process recording data files
    in CSV, Parquet, or HDF5 format.  It supports lazy file reading, time-slicing based
    on timestamps, spatial cropping, and outlier-pixel removal.  The function validates
    parameters for mutual exclusivity, reads the file, and applies any requested
    transformations to the resulting dataframe.

    Logical flow
    ------------
    1. Verifies that only one slicing method is used at a time
       (``remove_from_start``, ``remove_from_end``, ``from_start``, ``rand_seg``, or
       ``seg``).
    2. Validates the file path via :func:`noise2params.utils.file_checker` and reads
       the file according to ``file_type``.
    3. Under ``lazy=True`` for parquet, operations are deferred until ``.collect()``.
    4. Applies the requested slicing / cropping.
    5. If ``remove_outliers=True``, removes the pixels listed in the dataset's
       ``Extreme Outliers.txt`` and ``Hot Pixels Column 30.txt`` files.

    Parameters
    ----------
    input_file : str
        Path to the input file, or file stem resolved via ``file_checker``.
    file_type : {'parquet', 'csv', 'h5'}, default 'parquet'
        File format.
    lazy : bool, default False
        If True, use :func:`polars.scan_parquet` and return a LazyFrame.
    remove_outliers : bool, default False
        If True, remove outlier pixels from the shipped outlier lists.
    remove_from_start, remove_from_end, from_start : float, optional
        Mutually exclusive time slices in seconds.
    rand_seg : float or tuple, optional
        Random segment extraction: ``float`` specifies length in seconds;
        ``tuple`` specifies ``(length, start_offset, end_offset)``.
    seg : tuple of (float, float), optional
        Explicit segment bounds (in seconds from start_time).
    crop_params : tuple of (x_min, x_max, y_min, y_max), optional
        Spatial crop.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        Dataframe with columns ``x, y, polarity, timestamp``.

    Raises
    ------
    ValueError
        When more than one mutually exclusive slicing parameter is given.
    """
    if len([x for x in (remove_from_start, remove_from_end, from_start, rand_seg, seg) if x is not None]) > 1:
        raise ValueError('Only one of remove_from_start, remove_from_end, from_start, rand_seg, and seg can be used at a time')

    starttime = time.time()

    file_path=ut.file_checker(input_file, file_type)

    if file_type == 'csv':
        df = pl.read_csv(file_path, has_header=False,
                         dtypes={"x": pl.Int64, "y": pl.Int64, "polarity": pl.Int64,
                                 "timestamp": pl.Int64}, truncate_ragged_lines=True)
    if file_type == 'parquet':
        if lazy==False:
            df = pl.read_parquet(file_path)
            df = df.rename({
                df.columns[0]: "x",
                df.columns[1]: "y",
                df.columns[2]: "polarity",
                df.columns[3]: "timestamp"
            })
        if lazy==True:
            df = pl.scan_parquet(file_path)
            column_names = df.collect_schema().names()
            df = df.rename({
                column_names[0]: "x",
                column_names[1]: "y",
                column_names[2]: "polarity",
                column_names[3]: "timestamp"
            })
    if file_type == 'h5':
        df=read_hf_recording(file_path, lazy)

    print('reading the file took {} seconds'.format(time.time() - starttime))

    if lazy == False:
        start_time = df["timestamp"][0]
        end_time = df["timestamp"][-1]

    if lazy == True:
        start_time = df.select(pl.col("timestamp").first()).collect().item()
        end_time = df.select(pl.col("timestamp").last()).collect().item()

    if remove_from_start is not None:
        remove_from_start = remove_from_start * 1e6
        df = df.filter(pl.col("timestamp") >= (remove_from_start + start_time))
    if remove_from_end is not None:
        remove_from_end = remove_from_end * 1e6
        df = df.filter(pl.col("timestamp") <= (end_time - remove_from_end))
    if from_start is not None:
        from_start = from_start * 1e6
        df = df.filter(pl.col("timestamp") <= (from_start + start_time))
    if rand_seg != None:
        if type(rand_seg) in [float, int]:
            seg_start = np.random.randint(start_time, end_time - rand_seg * 1e6)
            seg_end = seg_start + (rand_seg * 1e6)
            df = df.filter((pl.col("timestamp") >= seg_start) & (pl.col("timestamp") <= seg_end))

        if type(rand_seg) == tuple:
            # (length of random segment, start of range of random segment after start,
            #  end of range of random segment before end)
            rand_seg_length = rand_seg[0]
            rand_seg_start = rand_seg[1]
            if len(rand_seg) == 2:
                rand_seg_end = 0
            if len(rand_seg) == 3:
                rand_seg_end = rand_seg[2]

            seg_start = np.random.randint(
                start_time + rand_seg_start * 1e6, end_time - (rand_seg_length * 1e6) - (rand_seg_end * 1e6)
            )
            seg_end = seg_start + (rand_seg_length * 1e6)
            df = df.filter((pl.col("timestamp") >= seg_start) & (pl.col("timestamp") <= seg_end))
    if type(seg)==tuple and len(seg)==2:
            seg_start = start_time + seg[0]*1e6
            seg_end = start_time + seg[1]*1e6
            df = df.filter((pl.col("timestamp") >= seg_start) & (pl.col("timestamp") <= seg_end))

    if type(crop_params)==tuple:
        if len(crop_params)!=4:
            raise ValueError('crop_params must specify all four of (x_min, x_max, y_min, y_max)')
        df = crop_recording(df, crop_params[0], crop_params[1], crop_params[2], crop_params[3], lazy=lazy)

    if remove_outliers==True:
        # Load the shipped outlier coordinate lists.  See the module docstring for
        # the file names / locations.
        pixel_list = load_default_pixel_list()
        print(f'length of outlier pixel_list: {len(pixel_list)}')
        df=remove_pixels_df(df,pixel_list)

    if type(remove_all_but)==list:
        df=remove_pixels_df(df,pixel_list=remove_all_but, remove_all_but=True)

    if return_outlier_count==False:
        return df

    if return_outlier_count==True:
        return df, len(pixel_list)


def read_hf_recording(file_path, lazy=False):
    """Read a Noise2Image-format HDF5 event recording into a polars
    dataframe.  Retained for completeness; the public dataset ships only
    ``.parquet`` recordings."""
    import h5py
    with h5py.File(file_path, "r") as f:
        events = f["CD"]["events"]
        x = events["x"][:]
        y = events["y"][:]
        p = events["p"][:]
        t = events["t"][:]
    df = pl.DataFrame({"x": x.astype(np.int64),
                       "y": y.astype(np.int64),
                       "polarity": p.astype(np.int64),
                       "timestamp": t.astype(np.int64)})
    if lazy:
        df = df.lazy()
    return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2D event count histogramming

@njit
def accumulate_counts_numba(x_positions, y_positions, width, height):
    """
    Accumulate a 2D event count histogram from x/y coordinate arrays.

    Numba-JIT implementation of a per-pixel event counter: for each input
    ``(x, y)`` pair, increment the corresponding element of a zero-initialised
    ``(height, width)`` :mod:`numpy` integer array.

    Parameters
    ----------
    x_positions, y_positions : ndarray
        1-D integer arrays of pixel coordinates.
    width, height : int
        Output histogram dimensions (in pixels).

    Returns
    -------
    ndarray
        2D (height, width) int32 histogram of event counts.
    """
    counts = np.zeros((height, width), dtype=np.int32)
    n = x_positions.shape[0]
    for i in range(n):
        counts[y_positions[i], x_positions[i]] += 1
    return counts


def counts_pixel_array(input_file, polarity=None, file_type='parquet', lazy=True,
                       remove_from_start=None, remove_from_end=None, from_start=None,
                       rand_seg=None, seg=None,
                       remove_outliers=True):
    """
    Build a 2D (height, width) event-count array from an event recording.

    Each pixel of the output array holds the number of events recorded at
    that pixel over the (optionally sliced) event stream.

    Parameters
    ----------
    input_file : str, polars.DataFrame, or polars.LazyFrame
        Event recording file path or an already-loaded dataframe.
    polarity : {None, 0, 1, 'both'}
        ``None``: all events.  ``0``: negative only.  ``1``: positive only.
        ``'both'``: return ``(counts_pos, counts_neg)`` as a tuple.
    file_type, lazy, remove_from_start, remove_from_end, from_start, rand_seg, seg, remove_outliers
        Forwarded to :func:`recording_reader`.

    Returns
    -------
    ndarray or (ndarray, ndarray)
        (H, W) int32 histogram, or tuple of (pos, neg) histograms.
    """
    if type(input_file) == str:
        df = recording_reader(input_file, file_type, remove_outliers=remove_outliers, lazy=lazy,
                              remove_from_start=remove_from_start, remove_from_end=remove_from_end,
                              from_start=from_start, rand_seg=rand_seg, seg=seg,
                              )
    if type(input_file) == pl.DataFrame:
        df=input_file.lazy().copy()

    if type(input_file) == pl.LazyFrame:
        df=input_file.copy()

    x_max = df.select(pl.max('x')).collect()[0, 0] + 1
    x_min = df.select(pl.min('x')).collect()[0, 0]
    width = x_max - x_min

    y_max = df.select(pl.max('y')).collect()[0, 0] + 1
    y_min = df.select(pl.min('y')).collect()[0, 0]
    height = y_max - y_min

    event_counts_array = np.zeros((height, width), dtype=np.int32)

    if polarity in [None, 0, 1]:
        if polarity == 0:
            df = df.filter(pl.col("polarity") == 0)
        if polarity == 1:
            df = df.filter(pl.col("polarity") == 1)

        coords = df.select(["x", "y"]).collect().to_numpy().astype(np.int64)
        x_positions = coords[:, 0] - x_min
        y_positions = coords[:, 1] - y_min

        event_counts_array = accumulate_counts_numba(x_positions, y_positions, width, height)
        return event_counts_array

    if polarity == 'both':
        df_pos = df.filter(pl.col("polarity") == 1)
        df_neg = df.filter(pl.col("polarity") == 0)

        coords_pos = df_pos.select(["x", "y"]).collect().to_numpy().astype(np.int64)
        x_positions_pos = coords_pos[:, 0] - x_min
        y_positions_pos = coords_pos[:, 1] - y_min

        coords_neg = df_neg.select(["x", "y"]).collect().to_numpy().astype(np.int64)
        x_positions_neg = coords_neg[:, 0] - x_min
        y_positions_neg = coords_neg[:, 1] - y_min

        counts_pos = accumulate_counts_numba(x_positions_pos, y_positions_pos, width, height)
        counts_neg = accumulate_counts_numba(x_positions_neg, y_positions_neg, width, height)

        return counts_pos, counts_neg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dataframe pixel / spatial filtering

def remove_pixels_df(input_data, pixel_list, remove_all_but=False):
    """
    Remove rows with ``(x, y)`` in ``pixel_list`` from an event dataframe.

    Implemented as an anti-join against a small DataFrame constructed from
    the pixel coordinate list.  Preserves laziness: if ``input_data`` is a
    :class:`polars.LazyFrame`, returns a LazyFrame.

    Parameters
    ----------
    input_data : polars.DataFrame or polars.LazyFrame
        Recording dataframe with at least columns ``x`` and ``y``.
    pixel_list : list of [x, y]
        Coordinates to drop.
    remove_all_but : bool, default False
        If True, instead retain only rows matching ``pixel_list`` (semi-join).
    """
    pixels_df = pl.DataFrame(pixel_list, schema=["x", "y"], orient="row")

    if isinstance(input_data, pl.LazyFrame):
        pixels_df = pixels_df.lazy()

    if remove_all_but==False:
        return input_data.join(pixels_df, on=["x", "y"], how="anti")

    if remove_all_but==True:
        return input_data.join(pixels_df, on=["x", "y"], how="semi")


def crop_recording(input_data, x_min, x_max, y_min, y_max, file_type='parquet', lazy=False):
    """
    Spatially crop an event recording to the given pixel bounding box.

    Does not ``.collect()`` a lazy frame.
    """
    if type(input_data) not in [str, pl.DataFrame, pl.LazyFrame]:
        raise TypeError('input_data must either a string (of a filename) or a Polars data or lazy frame.')

    if type(input_data)==str:
        df = recording_reader(input_data, lazy=lazy, remove_outliers=True)

    if type(input_data) in [pl.DataFrame, pl.LazyFrame]:
        df = input_data

    df = df.filter((pl.col("x") >= x_min) & (pl.col("x") < x_max) &
                   (pl.col("y") >= y_min) & (pl.col("y") < y_max))
    return df
