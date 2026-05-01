"""
noise2params.outlier_masks — load the shipped outlier pixel coordinate lists.

The Noise2Params public dataset ships two plain-text files that list
coordinates of pixels that should be excluded from event-count
histograms and subsequent analysis:

    ``Extreme Outliers.txt``
        Pixels that are consistently extreme outliers across all bias
        settings the lab has tested.  These are always safe to remove.

    ``Hot Pixels Column 30.txt``
        Per-bias-setting hot pixel list for the default bias settings
        (bias_diff_on / bias_diff_off at their factory defaults).  All
        shipped recordings were made with those default settings, so this
        is the list to use with the shipped data.

Each file is a plain-text Python ``list`` literal whose elements are
``[x, y]`` coordinate pairs.  :func:`load_default_pixel_list` reads and
concatenates them (dropping duplicates from the second).

Customisation
-------------
If you record at non-default biases, substitute the matching
``Hot Pixels Column N.txt`` (N = bias-difference setting) by passing
a different ``per_bias_file`` to :func:`load_default_pixel_list`.
"""

from typing import List
from noise2params import utils as ut


def load_default_pixel_list(
    extreme_file: str = 'Extreme Outliers',
    per_bias_file: str = 'Hot Pixels Column 30',
) -> List[list]:
    """Load the outlier pixel coordinate list used by the shipped dataset.

    Parameters
    ----------
    extreme_file : str
        Stem of the extreme-outliers text file.  Resolved via
        :func:`noise2params.utils.file_checker`; pass a full path to
        bypass that search.
    per_bias_file : str
        Stem of the per-bias hot-pixels file (default matches the
        shipped recordings' bias settings).

    Returns
    -------
    list of [int, int]
        Deduplicated list of (x, y) pixel coordinates to drop.
    """
    extreme_path = ut.file_checker(extreme_file, file_type='txt')
    per_bias_path = ut.file_checker(per_bias_file, file_type='txt')

    extreme_list = ut.text_to_data_format(extreme_path)
    per_bias_list = ut.text_to_data_format(per_bias_path)

    # Deduplicate per-bias list and merge with extreme list.
    per_bias_list = [list(t) for t in {tuple(p) for p in per_bias_list
                                       if isinstance(p, (list, tuple)) and len(p) == 2}]
    merged = list(extreme_list) + per_bias_list
    return merged
