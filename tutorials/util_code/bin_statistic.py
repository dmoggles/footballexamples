"""
This is a fork of mplsoccer code from 
https://github.com/andrewRowlinson/mplsoccer/blob/main/mplsoccer/heatmap.py

This code is authored and owned by Andrew Rowlinson. Its been copied here to apply
modifications to allow the use of gaussian_filter and zoom
"""

import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter, zoom
from dataclasses import asdict
from mplsoccer.heatmap import _nan_safe, BinnedStatisticResult


def bin_statistic(x, y, values=None, dim=None, statistic='count',
                  bins=(5, 4), normalize=False, standardized=False,
                  gaussian_filter_value=None, zoom_value=None):
                  
    """ Calculates binned statistics using scipy.stats.binned_statistic_2d.

    This method automatically sets the range, changes the scipy defaults,
    and outputs the grids and centers for plotting.

    The default statistic has been changed to count instead of mean.
    The default bins have been set to (5,4).

    Parameters
    ----------
    x, y, values : array-like or scalar.
        Commonly, these parameters are 1D arrays.
        If the statistic is 'count' then values are ignored.
    dim : mplsoccer pitch dimensions
        One of FixedDims, MetricasportsDims, VariableCenterDims, or CustomDims.
        Automatically populated when using Pitch/ VerticalPitch class
    statistic : string or callable, optional
        The statistic to compute (default is 'count').
        The following statistics are available: 'count' (default),
        'mean', 'std', 'median', 'sum', 'min', 'max', 'circmean' or a user-defined function. See:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_2d.html
    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification.
          * the number of bins for the two dimensions (nx = ny = bins),
          * the number of bins in each dimension (nx, ny = bins),
          * the bin edges for the two dimensions (x_edge = y_edge = bins),
          * the bin edges in each dimension (x_edge, y_edge = bins).
            If the bin edges are specified, the number of bins will be,
            (nx = len(x_edge)-1, ny = len(y_edge)-1).
    normalize : bool, default False
        Whether to normalize the statistic by dividing by the total.
    standardized : bool, default False
        Whether the x, y values have been standardized to the
        'uefa' pitch coordinates (105m x 68m)
    gaussian_filter_value : float, default None
        The standard deviation for the Gaussian kernel. If None, no filtering is performed.
    zoom_value : float, default None
        The zoom factor. If None, no zooming is performed.

    Returns
    -------
    bin_statistic : dict.
        The keys are 'statistic' (the calculated statistic),
        'x_grid' and 'y_grid (the bin's edges), cx and cy (the bin centers)
        and 'binnumber' (the bin indices each point belongs to).
        'binnumber' is a (2, N) array that represents the bin in which the observation falls
        if the observations falls outside the pitch the value is -1 for the dimension. The
        binnumber are zero indexed and start from the top and left handside of the pitch.

    Examples
    --------
    >>> from mplsoccer import Pitch
    >>> import numpy as np
    >>> pitch = Pitch(line_zorder=2, pitch_color='black')
    >>> fig, ax = pitch.draw()
    >>> x = np.random.uniform(low=0, high=120, size=100)
    >>> y = np.random.uniform(low=0, high=80, size=100)
    >>> stats = pitch.bin_statistic(x, y)
    >>> pitch.heatmap(stats, edgecolors='black', cmap='hot', ax=ax)
    """
    x = np.ravel(x)
    y = np.ravel(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    statistic = _nan_safe(statistic)
    if (values is None) & (statistic == 'count'):
        values = x
    if (values is None) & (statistic != 'count'):
        raise ValueError("values on which to calculate the statistic are missing")
    if standardized:
        pitch_range = [[0, 105], [0, 68]]
    elif dim.invert_y:
        pitch_range = [[dim.left, dim.right], [dim.top, dim.bottom]]
        y = dim.bottom - y
    else:
        pitch_range = [[dim.left, dim.right], [dim.bottom, dim.top]]
    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(x, y, values, statistic=statistic,
                                                               bins=bins, range=pitch_range,
                                                               expand_binnumbers=True)

    if gaussian_filter_value is not None:
        statistic = gaussian_filter(statistic, gaussian_filter_value)
    if zoom_value is not None:
        statistic = zoom(statistic, zoom_value)
        x_edge = np.linspace(x_edge[0], x_edge[-1], int(bins[0] * zoom_value) + 1)
        y_edge = np.linspace(y_edge[0], y_edge[-1], int(bins[1] * zoom_value) + 1)
        # Determine the bin indices in the new grid for each data point
        x_bin_indices = np.digitize(x, x_edge) - 1  # `-1` to match zero-based indexing
        y_bin_indices = np.digitize(y, y_edge) - 1
        binnumber = np.array([x_bin_indices, y_bin_indices])



    statistic = np.flip(statistic.T, axis=0)
    if statistic.ndim == 3:
        num_y, num_x, _ = statistic.shape
    else:
        num_y, num_x = statistic.shape
    if normalize:
        statistic = statistic / statistic.sum()
    binnumber[1, :] = num_y - binnumber[1, :] + 1
    x_grid, y_grid = np.meshgrid(x_edge, y_edge)
    cx, cy = np.meshgrid(x_edge[:-1] + 0.5 * np.diff(x_edge), y_edge[:-1] + 0.5 * np.diff(y_edge))

    if not dim.invert_y or standardized is not False:
        y_grid = np.flip(y_grid, axis=0)
        cy = np.flip(cy, axis=0)

    # if outside the pitch set the bin number to minus one
    # else zero index the results by removing one
    mask_x_out = np.logical_or(binnumber[0, :] == 0,
                               binnumber[0, :] == num_x + 1)
    binnumber[0, mask_x_out] = -1
    binnumber[0, ~mask_x_out] = binnumber[0, ~mask_x_out] - 1

    mask_y_out = np.logical_or(binnumber[1, :] == 0,
                               binnumber[1, :] == num_y + 1)
    binnumber[1, mask_y_out] = -1
    binnumber[1, ~mask_y_out] = binnumber[1, ~mask_y_out] - 1
    inside = np.logical_and(~mask_x_out, ~mask_y_out)
    return asdict(BinnedStatisticResult(statistic, x_grid, y_grid,
                                        cx, cy, binnumber=binnumber,
                                        inside=inside))
