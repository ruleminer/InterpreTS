from itertools import groupby
import numpy as np
import pandas as pd


def calculate_flat_spots(data, window_size=5):
    """
    Calculate the number of flat spots in the time series.
    
    Flat spots are defined as maximum run-lengths across equally-sized segments of the time series.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        The time series data for which flat spots are to be calculated.
    window_size : int, optional
        The size of the window to look for flat spots (default is 5).

    Returns
    -------
    int
        The number of flat spots in the time series.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.Series([1, 1, 1, 2, 3, 3, 4, 4, 4, 4, 4, 5, 1, 1])
    >>> calculate_flat_spots(data)
    4
    """
    run_lengths = [
        len(list(group))
        for i in range(0, len(data), window_size)
        for _, group in groupby(data[i:i + window_size])
    ]
    max_flat_spot = max(run_lengths, default=0)

    return max_flat_spot
