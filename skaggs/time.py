"""
Handle time and time intervals.
"""

from functools import reduce
import operator as op

import numpy as np

from roto import arrays


def select_from(t, intervals):
    """Index for selecting time intervals from a time array.

    Arguments:
    t -- time array from which times are selected
    intervals -- list of (start, end) tuples of time intervals

    Returns:
    Boolean array indicating time points from the intervals
    """
    return reduce(op.add, (np.logical_and(t>=u, t<v) for u,v in intervals),
            np.zeros_like(t, '?'))

def exclude_from(t, intervals):
    """Index for excluding time intervals from a time array.

    Arguments and return values similar to `select_from`.
    """
    return np.logical_not(select_from(t, intervals))

def intervals_where(t, condns, min_dur=0.0, gap_tol=None, freq=None):
    """Find the time intervals for which a set of conditions are met.

    Arguments:
    t -- time array from which intervals are extracted
    condns -- binary indicator arrays of conditions that will be ANDed

    Keyword arguments:
    min_size -- minimum interval duration
    gap_tol -- optionally merge adjacent intervals up to `gap_tol` apart
    freq -- sampling frequency of the time array (default to median diff)

    Returns:
    [(start, end), ...] array of timing interval tuples
    """
    freq = 1/np.median(np.diff(t)) if freq is None else freq
    ix = reduce(op.mul, condns, np.ones_like(t, '?'))
    sz = max(1, int(np.ceil(min_dur*freq)))
    grps = arrays.find_groups(ix, min_size=sz)
    if gap_tol is not None:
        tol = int(np.ceil(gap_tol*freq))
        grps = arrays.merge_adjacent_groups(grps, tol=tol)
    if not grps.size:
        return grps
    grps[-1,-1] -= int(grps[-1,-1] == t.size)
    return t[grps]
