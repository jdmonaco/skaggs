"""
Functions for computing characteristics of spatial maps.
"""

import numpy as np

from roto.radians import cdiff


def ratemap_allocentricity(R, Rdir):
    """Compute allocentricity index for directional firing-rate maps.

    The index is based on the normalized vector angle similarity between the
    rate values in the directional maps and the reference map.

    If R has shape (nx,ny), then Rdir should have shape (n,nx,ny), where n is
    the number of directional maps.

    Arguments:
    R -- 2D array, the reference spatial map
    Rdir -- 3D array, the set of directional spatial maps

    Returns:
    float -- allocentricity index
    """
    N = Rdir.shape[0]
    cos = np.zeros(N)  # missing values contribute to 0 index

    for i in range(N):
        valid = np.logical_and(np.isfinite(Rdir[i]), np.isfinite(R))
        if not valid.any():
            continue

        r = Rdir[i][valid]
        f = R[valid]
        r_dot_f = np.dot(r, f)
        rnorm = np.sqrt(np.dot(r, r))
        fnorm = np.sqrt(np.dot(f, f))
        maxnorm = max(rnorm, fnorm)
        if not maxnorm:
            continue

        cos[i] = r_dot_f / maxnorm**2

    index = 1 - 2 * np.mean(np.arccos(cos)) / np.pi

    return index

def phasemap_allocentricity(P, Pdir):
    """Compute allocentricity index for directional firing-phase maps.

    The index is based on similarity calculated as an inverse normalized RMS
    deviation of the average phase values in the directional maps from the
    reference map.

    Arguments similar to `ratemap_allocentricity`.
    """
    N = Pdir.shape[0]
    rmsd = np.zeros(N) + np.pi / 2  # missing values contribute to 0 index

    for i in range(N):
        valid = np.logical_and(np.isfinite(Pdir[i]), np.isfinite(P))
        if not valid.any():
            continue

        p = Pdir[i][valid]
        f = P[valid]
        n = valid.sum()
        if not n:
            continue

        rmsd[i] = np.sqrt(np.sum(cdiff(p, f)**2) / n)

    index = 1 - 2 * np.mean(rmsd) / np.pi

    return index
