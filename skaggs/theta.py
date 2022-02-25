"""
Functions for computing characteristics of theta-rhythmic activity.
"""

import numpy as np
import scipy.optimize as opt
import scipy.special as sp

from pouty import debug as log
from roto import circstats, arrays

from . import labels, session
from ..tools import spikes


def phaseregress(x, theta):
    """Compute circular-linear regression of phase against e.g. distance."""
    return KempterPhaseCorrelation(x, theta).regress()


class KempterPhaseCorrelation(object):

    """
    Implementation of Kempter et al. (2012) circular-linear correlation.
    """

    def __init__(self, x, phase, maxslope=1.0):
        x, phase = np.atleast_1d(x, phase)
        valid = np.logical_and(np.isfinite(x), np.isfinite(phase))
        self.phi = phase[valid]
        self.x = x[valid]
        self.xnorm = (self.x - self.x.min()) / self.x.ptp()
        self.alim = (-maxslope, maxslope)

    def regress(self):
        """Compute the circular-linear correlation.

        Returns:
        (slope, intercept, correlation r, p-value) tuple
        """
        self.maximize_residuals()
        self.estimate_intercept()
        self.correlation_and_pvalue()
        return 2 * np.pi * self.a_hat, self.phi0, self.rho_hat, self.p_value

    def _R(self, a):
        res = self.phi - 2 * np.pi * self.xnorm * a
        return -1.0 * np.sqrt(np.cos(res).mean()**2 + np.sin(res).mean()**2)

    def maximize_residuals(self):
        """Equation (1): Maximize residuals to estimate slope."""
        res = opt.minimize_scalar(self._R, bounds=self.alim, method='bounded')
        if not res.success:
            log('Error: {}', res.message, error=True)
            self.a_hat = 0.0
            return
        self.a_hat = res.x / self.x.ptp()  # rescale to data range

    def estimate_intercept(self):
        """Equation (2): Estimate the intercept."""
        res = self.phi - 2 * np.pi * self.x * self.a_hat
        self.phi0 = circstats.mean(res)

    def correlation_and_pvalue(self):
        """Equations (3-4): Calculate correlation strength and p-value."""
        sdphi = np.sin(self.phi - circstats.mean(self.phi))
        sdphi2 = np.power(sdphi, 2)

        theta = 2 * np.pi * np.abs(self.a_hat) * self.x
        sdtheta = np.sin(theta - circstats.mean(theta))
        sdtheta2 = np.power(sdtheta, 2)

        lambda20 = np.sum(sdphi2)
        lambda02 = np.sum(sdtheta2)
        lambda22 = np.sum(sdphi2 * sdtheta2)

        self.rho_hat = (sdphi * sdtheta).sum() / np.sqrt(lambda20 * lambda02)

        z = self.rho_hat * np.sqrt(lambda20 * lambda02 / lambda22)
        self.p_value = sp.erfc(np.abs(z) / np.sqrt(2))


def burst_frequency_estimate(st_or_corr, maxlag=0.25, bins=128, debug=False,
    **kwargs):
    """Autocorrelation-based estimate of theta burst frequency."""
    kwargs['H'] = kwargs.get('H', 0.75)
    kwargs['theta_frac'] = kwargs.get('theta_frac', 1.0)

    check = lambda n, mn, mx: len(mn) == 2 and len(mx) == 1 and (
        mn[0] < mx[0] < mn[1] < n - 2)

    est = _theta_autocorr_estimates(st_or_corr, check, maxlag=maxlag,
        bins=bins, debug=debug, **kwargs)

    if est is None:
        if debug:
            log('Unable to find theta burst peak', error=True)
        return -1

    mins, maxs, lags, tix, Cg = est

    lagt = lags[tix]
    tau_MAP = lagt[maxs[0]]
    f_MAP = 1 / tau_MAP

    b = np.logical_and(lagt >= lagt[mins[0]], lagt <= lagt[mins[1]])
    W = Cg[tix][b] - Cg[tix][b].min()
    f_hat = 1 / np.dot(W / W.sum(), lags[tix][b])

    return (f_MAP + f_hat) / 2

def rhythmicity_estimate(st_or_corr, maxlag=0.25, bins=128, debug=False,
    **kwargs):
    """Autocorrelation-based estimate for an index of theta rhythmicity."""
    kwargs['H'] = kwargs.get('H', 2/3)
    kwargs['theta_frac'] = kwargs.get('theta_frac', 0.75)

    check = lambda n, mn, mx: len(mn) == 1 and len(mx) == 1 and (
        mn[0] < mx[0] and mn[0] > 1 and mx[0] < n - 2)

    est = _theta_autocorr_estimates(st_or_corr, check,
        maxlag=maxlag, bins=bins, debug=debug, **kwargs)

    if est is None:
        if debug:
            log('Unable to determine theta rhythmicity', error=True)
        return -1

    mins, maxs, lags, tix, Cg = est

    C_trough = Cg[tix][mins[0]]
    C_peak = Cg[tix][maxs[0]]

    return (C_peak - C_trough) / C_peak

def _theta_autocorr_estimates(st_or_corr, check, maxlag=0.25, bins=128, H=0.75,
    f0=7.5, f_theta_min=4.0, f_theta_max=10.0, f_theta_inc=0.25,
    tau_blur_min=0.002, tau_blur_max=0.02, tau_blur_inc=0.001,
    theta_frac=1.0, debug=False):
    """Autocorrelation-based estimates of theta peaks and troughs."""
    if type(st_or_corr) is tuple and len(st_or_corr) == 2:
        C, lags = st_or_corr
    else:
        C, lags = spikes.acorr(st_or_corr, maxlag=maxlag, bins=bins,
            parallel=st_or_corr.size>2e4)

    posix = lags > 0
    lags = lags[posix]
    C = C[posix]

    N = 0
    mins = maxs = []
    nblur = (tau_blur_max - tau_blur_min) / tau_blur_inc + 1
    nf = (f_theta_max - f_theta_min) / f_theta_inc + 1
    tau_blur_opt = np.inf
    f_theta_opt = np.inf
    retval = None

    for tau_blur in np.linspace(tau_blur_min, tau_blur_max, nblur):
        Cg = arrays.blur(C, (bins / (2 * maxlag)) * tau_blur, padding='same')

        for f_theta in np.linspace(f_theta_min, f_theta_max, nf):
            tau_theta = theta_frac / f_theta
            taulim = (1 - H) * tau_theta, (1 + H) * tau_theta
            tix = np.logical_and(lags >= taulim[0], lags <= taulim[1])
            N = tix.size

            mins = arrays.minima(Cg[tix])
            maxs = arrays.maxima(Cg[tix])

            if check(N, mins, maxs):
                if tau_blur < tau_blur_opt or (tau_blur == tau_blur_opt and (
                    np.abs(f_theta - f0) < np.abs(f_theta_opt - f0))):
                    tau_blur_opt = tau_blur
                    f_theta_opt = f_theta
                    retval = mins, maxs, lags, tix, Cg

    if debug:
        if retval is None:
            print('Peak not found')
        else:
            print('Peak found using blur = {}, f = {} Hz'.format(
                    tau_blur_opt, f_theta_opt))

    return retval
