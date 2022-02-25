"""
Functions for computing characteristics of spike trains.
"""

from ipyparallel import interactive
from scipy import signal
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np

import tenko.parallel as tp
import pouty as pty


BURST_ISI = 0.009
SIM_BURST_ISI = 0.035


class FiringPhaseEstimate(object):

    """
    Compute a continous signal estimate of theta phase of firing.
    """

    def __init__(self, st, tlim=None, ntaps=69, corr_baseline=1.0,
        corr_maxlag=3.0, Fs=250.0, parallel=False):
        if tlim is None:
            self.tlim = (st[0], st[-1])
            self.spikes = st
        else:
            self.tlim = tlim
            self.spikes = st[np.logical_and(st >= tlim[0], st <= tlim[1])]
        self.duration = tlim[1] - tlim[0]
        self.maxlag = corr_maxlag
        self.baseline = corr_baseline
        self.Fs = Fs
        self.ntaps = ntaps
        self.parallel = parallel

    def compute(self):
        """Compute the phase estimate time series for the spike train."""
        self._matched_filter()
        self._convolve_template()
        self._filter_response()
        self._compute_phase()
        return self.ts, self.phase

    def _matched_filter(self):
        """Generate the matched filter based on autocorrelation."""
        corrbins = round(2 * self.maxlag * self.Fs)
        autocorr, lags = acorr(self.spikes, maxlag=self.maxlag, bins=corrbins,
                parallel=self.parallel)
        baseline = autocorr[lags >= self.maxlag - self.baseline].mean()
        autocorr -= baseline
        autocorr /= autocorr.sum()
        self.template = autocorr

    def _convolve_template(self):
        """Convolve the matched-filter template with the spike train."""
        bins = round(self.Fs * self.duration)
        st = np.histogram(self.spikes, bins=bins, range=self.tlim)[0]
        self.y = signal.fftconvolve(st, self.template, mode='same')
        self.ts = np.linspace(self.tlim[0], self.tlim[1], bins)

    def _filter_response(self, theta=(4, 12)):
        """Filter template response with a theta band-pass filter."""
        b = signal.firwin(self.ntaps, theta, pass_zero=False, nyq=self.Fs/2)
        self.yf = signal.filtfilt(b, [1], self.y)

    def _compute_phase(self):
        """Compute phase of filtered response using the Hilbert transform."""
        N = self.yf.size
        M = pow(2, int(np.ceil(np.log2(N))))
        H = signal.hilbert(np.r_[self.yf, np.zeros(M - N)])[:N]
        self.phase = np.arctan2(np.imag(H), self.yf)


class FiringRateEstimate(object):

    """
    Compute a binless, kernel-based estimate of instantaneous firing rate.
    """

    def __init__(self, spikes, duration, width=0.1, kernel='gaussian',
        Fs_norm=60.0):
        self.out = pty.ConsolePrinter(prefix=self.__class__.__name__)
        if spikes.ndim != 2:
            spikes = np.atleast_2d(spikes).T

        self.spikes = spikes
        self.avg_rate = spikes.size / duration
        self.sigma = sigma = width / np.sqrt(12)
        self.model = neighbors.KernelDensity(bandwidth=sigma, kernel=kernel,
            rtol=1e-3)
        self.model.fit(spikes)
        if Fs_norm == 0.0:
            self.norm = 1.0
        else:
            self._normalize(Fnorm=Fs_norm)

    def _normalize(self, Fnorm=60.0):
        self.out('Normalizing firing rate estimate ({} spikes)', self.spikes.size)
        smin, smax = self.spikes[0], self.spikes[-1]
        t0, t1 = smin - 2 * self.sigma, smax + 2 * self.sigma
        tp = np.linspace(t0, t1, int(Fnorm * (t1 - t0)))
        tp = np.atleast_2d(tp).T
        pf = np.exp(self.model.score_samples(tp)).squeeze()
        self.norm = self.avg_rate / pf.mean()

    def evaluate(self, tp):
        """Evaluate the firing-rate estimate at an array of time points."""
        if tp.ndim != 2:
            tp = np.atleast_2d(tp).T

        logp = self.model.score_samples(tp).squeeze()
        return self.norm * np.exp(logp)

    __call__ = evaluate


def burst_onset(ts, isi_thresh=BURST_ISI):
    """Find burst onsets within the given spike train.

    Arguments:
    ts -- (n_spikes,) array of spike times in seconds

    Keyword arguments:
    isi_thresh -- maximum ISI cutoff for intraburst spikes

    Returns:
    boolean index array of burst onset spikes
    """
    isi = np.diff(ts)
    return np.r_[True, isi > isi_thresh]

def burst_filter(ts, isi_thresh=BURST_ISI):
    """Filter spike train for spikes that initiate bursts.

    Arguments:
    ts -- (n_spikes,) array of spike times in seconds

    Keyword arguments:
    isi_thresh -- maximum ISI cutiff for intraburst spikes

    Returns:
    array of timestamps for initial burst spikes
    """
    ts = np.asarray(ts)
    return ts[burst_onset(ts, isi_thresh)]

def acorr(t, **kwargs):
    """Compute spike train autocorrelograms."""
    return xcorr(t, t, **kwargs)

def xcorr(a, b, maxlag=1.0, bins=128, side=None, parallel=False):
    """Compute the spike train correlogram of two spike train arrays

    Arguments:
    a, b -- compute correlations of spike train b relative to spike train a

    Keyword arguments:
    maxlag -- range of lag times (+/-) to be returned
    bins -- number of discrete lag bins to use for the histogram
    side -- None|'left'|'negative'|'right'|'positive', restrict lag range
    parallel -- use ipyparallel implementation

    Returns:
    (counts, bin_centers) tuple of float arrays
    """
    lagmin, lagmax = -maxlag, maxlag
    if side is None:
        if bins % 2 == 0:  # zero-center range should
            bins += 1      # include a zero-center bin
            pty.debug('xcorr: zero-center correlation, so '
                      'changing bins from {} to {}'.format(bins-1, bins))
    elif side in ('left', 'negative'):
        lagmax = 0.0
    elif side in ('right', 'positive'):
        lagmin = 0.0
    else:
        pty.debug('xcorr: ignoring unknown side value ({})'.format(repr(side)))
    return _xcorr(a, b, lagmin, lagmax, bins, parallel)

def _xcorr(a, b, lagmin, lagmax, bins, parallel):
    edges = np.linspace(lagmin, lagmax, bins + 1)

    if parallel:
        rc = tp.client()
        dview = rc[:]
        dview.block = True
        dview.execute('import numpy as np')
        dview.scatter('_xcorr_spiketrain', a)
        dview['_xcorr_reference'] = b
        dview['_xcorr_edges'] = edges
        dview['_xcorr_kernel'] = _xcorr_kernel
        ar = dview.apply_async(_xcorr_parallel, lagmin, lagmax, bins)
        H = np.array(ar.get()).sum(axis=0)
    else:
        H = _xcorr_kernel(a, b, lagmin, lagmax, bins, edges)

    centers = (edges[:-1] + edges[1:]) / 2
    return H, centers

@interactive
def _xcorr_parallel(lagmin, lagmax, bins):
    a, b, edges = _xcorr_spiketrain, _xcorr_reference, _xcorr_edges
    return _xcorr_kernel(a, b, lagmin, lagmax, bins, edges)

def _xcorr_kernel(a, b, lagmin, lagmax, bins, edges):
    nb = b.size
    start = end = 0
    H = np.zeros(bins)
    for t in a:
        while start < nb and b[start] < t + lagmin:
            start += 1
        if start == nb:
            break
        while end < nb and b[end] <= t + lagmax:
            end += 1
        H += np.histogram(b[start:end] - t, bins=edges)[0]
    return H

def maxlag_from_lags(lags):
    """From an array of lag bin centers, back-calculate the maxlag."""
    lags = np.asarray(lags)
    bins = lags.size
    maxlag = lags.max() + (lags.ptp() / (bins - 1) / 2)
    return maxlag

def corrplot(data, ax=None, style='verts', zero_line=True, norm=False, **fmt):
    """Plot spike train correlogram to the specified axes.

    Remaining keyword arguments are passed to `Axes.plot(...)` (lines, steps)
    or `Axes.vlines(...)` (verts).

    Arguments:
    data -- spike train array (acorr) or tuple of two arrays (xcorr)

    Keyword arguments:
    ax -- axes object to draw autocorrelogram into
    style -- plot style: can be 'verts' (vlines), 'steps', or 'lines'
    zero_line -- draw a vertical dotted line at zero lag for reference
    norm -- normalize so that the peak correlation is 1

    Returns:
    plot handle
    """
    ax = ax is None and plt.gca() or ax
    C, lags = data
    if norm:
        C = C.astype(float) / C.max()

    if style in ('lines', 'steps'):
        fmt.update(lw=fmt.get('lw', 2))
        if style == 'steps':
            fmt.update(drawstyle='steps-mid')
        h = ax.plot(lags, C, **fmt)
        h = h[0]
    elif style == 'verts':
        fmt.update(colors=fmt.get('colors', 'k'), lw=fmt.get('lw',2))
        h = ax.vlines(lags, [0], C, **fmt)

    ax.axhline(color='k')
    if zero_line:
        ax.axvline(color='k', ls=':')
    ax.set_xlim(lags.min(), lags.max())

    plt.draw_if_interactive()
    return h

def interval_firing_rate(t, duration=None):
    """ISI-based firing rate computation for a given spike train

    Arguments:
    t -- spike train (or segment) for which to compute ISI firing rate
    duration -- total sampling duration of the spike train; defaults to elapsed
        time of the spike train
    """
    if duration is None:
        duration = t.ptp()
    if t.size == 0:
        return 0.0
    elif t.size == 1:
        return 1 / duration
    else:
        return 1 / np.diff(t).mean()
