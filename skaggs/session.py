"""
Primary interface for handling recording session data.
"""

from functools import reduce
import operator as op

from scipy.interpolate import interp1d
import numpy as np

from roto import arrays
from roto.decorators import memoize, lazyprop
from pouty import debug

from . import time, motion, lfp, cluster
from .. import store
from ..tools import binned, adaptive, spikes
from .data import circinterp1d, Arenas
from .parsers import parse_session, parse_ttc


INFO_PVAL_SAMPLES = 1000
SHUFFLE_MIN_OFFSET = 20.0


def record(s_id):
    """Get a dict with the /sessions row for the given session id."""
    return parse_session(s_id)

@memoize
def get(*args, **kwargs):
    return RecordingSession(*args, **kwargs)


class RecordingSession(object):

    """
    Load recording session data and perform session-wide computations.
    """

    def __init__(self, index, **mapargs):
        # Store attributes from /sessions row data
        self.attrs = parse_session(index, raise_on_fail=True)
        keymap = {k:k for k in self.attrs}
        keymap['path'] = 'folder'
        keymap['group'] = 'path'
        for key in self.attrs:
            setattr(self, keymap[key], self.attrs[key])

        # Load session-level data sources
        self.arena = Arenas(self.attrs['arena'])
        self.motion = motion.MotionData(self.path)
        self.lfp = lfp.ContinuousData(self.path)

        # Adaptive maps
        self.adaptive_ratemap = adaptive.AdaptiveRatemap(self.motion,
                **mapargs)
        self.adaptive_phasemap = adaptive.AdaptivePhasemap(self.motion,
                **mapargs)

        # Session timing
        self.tstart = max(self.motion.t[0], self.lfp.t[0])
        self.tend = min(self.motion.t[-1], self.lfp.t[-1])
        self.tlim = (self.tstart, self.tend)
        self.duration = self.tend - self.tstart

        # Load spiking data for each cluster
        f = store.get()
        self.clusters = {}
        for C in f.root.recordings.where("session_id==%d" % self.id):
            cdata = cluster.ClusterData(self.path, C)
            self.clusters[cdata.ttc] = cdata

    def get_cluster(self, cell):
        """Get the cluster data for a given cell."""
        ttc = parse_ttc(cell)
        if ttc not in self.clusters:
            raise KeyError('no cluster data for cluster %s' % ttc)
        return self.clusters[ttc]

    # Mapping methods

    def dirmap(self, cell, bursts=False, speed='fast', intervals=None,
        bins=None):
        """Compute a directional rate map (1D) for the given cell.

        Arguments:
        cell -- (tetrode, cluster) key for the cluster

        Keyword arguments:
        bursts -- restrict spike train to burst onset timing
        speed -- ('fast'|'slow'|'still'|None) velocity filter
        intervals -- restrict data to the given list of time intervals
        bins -- int, number of directional bins (default {0})

        Returns:
        rates -- (bins,)-shaped array
        angles -- (bins,)-shaped array
        """
        (_, ds), (_, dp) = self._dirdata(cell, speed, intervals, bursts)

        R, phi = binned.dirmap(ds, dp, bins=bins, freq=self.motion.Fs)

        return R, phi

    dirmap.__doc__ = dirmap.__doc__.format(binned.DEFAULT_DIR_BINS)

    def speedmap(self, cell, bursts=False, intervals=None, min_occ=None,
        bins=None, slim=None):
        """Compute a speed rate map (1D) for the given cell.

        Arguments:
        cell -- (tetrode, cluster) key for the cluster

        Keyword arguments:
        bursts -- restrict spike train to burst onset timing
        intervals -- restrict data to the given list of time intervals
        min_occ -- minimum bin occupancy for masking (default {0})
        bins -- int, number of speed bins (default {1})
        slim -- (smin, smax) speed limits tuple (default ({2}, {3}))

        Returns:
        rates -- (bins,)-shaped array
        speeds -- (bins,)-shaped array
        """
        speed = None
        smin, smax = (None, None) if slim is None else slim
        (_, vs), (_, vp) = self._speeddata(cell, speed, intervals, bursts)

        R, speeds = binned.speedmap(vs, vp, bins=bins, smin=smin, smax=smax,
                min_occ=min_occ, freq=self.motion.Fs)

        return R, speeds

    speedmap.__doc__ = speedmap.__doc__.format(binned.DEFAULT_SPEED_MIN_OCC,
            binned.DEFAULT_SPEED_BINS, binned.DEFAULT_SPEED_MIN,
            binned.DEFAULT_SPEED_MAX)

    def ratemap(self, cell, bursts=False, adaptive=True, speed='fast',
        intervals=None, min_occ=None, bins=None):
        """Compute a spatial rate map for the given cell.

        Note: `min_occ` and `bins` are ignored for `adaptive=True`. The value
        for `min_occ` (default {0}) should be reduced as `bins` (default {1}) is
        increased above the default number.

        Arguments:
        cell -- (tetrode, cluster) key for the cluster

        Keyword arguments:
        bursts -- restrict spike train to burst onset timing
        adaptive -- boolean, use adaptive kernel smoothing
        speed -- ('fast'|'slow'|'still'|None) velocity filter
        intervals -- restrict data to the given list of time intervals
        min_occ -- minimum bin occupancy for spatial masking (default {0})
        bins -- int, number of spatial bins along each dimension (default {1})

        Returns:
        (bins,bins)-shaped array
        """
        (_, xs, ys), (_, xp, yp) = self._posdata(cell, speed, intervals, bursts)

        if adaptive:
            R = self.adaptive_ratemap(xs, ys, xp, yp)
        else:
            R = binned.ratemap(xs, ys, xp, yp, bins=bins, min_occ=min_occ,
                    freq=self.motion.Fs)

        return R

    ratemap.__doc__ = ratemap.__doc__.format(binned.DEFAULT_MIN_OCC,
            binned.DEFAULT_BINS)

    def phasemap(self, cell, bursts=False, adaptive=True, speed='fast',
        intervals=None, min_spikes=5, bins=None):
        """Compute a spatial phase map for the given cell.

        Arguments are the same as `ratemap`.
        """
        ts, xs, ys = self._spkposdata(cell, speed, intervals, bursts)
        phase = self.lfp.F('phase', ts)

        if adaptive:
            P = self.adaptive_phasemap(xs, ys, phase)
        else:
            P = binned.phasemap(xs, ys, phase, bins=bins,
                    min_spikes=min_spikes, freq=self.motion.Fs)

        return P

    # Information rate methods

    def directional_info(self, cell, speed='fast', intervals=None, bursts=False,
        pvalue=False):
        """Compute directional information for a cell with optional p-value."""
        (ts, ds), (tm, dm) = self._dirdata(cell, speed, intervals, bursts)

        I = binned.dirinfo(ds, dm)

        if not pvalue:
            return I

        debug('spatial_info: computing shuffled direction information values')
        self._compress_intervals(ts, tm, speed=speed)
        dstar = circinterp1d(tm, dm, zero_centered=False, copy=False)
        Istar = [binned.dirinfo(dstar(tstar), dm) for tstar in
                self._shuffle_spikes(ts, tm)]
        pval = self._shuffle_pvalue(I, Istar)

        return I, pval

    def speed_info(self, cell, intervals=None, bursts=False, pvalue=False):
        """Compute speed information for a cell with optional p-value."""
        speed = None
        (ts, vs), (tm, vm) = self._speeddata(cell, speed, intervals, bursts)

        I = binned.speedinfo(vs, vm)

        if not pvalue:
            return I

        debug('spatial_info: computing shuffled speed information values')
        self._compress_intervals(ts, tm, speed=speed)
        vstar = interp1d(tm, vm, copy=False)
        Istar = [binned.speedinfo(vstar(tstar), vm) for tstar in
                self._shuffle_spikes(ts, tm)]
        pval = self._shuffle_pvalue(I, Istar)

        return I, pval

    def spatial_info(self, cell, speed='fast', intervals=None, bursts=False,
        pvalue=False):
        """Compute spatial information for a cell with optional p-value."""
        (ts, xs, ys), (tm, xp, yp) = self._posdata(cell, speed, intervals,
                bursts)

        I = binned.rateinfo(xs, ys, xp, yp)

        if not pvalue:
            return I

        debug('spatial_info: computing shuffled spatial information values')
        self._compress_intervals(ts, tm, speed=speed)
        xstar = interp1d(tm, xp, copy=False)
        ystar = interp1d(tm, yp, copy=False)
        Istar = [binned.rateinfo(xstar(tstar), ystar(tstar), xp, yp)
                for tstar in self._shuffle_spikes(ts, tm)]
        pval = self._shuffle_pvalue(I, Istar)

        return I, pval

    def phase_space_info(self, cell, speed='fast', intervals=None, bursts=False,
        pvalue=False):
        """Compute phase-position mutual information with optional p-value."""
        ts, xs, ys = self._spkposdata(cell, speed, intervals, bursts)
        phase = self.lfp.F('phase', ts)

        I = binned.phaseinfo(xs, ys, phase)

        if not pvalue:
            return I

        debug('spatial_info: computing shuffled phase information values')
        Istar = np.empty(INFO_PVAL_SAMPLES)
        for i in range(INFO_PVAL_SAMPLES):
            np.random.shuffle(phase)
            Istar[i] = binned.phaseinfo(xs, ys, phase)
        pval = self._shuffle_pvalue(I, Istar)

        return I, pval

    # Filtering methods

    def spike_filter(self, cell, speed='fast', intervals=None, bursts=False):
        """Spike times filtered by speed, time intervals, and/or bursting."""
        return self._spkfilt(cell, speed, intervals, bursts)

    def motion_filter(self, speed='fast', intervals=None):
        """Motion index array for filtering by speed or time intervals."""
        return self._occfilt(speed, intervals)

    # Private methods for shuffle-testing spike trains

    def _compress_intervals(self, spike_t, motion_t, speed='fast'):
        """Remove speed interval gaps in spike and motion timing in place."""
        if speed is None:
            return

        ints = getattr(self.motion, '%s_intervals' % speed)
        t0 = motion_t[-1] - motion_t[0]
        for i in range(ints.shape[0] - 1, 1, -1):
            start, prev = ints[i, 0], ints[i - 1, 1]
            gap = start - prev
            spike_t[spike_t>=start] -= gap
            motion_t[motion_t>=start] -= gap
        debug('_compress_intervals: {:.1f}% compression',
                100 - 100 * (motion_t[-1] - motion_t[0]) / t0)

    def _shuffle_spikes(self, spike_t, motion_t, samples=INFO_PVAL_SAMPLES,
        min_offset=SHUFFLE_MIN_OFFSET):
        """Generate random offset shuffles of spike times."""
        start, end = motion_t[0], motion_t[-1]
        dur = end - start
        shuffled = np.empty_like(spike_t)
        for i in range(samples):
            offset = min_offset + (dur - 2 * min_offset) * np.random.rand()
            shuffled[:] = spike_t + offset
            shuffled[shuffled > end] -= dur
            yield shuffled

    def _shuffle_pvalue(self, observed, shuffled):
        """Compute a p-value against a shuffled distribution."""
        shuffled = np.asarray(shuffled)
        pval = ((shuffled >= observed).sum() + 1) / shuffled.size
        pval = min(pval, 1.0)  # avoid 1.001
        return pval

    # Private methods for filtered spike and occupancy map data

    def _posdata(self, cell, speed, intervals, bursts):
        """Retrieve filtered positional spike and occupancy data."""
        spk = self._spkposdata(cell, speed, intervals, bursts)
        occ = self._occposdata(speed, intervals)
        return spk, occ

    def _dirdata(self, cell, speed, intervals, bursts):
        """Retrieve filtered directional spike and occupancy data."""
        spk = self._spkdirdata(cell, speed, intervals, bursts)
        occ = self._occdirdata(speed, intervals)
        return spk, occ

    def _speeddata(self, cell, speed, intervals, bursts):
        """Retrieve filtered speed spike and occupancy data."""
        spk = self._spkspeeddata(cell, speed, intervals, bursts)
        occ = self._occspeeddata(speed, intervals)
        return spk, occ

    def _spkposdata(self, *args):
        """Filtered positional spike data."""
        st = self._spkfilt(*args)
        return st, self.motion.F('x', st), self.motion.F('y', st)

    def _spkdirdata(self, *args):
        """Filtered directional spike data."""
        st = self._spkfilt(*args)
        return st, self.motion.F('md', st)

    def _spkspeeddata(self, *args):
        """Filtered speed spike data."""
        st = self._spkfilt(*args)
        return st, self.motion.F('speed_cm', st)

    def _spkfilt(self, cell, speed, intervals, bursts):
        """Generate a filtered spike timing array."""
        st = self.get_cluster(cell).spikes
        filters = []

        if speed is not None:
            filters.append(self.motion.speed_filter(st, speed=speed))

        if intervals is not None:
            filters.append(time.select_from(st, intervals))

        if bursts:
            if type(bursts) is float:
                bf = spikes.burst_onset(st, isi_thresh=bursts)
            else:
                bf = spikes.burst_onset(st)
            filters.append(bf)

        session_timing = np.logical_and(st >= self.tstart, st <= self.tend)

        ix = reduce(op.and_, filters, session_timing)

        return st[ix]

    def _occposdata(self, *args):
        """Filtered positional occupancy data."""
        ix = self._occfilt(*args)
        return self.motion.t[ix], self.motion.x[ix], self.motion.y[ix]

    def _occdirdata(self, *args):
        """Filtered directional occupancy data."""
        ix = self._occfilt(*args)
        return self.motion.t[ix], self.motion.md[ix]

    def _occspeeddata(self, *args):
        """Filtered speed occupancy data."""
        ix = self._occfilt(*args)
        return self.motion.t[ix], self.motion.speed_cm[ix]

    def _occfilt(self, speed, intervals):
        """Create an occupancy index filter."""
        filters = []

        if speed is not None:
            if speed in motion.SPEED_LIMITS:
                filters.append(getattr(self.motion, '%s_index' % speed))
            else:
                filters.append(self.motion.speed_index(speed))

        if intervals is not None:
            filters.append(time.select_from(self.motion.t, intervals))

        session_timing = np.logical_and(self.motion.t >= self.tstart,
            self.motion.t <= self.tend)

        ix = reduce(op.and_, filters, session_timing)

        return ix
