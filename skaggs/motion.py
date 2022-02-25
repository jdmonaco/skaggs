"""
Functions and classes for handling trajectory and motion tracking data.
"""

import tables as tb
import numpy as np

from roto.arrays import boxcar
from roto.decorators import lazyprop

from .. import store
from . import time
from .data import AbstractDataSource, Arenas


NORM_POS_MAX = 100
NORM_POS_EXTENT = (0, NORM_POS_MAX, 0, NORM_POS_MAX)
NORM_POS_RANGE = [(0, NORM_POS_MAX), (0, NORM_POS_MAX)]

LARGE_BOX_SIZE = 115.0
CIRCLE_AND_BOX_SIZE = 80.0
CIRCLE_DIAMETER = CIRCLE_AND_BOX_SIZE
BOX_SIZE = CIRCLE_AND_BOX_SIZE

SLOW_STILL_LIMIT = 2.5  # cm/s
FAST_SLOW_LIMIT = 5.0  # cm/s
SPEED_LIMITS = {
    'still': (None, SLOW_STILL_LIMIT),
    'slow': (SLOW_STILL_LIMIT, FAST_SLOW_LIMIT),
    'fast': (FAST_SLOW_LIMIT, None)
}

MDIR_RANGE = (0, 2*np.pi)
BOXCAR_LEN = 4


float_or_none = lambda x: None if x is None else float(x)

def arena_size(which):
    """Arena size in cm for an arena type (enum value or name)."""
    if which in (Arenas.large, 'large'):
        return LARGE_BOX_SIZE
    return CIRCLE_AND_BOX_SIZE

def arena_extent(which):
    """Arena extent (tuple) in cm for an arena type."""
    sz = arena_size(which)
    return 0, sz, 0, sz

def speed_limits(speed):
    """Parse speed specification into (low, high) optional limits."""
    if speed is None:
        return None, None

    try:
        smin, smax = SPEED_LIMITS[speed]
    except (KeyError, TypeError):
        pass
    else:
        return smin, smax

    try:
        smin, smax = list(map(float_or_none, speed))
    except (TypeError, ValueError):
        pass
    else:
        return smin, smax

    raise ValueError('bad speed value: {}'.format(speed))


class MotionData(AbstractDataSource):

    """
    Data source for motion tracking data for a recording session.
    """

    _subgroup = 'tracking'

    def __init__(self, where):
        super().__init__(where)
        try:
            stable = store.get().root.sessions
        except tb.NoSuchNodeError:
            arena = Arenas.circle
        else:
            try:
                s_id = int(where)
            except ValueError:
                where = str(where)
                s_id = stable.get_where_list('group==b"{}"'.format(where))[0]
            arena = stable[s_id]['arena']
        self.arena = Arenas(arena)

    def to_cm(self, x):
        return self.scale_cm * x

    @lazyprop
    def scale_cm(self):
        return self.width_cm / NORM_POS_MAX

    @lazyprop
    def width_cm(self):
        """Arena width in cm."""
        return arena_size(self.arena)

    @lazyprop
    def extent_cm(self):
        """Arena data extent in cm."""
        return arena_extent(self.arena)

    @lazyprop
    def t(self):
        return self._read_array('TS')

    @lazyprop
    def Fs(self):
        return 1.0 / np.median(np.diff(self.t))

    @lazyprop
    def x(self):
        return self._read_array('X')

    @lazyprop
    def x_cm(self):
        """X-coordinate in cm."""
        return self.to_cm(self.x)

    @lazyprop
    def y(self):
        return self._read_array('Y')

    @lazyprop
    def y_cm(self):
        """Y-coordinate in cm."""
        return self.to_cm(self.y)

    @lazyprop
    def hd_orig(self):
        return self._read_array('HD')

    hd_orig_attrs = { 'circular': True, 'radians': False,
            'zero_centered': False }

    @lazyprop
    def hd(self):
        return (np.pi / 180.0) * ((90.0 - self.hd_orig) % 360.0)

    hd_attrs = { 'circular': True, 'radians': True, 'zero_centered': False }

    @lazyprop
    def av(self):
        """Calculated head-direction angular velocity in degrees/s."""
        dhd = boxcar(np.diff(np.unwrap(self.hd)), M=BOXCAR_LEN)
        av = (180.0 / np.pi) * dhd * self.Fs
        return av

    @lazyprop
    def md(self):
        """Calculated movement direction from normalized position change."""
        dx = boxcar(np.diff(self.x), M=BOXCAR_LEN)
        dy = boxcar(np.diff(self.y), M=BOXCAR_LEN)
        md = np.zeros_like(self.x)
        md[1:] = np.arctan2(dy, dx)
        md[0] = md[1]
        md[md<0] += 2*np.pi
        return md

    md_attrs = { 'circular': True, 'radians': True, 'zero_centered': False }

    @lazyprop
    def speed(self):
        """Calculated speed as normalized position change per second."""
        s = np.sqrt(self.x_speed**2 + self.y_speed**2)
        return s

    @lazyprop
    def x_speed(self):
        """Calculated speed as normalized position change per second."""
        dx = boxcar(np.diff(self.x), M=BOXCAR_LEN)
        xs = np.zeros_like(self.x)
        xs[1:] = dx * self.Fs
        xs[0] = xs[1]
        return xs

    @lazyprop
    def y_speed(self):
        """Calculated speed as normalized position change per second."""
        dy = boxcar(np.diff(self.y), M=BOXCAR_LEN)
        ys = np.zeros_like(self.y)
        ys[1:] = dy * self.Fs
        ys[0] = ys[1]
        return ys

    @lazyprop
    def speed_cm(self):
        return self.to_cm(self.speed)

    @lazyprop
    def x_speed_cm(self):
        return self.to_cm(self.x_speed)

    @lazyprop
    def y_speed_cm(self):
        return self.to_cm(self.y_speed)

    @lazyprop
    def radius(self):
        x0 = np.mean(NORM_POS_RANGE[0])
        y0 = np.mean(NORM_POS_RANGE[1])
        rad = np.hypot(self.x - x0, self.y - y0)
        return rad

    @lazyprop
    def radius_cm(self):
        return self.to_cm(self.radius)

    @lazyprop
    def wall_distance(self):
        if self.arena == 'circle':
            wall = NORM_POS_MAX / 2
            walld = wall - self.radius
        else:
            D = np.c_[self.x, self.y,
                      NORM_POS_RANGE[0][1] - self.x,
                      NORM_POS_RANGE[1][1] - self.y]
            walld = np.min(D, axis=1)
        return walld

    @lazyprop
    def wall_distance_cm(self):
        return self.to_cm(self.wall_distance)

    # Speed-filtered intervals and indexes

    @lazyprop
    def fast_intervals(self):
        return self.speed_intervals('fast')

    @lazyprop
    def fast_index(self):
        return self.speed_index('fast')

    @lazyprop
    def slow_intervals(self):
        return self.speed_intervals('slow')

    @lazyprop
    def slow_index(self):
        return self.speed_index('slow')

    @lazyprop
    def still_intervals(self):
        return self.speed_intervals('still')

    @lazyprop
    def still_index(self):
        return self.speed_index('still')

    # Methods for speed filtering

    def speed_filter(self, t, speed='fast'):
        """Boolean index array for speed-filtering an array of timestamps.

        Arguments:
        t -- array of timestamps
        speed -- 'still'|'slow'|'fast' or (low|None, high|None) speed bin
        """
        speed_t = self.F('speed_cm', t)
        return self._speed_select(t, speed_t, speed)

    def speed_intervals(self, speed):
        """Time intervals corresponding to a speed filter.

        Arguments:
        speed -- 'still'|'slow'|'fast' or (low|None, high|None) speed bin

        Returns:
        (N,2)-shaped array of N time intervals
        """
        return time.intervals_where(self.t, [self.speed_index(speed)],
                    freq=self.Fs)

    def speed_index(self, speed):
        """Boolean index array for speed-filtering motion data.

        Arguments:
        speed -- 'still'|'slow'|'fast' or (low|None, high|None) speed bin

        Returns:
        motion-shaped boolean index array
        """
        return self._speed_select(self.t, self.speed_cm, speed, inclusive=True)

    def _speed_select(self, t, speed_t, speed, inclusive=False):
        """Boolean index array for speed-filtering a speed time-series.

        This is a private method for `speed_filter` and `speed_index`.

        Arguments:
        t -- array of timestamps
        speed_t -- array of speed values
        speed -- 'still'|'slow'|'fast' or (low|None, high|None) speed bin

        Keyword arguments:
        inclusive -- for time-series data, include the bounding samples
        """
        low, hi = speed_limits(speed)
        ix = np.ones_like(t, '?') if low is None else speed_t >= low
        ix = ix if hi is None else np.logical_and(ix, speed_t < hi)
        if inclusive:
            delta = np.diff(ix.astype('i'))
            ix[(delta == 1).nonzero()[0]] = True
            ix[1 + (delta == -1).nonzero()[0]] = True
        return ix

    # Original data traces that have been deprecated

    @lazyprop
    def md_orig(self):
        return self._read_array('MD')

    md_orig_attrs = { 'circular': True, 'radians': False,
                      'zero_centered': False, 'bad_value': -1 }

    @lazyprop
    def speed_orig(self):
        return self._read_array('speed')

    @lazyprop
    def av_orig(self):
        return self._read_array('AV')
