"""
Functions and classes for retrieving and computing with LFP data traces.
"""

from numpy import pi as PI, nan
import tables as tb

from pouty import debug
from roto.decorators import lazyprop

from .. import store
from .data import AbstractDataSource


PHASE_RANGE = -PI, PI
PHASE_MIN, PHASE_MAX = PHASE_RANGE

USE_TT9_PHASE_ESTIMATE = True
PHASE_TT9 = 'thetaphase_tt9'
TS_TT9 = 'TS_tt9'
PHASE = 'thetaphase'
TS = 'TS'


class ContinuousData(AbstractDataSource):

    """
    Load continuous neural recording data traces.
    """

    _subgroup = "LFP"

    @lazyprop
    def t(self):
        if USE_TT9_PHASE_ESTIMATE:
            f = store.get()
            try:
                tt9_data = f.get_node(self.path, TS_TT9)
            except tb.NoSuchNodeError:
                pass
            else:
                debug('LFP: using {} for {}', TS_TT9, self.path)
                return tt9_data.read()
        return self._read_array(TS)

    @lazyprop
    def phase(self):
        if USE_TT9_PHASE_ESTIMATE:
            f = store.get()
            try:
                tt9_data = f.get_node(self.path, PHASE_TT9)
            except tb.NoSuchNodeError:
                pass
            else:
                debug('LFP: using {} for {}', PHASE_TT9, self.path)
                return tt9_data.read()
        return self._read_array(PHASE)

    phase_attrs = { 'circular': True, 'fill_value': nan }
