"""
Functions and classes for loading cell and cluster data.
"""

import numpy as np

from roto.decorators import lazyprop

from .data import AbstractDataSource
from .parsers import parse_ttc


class ClusterData(AbstractDataSource):

    """
    Data source for spike cluster data.
    """

    _subgroup = 'spikes'

    def __init__(self, where, ttc):
        AbstractDataSource.__init__(self, where)
        self.ttc = parse_ttc(ttc)
        self.name = 'tt%d_c%d' % (self.ttc)

    @lazyprop
    def spikes(self):
        return self._read_array(self.name)

    # Pre-filtered spikes are deprecated

    @lazyprop
    def all_spikes(self):
        return np.unique(np.r_[self.fast_spikes, self.slow_spikes,
                               self.still_spikes])

    @lazyprop
    def fast_spikes(self):
        return self._read_array('%s_fast' % self.name)

    @lazyprop
    def slow_spikes(self):
        return self._read_array('%s_slow' % self.name)

    @lazyprop
    def still_spikes(self):
        return self._read_array('%s_still' % self.name)
