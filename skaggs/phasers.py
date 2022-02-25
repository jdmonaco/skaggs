"""
Functions for selecting and creating datasets of phaser cells.
"""

from functools import reduce

import numpy as np

from pouty import ConsolePrinter
from roto.strings import snake2title

from . import data, labels


# Phaser cell criteria

PHASER_IPHASE_IMIN = 0.1
PHASER_IPHASE_PMAX = 0.02
PHASER_RMIN = 3.5
PHASER_RP_CMIN = 0.2
PHASER_RP_PMAX = 0.02
PHASER_RP_SMIN = np.pi / 4


# Validation

def validate_cells(cell_list,
    rmin=PHASER_RMIN,
    pmax=PHASER_IPHASE_PMAX,
    imin=PHASER_IPHASE_IMIN,
    cmin=PHASER_RP_CMIN,
    cpmax=PHASER_RP_PMAX,
    smin=PHASER_RP_SMIN):
    """List how cells fulfill or fail the phaser criteria."""
    P = properties_dataframe()
    out = ConsolePrinter(prefix=snake2title(__name__),
            prefix_color='green')

    for c_id in cell_list:
        cell = P.loc[c_id]
        tag = labels.session_cell_id(c_id)

        if cell.ratemap_max < rmin:
            out('{}: Failed: ratemap_max [{:.2f}] < {}',
                    tag, cell.ratemap_max, rmin)
        elif cell.I_phase < imin:
            out('{}: Failed: I_phase [{:.4f}] < {}',
                    tag, cell.I_phase, imin)
        elif cell.I_phase_p > pmax:
            out('{}: Failed: I_phase_p [{:.3f}] < {}',
                    tag, cell.I_phase_p, pmax)
        elif np.abs(cell.C_rp_r) < cmin:
            out('{}: Failed: |C_rp_r| [{:.4f}] < {}',
                    tag, cell.C_rp_r, cmin)
        elif cell.C_rp_p > cpmax:
            out('{}: Failed: C_rp_p [{:.3f}] < {}',
                    tag, cell.C_rp_p, cpmax)
        elif np.abs(cell.rp_shift) < smin:
            out('{}: Failed: |rp_shift| [{:.3f}] < {}',
                    tag, cell.rp_shift, smin)
        else:
            out('{}: Phaser!', tag)


# Functions for phaser cell data

def properties_dataframe(
    rmin=PHASER_RMIN,
    pmax=PHASER_IPHASE_PMAX,
    imin=PHASER_IPHASE_IMIN,
    cmin=PHASER_RP_CMIN,
    cpmax=PHASER_RP_PMAX,
    smin=PHASER_RP_SMIN):
    """Cell properties dataframe with subtype/phaser columns based on the
    phaser-cell phase-coding criteria.
    """
    P = data.properties_dataframe()
    P['phaser'] = reduce(np.logical_and, [
            P.ratemap_max >= rmin,
            P.I_phase >= imin,
            P.I_phase_p <= pmax,
            np.abs(P.C_rp_r) >= cmin,
            P.C_rp_p <= cpmax,
            np.abs(P.rp_shift) >= smin
    ])

    P['subtype'] = 'none'
    P.loc[P.phaser, 'subtype'] = [
            {True: 'positive', False: 'negative'}[sl > 0]
                for sl in P.loc[P.phaser, 'C_rp_sl']]

    return P

def filtered_dataframe(**kwds):
    """Cell properties dataframe of only phaser cells."""
    P = properties_dataframe(**kwds)
    P = P.loc[P.phaser]
    return P
