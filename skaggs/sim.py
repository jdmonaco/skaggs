"""
Support functions for the spatial network simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator

import roto.axis as ra
from roto.dicts import merge_two_dicts
from roto.generators import unique
from roto.images import tiling_dims
from roto.arrays import circblur
from roto.radians import RadianFormatter
from roto.plots import condmap

from . import motion, theta
from ..tools import adaptive, maps, binned

from brian2 import TimedArray, NeuronGroup, Equations
from brian2.units import *


PI = np.pi
TWO_PI = 2*np.pi
CIRCLE_RADIUS = 0.5 * motion.CIRCLE_DIAMETER
CENTER_X = CENTER_Y = CIRCLE_RADIUS


def Emax_winners(W, Emax):
    """Normalized sparse competitive weights.

    Arguments:
    W -- array, raw weight vector to be sparsified and normalized
    Emax -- float [0,1], fraction of winning weights

    Returns (N,1)-shaped column vector of winning weights on [0,1].
    """
    assert Emax >= 0 and Emax <= 1, "Emax must be between 0 and 1"
    W = np.squeeze(W).copy()
    W -= np.sort(W)[max(0, int((1 - Emax) * W.size) - 1)]
    W[W<0] = 0.0
    W /= W.max()
    return W.reshape(-1,1)

def triangle_wave(t0, p):
    """Triangle wave on [0,1] with period p for time points t0."""
    t = t0 - p/4
    a = p/2
    b = np.floor(t/a + 1/2)
    return 0.5 + (1/a) * (t - a*b) * (-1)**b

def Fvco(xgrid, ygrid, phi_pref, beta, phi0, x0, y0):
    """Evaluate spatial phase function of example VCO cell."""
    # Phase gain over distance components for phase delta
    dx = (xgrid - x0)*np.cos(phi_pref)
    dy = (ygrid - y0)*np.sin(phi_pref)
    delta = TWO_PI*beta*(dx + dy)

    # Zero-center the absolute phase
    phase = (phi0 + delta) % TWO_PI
    phase[phase > np.pi] -= TWO_PI

    # Negate the phase to align with external theta reference
    # (That is, a positive phase modulation entails a phase advance against the
    # reference wave; advances are negative by convention.)
    phase *= -1

    return phase

def Fplot(ax, xgrid, ygrid, args, mapkw={}, cbticks=None):
    """Plot the spatial response function to the given axis."""
    P = Fvco(xgrid, ygrid, *args)
    rad2 = np.power(xgrid - CENTER_X, 2) + np.power(ygrid - CENTER_Y, 2)
    P[rad2 > CIRCLE_RADIUS**2] = np.nan

    radticks = FixedLocator([-np.pi,0,np.pi])
    radcbargs = merge_two_dicts(mapkw['cbargs'], dict(ticks=radticks))
    mapkw = merge_two_dicts(mapkw, dict(circular=True, cbargs=radcbargs))

    _, cb = maps.plot(P, ax=ax, **mapkw)

    if cbticks is None:
        cbticks = dict(pad=0, length=0)
    cb.ax.tick_params(**cbticks)
    cb.ax.yaxis.tick_right()
    ax.set_axis_off()
    return ax

def add_wrapped_phase_column(df, phase_col='phase', new_col='phase0'):
    """Add new column with zero-center wrapped phases.

    Arguments:
    df -- DataFrame of spike/state data containing phase values
    phase_col -- name of the column containing unwrapped phase values
    new_col -- name of new column to contain the zero-center phases
    """
    phase = df.loc[:,phase_col] % TWO_PI
    phase[phase > PI] -= TWO_PI
    df[new_col] = phase

def plot_phaser_maps_figure(ctx, traj, spikes, figname='phaser-layer',
    figtitle='Phaser Layer Ratemaps and Phasemaps', maxN=36, levels=12,
    mapres=112, figsize=(9,9)):
    """Create small-multiples figure of phaser simulation rate/phasemaps.

    Arguments:
    ctx -- SPCSimulation context object
    traj -- trajectory recording DataFrame as loaded with `read_simulation`
    spikes -- a spikes recording DataFrame as load with `read_simulation`
    figname -- figure name (default, 'phaser-layer')
    figtitle -- optional, figure title to display at top of figure
    maxN -- maximum number of cells to plot regardless of input data
    levels -- discrete colormap levels for maps
    mapres -- pixel resolution of maps to compute and display
    figsize -- figure size (w, h) tuple in inches
    """
    add_wrapped_phase_column(spikes)

    was_interactive = plt.isinteractive()
    plt.ioff()
    f = ctx.figure(figname, clear=True, figsize=figsize, title=figtitle)

    # Compute adpative rate/phase maps
    mdata = motion.MotionData(ctx.c.traj_session_id)
    amapkw = dict(res=mapres, alim=(8.0, 24.0), scale='cm')
    ratemap = adaptive.AdaptiveRatemap(mdata, **amapkw)
    phasemap = adaptive.AdaptivePhasemap(mdata, **amapkw)

    # Map parameters
    mapkw = dict(cmin=0.0, cbar=False, mask_color='0.9', levels=levels)
    pmapkw = dict(cmin=-PI, cmax=PI, cbar=False, levels=levels,
            mask_color='w', satnorm=True)

    j = 1
    neurons = sorted(unique(spikes.neuron))
    N = min(len(neurons), maxN)
    r, c = tiling_dims(2*N)
    if c % 2 != 0:
        c += 1
        r = np.ceil(2*N/c)

    ctx.out('Computing and plotting rate/phasemaps...')
    for i in neurons[:N]:
        spiketrain = spikes.loc[spikes.neuron == i]
        ctx.box(color='green')
        R = ratemap(spiketrain.x, spiketrain.y, traj.x, traj.y)
        ctx.box(color='purple')
        P = phasemap(spiketrain.x, spiketrain.y, spiketrain.phase)

        axr = f.add_subplot(r, c, j)
        axp = f.add_subplot(r, c, j+1)

        maps.plot(R, ax=axr, **mapkw)
        maps.plot(P, ax=axp, **pmapkw)

        axr.set_axis_off()
        axp.set_axis_off()

        j += 2
    ctx.newline()

    if was_interactive:
        plt.ion()
        plt.show()

def plot_vco_figure(ctx, traj, spikes, figname='vco-maps',
    figtitle='VCO Simulation Maps', levels=18, mapres=128, Fbins=256,
    figsize=(8,10)):
    """Plot figure showing results from example phaser simulation."""
    if 'phase0' not in spikes:
        add_wrapped_phase_column(spikes)
    if 'theta0' not in spikes:
        add_wrapped_phase_column(spikes, 'theta', 'theta0')

    was_interactive = plt.isinteractive()
    plt.ioff()
    f = ctx.figure(figname, clear=True, figsize=figsize, title=figtitle)

    # Compute adpative rate/phase maps
    mdata = motion.MotionData(ctx.c.traj_session_id)
    amapkw = dict(res=mapres, alim=(8.0, 24.0), scale='cm')
    ratemap = adaptive.AdaptiveRatemap(mdata, **amapkw)
    phasemap = adaptive.AdaptivePhasemap(mdata, **amapkw)

    # Axis and plot formatting
    gshape = (3,2)
    trajkw = dict(ls='-', c='k', alpha=0.5, lw=1, zorder=-1)
    spkkw = dict(marker='o', s=12, c='#FF5555', alpha=0.3, linewidths=0)
    cbtickkw = dict(axis='y', labelsize='small', pad=-3, length=3)
    cbargs = dict(pad=0.01, shrink=0.7, aspect=12)
    mapkw = dict(cmin=0.0, cbar=True, mask_color='0.8', levels=levels)
    histkw = dict(color='r', lw=0, alpha=0.9, zorder=0)
    pdistkw = dict(c='k', ls='-', lw=1.5, alpha=0.7, zorder=1)

    # Radian colorbar parameters
    radfmt = RadianFormatter(inline=False)
    radticks = FixedLocator([-PI,0,PI])
    radcbargs = merge_two_dicts(cbargs, dict(ticks=radticks, format=radfmt))
    phasekw = dict(cmin=-PI, cmax=PI, cbar=True, levels=levels,
            mask_color='0.9', cbargs=radcbargs)
    maxfivebins = MaxNLocator(prune='both', nbins=5, integer=True)
    Fmapkw = merge_two_dicts(phasekw, dict(levels=None))

    # Generate spatial model function and pixel evaluation grid
    pts = np.linspace(0, motion.CIRCLE_DIAMETER, Fbins)
    xgrid, ygrid = np.meshgrid(pts, pts, indexing='ij')

    def plot_spatial_function(ax):
        args = (ctx.c.phi_pref, ctx.c.beta, ctx.c.phi0, ctx.c.x0, ctx.c.y0)
        Fplot(ax, xgrid, ygrid, args, mapkw=Fmapkw, cbticks=cbtickkw)
    plot_spatial_function(plt.subplot2grid(gshape, (0,0)))

    def plot_spike_trajectory(ax):
        plt.plot(traj.x, traj.y, **trajkw)
        plt.scatter(spikes.x, spikes.y, **spkkw)
        ax.axis('scaled')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 80)
        ax.set_axis_off()
    plot_spike_trajectory(plt.subplot2grid(gshape, (0,1)))

    R = ratemap(spikes.x, spikes.y, traj.x, traj.y)
    Pvco = phasemap(spikes.x, spikes.y, spikes.theta0)
    Plfp = phasemap(spikes.x, spikes.y, spikes.phase0)
    def plot_phasemap(ax, P, title):
        _, cb = maps.plot(P, satnorm=True, **phasekw)
        cb.ax.tick_params(**cbtickkw)
        ax.set_axis_off()
        ra.quicktitle(ax, title)
    plot_phasemap(plt.subplot2grid(gshape, (1,1)), Pvco, 'VCO Phase')
    plot_phasemap(plt.subplot2grid(gshape, (1,0)), Plfp, 'LFP Phase')

    def plot_phase_dist(ax, phase, label, bins=binned.DEFAULT_PHASE_BINS):
        edges = np.linspace(-PI, PI, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        H, _, _ = ax.hist(phase, bins=edges, **histkw)
        pdistsm = circblur(H, radians=TWO_PI/36)
        ax.plot(centers, pdistsm, **pdistkw)
        ax.axvline(x=0, ls=':', c='k', alpha=0.8, zorder=2)
        ax.xaxis.set_major_formatter(radfmt)
        ax.set_xticks([-PI, -PI/2, 0, PI/2, PI])
        ax.set_yticks([])
        ax.set(xlim=(-PI, PI), xlabel='%s Phase (rad)' % label,
            ylim=(0, 1.2 * np.max(pdistsm)))
        ra.despine(ax, left=False)
    plot_phase_dist(plt.subplot2grid(gshape, (2,0)), spikes.phase0, 'Theta')
    plot_phase_dist(plt.subplot2grid(gshape, (2,1)), spikes.theta0, 'VCO')

    if was_interactive:
        plt.ion()
        plt.show()

class TrajectoryModel(object):

    """
    Handle a simulated trajectory based on real tracking data
    """

    def __init__(self, session_id):
        self.session_id = session_id
        self._setup_trajectory()

    def _setup_trajectory(self):
        """Initialize timed arrays and other data for simulated trajectory."""
        mdata = motion.MotionData(self.session_id)
        T = self.T = mdata.t
        X = self.X = mdata.x_cm
        Y = self.Y = mdata.y_cm
        R = self.R = mdata.radius_cm
        S = self.S = mdata.speed_cm
        D = self.D = mdata.md  # radians

        self.dt = dt = np.median(T[1:] - T[:-1]) * second
        self.namespace = {
            'X': TimedArray(X, dt, name='TrajX'),
            'Y': TimedArray(Y, dt, name='TrajY'),
            'R': TimedArray(R, dt, name='TrajRadius'),
            'S': TimedArray(S, dt, name='TrajSpeed'),
            'D': TimedArray(D, dt, name='TrajDirection')
        }

        self.eqns = Equations("""
            x = X(t) : 1 (constant over dt)
            y = Y(t) : 1 (constant over dt)
            r = R(t) : 1 (constant over dt)
            s = S(t) : 1 (constant over dt)
            d = D(t) : 1 (constant over dt)
        """)

    def new_group(self):
        """Create a new simulation group object for this trajectory."""
        grp = NeuronGroup(1, dt=self.dt, model=self.eqns, order=-1,
                name='Trajectory', namespace=self.namespace)
        return grp
