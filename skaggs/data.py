"""
Handle writing tables and arrays to the hierarchical data file
"""

import os
import re

from scipy.interpolate import interp1d
import numpy as np
import tables as tb
import pandas as pd

from pouty import log

from .. import store
from .anatomy import area_for_code
from .parsers import parse_session


arena_types = ['circle', 'square', 'large']
Arenas = tb.Enum(arena_types)

sessions_desc = {
        "id": tb.UInt16Col(pos=1),
        "rat": tb.UInt16Col(pos=2),
        "day": tb.UInt16Col(pos=3),
        "arena": tb.EnumCol(Arenas, 'circle', base='uint8', pos=4),
        "comment": tb.StringCol(itemsize=32, pos=5),
        "path": tb.StringCol(itemsize=256, pos=6),
        "group": tb.StringCol(itemsize=256, pos=7)
    }

recordings_desc = {
        "id": tb.UInt32Col(pos=1),
        "session_id": tb.UInt16Col(pos=2),
        "tetrode": tb.UInt8Col(pos=3),
        "cluster": tb.UInt8Col(pos=4),
        "ttc": tb.StringCol(itemsize=12, pos=5),
        "cell_id": tb.StringCol(itemsize=12, pos=6),
        "site": tb.StringCol(itemsize=16, pos=7),
        "area": tb.EnumCol(area_for_code, 'other', base='uint8', pos=8),
        "missing": tb.BoolCol(pos=9),
        "fast_spikes" : tb.UInt32Col(pos=10),
        "slow_spikes" : tb.UInt32Col(pos=11),
        "still_spikes" : tb.UInt32Col(pos=12)
    }


def dataframe_from_table(where, name=None, h5file=None, **from_recs_kwds):
    """Return a pandas dataframe for the specified table.

    Arguments:
    where -- Table object to be directly converted to a dataframe
    where/name -- group/path for table with optional name
    h5file -- HDF file [default: project store]

    Remaining keyword arguments are passed to `pandas.DataFrame.from_records`.
    """
    if type(where) is tb.Table:
        table = where
    else:
        f = store.get() if h5file is None else h5file
        table = f.get_node(where, name=name, classname='Table')

    df = pd.DataFrame.from_records(table.read(), **from_recs_kwds)

    # String columns should be decoded from bytes arrays
    for colname, coltype in table.coltypes.items():
        if coltype == 'string':
            df[colname] = df[colname].apply(lambda x: x.decode())

    return df

def properties_dataframe(exclude_tt9ref=True, add_logp_cols=False):
    """Return a pandas dataframe for the /tables/properties table.

    Optionally monkey-patches in some `-log(p)` columns for the p-value fields
    in the dataframe, which are better for visualization, and also removes TT9
    reference cell recordings.
    """
    root = store.get().root
    df = pd.DataFrame.from_records(root.tables.properties.read(), index='id')
    if add_logp_cols:
        for col in df.columns:
            if col.endswith('_p'):
                df[col[:-2] + '_logp'] = -np.log10(df[col])
    if exclude_tt9ref:
        recs = root.recordings
        isnotref = lambda c_id: not recs[c_id]['tt9ref']
        return df.iloc[df.index.map(isnotref)]
    return df


class circinterp1d(object):

    """
    Wrapper for interp1d for circular-linear interpolation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize as `scipy.interpolate.interp1d`."""
        t, angle = args
        self._rads = kwargs.pop('radians', True)
        self._zc = kwargs.pop('zero_centered', True)
        _bad = kwargs.pop('bad_value', None)
        if _bad is not None:
            valid = (angle != _bad)
            t, angle = t[valid], angle[valid]
        self._f = interp1d(t, np.unwrap(angle), **kwargs)

    def __call__(self, t):
        """Perform circular-linear interpolation for the given time points."""
        twopi = self._rads and 2 * np.pi or 360.0
        wrapped = self._f(t) % twopi
        if self._zc:
            if np.iterable(wrapped):
                wrapped[wrapped > 0.5 * twopi] -= twopi
            elif wrapped > 0.5 * twopi:
                wrapped -= twopi
        return wrapped


class AbstractDataSource(object):

    """
    Abstract base class for loading data traces from a session subgroup.

    Subclasses must set a class attribute called `_subgroup` with the name
    of the group (child of session group) holding the data array nodes.

    Data arrays can have an additional class attribute `<name>_attrs` set
    to a dict of arguments that are passed to `interp1d`.

    For circular data arrays, set boolean key `circular` and, optionally,
    `radians` (default True) and `zero_centered` (default True) to control
    the circular wrapping used in circular-linear interpolation.

    Methods:
    _read_array -- subclasses should use this to read array nodes
    F -- clients should use this to interpolate time-series values
    """

    _subgroup = ''

    def __init__(self, where_or_index):
        """Set up group paths pointing to the recording data.

        Arguments:
        where_or_index -- recording session path, group, or index
        """
        if type(where_or_index) is str:
            where = where_or_index
        else:
            where = parse_session(where_or_index)['group']

        f = store.get()
        session = f.get_node(where)
        group = f.get_node(where, name=self.__class__._subgroup)

        self.session_path = str(session._v_pathname)
        self.path = str(group._v_pathname)

        self._interpolators = {}

    def _read_array(self, name):
        """Subclass property attributes should call this to read data."""
        f = store.get()
        try:
            arr = f.get_node(self.path, name)
        except tb.NoSuchNodeError:
            log('array \'{}\' does not exist at {}', name, self.path,
                    error=True)
            return np.array([])
        return arr.read()

    def F(self, which, t_i):
        """Interpolate stored data traces for given time points."""
        try:
            return self._interpolators[which](t_i)
        except KeyError:
            pass

        if not hasattr(self, which):
            raise ValueError("no attribute named %s" % which)

        trace = getattr(self, which)
        if trace.shape[-1] != self.t.size:
            raise ValueError("size mismatch for %s along axis" % which)

        attrs = getattr(self, '%s_attrs' % which, {})
        circular = attrs.get('circular', False)
        bad = attrs.get('bad_value', None)

        ikw = dict(copy=False, bounds_error=False, fill_value=0.0)
        ikw.update({k:attrs[k] for k in attrs
            if k not in ('circular', 'bad_sample')})

        intp = circular and circinterp1d or interp1d
        valid = (bad is None) and slice(None) or (trace != bad)

        self._interpolators[which] = f = \
                intp(self.t[valid], trace[valid], **ikw)

        return f(t_i)


def new_table(where, name, description, force=False, h5file=None, **kwargs):
    """Add `force` keyword to tables.File.create_table().

    Arguments:
    force -- force erasure of existing table without asking
    h5file -- alternate file in which to create the array

    Returns `tables.Table` node.
    """
    f = h5file is None and store.get(False) or h5file

    try:
        table = f.get_node(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        if force:
            log('new_table: Erasing {} table', table._v_pathname)
            do_erase = 'y'
        else:
            do_erase = input(
                'new_table: Erase %s table? (y/N) ' % table._v_pathname)
        if not do_erase.lower().strip().startswith('y'):
            raise tb.NodeError('%s table already exists' % table._v_pathname)
        f.remove_node(table)

    kwargs.update(description=description)
    return f.create_table(where, name, **kwargs)

def new_array(where, name, x, overwrite=True, h5file=None, **kwargs):
    """Add `overwrite` keyword to tables.File.create_array().

    Arguments:
    overwrite -- automatically remove array if it already exists
    h5file -- alternate file in which to create the array

    Returns `tables.Array` node.
    """
    f = h5file is None and store.get(False) or h5file

    try:
        array = f.get_node(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        if not overwrite:
            raise tb.NodeError('%s array already exists' % array._v_pathname)
        f.remove_node(array)

    return f.create_array(where, name, obj=x, **kwargs)

def new_group(where, name, h5file=None, **kwargs):
    """Enforce `createparents=True` in creating a new group."""
    f = h5file is None and store.get(False) or h5file
    kwargs.update(createparents=True)

    try:
        group = f.create_group(where, name, **kwargs)
    except tb.NodeError:
        group = f.get_node(where, name=name)
        log('{} already exists', group._v_pathname, error=True)
    return group

def clone_node(node, destfile, parent=None, name=None):
    """Clone a node from one file to another file."""
    name = node._v_name if name is None else name
    parent = node._v_parent._v_pathname if parent is None else parent
    destpath = os.path.join(parent, name)

    if isinstance(node, tb.Array):
        log('Cloning array: {} to {}', node._v_pathname, destpath)
        arr = clone = new_array(parent, name, node.read(), h5file=destfile,
            createparents=True, title=node.title)
        for k in node._v_attrs._v_attrnames:
            setattr(arr._v_attrs, k, node._v_attrs[k])
    elif isinstance(node, tb.Table):
        log('Cloning table: {} to {}', node._v_pathname, destpath)
        tbl = clone = new_table(parent, name, node.description,
            h5file=destfile, createparents=True, title=node.title)
        row = tbl.row
        for record in node.iterrows():
            for col in node.colnames:
                row[col] = record[col]
            row.append()
        for k in node._v_attrs._v_attrnames:
            setattr(tbl._v_attrs, k, node._v_attrs[k])
        tbl.flush()

    return clone

def clone_subtree(tree, destfile, srcfile=None, destroot='/', classname=None):
    """Clone a subtree within one file into a different (possibly new) file."""
    if srcfile is None:
        srcfile = store.get()
    elif type(srcfile) is str:
        srcfile = tb.open_file(srcfile, 'r')

    if not srcfile.isopen:
        raise IOError('Source file is not open')

    if type(destfile) is str:
        destfile = tb.open_file(destfile, 'a')

    if not (destfile.isopen and destfile._iswritable()):
        raise IOError('Destination file is not open and writable')

    try:
        srctree = srcfile.get_node(tree)
    except tb.NoSuchNodeError as e:
        log('Source group does not exist: {}', tree, error=True)
        return

    for node in srcfile.walk_nodes(srctree, classname=classname):
        if isinstance(node, tb.Group):
            continue
        destpath = os.path.join(destroot, node._v_pathname[1:])
        parent, name = os.path.split(destpath)
        clone_node(node, destfile, parent=parent, name=name)

    log('Finished cloning subtree: {}', srctree._v_pathname)
    return destfile.get_node(destroot)

def session_path(info):
    """Construct a session group path from a dict of values."""
    return "/data/rat{rat:02d}/day{day:02d}/{comment}".format(**info)
