"""
Functions for parsing or transforming data keys.
"""

import re

import tables as tb

from .. import store


TTC_PATTERN = re.compile('tt(\d+)_c(\d+)')
STR_PATTERN = re.compile('^b\'(.*)\'')


class ParserError(Exception):
    pass


def raise_or_none(raze, msg):
    if raze:
        raise ParserError(msg)
    return None

def _process_table_row(row, cols):
    """Convert a table row with bytes-strings to a dict with normal strings."""
    values = {}
    for name, coltype in cols.items():
        if coltype == 'string':
            try:
                val = re.match(STR_PATTERN, str(row[name])).groups()[0]
            except AttributeError:
                val = str(row[name])
                val = val[2:-1]  # remove "^b'" and "'$"
        else:
            val = row[name]
        values[name] = val
    return values

def parse_session(index, raise_on_fail=False):
    """Get dict of session info for /sessions index or row object."""
    return _parse_table_row(store.get().root.sessions, index, raise_on_fail)

def parse_recording(index, raise_on_fail=False):
    """Get dict of cell info for /recordigns index or row object."""
    return _parse_table_row(store.get().root.recordings, index, raise_on_fail)

def _parse_table_row(table, index, rof):
    if type(index) is tb.tableextension.Row:
        row = index
        table = row.table
    else:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return raise_or_none(rof, "bad session index: '%s'" % str(index))
        else:
            row = table[index]

    return _process_table_row(row, table.coltypes)

def parse_ttc(ttc, raise_on_fail=False):
    """Get (tetrode, cluster) integer tuple for any specification."""
    if type(ttc) is str:
        return parse_ttc_str(ttc, raise_on_fail=raise_on_fail)

    try:
        tt, cl = int(ttc['tetrode']), int(ttc['cluster'])
    except (KeyError, TypeError):
        pass
    else:
        return (tt, cl)

    try:
        tt, cl = int(ttc['tt']), int(ttc['cl'])
    except (KeyError, TypeError):
        pass
    else:
        return (tt, cl)

    try:
        tt, cl = int(ttc.tetrode), int(ttc.cluster)
    except (AttributeError, TypeError):
        pass
    else:
        return (tt, cl)

    try:
        tt, cl = int(ttc.tt), int(ttc.cl)
    except (AttributeError, TypeError):
        pass
    else:
        return (tt, cl)

    try:
        tt, cl = list(map(int, ttc))
    except (ValueError, TypeError):
        pass
    else:
        return (tt, cl)

    return raise_or_none(raise_on_fail, "invalid ttc: %s" % str(ttc))

def parse_ttc_str(ttc_str, raise_on_fail=False):
    """Convert ttc string (e.g. 'tt11_c3') -> (11, 3) tuple."""
    match = re.match(TTC_PATTERN, ttc_str)
    if match:
        return tuple(map(int, match.groups()))
    return raise_or_none(raise_on_fail,
            "could not parse ttc string '{}'".format(ttc_str))
