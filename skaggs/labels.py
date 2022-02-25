"""
Functions for creating labels for data.
"""

from .. import store


def session(session_id):
    """A label describing a session."""
    df = store.get()
    session = df.root.sessions[session_id]
    label = 'Rat {}, Day {} ({})'.format(
        session['rat'], session['day'], session['comment'].decode('utf-8'))
    return label

def session_id(session_id):
    """A label describing a session and its id."""
    return '{} [#{}]'.format(session(session_id), session_id)

def cell(c_id):
    """A label describing a cell."""
    return store.get().root.recordings[c_id]['ttc'].decode('utf-8')

def session_cell(c_id):
    """A label describing a session and a cell."""
    df = store.get()
    cell = df.root.recordings[c_id]
    label = '{}, Cell {}'.format(session(cell['session_id']),
            cell['ttc'].decode('utf-8'))
    return label

def rat_cell_id(c_id):
    """A label describing a rat and cell name/id."""
    df = store.get()
    s_id = df.root.recordings[c_id]['session_id']
    rat = df.root.sessions[s_id]['rat']
    return 'Rat {}, {}'.format(rat, cell_id(c_id))

def session_cell_id(c_id):
    """A label describing a session and a cell with its id."""
    return '{} [#{}]'.format(session_cell(c_id), c_id)

def cell_id(c_id):
    """A label describing a cell with its id."""
    return '{} [#{}]'.format(cell(c_id), c_id)
