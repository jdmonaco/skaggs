"""
Functions for handling cells and cell data.
"""

from .. import store
from . import session, labels, data


def record(cell_id):
    """Get the /recordings record for the given cell id."""
    return store.get().root.recordings[cell_id]

def session_for(cell_id, **mapkwds):
    """Get the session object for the cell with the given id."""
    cell = record(cell_id)
    return session.get(cell['session_id'], **mapkwds)

def where_session(query):
    """Generator to produce /recordings rows for /sessions queries."""
    df = store.get()
    recs = df.root.recordings
    for session in df.root.sessions.where(query):
        for cell in recs.where('session_id==%d' % session['id']):
            yield cell

def spikes(cell_id, speed='fast'):
    """Speed-filtered spike train for the cell with the given id."""
    S = session_for(cell_id)
    cell = record(cell_id)
    return S.spike_filter(cell, speed=speed)

def tag_and_spikes(cell_id, speed='fast'):
    """Convenience function to get a full tag and spike train."""
    tag = labels.session_cell_id(cell_id)
    st = spikes(cell_id, speed=speed)
    return tag, st

#
# The valid_phase generator is not necessary as of 07-16-2015 since there
# is complete overlap between rat 11 circle sessions and sessions that
# have been reclocked against the simultaneous TT9 theta cell reference.
# However, care must be taken to not analyze TT9 itself.
#

def valid_phase(session_query=None):
    """Generator for /recordings rows of cells with valid theta phase."""
    if session_query is None:
        session_query = 'arena==%d' % data.Arenas.circle
    df = store.get()
    for sess in df.root.sessions.where(session_query):
        rat = sess['rat']
        s_id = sess['id']
        for rec in df.root.recordings.where('session_id==%d' % s_id):
            if rat == 11:
                if rec['tt9clocked'] and not rec['tt9ref']:
                    yield rec
            else:
                yield rec
