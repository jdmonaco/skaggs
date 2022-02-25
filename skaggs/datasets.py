"""
Traverse the raw on-disk dataset
"""

import os
import re

from pouty import log

from .. import RAW_DATA_ROOT


SPIKES_FILE = 'allspikes.mat'
RAWSPIKES_FILE = 'rawspikes.mat'
DATASET_FILES = [
        'posxdata.mat',
        'trackdata.mat',
        'LFPphase.mat'
    ]

RAT_PATTERN = r"(?i)o?theta(\d+)"
DATASET_PATTERN = r"(?i)o?theta(\d+)_d{1,2}ay(\d+)_?(\w+)?"


def find_rat_paths(root=RAW_DATA_ROOT):
    """Find folders corresponding to individual rats.

    Arguments:
    root -- parent folder containing all of the per-rat datasets

    Returns:
    {ratnum: path} dictionary of rat folders
    """
    pat = re.compile(RAT_PATTERN)
    paths = {}
    for dirname in os.listdir(root):
        match = re.match(pat, os.path.split(dirname)[1])
        if match:
            rat = int(match.groups()[0])
            paths[rat] = os.path.join(root, dirname)
    return paths

def find_dataset_paths(root=RAW_DATA_ROOT, use_rawspikes=False):
    """Find subfolders that contain a complete recording session dataset.

    Arguments:
    root -- parent folder containing all of the per-rat datasets
    use_rawspikes -- check for `rawspikes.mat` instead of `allspikes.mat`

    Returns:
    list of dataset attribute dictionaries
    """
    spikes_fn = use_rawspikes and RAWSPIKES_FILE or SPIKES_FILE
    pat = re.compile(DATASET_PATTERN)
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        match = re.match(pat, os.path.split(dirpath)[1])
        if match is None:
            continue

        if spikes_fn not in filenames:
            log('missing spikes: {}', dirpath, error=True)
            continue

        missing = False
        for dataset_fn in DATASET_FILES:
            if dataset_fn not in filenames:
                log('missing {} in {}', dataset_fn, dirpath, error=True)
                missing = True
                break
        if missing:
            continue

        rat, day, cmt = match.groups()
        rat = int(rat)
        day = int(day)
        cmt = cmt or "std"
        paths.append({'rat': rat, 'day': day, 'comment': cmt, 'path': dirpath})

    log('Found {} dataset paths.', len(paths))
    return paths
