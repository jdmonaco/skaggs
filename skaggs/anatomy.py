"""
Anatomical codes for recordings sites.
"""


area_for_code = {
    'septum': 1, 'hippocampus': 2, 'thalamus': 3, 'midbrain': 4, 'other': 5
}

code_for_area = {
    1: 'septum', 2: 'hippocampus', 3: 'thalamus', 4: 'midbrain', 5: 'other'
}


def area_code(name):
    """Get numeric code for general recording area based on name."""
    return area_for_code[name]

def area_name(code):
    """Get name of general recording area based on numeric code."""
    return code_for_area[int(code)]
