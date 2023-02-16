import numpy as np
from copy import deepcopy
import re

SPLIT_SUFFIXES = ['x', 'y', 'z']

def is_split_position(pt_col, df):
    if pt_col in df.columns:
        return False
    prefix_found = []
    for suf in SPLIT_SUFFIXES:
        prefix_found.append( np.any([re.search(f"^{pt_col}_{suf}", col) is not None for col in df.columns]) )
    if np.all(prefix_found):
        return True
    else:
        raise ValueError(f'Point column "{pt_col}" not found directory or as split position')

def assemble_split_points(pt_col, df):
    cols = [f"{pt_col}_{suf}" for suf in SPLIT_SUFFIXES]
    return np.vstack(df[cols].values)


def get_dataframe_points(pt_col, df):
    if is_split_position(pt_col, df):
        return assemble_split_points(pt_col, df)
    else:
        return np.vstack(df[pt_col].values)