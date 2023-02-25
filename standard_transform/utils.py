import numpy as np
from copy import deepcopy
import re

SPLIT_SUFFIXES = ['x', 'y', 'z']

def _t1_split_column(pt_col):
    return [f"{pt_col}_{suf}" for suf in SPLIT_SUFFIXES]

def _t2_split_column(pt_col):
    t2_comps = pt_col.split('_')
    return [f"{'_'.join(t2_comps[:-1])}_{suf}_{t2_comps[-1]}" for suf in SPLIT_SUFFIXES]

def split_position_columns(pt_col, df):
    prefix_found_t1 = []            # type 1: pt_position_x/y/z
    t1_col_guess = _t1_split_column(pt_col)
    for colg in t1_col_guess:
        prefix_found_t1.append( colg in df.columns )

    t2_col_guess = _t2_split_column(pt_col)
    prefix_found_t2 = []            # type 2: pt_position_x_suffix
    for colg in t2_col_guess:
        prefix_found_t2.append(colg in df.columns)

    if np.all(prefix_found_t1) and not np.all(prefix_found_t2):
        return t1_col_guess
    elif np.all(prefix_found_t2) and not np.all(prefix_found_t1):
        return t2_col_guess
    elif np.all(prefix_found_t2) and np.all(prefix_found_t1):
        raise ValueError(f"Both final and intermediate pre-final suffixes are possible within the dataframe")
    else:
        raise ValueError(f'Point column "{pt_col}" not found directory or as split position')

def is_split_position(pt_col, df):
    if pt_col in df.columns:
        return False
    return split_position_columns(pt_col, df) is not None

def assemble_split_points(pt_col, df, suffix_type=None):
    if suffix_type is None:
        cols = split_position_columns(pt_col, df)
    elif suffix_type == 1:
        cols = _t1_split_column(pt_col)
    elif suffix_type == 2:
        cols = _t2_split_column(pt_col)
    else:
        raise ValueError("If specified, suffix type must be 1 ('pt_position_suf_x') or 2 ('pt_position_x_suf')")
    return np.vstack(df[cols].values)


def get_dataframe_points(pt_col, df):
    if is_split_position(pt_col, df):
        return assemble_split_points(pt_col, df)
    else:
        return np.vstack(df[pt_col].values)