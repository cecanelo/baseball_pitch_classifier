import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from preprocess import exclude_pitch_types, exclude_below_threshold, select_features


# ── exclude_pitch_types ───────────────────────────────────────────────────────

def test_exclude_pitch_types_removes_excluded_codes():
    df = pd.DataFrame({'pitch_type': ['FF', 'PO', 'SL', 'UN', 'FF']})
    result = exclude_pitch_types(['PO', 'UN'], df)
    assert 'PO' not in result['pitch_type'].values
    assert 'UN' not in result['pitch_type'].values

def test_exclude_pitch_types_removes_nulls():
    df = pd.DataFrame({'pitch_type': ['FF', None, 'SL', np.nan, 'CU']})
    result = exclude_pitch_types([], df)
    assert result['pitch_type'].isna().sum() == 0

def test_exclude_pitch_types_keeps_valid_rows():
    df = pd.DataFrame({'pitch_type': ['FF', 'PO', 'SL']})
    result = exclude_pitch_types(['PO'], df)
    assert set(result['pitch_type'].values) == {'FF', 'SL'}
    assert len(result) == 2


# ── exclude_below_threshold ───────────────────────────────────────────────────

def test_exclude_below_threshold_removes_rare_types():
    df = pd.DataFrame({'pitch_type': ['FF'] * 100 + ['FS'] * 10})
    result = exclude_below_threshold(df, threshold=50)
    assert 'FS' not in result['pitch_type'].values

def test_exclude_below_threshold_keeps_common_types():
    df = pd.DataFrame({'pitch_type': ['FF'] * 100 + ['SL'] * 60})
    result = exclude_below_threshold(df, threshold=50)
    assert 'FF' in result['pitch_type'].values
    assert 'SL' in result['pitch_type'].values

def test_exclude_below_threshold_boundary_keeps_exactly_threshold():
    # A pitch type with exactly threshold samples must be kept
    df = pd.DataFrame({'pitch_type': ['FF'] * 100 + ['FS'] * 50})
    result = exclude_below_threshold(df, threshold=50)
    assert 'FS' in result['pitch_type'].values


# ── select_features ───────────────────────────────────────────────────────────

def test_select_features_keeps_only_selected_columns():
    df = pd.DataFrame({
        'release_speed': [95.0],
        'pfx_z': [1.4],
        'irrelevant_col': [999],
        'pitch_type': ['FF']
    })
    result = select_features(['release_speed', 'pfx_z'], 'pitch_type', df)
    assert set(result.columns) == {'release_speed', 'pfx_z', 'pitch_type'}

def test_select_features_drops_other_columns():
    df = pd.DataFrame({
        'release_speed': [95.0],
        'pfx_z': [1.4],
        'irrelevant_col': [999],
        'pitch_type': ['FF']
    })
    result = select_features(['release_speed', 'pfx_z'], 'pitch_type', df)
    assert 'irrelevant_col' not in result.columns
