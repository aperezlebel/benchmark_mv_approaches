"""Test the statistics computation."""
import numpy as np
import pandas as pd

from statistics.statistics import get_indicators_mv, compute_correlation


mv = pd.DataFrame({
    'F1': [0, 0, 0, 0],
    'F2': [1, 0, 0, 0],
    'F3': [2, 0, 0, 0],
    'F4': [1, 2, 0, 0],
    'F5': [0, 2, 2, 0],
})


def test_indicators():
    """Test all the indicators on a hand-crafted missing values df."""
    indicators = get_indicators_mv(mv)

    # Merge all the indicators except feature-wise which has a different
    # way to store the information
    df = pd.concat([
        indicators['global'],
        indicators['features'],
        indicators['rows'],
        indicators['rm_rows'],
        indicators['rm_features'],
    ], axis=1)

    df3 = indicators['feature-wise']

    # 1st indicator
    assert df.at[0, 'n_rows'] == 4
    assert df.at[0, 'n_cols'] == 5
    assert df.at[0, 'n_values'] == 20
    assert df.at[0, 'n_mv'] == 6
    assert df.at[0, 'n_mv1'] == 2
    assert df.at[0, 'n_mv2'] == 4
    assert df.at[0, 'n_not_mv'] == 14
    assert df.at[0, 'f_mv'] == 30
    assert df.at[0, 'f_mv1'] == 10
    assert df.at[0, 'f_mv2'] == 20
    assert df.at[0, 'f_not_mv'] == 70

    # 2nd indicator
    assert df.at[0, 'n_f_w_mv'] == 4
    assert df.at[0, 'n_f_w_mv1_o'] == 1
    assert df.at[0, 'n_f_w_mv2_o'] == 2
    assert df.at[0, 'n_f_w_mv_1a2'] == 1
    assert df.at[0, 'n_f_wo_mv'] == 1
    assert df.at[0, 'f_f_w_mv'] == 80
    assert df.at[0, 'f_f_w_mv1_o'] == 20
    assert df.at[0, 'f_f_w_mv2_o'] == 40
    assert df.at[0, 'f_f_w_mv_1a2'] == 20
    assert df.at[0, 'f_f_wo_mv'] == 20

    # 3rd indicator
    assert df3.at['F1', 'N MV'] == 0
    assert df3.at['F1', 'N MV1'] == 0
    assert df3.at['F1', 'N MV2'] == 0
    assert df3.at['F1', 'F MV'] == 0
    assert df3.at['F1', 'F MV1'] == 0
    assert df3.at['F1', 'F MV2'] == 0

    assert df3.at['F2', 'N MV'] == 1
    assert df3.at['F2', 'N MV1'] == 1
    assert df3.at['F2', 'N MV2'] == 0
    assert df3.at['F2', 'F MV'] == 25
    assert df3.at['F2', 'F MV1'] == 25
    assert df3.at['F2', 'F MV2'] == 0

    assert df3.at['F3', 'N MV'] == 1
    assert df3.at['F3', 'N MV1'] == 0
    assert df3.at['F3', 'N MV2'] == 1
    assert df3.at['F3', 'F MV'] == 25
    assert df3.at['F3', 'F MV1'] == 0
    assert df3.at['F3', 'F MV2'] == 25

    assert df3.at['F4', 'N MV'] == 2
    assert df3.at['F4', 'N MV1'] == 1
    assert df3.at['F4', 'N MV2'] == 1
    assert df3.at['F4', 'F MV'] == 50
    assert df3.at['F4', 'F MV1'] == 25
    assert df3.at['F4', 'F MV2'] == 25

    assert df3.at['F5', 'N MV'] == 2
    assert df3.at['F5', 'N MV1'] == 0
    assert df3.at['F5', 'N MV2'] == 2
    assert df3.at['F5', 'F MV'] == 50
    assert df3.at['F5', 'F MV1'] == 0
    assert df3.at['F5', 'F MV2'] == 50

    # 4th indicator
    assert df.at[0, 'n_r_w_mv'] == 3
    assert df.at[0, 'n_r_w_mv1_o'] == 0
    assert df.at[0, 'n_r_w_mv2_o'] == 2
    assert df.at[0, 'n_r_w_mv_1a2'] == 1
    assert df.at[0, 'n_r_wo_mv'] == 1
    assert df.at[0, 'f_r_w_mv'] == 75
    assert df.at[0, 'f_r_w_mv1_o'] == 0
    assert df.at[0, 'f_r_w_mv2_o'] == 50
    assert df.at[0, 'f_r_w_mv_1a2'] == 25
    assert df.at[0, 'f_r_wo_mv'] == 25

    # 5th indicator
    assert df.at[0, 'n_r_a_rm_mv1'] == 3
    assert df.at[0, 'n_r_a_rm_mv2'] == 4
    assert df.at[0, 'n_r_a_rm_mv_1o2'] == 4
    assert df.at[0, 'n_r_a_rm_mv1_o'] == 3
    assert df.at[0, 'n_r_a_rm_mv2_o'] == 4
    assert df.at[0, 'n_r_a_rm_mv_1a2'] == 2
    assert df.at[0, 'f_r_a_rm_mv1'] == 75
    assert df.at[0, 'f_r_a_rm_mv2'] == 100
    assert df.at[0, 'f_r_a_rm_mv_1o2'] == 100
    assert df.at[0, 'f_r_a_rm_mv1_o'] == 75
    assert df.at[0, 'f_r_a_rm_mv2_o'] == 100
    assert df.at[0, 'f_r_a_rm_mv_1a2'] == 50

    # 6th indicator
    assert df.at[0, 'n_v_lost_mv1'] == 5
    assert df.at[0, 'n_v_lost_mv2'] == 7
    assert df.at[0, 'n_v_lost_mv_1o2'] == 10
    assert df.at[0, 'n_v_lost_mv1_o'] == 3
    assert df.at[0, 'n_v_lost_mv2_o'] == 5
    assert df.at[0, 'n_v_lost_mv_1a2'] == 2
    assert df.at[0, 'f_v_lost_mv1'] == 25
    assert df.at[0, 'f_v_lost_mv2'] == 35
    assert df.at[0, 'f_v_lost_mv_1o2'] == 50
    assert df.at[0, 'f_v_lost_mv1_o'] == 15
    assert df.at[0, 'f_v_lost_mv2_o'] == 25
    assert df.at[0, 'f_v_lost_mv_1a2'] == 10


def test_correlation():
    np.random.seed(0)
    X = np.random.uniform(-100, 100, size=(10000, 100))
    R1 = compute_correlation(X.T)
    R2 = np.corrcoef(X.T)

    assert np.isclose(R1, R2).all()
