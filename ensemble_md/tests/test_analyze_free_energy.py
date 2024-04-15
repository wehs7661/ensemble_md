####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
Unit tests for the module analyze_free_energy.py.
"""
import numpy as np
from unittest.mock import patch, MagicMock
from ensemble_md.analysis import analyze_free_energy


@patch('ensemble_md.analysis.analyze_free_energy.alchemlyb')
def test_preprocess_data(mock_alchemlyb):
    pass
    """
    # Mock DataFrame setup
    df_mock = MagicMock(spec=pd.DataFrame)
    df_mock.index.values.return_value = [(1,)]
    df_mock.columns.values = ['A', 'B', 'C']

    mock_alchemlyb.concat.return_value = df_mock
    mock_alchemlyb.parsing.gmx.extract_u_nk.return_value = df_mock
    mock_alchemlyb.parsing.gmx.extract_dHdl.return_value = df_mock
    mock_alchemlyb.preprocessing.subsampling.u_nk2series.return_value = MagicMock(spec=pd.DataFrame)
    mock_alchemlyb.preprocessing.subsampling.dHdl2series.return_value = MagicMock(spec=pd.DataFrame)

    files = [[f'ensemble_md/tests/data/dhdl/simulation_example/sim_{i}/iteration_{j}/dhdl.xvg' for j in range(3)] for i in range(4)]  # noqa: E501
    results = analyze_free_energy.preprocess_data(files, 300, 'u_nk')

    mock_alchemlyb.concat.called()
    mock_alchemlyb.extract_u_nk.called()
    mock_alchemlyb.subsampling.u_nk2series.called()

    assert mock_alchemlyb.concat.call_count == 4
    assert mock_alchemlyb.extract_u_nk.call_count == 12
    """


# @patch(ensemble_md.analysis.analyze_free_energy.MBAR)
# @patch(ensemble_md.analysis.analyze_free_energy.BAR)
# @patch(ensemble_md.analysis.analyze_free_energy.TI)
# def test_apply_estimators(mock_ti, mock_bar, mock_mbar):
#     estimator_mock = MagicMock()
#     estimator_mock.fit.return_value = estimator_mock


def test_apply_estimators():
    pass


def test_calculate_df_adjacent():
    pass


def test_combine_df_adjacent():
    pass


def test_calculate_free_energy():
    pass


def test_calculate_df_rmse():
    # Mock estimators setup
    estimator1 = MagicMock()
    estimator2 = MagicMock()

    # Mock the delta_f_ DataFrame-like attributes
    # Using np.array to simulate the DataFrame iloc functionality
    estimator1.delta_f_ = MagicMock()
    estimator1.delta_f_.iloc.__getitem__.side_effect = lambda x: np.array([0, 1, 2, 3]) if x == 0 else np.array([])  # so estimator1.delta_f_.iloc[0] will be np.array([0, 1, 2, 3])  # noqa: E501
    estimator2.delta_f_ = MagicMock()
    estimator2.delta_f_.iloc.__getitem__.side_effect = lambda x: np.array([0, 1.5, 2.5, 3.5]) if x == 0 else np.array([])  # so estimator1.delta_f_.iloc[0] will be np.array([0, 1.5, 2.5, 3.5])  # noqa: E501
    estimators = [estimator1, estimator2]

    # Reference free energies
    df_ref = [0, 1, 2, 3, 4, 5]
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]  # Indices into df_ref

    expected_rmse1 = np.sqrt(np.mean((np.array([0, 1, 2, 3]) - np.array([0, 1, 2, 3])) ** 2))
    expected_rmse2 = np.sqrt(np.mean((np.array([0, 1.5, 2.5, 3.5]) - np.array([0, 1, 2, 3])) ** 2))
    expected_rmse = [expected_rmse1, expected_rmse2]

    rmse_list = analyze_free_energy.calculate_df_rmse(estimators, df_ref, state_ranges)
    np.testing.assert_allclose(rmse_list, expected_rmse, rtol=1e-5)


@patch('ensemble_md.analysis.analyze_free_energy.plt')
def test_plot_free_energy(mock_plt):
    analyze_free_energy.plot_free_energy([1, 2, 3], [0.1, 0.1, 0.1], 'test.png')
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called_once_with(range(3), [1, 2, 3], 'o-', c='#1f77b4')
    mock_plt.errorbar.assert_called_once_with(range(3), [1, 2, 3], yerr=[0.1, 0.1, 0.1], fmt='o', capsize=2, c='#1f77b4')  # noqa: E501
    mock_plt.xlabel.assert_called_once_with('State')
    mock_plt.ylabel.assert_called_once_with('Free energy (kT)')
    mock_plt.grid.assert_called_once()
    mock_plt.savefig.assert_called_once_with('test.png', dpi=600)


def test_average_weights(capfd):
    g_vecs = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    dg_avg, dg_avg_err = analyze_free_energy.average_weights(g_vecs, frac=0.3)
    out, err = capfd.readouterr()
    assert dg_avg == 3
    assert dg_avg_err == 1
    assert 'The number of samples to be averaged is less than 2, so all samples will be averaged.' in out
