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
from unittest.mock import patch, call, MagicMock
from ensemble_md.analysis import analyze_free_energy


@patch('ensemble_md.analysis.analyze_free_energy.alchemlyb')
@patch('ensemble_md.analysis.analyze_free_energy.subsampling')
@patch('ensemble_md.analysis.analyze_free_energy.extract_u_nk')
@patch('ensemble_md.analysis.analyze_free_energy.extract_dHdl')
@patch('ensemble_md.analysis.analyze_free_energy.detect_equilibration')
@patch('ensemble_md.analysis.analyze_free_energy.subsample_correlated_data')
def test_preprocess_data(mock_corr, mock_equil, mock_extract_dHdl, mock_extract_u_nk, mock_subsampling, mock_alchemlyb, capfd):  # noqa: E501
    mock_data, mock_data_series = MagicMock(), MagicMock()
    mock_alchemlyb.concat.return_value = mock_data
    mock_subsampling.u_nk2series.return_value = mock_data_series
    mock_subsampling._prepare_input.return_value = (mock_data, mock_data_series)
    mock_equil.return_value = (10, 5, 50)  # t, g, Neff_max
    mock_data_series.__len__.return_value = 100   # For one of the print statements

    # Set slicing to return different mock objects based on input
    def slicing_side_effect(*args, **kwargs):
        if args[0] == mock_data:
            return mock_data  # Return mock_data when the data object is sliced
        elif args[0] == mock_data_series:
            return mock_data_series  # Return mock_data_series when the data_series object is sliced

    mock_subsampling.slicing.side_effect = slicing_side_effect

    def generic_list_slicing(key):
        # Function like data_series[t:] internally uses the slice object
        if isinstance(key, slice):
            # Create a new mock to represent the sliced part if needed
            # This allows for separate tracking or specific return values for the slice
            slice_mock = MagicMock()
            return slice_mock

    mock_data.__getitem__.side_effect = generic_list_slicing  # so that we can use mock_data[t:]
    mock_data_series.__getitem__.side_effect = slicing_side_effect  # so that we can use mock_data_series[t:]
    mock_data_series_equil = mock_data_series[10:]  # Mock the equilibrated data series, given t=10

    files = [[f'ensemble_md/tests/data/dhdl/simulation_example/sim_{i}/iteration_{j}/dhdl.xvg' for j in range(3)] for i in range(4)]  # noqa: E501
    results = analyze_free_energy.preprocess_data(files, 300, 'u_nk')

    out, err = capfd.readouterr()

    assert mock_alchemlyb.concat.call_count == 4
    assert mock_extract_u_nk.call_count == 12
    assert mock_subsampling._prepare_input.call_count == 4
    assert mock_subsampling.slicing.call_count == 8
    assert mock_equil.call_count == 4

    for i in range(4):
        for j in range(3):
            assert mock_extract_u_nk.call_args_list[i * 3 + j] == call(files[i][j], T=300)
        assert mock_subsampling._prepare_input.call_args_list[i] == call(mock_data, mock_data_series, drop_duplicates=True, sort=True)  # noqa: E501
        assert mock_subsampling.slicing.call_args_list[2 * i] == call(mock_data, step=1)
        assert mock_subsampling.slicing.call_args_list[2 * i + 1] == call(mock_data_series, step=1)
        assert 'Subsampling and decorrelating the concatenated u_nk data ...' in out
        assert '  Adopted spacing: 1' in out
        assert '  10.0% of the u_nk data was in the equilibrium region and therfore discarded.' in out  # noqa: E501
        assert '  Statistical inefficiency of u_nk: 5.0' in out
        assert '  Number of effective samples: 50' in out
        assert mock_corr.call_args_list[i] == call(mock_data_series_equil, g=5)

    assert len(results[0]) == 4
    assert results[1] == [10, 10, 10, 10]
    assert results[2] == [5, 5, 5, 5]


# @patch(ensemble_md.analysis.analyze_free_energy.MBAR)
# @patch(ensemble_md.analysis.analyze_free_energy.BAR)
# @patch(ensemble_md.analysis.analyze_free_energy.TI)
# def test_apply_estimators(mock_ti, mock_bar, mock_mbar):
#     estimator_mock = MagicMock()
#     estimator_mock.fit.return_value = estimator_mock


def test_apply_estimators():
    pass


def test_calculate_df_adjacent():
    estimator1 = MagicMock()
    estimator2 = MagicMock()

    # Setup delta_f_ and d_delta_f_ matrices for two estimators
    estimator1.delta_f_ = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    estimator1.d_delta_f_ = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.1]])
    estimator2.delta_f_ = np.array([[0, 4, 5], [4, 1, 6], [5, 6, 2]])
    estimator2.d_delta_f_ = np.array([[0.2, 0.3, 0.5], [0.3, 0.2, 0.6], [0.5, 0.6, 0.2]])
    estimators = [estimator1, estimator2]

    df_adjacent, df_err_adjacent = analyze_free_energy._calculate_df_adjacent(estimators)

    assert df_adjacent == [[1, 3], [4, 6]]
    assert df_err_adjacent == [[0.2, 0.4], [0.3, 0.6]]


def test_combine_df_adjacent():
    pass


def test_calculate_free_energy():
    # state_ranges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    pass


def test_calculate_df_rmse():
    # Mock estimators setup
    estimator1 = MagicMock()
    estimator2 = MagicMock()

    # Mock the delta_f_ DataFrame-like attributes
    # Using np.array to simulate the DataFrame iloc functionality
    estimator1.delta_f_ = MagicMock()

    # vals = np.array([0, 1, 2, 3])

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
