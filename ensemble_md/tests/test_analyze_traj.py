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
Unit tests for the module analyze_traj.py.
"""
import os
import numpy as np
from unittest.mock import patch, MagicMock
from ensemble_md.analysis import analyze_traj

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def assert_plt_calls(mock_plt, call_specs):
    """
    Assert calls to matplotlib functions with specified parameters.

    Parameters
    ----------
    mock_plt : :code:`MagicMock` object
        Mock object of :code:`matplotlib.pyplot`.
    call_specs : list
        A list of lists that contains the following four elements:
          - The name of the matplotlib function (as :code:`str`) that was called.
          - The assert method (as :code:`str`), e.g., :code:`assert_called_once_with`.
          - The positional arguments (as :code:`tuple`) passed to the matplotlib function.
          - The keyword arguments (as :code:`dict`) passed to the matplotlib function.
    """
    for call_spec in call_specs:
        plt_func = call_spec[0]
        assert_method = call_spec[1]
        plt_args = call_spec[2]
        plt_kwargs = call_spec[3]

        mock_func = getattr(mock_plt, plt_func)
        assert_func = getattr(mock_func, assert_method)
        assert_func(*plt_args, **plt_kwargs)  # call the assertion method


def test_extract_state_traj():
    traj, t = analyze_traj.extract_state_traj(os.path.join(input_path, 'dhdl/dhdl_0.xvg'))
    state_list = [
        0, 0, 3, 1, 4, 4, 5, 4, 5, 5,
        4, 4, 5, 4, 2, 4, 5, 2, 1, 2,
        3, 1, 2, 4, 1, 0, 2, 4, 3, 2,
        1, 3, 3, 4, 2, 3, 1, 1, 0, 1,
        2, 3, 1, 0, 1, 4, 3, 1, 3, 2, 5]
    t_true = [0.02 * i + 3 for i in range(len(state_list))]
    assert traj == state_list
    assert np.allclose(t, t_true)


def test_stitch_time_series():
    pass


def test_convert_npy2xvg():
    # Create dummy input data
    trajs = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=int)
    dt = 0.1  # ps
    subsampling = 2

    os.chdir('ensemble_md/tests/data')
    analyze_traj.convert_npy2xvg(trajs, dt, subsampling)

    assert os.path.exists('traj_0.xvg')
    assert os.path.exists('traj_1.xvg')

    with open('traj_0.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. State index\n'
        assert content[2] == '0.0         0\n'
        assert content[3] == '0.2         2\n'

    with open('traj_1.xvg', 'r') as f:
        content = f.readlines()
        assert content[0] == '# This file was created by ensemble_md\n'
        assert content[1] == '# Time (ps) v.s. State index\n'
        assert content[2] == '0.0         4\n'
        assert content[3] == '0.2         6\n'

    os.remove('traj_0.xvg')
    os.remove('traj_1.xvg')
    os.chdir('../../../')


def test_stitch_time_series_for_sim():
    pass


def test_stitch_trajs():
    pass


def test_traj2transmtx():
    traj = [0, 1, 2, 1, 0, 3]
    N = 4  # matrix size

    # Case 1: normalize=False
    array = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]])
    np.testing.assert_array_equal(analyze_traj.traj2transmtx(traj, N, normalize=False), array)

    # Case 2: normalize=True
    array = np.array([
        [0, 0.5, 0, 0.5],
        [0.5, 0, 0.5, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]])
    np.testing.assert_array_equal(analyze_traj.traj2transmtx(traj, N, normalize=True), array)


@patch('ensemble_md.analysis.analyze_traj.plt')  # to mock and replace plt (which is matplotlib.pyplot) in analyze_traj
def test_plot_rep_trajs(mock_plt):
    # Not much we can do to test the plot function except to verify if the expected matplotlib functions were called.
    fig_name = 'ensemble_md/tests/data/test.png'
    cmap = mock_plt.cm.ocean

    # Case 1: Short trajs without dt and stride
    trajs = np.array([[0, 1, 1, 0, 2], [1, 0, 1, 2, 0], [2, 0, 1, 0, 2]], dtype=int)
    n_sim = len(trajs)
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    analyze_traj.plot_rep_trajs(trajs, fig_name)

    x_input = np.array([0, 1, 2, 3, 4])
    y_input_1 = np.array([0, 1, 1, 0, 2])
    y_input_2 = np.array([1, 0, 1, 2, 0])
    y_input_3 = np.array([2, 0, 1, 0, 2])

    # Verify that the expected matplotlib functions were called
    # mock_plt.figure.assert_called_once()
    # mock_plt.plot.assert_called()
    # mock_plt.xlabel.assert_called_with('MC moves')
    # mock_plt.ylabel.assert_called_with('Replica')
    # mock_plt.grid.assert_called_once()
    # mock_plt.legend.assert_called_once()
    # mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    
    # Using assert_plt_calls, the lines above can be written as below
    call_specs = [
        ['figure', 'assert_called_once', (), {}],
        ['plot', 'assert_called', (), {}],
        ['xlabel', 'assert_called_with', ('MC moves',), {}],
        ['ylabel', 'assert_called_with', ('Replica',), {}],
        ['grid', 'assert_called_once', (), {}],
        ['legend', 'assert_called_once', (), {}],
        ['savefig', 'assert_called_once_with', (fig_name,), {'dpi': 600}]
    ]
    assert_plt_calls(mock_plt, call_specs)
    


    assert mock_plt.plot.call_count == len(trajs)

    # mock_plt.plot.assert_any_call(x_input, y_input_1, color=colors[0], label='Trajectory 0')
    # There is a bug in unittest.mock such that there will be an ValueError upon comparisons of two arrays.
    # Therefore, the line above would fail and we can only use assert_called (as shown above) and compare
    # x_input, y_input_1, etc. with the called arguments.
    # Here mock_plt.plot.call_args_list prints as below. Note that a `call` object can be indexed like a tuple
    # where index 0 contains the positional arguments, and index 1 contains the keyword arguments.
    # [
    #     call(array([0, 1, 2, 3, 4]), array([0, 1, 1, 0, 2]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 0', marker='o'),  # noqa: E501
    #     call(array([0, 1, 2, 3, 4]), array([1, 0, 1, 2, 0]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 1', marker='o'),  # noqa: E501
    #     call(array([0, 1, 2, 3, 4]), array([2, 0, 1, 0, 2]), color=<MagicMock name='plt.cm.ocean()' id='140575569521536'>, label='Trajectory 2', marker='o')   # noqa: E501
    # ]

    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, y_input_1)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, y_input_2)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[2][0], (x_input, y_input_3)))
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2', 'marker': 'o'}

    # Case 2: Short trajs with dt and stride
    mock_plt.reset_mock()

    dt = 0.2  # ps
    stride = 2
    analyze_traj.plot_rep_trajs(trajs, fig_name, dt, stride)
    x_input = np.array([0, 0.4, 0.8])
    y_input_1 = np.array([0, 1, 2])
    y_input_2 = np.array([1, 1, 0])
    y_input_3 = np.array([2, 1, 2])

    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Time (ps)')
    mock_plt.ylabel.assert_called_with('Replica')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    assert mock_plt.plot.call_count == len(trajs)

    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[0][0], (x_input, y_input_1)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[1][0], (x_input, y_input_2)))
    assert all(np.array_equal(a, b) for a, b in zip(mock_plt.plot.call_args_list[2][0], (x_input, y_input_3)))
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1', 'marker': 'o'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2', 'marker': 'o'}

    # Case 3: Long trajs with dt and without stride
    mock_plt.reset_mock()

    trajs = np.random.randint(low=0, high=2, size=(3, 2000000))
    analyze_traj.plot_rep_trajs(trajs, fig_name, dt)
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()
    mock_plt.xlabel.assert_called_with('Time (ns)')
    mock_plt.ylabel.assert_called_with('Replica')
    mock_plt.grid.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.savefig.assert_called_once_with(fig_name, dpi=600)
    assert mock_plt.plot.call_count == len(trajs)

    # Here we only check the lengths of x and y inputs
    assert len(mock_plt.plot.call_args_list[0][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[0][0][1]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[1][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[1][0][1]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[2][0][0]) == 2000000 / 100
    assert len(mock_plt.plot.call_args_list[2][0][1]) == 2000000 / 100
    assert mock_plt.plot.call_args_list[0][1] == {'color': colors[0], 'label': 'Trajectory 0'}
    assert mock_plt.plot.call_args_list[1][1] == {'color': colors[1], 'label': 'Trajectory 1'}
    assert mock_plt.plot.call_args_list[2][1] == {'color': colors[2], 'label': 'Trajectory 2'}


@patch('ensemble_md.analysis.analyze_traj.plt')
def test_plot_state_trajs(mock_plt):
    state_ranges = [[0, 1, 2, 3], [2, 3, 4, 5]]
    fig_name = 'ensemble_md/tests/data/test.png'
    cmap = mock_plt.cm.ocean
    n_sim = len(state_ranges)
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    # Mock the return value of plt.subplots to return a tuple of two mock objects
    # We need this because plot_state_trajs calls _, ax = plt.subplots(...). When we mock 
    # matplolib.pyplot using mock_plt, plt.subplots will be replaced by mock_plt.subplots
    # and will return a mock object, not the tuple of figure and axes objects that the real plt.subplots returns.
    # This would in turn lead to an ValueError. To avoid this, we need to mock the return values of plt.subplots.
    mock_figure = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_figure, mock_axes)

    # Case 1: Short trajs without dt and stride
    trajs = np.array([[0, 1, 0, 2, 3, 4, 3, 4, 5, 4], [2, 3, 4, 5, 4, 3, 2, 1, 0, 1]], dtype=int)

    analyze_traj.plot_state_trajs(trajs, state_ranges, fig_name)

    x_input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    mock_plt.subplots.assert_called_once_with(nrows=1, ncols=2, figsize=(5, 2.5))
    mock_plt.subplot.assert_called()
    mock_plt.plot.assert_called()
    mock_plt.fill_between.assert_called()
    mock_plt.xlabel.assert_called_with('MC moves')
    mock_plt.ylabel.assert_called_with('State')
    mock_plt.grid.assert_called()

    assert mock_plt.subplot.call_count == len(state_ranges)
    assert mock_plt.plot.call_count == len(state_ranges)
    assert mock_plt.grid.call_count == len(state_ranges)
    assert mock_plt.fill_between.call_count == len(state_ranges) ** 2


def test_plot_state_hist():
    pass


def test_calculate_hist_rmse():
    pass


def plot_transit_time():
    pass


def test_plot_g_vecs():
    pass


def test_get_swaps():
    pass


def test_plot_swaps():
    pass


def test_get_g_evolution():
    pass


def test_get_dg_evoluation():
    pass


def test_plot_dg_evolution():
    pass


def test_get_delta_w_updates():
    pass
