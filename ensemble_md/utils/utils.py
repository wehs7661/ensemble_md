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
The :obj:`.utils` module provides useful utility functions.
"""
import sys
import glob
import natsort
import datetime
import collections
import numpy as np
from itertools import combinations
from ensemble_md.utils import gmx_parser


class Logger:
    """
    Redirects the STDOUT and STDERR to a specified output file while preserving them on screen.

    Parameters
    ----------
    logfile : str
        Name of the output file to write the logged messages.

    Attributes
    ----------
    terminal : file object
        The file object that represents the original STDOUT (i.e., the screen).
    log : file object
        The file object that represents the logfile where messages will be written.
    """

    def __init__(self, logfile):
        """
        Initializes a Logger instance.

        Parameters
        ----------
        logfile : str
            Name of the output file to write the logged messages.
        """
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        """
        Writes the given message to both the STDOUT and the logfile.

        Parameters
        ----------
        message : str
            The message to be written to STDOUT and logfile.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        This method is needed for Python 3 compatibility. This handles the flush command by doing nothing.
        You might want to specify some extra behavior here.
        """
        # self.terminal.log()
        pass


def compare_MDPs(mdp_list):
    """
    Given a list of MDP files, identify the parameters for which not all MDP
    files have the same values. (Currently, this function is not used in the
    workflow adopted in :code:`run_EEXE.py` but it might be useful in some places,
    so we decided to keep it.)

    Returns
    -------
    diff_params : list
        The list of parameters differing between the input MDP files.
    """
    compare_list = list(combinations(mdp_list, r=2))
    diff_params = []
    for i in range(len(compare_list)):
        mdp_1 = gmx_parser.MDP(compare_list[i][0])
        mdp_2 = gmx_parser.MDP(compare_list[i][1])

        # First figure out the union set of the parameters and exclude blanks and comments
        all_params = set(list(mdp_1.keys()) + list(mdp_2.keys()))
        all_params = [p for p in all_params if not p.startswith(('B', 'C'))]

        for p in all_params:
            if p in diff_params:
                pass  # already in the list, no need to compare again
            else:
                if p not in mdp_1 or p not in mdp_2:
                    diff_params.append(p)
                else:
                    # the parameter is in both MDP files
                    if mdp_1[p] != mdp_2[p]:
                        diff_params.append(p)

    return diff_params


def format_time(t):
    """
    Converts time in seconds to the "most readable" format.

    Parameters
    ----------
    t : float
        The time in seconds.

    Returns
    -------
    t_str : str
        A string in the format of "XX day XX hour(s) XX minute(s) XX second(s)".
    """
    hh_mm_ss = str(datetime.timedelta(seconds=t)).split(":")

    if "day" in hh_mm_ss[0]:
        # hh_mm_ss[0] will contain "day" and cannot be converted to float
        hh, mm, ss = hh_mm_ss[0], float(hh_mm_ss[1]), float(hh_mm_ss[2])
        t_str = f"{hh} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"
    else:
        hh, mm, ss = float(hh_mm_ss[0]), float(hh_mm_ss[1]), float(hh_mm_ss[2])
        if hh == 0:
            if mm == 0:
                t_str = f"{ss:.1f} second(s)"
            else:
                t_str = f"{mm:.0f} minute(s) {ss:.0f} second(s)"
        else:
            t_str = f"{hh:.0f} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"

    return t_str


def autoconvert(s):
    """
    Converts input to a numerical type if possible. Used for the MDP parser.
    Modified from `utilities.py in GromacsWrapper <https://github.com/Becksteinlab/GromacsWrapper>`_.
    Copyright (c) 2009 Oliver Beckstein <orbeckst@gmail.com>

    Parameters
    ----------
    s : str or any
        The input value to be converted to a numerical type if possible. If :code:`s` is not a string,
        it is returned as is.

    Returns
    -------
    numerical : int, float, numpy.ndarray, or any
        The converted numerical value. If :code:`s` can be converted to a single numerical value,
        that value is returned as an :code:`int` or :code:`float`. If :code:`s` can be converted to
        multiple numerical values, a :code:`numpy.ndarray` containing those values is returned.
        If :code:`s` cannot be converted to a numerical value, :code:`s` is returned as is.

    Raises
    ------
    ValueError
        If :code:`s` cannot be converted to a numerical value.
    """
    if type(s) is not str:
        return s
    for converter in int, float, str:  # try them in increasing order of lenience
        try:
            s = [converter(i) for i in s.split()]
            if len(s) == 1:
                return s[0]
            else:
                return s
                """
                if len(s) != 0 and type(s[0]) == str:
                    # For the case like pull_coord1_dim = Y Y Y
                    return s
                else:
                    return np.array(s)
                """
        except (ValueError, AttributeError):
            pass
    raise ValueError("Failed to autoconvert {0!r}".format(s))


def get_subplot_dimension(n_panels):
    """
    Gets the numbers of rows and columns in a subplot such that
    the arrangement of the .

    Parameters
    ----------
    n_panels : int
        The number of panels in the subplot.
    """
    if int(np.sqrt(n_panels) + 0.5) ** 2 == n_panels:
        # perfect square number
        n_cols = int(np.sqrt(n_panels))
    else:
        n_cols = int(np.floor(np.sqrt(n_panels))) + 1

    if n_panels % n_cols == 0:
        n_rows = int(np.floor(n_panels / n_cols))
    else:
        n_rows = int(np.floor(n_panels / n_cols)) + 1

    return n_rows, n_cols


def weighted_mean(vals, errs):
    """
    Calculates the inverse-variance-weighted mean. Note that if
    any error is 0, the simple mean will be returned.

    Parameters
    ----------
    vals : list
        A list of values to be averaged.
    errs : list
        A list of errors corresponding to the given values

    Returns
    -------
    mean : float
        The inverse-variance-weighted mean.
    err : float
        The propgated error of the mean.
    """
    if 0 in errs:
        # This could happen in the very beginning of a simulation, which could lead to an ZeroDivision warning/error.
        # In this case, we just ignore the w vector and just calculate the simple average.
        mean, err = np.mean(vals), None
        return mean, err

    w = [1 / (i ** 2) for i in errs]
    wx = [w[i] * vals[i] for i in range(len(vals))]
    mean = np.sum(wx) / np.sum(w)
    err = np.sqrt(1 / np.sum(w))

    return mean, err


def get_time_metrics(log):
    """
    Gets the time-based metrics from a log file, including the core time (s),
    wall time, and performance (ns/day).

    Parameters
    ----------
    log : str
        The input log file.

    Returns
    -------
    t_metrics : dict
        A dictionary having following keys: :code:`t_core`, :code:`t_wall`, :code:`performance`.
    """
    t_metrics = {}
    with open(log, 'r') as f:
        # Using deque is much faster than using readlines and reversing it since we only need the last few lines.
        lines = collections.deque(f, 20)  # get the last 20 lines should be enough

    for l in lines:  # noqa: E741
        if 'Performance: ' in l:
            t_metrics['performance'] = float(l.split()[1])  # ns/day
        if 'Time: ' in l:
            t_metrics['t_core'] = float(l.split()[1])  # s
            t_metrics['t_wall'] = float(l.split()[2])  # s
            break

    return t_metrics


def analyze_EEXE_time(log_files=None):
    """
    Perform simple data analysis on the wall times and performances of all iterations of an EEXE simulation.

    Parameters
    ----------
    log_files : None or list
        A list of sorted file names of all log files.

    Returns
    -------
    t_wall_tot : float
        The total wall time GROMACS spent to finish all iterations for the EEXE simulation.
    t_sync : float
        The total time spent in synchronizing all replicas, which is the sum of the differences
        between the longest and the shortest time elapsed to finish a iteration.
    t_wall_list : list
        The list of wall times of finishing each mdrun command.
    """
    n_iter = len(glob.glob('sim_0/iteration_*'))
    if log_files is None:
        log_files = [natsort.natsorted(glob.glob(f'sim_*/iteration_{i}/*log')) for i in range(n_iter)]

    t_wall_list = []
    t_wall_tot, t_sync = 0, 0
    for i in range(n_iter):
        t_wall = [get_time_metrics(log_files[i][j])['t_wall'] for j in range(len(log_files[i]))]
        t_wall_list.append(t_wall)
        t_wall_tot += max(t_wall)
        t_sync += (max(t_wall) - min(t_wall))

    return t_wall_tot, t_sync, t_wall_list
