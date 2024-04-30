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
The :obj:`.utils` module provides useful utility functions for running or analyzing REXEE simulations.
"""
import sys
import glob
import natsort
import datetime
import subprocess
import collections
import numpy as np


class Logger:
    """
    A logger class that redirects the STDOUT and STDERR to a specified output file while
    preserving the output on screen. This is useful for logging terminal output to a file
    for later analysis while still seeing the output in real-time during execution.

    Parameters
    ----------
    logfile : str
        The file path of which the standard output and standard error should be logged.

    Attributes
    ----------
    terminal : :code:`io.TextIOWrapper` object
        The original standard output object, typically :code:`sys.stdout`.
    log : :code:`io.TextIOWrapper` object
        File object used to log the output in append mode.
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        """
        Writes a message to the terminal and to the log file.

        Parameters
        ----------
        message : str
            The message to be written to STDOUT and the log file.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        This method is needed for Python 3 compatibility. This handles the flush command by doing nothing.
        Some extra behaviors may be specified here.
        """
        # self.terminal.log()
        pass


def run_gmx_cmd(arguments, prompt_input=None):
    """
    Runs a GROMACS command through a subprocess call.

    Parameters
    ----------
    arguments : list
        A list of arguments that compose of the GROMACS command to run, e.g.,
        :code:`['gmx', 'mdrun', '-deffnm', 'sys']`.
    prompt_input : str or None, Optional
        The input to be passed to the interative prompt launched by the GROMACS command, if any.

    Returns
    -------
    return_code : int
        The exit code of the GROMACS command. Any number other than 0 indicates an error.
    stdout : str or None
        The STDOUT of the process.
    stderr: str or None
        The STDERR or the process.
    """
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, input=prompt_input, check=True)
        return_code, stdout, stderr = result.returncode, result.stdout, None
    except subprocess.CalledProcessError as e:
        return_code, stdout, stderr = e.returncode, None, e.stderr

    return return_code, stdout, stderr


def format_time(t):
    """
    Converts time in seconds to a more readable format.

    Parameters
    ----------
    t : float
        The time in seconds.

    Returns
    -------
    t_str : str
        A string representing the time duration in a format of "X hour(s) Y minute(s) Z second(s)", adjusting the units
        as necessary based on the input duration, e.g., 1 hour(s) 0 minute(s) 0 second(s) for 3600 seconds and
        15 minute(s) 30 second(s) for 930 seconds.
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


def _convert_to_numeric(s):
    """
    Converts the input to a numerical type when possible. This internal function is used for the MDP parser.

    Parameters
    ----------
    s : any
        The input value to be converted to a numerical type if possible. The data type of :code:`s` is
        usually :code:`str` but can be any. However, if :code:`s` is not a string, it will be returned as is.

    Returns
    -------
    numerical : any
        The converted numerical value. If :code:`s` can be converted to a single numerical value,
        that value is returned as an :code:`int` or :code:`float`. If :code:`s` can be converted to
        multiple numerical values, a list containing those values is returned.
        If :code:`s` cannot be converted to a numerical value, :code:`s` is returned as is.
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
        except (ValueError, AttributeError):
            pass


def _get_subplot_dimension(n_panels):
    """
    Gets the number of rows and columns for a subplot based on the number of panels such
    that the subplots are arranged in a grid that is as square as possible. A greater number
    of columns is preferred to a greater number of rows.

    Parameters
    ----------
    n_panels : int
        The number of panels to be arranged in subplots.

    Example
    -------
    >>> from ensemble_md.utils import utils
    >>> utils._get_subplot_dimension(10)
    (4, 3)
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


def calc_rmse(data, ref):
    """
    Calculates the root mean square error (RMSE) of the given data
    with respect to the reference data.

    Parameters
    ----------
    data : list
        A list of values to be compared with the reference data.
    ref : list
        A list of reference values.

    Returns
    -------
    rmse : float
        The root mean square error.
    """
    rmse = np.sqrt(np.mean((np.array(data) - np.array(ref)) ** 2))

    return rmse


def get_time_metrics(log):
    """
    Gets the time-based metrics from a log file of a REXEE simulation, including the core time,
    wall time, and performance (ns/day).

    Parameters
    ----------
    log : str
        The file path of the input log file.

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

    return t_metrics


def analyze_REXEE_time(n_iter=None, log_files=None):
    """
    Performs simple data analysis on the wall times and performances of all iterations of an REXEE simulation.

    Parameters
    ----------
    n_iter : None or int, Optional
        The number of iterations in the REXEE simulation. If None, the function will try to find the number of
        iterations by counting the number of directories named in the format of :code`iteration_*` in the simulation
        directory (specifically :code:`sim_0`) in the current working directory or where the log files are located.
    log_files : None or list, Optional
        A list of lists of log paths with the shape of :code:`(n_iter, n_replicas)`. If None, the function will try to
        find the log files by searching the current working directory.

    Returns
    -------
    t_wall_tot : float
        The total wall time GROMACS spent to finish all iterations for the REXEE simulation.
    t_sync : float
        The total time spent in synchronizing all replicas, which is the sum of the differences
        between the longest and the shortest time elapsed to finish a iteration.
    t_wall_list : list
        The list of wall times for finishing each GROMACS mdrun command.
    """
    if n_iter is None:
        if log_files is None:
            n_iter = len(glob.glob('sim_0/iteration_*'))
        else:
            n_iter = len(log_files)

    if log_files is None:
        log_files = [natsort.natsorted(glob.glob(f'sim_*/iteration_{i}/*log')) for i in range(n_iter)]

    if len(log_files) == 0:
        raise FileNotFoundError("No sim/iteration directories found.")

    t_wall_list = []
    t_wall_tot, t_sync = 0, 0
    for i in range(n_iter):
        t_wall = [get_time_metrics(log_files[i][j])['t_wall'] for j in range(len(log_files[i]))]
        t_wall_list.append(t_wall)
        t_wall_tot += max(t_wall)
        t_sync += (max(t_wall) - min(t_wall))

    return t_wall_tot, t_sync, t_wall_list
