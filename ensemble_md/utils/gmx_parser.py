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
The :code:`gmx_parser` module provides functions for parsing GROMACS files.
"""
import os
import re
import six
import logging
import warnings
from collections import OrderedDict as odict

from ensemble_md.utils import utils
from ensemble_md.utils.exceptions import ParseError


def parse_log(log_file):
    """
    This function parses a log file generated by expanded ensemble and provides
    important information, especially for running new iterations in REXEE.
    Typically, there are three types of log files from an expanded ensemble simulation:

    - **Case 1**: The weights are still updating in the simulation and have never been equilibrated.

        - The output :code:`equil_time` should always be -1.

    - **Case 2**: The weights were equilibrated during the simulation.

        - The output :code:`equil_time` should be the time (in ps) it took to get the weights equilibrated.
        - The final weights (:code:`weights`) will just be the equilibrated weights.

    - **Case 3**: The weights were fixed in the simulation.

        - The output :code:`equil_time` should always be 0.
        - The final weights (which never change during the simulation) and the final counts will
          still be returned.

    Parameters
    ----------
    log_file : str
        The log file to be parsed.

    Returns
    -------
    weights : list
        In all cases, :code:`weights` should be a list of lists (of weights).

        - In Case 1, a list of list of weights as a function of time since the last update of the Wang-Landau
          incrementor will be returned.
        - In Case 2, a list of list of weights as a function of time since the last update of the Wang-Landau
          incrementor up to equilibration will be returned.
        - In Case 3, the returned list will only have one list inside, which is the list of the final weights.

        That is, for all cases, :code:`weights[-1]` will be the final weights, which are useful in REXEE.
    counts : list
        The final histogram counts.
    wl_delta : float
        The final Wang-Landau incementor. In Cases 2 and 3, :code:`None` will be returned.
    equil_time : int or float
        - In Case 1, -1 will be returned, which means that the weights have not been equilibrated.
        - In Case 2, the time in ps that it took to equilibrate the weights will be returned.
        - In Case 3, 0 will be returned, which means that the weights were fixed during the simulation.
    """
    f = open(log_file, "r")
    lines = f.readlines()
    f.close()

    case = None  # itialized as None and should end up as 1 or 2 or 3.

    # First parse the MD parameters to tell the type of the simulation.
    for l in lines:  # noqa: E741
        if "n-lambdas" in l:
            N_states = int(l.split("=")[1])
        if "tinit" in l:
            tinit = float(l.split("=")[1])
        if "weight-equil-wl-delta" in l:
            # For Case 1 and Case 2
            cutoff = float(l.split("=")[1])
        if "lmc-stats" in l:
            if l.split("=")[1].split()[0] in ["no", "No"]:  # Case 3
                case = '3'
                equil_time = 0
                wl_delta = None
            else:
                pass  # Either Case 1 or Case 2
        if "dt  " in l:
            dt = float(l.split("=")[1])

    # For all cases, we need to find weights and counts
    weights = []
    if case == '3':  # could be either '3' or None
        # We only need the info at the end of the simulation, so it's faster to search from the bottom of the file.
        lines.reverse()
        n = -1
        counts = []
        for l in lines:  # noqa: E741
            n += 1
            if "Count   G(in kT)" in l:
                w = []  # the list of weights at this time frame
                # The first occurrence would be the final weights. (We've reversed the lines!)
                for i in range(1, N_states + 1):
                    if "<<" in lines[n - i]:
                        w.append(float(lines[n - i].split()[-3]))
                        counts.append(int(lines[n - i].split()[-4]))
                    else:
                        w.append(float(lines[n - i].split()[-2]))
                        counts.append(int(lines[n - i].split()[-3]))
                weights.append(w)
                break
    else:  # Case 1 and Case 2
        # Here we search from the top, since we need weights as a function of time anyway.
        n = -1
        find_equil, append_equil = False, False
        wl_delta_list = [None]  # We use None so that the change in wl-delta will always get deteced at the beginning
        for l in lines:  # noqa: E741
            n += 1
            if "Count   G(in kT)" in l:  # this line is lines[n]
                # We should first check if wl_delta is changed, set weights=[] if needed before we append new weights
                if "Wang-Landau incrementor is" in lines[n - 1]:  # Case 1 so far: wl_delta is still updating
                    case = '1'
                    equil_time = -1
                    current_wl_delta = float(lines[n - 1].split(":")[1])
                    if wl_delta_list[-1] != current_wl_delta and current_wl_delta > cutoff:  # when wl-delta changes
                        # Note that if the time step the equilibration is reached happens to be a multiple of nstlog,
                        # a wl-delta right below the cutoff will be printed. In that case, we don't want to reset
                        # `weights` so we get the time series since the last time when we still have wl_delta > cutoff
                        weights = []  # we want only time series since the latest update of wl_delta
                    wl_delta_list.append(float(lines[n - 1].split(":")[1]))

                w, counts = [], []  # the list of weights at this time frame
                for i in range(1, N_states + 1):
                    # counts will be constantly updated by new values
                    if "<<" in lines[n + i]:
                        w.append(float(lines[n + i].split()[-3]))
                        counts.append(int(lines[n + i].split()[-4]))
                    else:
                        w.append(float(lines[n + i].split()[-2]))
                        counts.append(int(lines[n + i].split()[-3]))

                if find_equil is False or append_equil is False:
                    weights.append(w)
                    if find_equil is True:
                        append_equil = True

            if "Weights have equilibrated" in l:
                case = '2'  # we don't break the loop even if equil_time is found, as we need the final counts.
                find_equil = True  # After this, we will append weights one last time, which are equilibrated weights.
                equil_step = int(l.split(":")[0].split("Step")[1])
                equil_time = equil_step * dt + tinit  # ps
                if wl_delta is not None and wl_delta < cutoff:
                    # Should only happen when equil_time % nstlog == 0, where the weights should have been appended
                    # Note that we additionally have wl_delta is not None. Since wl_delta could be None if the
                    # weights get equilibrated right after the simulation stop (before nstlog).
                    append_equil = True

            wl_delta = wl_delta_list[-1]

        if case == '2':
            wl_delta = None

    return weights, counts, wl_delta, equil_time


class FileUtils(object):
    """Mixin class to provide additional file-related capabilities.
    Modified from `utilities.py in GromacsWrapper <https://github.com/Becksteinlab/GromacsWrapper>`_.
    Copyright (c) 2009 Oliver Beckstein <orbeckst@gmail.com>
    """

    #: Default extension for files read/written by this class.
    default_extension = None

    def _init_filename(self, filename=None, ext=None):
        """Initialize the current filename :attr:`FileUtils.real_filename` of the object.

        Bit of a hack.

        - The first invocation must have ``filename != None``; this will set a
          default filename with suffix :attr:`FileUtils.default_extension`
          unless another one was supplied.

        - Subsequent invocations either change the filename accordingly or
          ensure that the default filename is set with the proper suffix.

        """

        extension = ext or self.default_extension
        filename = self.filename(
            filename, ext=extension, use_my_ext=True, set_default=True
        )
        #: Current full path of the object for reading and writing I/O.
        self.real_filename = os.path.realpath(filename)

    def filename(self, filename=None, ext=None, set_default=False, use_my_ext=False):
        """Supply a file name for the class object.

        Typical uses::

           fn = filename()             ---> <default_filename>
           fn = filename('name.ext')   ---> 'name'
           fn = filename(ext='pickle') ---> <default_filename>'.pickle'
           fn = filename('name.inp','pdf') --> 'name.pdf'
           fn = filename('foo.pdf',ext='png',use_my_ext=True) --> 'foo.pdf'

        The returned filename is stripped of the extension
        (``use_my_ext=False``) and if provided, another extension is
        appended. Chooses a default if no filename is given.

        Raises a ``ValueError`` exception if no default file name is known.

        If ``set_default=True`` then the default filename is also set.

        ``use_my_ext=True`` lets the suffix of a provided filename take
        priority over a default ``ext`` tension.
        """
        if filename is None:
            if not hasattr(self, "_filename"):
                self._filename = None  # add attribute to class
            if self._filename:
                filename = self._filename
            else:
                raise ValueError(
                    "A file name is required because no default file name was defined."
                )
            my_ext = None
        else:
            filename, my_ext = os.path.splitext(filename)
            if set_default:  # replaces existing default file name
                self._filename = filename
        if my_ext and use_my_ext:
            ext = my_ext
        if ext is not None:
            if ext.startswith(os.extsep):
                ext = ext[1:]  # strip a dot to avoid annoying mistakes
            if ext != "":
                filename = filename + os.extsep + ext
        return filename


class MDP(odict, FileUtils):
    """Class that represents a Gromacs mdp run input file.
    Modified from `GromacsWrapper <https://github.com/Becksteinlab/GromacsWrapper>`_.
    Copyright (c) 2009-2011 Oliver Beckstein <orbeckst@gmail.com>
    The MDP instance is an ordered dictionary.

      - *Parameter names* are keys in the dictionary.
      - *Comments* are sequentially numbered with keys Comment0001,
        Comment0002, ...
      - *Empty lines* are similarly preserved as Blank0001, ....

    When writing, the dictionary is dumped in the recorded order to a
    file. Inserting keys at a specific position is not possible.

    Currently, comments after a parameter on the same line are
    discarded. Leading and trailing spaces are always stripped.
    """

    default_extension = "mdp"
    logger = logging.getLogger("gromacs.formats.MDP")

    COMMENT = re.compile("""\s*;\s*(?P<value>.*)""")  # eat initial ws  # noqa: W605
    # see regex in cbook.edit_mdp()
    PARAMETER = re.compile(
        """
                            \s*(?P<parameter>[^=]+?)\s*=\s*  # parameter (ws-stripped), before '='  # noqa: W605
                            (?P<value>[^;]*)                # value (stop before comment=;)  # noqa: W605
                            (?P<comment>\s*;.*)?            # optional comment  # noqa: W605
                            """,
        re.VERBOSE,
    )

    def __init__(self, filename=None, autoconvert=True, **kwargs):
        """Initialize mdp structure.

        :Arguments:
          *filename*
              read from mdp file
          *autoconvert* : boolean
              ``True`` converts numerical values to python numerical types;
              ``False`` keeps everything as strings [``True``]
          *kwargs*
              Populate the MDP with key=value pairs. (NO SANITY CHECKS; and also
              does not work for keys that are not legal python variable names such
              as anything that includes a minus '-' sign or starts with a number).
        """
        super(MDP, self).__init__(
            **kwargs
        )  # can use kwargs to set dict! (but no sanity checks!)

        self.autoconvert = autoconvert

        if filename is not None:
            self._init_filename(filename)
            self.read(filename)

    def __eq__(self, other):
        """
        __eq__ inherited from FileUtils needs to be overridden if new attributes (autoconvert in
        this case) are assigned to the instance of the subclass (MDP in our case).
        See `this post by LGTM <https://lgtm.com/rules/9990086/>`_ for more details.
        """
        if not isinstance(other, MDP):
            return False
        return FileUtils.__eq__(self, other) and self.autoconvert == other.autoconvert

    def _transform(self, value):
        if self.autoconvert:
            return utils._autoconvert(value)
        else:
            return value.rstrip()

    def read(self, filename=None):
        """Read and parse mdp file *filename*."""
        self._init_filename(filename)

        def BLANK(i):
            return "B{0:04d}".format(i)

        def COMMENT(i):
            return "C{0:04d}".format(i)

        data = odict()
        iblank = icomment = 0
        with open(self.real_filename) as mdp:
            for line in mdp:
                line = line.strip()
                if len(line) == 0:
                    iblank += 1
                    data[BLANK(iblank)] = ""
                    continue
                m = self.COMMENT.match(line)
                if m:
                    icomment += 1
                    data[COMMENT(icomment)] = m.group("value")
                    continue
                # parameter
                m = self.PARAMETER.match(line)
                if m:
                    # check for comments after parameter?? -- currently discarded
                    parameter = m.group("parameter")
                    value = self._transform(m.group("value"))
                    data[parameter] = value
                else:
                    errmsg = "{filename!r}: unknown line in mdp file, {line!r}".format(
                        **vars()
                    )
                    self.logger.error(errmsg)
                    raise ParseError(errmsg)

        super(MDP, self).update(data)

    def write(self, filename=None, skipempty=False):
        """Write mdp file to *filename*.

        Parameters
        ----------
        filename : str
            Output mdp file; default is the filename the mdp was read from. If the filename
            is not supplied, the function will overwrite the file that the mdp was read from.
        skipempty : bool
            ``True`` removes any parameter lines from output that contain empty values [``False``]
        """
        # The line 'if skipempty and (v == "" or v is None):' below could possibly incur FutureWarning
        warnings.simplefilter(action='ignore', category=FutureWarning)

        with open(self.filename(filename, ext="mdp"), "w") as mdp:
            for k, v in self.items():
                if k[0] == "B":  # blank line
                    mdp.write("\n")
                elif k[0] == "C":  # comment
                    mdp.write("; {v!s}\n".format(**vars()))
                else:  # parameter = value
                    if skipempty and (v == "" or v is None):
                        continue
                    if isinstance(v, six.string_types) or not hasattr(v, "__iter__"):
                        mdp.write("{k!s} = {v!s}\n".format(**vars()))
                    else:
                        mdp.write("{} = {}\n".format(k, " ".join(map(str, v))))


def compare_MDPs(mdp_list, print_diff=False):
    """
    Given a list of MDP files, identify the parameters for which not all MDP
    files have the same values. Note that this function is not aware of the default
    values of GROMACS parameters. (Currently, this function is not used in the
    workflow adopted in :code:`run_REXEE.py` but it might be useful in some places,
    so we decided to keep it.)

    Parameters
    ----------
    mdp_list : list
        A list of MDP files.
    print_diff : bool
        If :code:`True`, print to screen the parameters that are different among the MDP files
        and the values of the parameters in the MDP files in a more readable format.

    Returns
    -------
    diff_params : dict
        A dictionary of parameters that are different among the MDP files.
        The keys are the parameter names and the values is a list of values of the
        parameters in the MDP files.
    """
    diff_params = {}
    for i in range(len(mdp_list)):
        mdps = [MDP(mdp_list[i]) for i in range(len(mdp_list))]
        params_dicts = [odict([(k.replace('-', '_'), v) if type(v) is not str else (k.replace('-', '_'), v.replace('-', '_')) for k, v in p.items()]) for p in mdps]  # noqa: E501

        # First figure out the union set of the parameters and exclude blanks and comments
        all_params = set([key for d in params_dicts for key in d.keys()])
        all_params = [p for p in all_params if not p.startswith(('B', 'C'))]

        for p in all_params:
            if p in diff_params:
                pass  # already in the dictionary, no need to compare again
            else:
                if not all(p in d for d in params_dicts):
                    # the parameter is not in all MDP files
                    diff_params[p] = [d[p] if p in d else None for d in params_dicts]
                else:
                    # the parameter is in all MDP files (Note that "set([1, 1, 1]={1}.)")
                    if isinstance(params_dicts[0][p], list):
                        # the parameter is a list, which is unhashable
                        if len(set([tuple(d[p]) for d in params_dicts])) > 1:
                            diff_params[p] = [d[p] for d in params_dicts]
                    else:
                        if len(set([d[p] for d in params_dicts])) > 1:
                            diff_params[p] = [d[p] for d in params_dicts]

    if print_diff:
        print("The following parameters are different among the MDP files:")
        for k, v in diff_params.items():
            print(k)
            for i in range(len(mdp_list)):
                print(f'  - {mdp_list[i]}: {v[i]}')
            print()

    return diff_params
