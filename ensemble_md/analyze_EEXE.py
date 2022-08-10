"""
The :code:`analyze_EEXE` module provides methods for performing data analysis for EEXE.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from ensemble_md.exceptions import ParseError


def parse_transmtx(log_file):
    """
    Parses the log file to get the final theoretical state transition matrix and empirical state transition matrix.

    Parameters
    ----------
    log_file : str
        The log file to be parsed.

    Returns
    -------
    theoretical : np.array
        The final theoretical state transition matrix.
    empirical : np.array
        The final empirical state transition matrix.
    diff_matrix : np.array
        The difference between the theortial and empirical state transition matrix (theoretical - empirical).
    """
    f = open(log_file, "r")
    lines = f.readlines()
    f.close()

    lines.reverse()

    n = -1
    theoretical_found, empirical_found = False, False
    for l in lines:  # noqa: E741
        n += 1
        if "Empirical Transition Matrix" in l:  # This will be found first
            empirical_found = True
            n_states = int(lines[n - 1].split()[-1])
            experimental = np.zeros([n_states, n_states])
            for i in range(n_states):
                experimental[i] = [float(k) for k in lines[n - 2 - i].split()[:-1]]

        if "Transition Matrix" in l and "Empirical" not in l:
            theoretical_found = True
            theoretical = np.zeros([n_states, n_states])
            for i in range(n_states):
                theoretical[i] = [float(k) for k in lines[n - 2 - i].split()[:-1]]

        if theoretical_found is True and empirical_found is True:
            break

    if theoretical_found is False and empirical_found is False:
        raise ParseError(f"No transition matrices found in {log_file}.")

    diff_matrix = theoretical - experimental

    return theoretical, experimental, diff_matrix


def calc_overall_transmtx(self):
    pass


def plot_matrix(matrix, png_name, title=None, start_idx=0):
    """
    Visualizes a matrix a in heatmap.

    Parameters
    ----------
    matrix : np.array
        The matrix to be visualized
    png_name : str
        The file name of the output PNG file (including the extension).
    title : str
        The title of the plot
    start_idx : int
        The starting value of the state index
    """
    sns.set_context(
        rc={"family": "sans-serif", "sans-serif": ["DejaVu Sans"], "size": 5}
    )

    K = len(matrix)
    plt.figure(figsize=(K / 1.5, K / 1.5))
    annot_matrix = np.zeros([K, K])  # matrix for annotating values

    mask = []
    for i in range(K):
        mask.append([])
        for j in range(len(matrix[0])):
            if matrix[i][j] < 0.005:
                mask[-1].append(True)
            else:
                mask[-1].append(False)

    for i in range(K):
        for j in range(K):
            annot_matrix[i, j] = round(matrix[i, j], 2)

    x_tick_labels = y_tick_labels = np.arange(start_idx, start_idx + K)
    ax = sns.heatmap(
        matrix,
        cmap="YlGnBu",
        linecolor="silver",
        linewidth=0.25,
        annot=annot_matrix,
        square=True,
        mask=matrix < 0.005,
        fmt=".2f",
        cbar=False,
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    cmap = cm.get_cmap("YlGnBu")  # to get the facecolor
    ax.set_facecolor(cmap(0))  # use the brightest color (value = 0)
    for _, spine in ax.spines.items():
        spine.set_visible(True)  # add frames to the heat map
    plt.annotate("$\lambda$", xy=(0, 0), xytext=(-0.45, -0.20))  # noqa: W605
    if title is None:
        pass
    else:
        plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout(pad=1.0)

    plt.savefig(png_name, dpi=600)
    # plt.show()
    plt.close()


def calc_spectral_gap(matrix):
    pass


class StateDiffusivity:
    def __init__(self):
        pass


class FreeEnergyCalculation:
    def __init__(self):
        pass
