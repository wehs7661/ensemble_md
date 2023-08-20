import numpy as np
from ensemble_md.utils.utils import run_gmx_cmd


def cluster_traj(gmx_executable, inputs, grps, method='linkage', cutoff=0.1, suffix=None):
    """
    Performs clustering analysis on a trajectory using the GROMACS command :code:`gmx cluster`.
    Note that only fully coupled configurations are considered.

    Parameters
    ----------
    gmx_executable : str
        The path to the GROMACS executable.
    inputs : dict
        A dictionary that contains the file names of the input trajectory file (XTC or TRR),
        the configuration file (TPR or GRO), the file that contains the time series of the
        state index, and the index file (NDX). The must include the keys :code:`traj`, :code:`config`,
        :code:`xvg`, and :code:`index`. Note that the value for the key :code:`index` can be :code:`None`.
    grps : dict
        A dictionary that contains the names of the groups in the index file (NDX) for
        centering the system, calculating the RMSD, and outputting. The keys are
        :code:`center`, :code:`rmsd`, and :code:`output`.
    method : str
        The method for clustering available for the GROMACS command :code:`gmx cluster`. The default is 'linkage'.
    cutoff : float
        The cutoff in RMSD for clustering. The default is 0.1.
    suffix : str
        The suffix for the output files. The default is :code:`None`, which means no suffix will be added.
    """
    outputs = {
        'nojump': 'nojump.xtc',
        'center': 'center.xtc',
        'rmsd-clust': 'rmsd-clust.xpm',
        'rmsd-dist': 'rmsd-dist.xvg',
        'cluster-log': 'cluster.log',
        'cluster-pdb': 'clusters.pdb',
        'rmsd': 'rmsd.xvg',  # inter-medoid RMSD
    }
    if suffix is not None:
        for key in outputs:
            outputs[key] = outputs[key].replace('.', f'_{suffix}.')

    print('Eliminating jumps across periodic boundaries for the input trajectory ...')
    args = [
        gmx_executable, 'trjconv',
        '-f', inputs['traj'],
        '-s', inputs['config'],
        '-o', outputs['nojump'],
        '-center', 'yes',
        '-pbc', 'nojump',
        '-drop', inputs['xvg'],
        '-dropover', '0'
    ]
    if inputs['index'] is not None:
        args.extend(['-n', inputs['index']])
    returncode, stdout, stderr = run_gmx_cmd(args, prompt_input=f'{grps["center"]}\n{grps["output"]}\n')
    if returncode != 0:
        print(f'Error with return code: {returncode}):\n{stderr}')

    print('Centering the system ...')
    args = [
        gmx_executable, 'trjconv',
        '-f', outputs['nojump'],
        '-s', inputs['config'],
        '-o', outputs['center'],
        '-center', 'yes',
        '-pbc', 'mol',
        '-ur', 'compact',
    ]
    if inputs['index'] is not None:
        args.extend(['-n', inputs['index']])
    returncode, stdout, stderr = run_gmx_cmd(args, prompt_input=f'{grps["center"]}\n{grps["output"]}\n')
    if returncode != 0:
        print(f'Error with return code: {returncode}):\n{stderr}')

    print('Performing clustering analysis ...')
    args = [
        gmx_executable, 'cluster',
        '-f', outputs['center'],
        '-s', inputs['config'],
        '-o', outputs['rmsd-clust'],
        '-dist', outputs['rmsd-dist'],
        '-g', outputs['cluster-log'],
        '-cl', outputs['cluster-pdb'],
        '-cutoff', str(cutoff),
        '-method', method,
    ]
    if inputs['index'] is not None:
        args.extend(['-n', inputs['index']])
    returncode, stdout, stderr = run_gmx_cmd(args, prompt_input=f'{grps["rmsd"]}\n{grps["output"]}\n')
    if returncode != 0:
        print(f'Error with return code: {returncode}):\n{stderr}')

    rmsd_range, rmsd_avg, n_clusters = get_cluster_info(outputs['cluster-log'])

    print(f'Range of RMSD values: from {rmsd_range[0]:.3f} to {rmsd_range[1]:.3f} nm')
    print(f'Average RMSD: {rmsd_avg:.3f} nm')
    print(f'Number of clusters: {n_clusters}')

    if n_clusters > 1:
        clusters, sizes = get_cluster_members(outputs['cluster-log'])
        for i in range(1, n_clusters + 1):
            print(f'  - Cluster {i} accounts for {sizes[i] * 100:.2f}% of the total configurations.')

        n_transitions, t_transitions = count_transitions(clusters)
        print(f'Number of transitions between the two biggest clusters: {n_transitions}')
        print(f'Time frames of the transitions (ps): {t_transitions}')

        print('Calculating the inter-medoid RMSD between the two biggest clusters ...')
        # Note that we pass outputs['cluster-pdb'] to -s so that the first medoid will be used as the reference
        args = [
            gmx_executable, 'rms',
            '-f', outputs['cluster-pdb'],
            '-s', outputs['cluster-pdb'],
            '-o', outputs['rmsd'],
        ]
        if inputs['index'] is not None:
            args.extend(['-n', inputs['index']])

        # Here we simply assume same groups for least-squares fitting and RMSD calculation
        returncode, stdout, stderr = run_gmx_cmd(args, prompt_input=f'{grps["rmsd"]}\n{grps["rmsd"]}\n')
        if returncode != 0:
            print(f'Error with return code: {returncode}):\n{stderr}')

        rmsd = np.transpose(np.loadtxt(outputs['rmsd'], comments=['@', '#']))[1][1]  # inter-medoid RMSD
        print(f'Inter-medoid RMSD between the two biggest clusters: {rmsd:.3f} nm')


def get_cluster_info(cluster_log):
    """
    Gets the metadata of the LOG file generated by the GROMACS :code:`gmx cluster` command.

    Parameters
    ----------
    cluster_log : str
        The LOG file generated by the GROMACS :code:`gmx cluster` command.

    Returns
    -------
    rmsd_range: list
        The range of RMSD values
    rmsd_avg: float
        The average RMSD value.
    n_clusters : int
        The number of clusters.
    """
    f = open(cluster_log, 'r')
    lines = f.readlines()
    f.close()

    rmsd_range = []
    for line in lines:
        if 'The RMSD ranges from' in line:
            rmsd_range.append(float(line.split('from')[-1].split('to')[0]))
            rmsd_range.append(float(line.split('from')[-1].split('to')[-1].split('nm')[0]))
        if 'Average RMSD' in line:
            rmsd_avg = float(line.split('is')[-1])
        if 'Found' in line:
            n_clusters = int(line.split()[1])
            break

    return rmsd_range, rmsd_avg, n_clusters


def get_cluster_members(cluster_log):
    """
    Gets the members of each cluster from the LOG file generated by the GROMACS :code:`gmx cluster` command.

    Parameters
    ----------
    cluster_log : str
        The LOG file generated by the GROMACS :code:`gmx cluster` command.

    Returns
    -------
    clusters : dict
        A dictionary that contains the cluster index (starting from 1) as the key and the list of members
        (configurations at different timeframes) as the value.
    sizes : dict
        A dictionary that contains the cluster index (starting from 1) as the key and the size of the cluster
        (in fraction) as the value.
    """
    clusters = {}
    current_cluster = 0
    start_processing = False

    f = open(cluster_log, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        # Start processing when we reach the line that starts with "cl."
        if line.strip().startswith("cl."):
            start_processing = True
            continue  # Skip this line and continue to the next iteration

        if start_processing:
            parts = line.split('|')
            try:
                current_cluster = int(parts[0].strip())
                clusters[current_cluster] = []
            except ValueError:
                pass

            # This is either a new cluster or continuation of it, add members
            members = parts[-1].split()
            clusters[current_cluster].extend([int(i) for i in members])

    sizes_list = [len(clusters[i]) for i in clusters]
    sizes = {i: sizes_list[i - 1] / sum(sizes_list) for i in clusters}

    return clusters, sizes


def count_transitions(clusters):
    """
    Counts the number of transitions between the two biggest clusters.

    Parameters
    ----------
    clusters : dict
        A dictionary that contains the cluster index (starting from 1) as the key and the list of members
        (configurations at different timeframes) as the value.

    Returns
    -------
    n_transitions : int
        The number of transitions between the two biggest clusters.
    t_transitions : list
        The list of time frames when the transitions occur.
    """
    # Combine and sort all cluster members for the first two biggest clusters while keeping track of their origin
    all_members = [(member, 1) for member in clusters[1]] + [(member, 2) for member in clusters[2]]
    all_members.sort()

    # Count transitions and record time frames
    n_transitions = 0
    t_transitions = []
    last_cluster = all_members[0][1]  # the cluster index of the last time frame in the previous iteration
    for member in all_members[1:]:
        if member[1] != last_cluster:
            n_transitions += 1
            last_cluster = member[1]
            t_transitions.append(member[0])

    return n_transitions, t_transitions
