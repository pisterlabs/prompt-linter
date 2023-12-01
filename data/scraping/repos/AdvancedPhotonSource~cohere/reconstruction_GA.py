# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_GA
=============================

This module controls a reconstructions using genetic algorithm (GA).
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI x or using command line scripts x.
"""

import numpy as np
import os
import cohere_core.controller.reconstruction_multi as multi
import cohere_core.utilities.utils as ut
from multiprocessing import Process, Queue
import shutil
import importlib
import cohere_core.controller.phasing as calc


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


class Tracing:
    def __init__(self, reconstructions, pars, dir):
        self.init_dirs = []
        self.report_tracing = []

        if 'init_guess' not in pars:
            pars['init_guess'] = 'random'
        if pars['init_guess'] == 'continue':
            continue_dir = pars['continue_dir']
            for sub in os.listdir(continue_dir):
                image, support, coh = ut.read_results(continue_dir + '/' + sub + '/')
                if image is not None:
                    self.init_dirs.append(continue_dir + '/' + sub)
                    self.report_tracing.append([continue_dir + '/' + sub])
            if len(self.init_dirs) < reconstructions:
                for i in range(reconstructions - len(self.init_dirs)):
                    self.report_tracing.append(['random' + str(i)])
                self.init_dirs = self.init_dirs + (reconstructions - len(self.init_dirs)) * [None]
        elif pars['init_guess'] == 'AI_guess':
            import cohere_core.controller.AI_guess as ai

            self.report_tracing.append(['AI_guess'])
            for i in range(reconstructions - 1):
                self.report_tracing.append(['random' + str(i)])
            # ai_dir = ai.start_AI(pars, datafile, dir)
            # if ai_dir is None:
            #     return
            self.init_dirs = [dir + '/results_AI'] + (reconstructions - 1) * [None]
        else:
            for i in range(reconstructions):
                self.init_dirs.append(None)
                self.report_tracing.append(['random' + str(i)])


    def set_map(self, map):
        self.map = map


    def append_gen(self, gen_ranks):
        for key in gen_ranks:
            self.report_tracing[self.map[key]].append(gen_ranks[key])


    def pretty_format_results(self):
        """
        Takes in a list of report traces and formats them into human-readable tables.
        Performs data conversion in 1 pass and formatting in a second to determine
        padding and spacing schemes.

        Parameters
        ----------
        none

        Returns
        -------
            report_output : str
            a string containing the formatted report
        """
        col_gap = 2

        num_gens = len(self.report_tracing[0]) - 1
        fitnesses = list(self.report_tracing[0][1][1].keys())

        report_table = []
        report_table.append(['start'] + [f'generation {i}' for i in range(num_gens)])
        report_table.append([''] * len(report_table[0]))

        data_col_width = 15
        start_col_width = 15
        for pop_data in self.report_tracing:
            report_table.append([str(pop_data[0])] + [str(ind_data[0]) for ind_data in pop_data[1:]])
            start_col_width = max(len(pop_data[0]), start_col_width)

            for fit in fitnesses:
                fit_row = ['']
                for ind_data in pop_data[1:]:
                    data_out = f'{fit} : {ind_data[1][fit]}'
                    data_col_width = max(len(data_out), data_col_width)
                    fit_row.append(data_out)
                report_table.append(fit_row)
            report_table.append([''] * len(report_table[0]))

        report_str = ''
        for row in report_table:
            report_str += row[0].ljust(start_col_width + col_gap)
            report_str += (' ' * col_gap).join([cell.ljust(data_col_width) for cell in row[1:]]) + '\n'

        return report_str


    def save(self, save_dir):
        try:
            report_str = self.pretty_format_results()
        except Exception as e:
            print(f'WARNING: Report formatting failed due to {type(e)}: {e}! Falling back to raw formatting.')
            report_str = '\n'.join([str(l) for l in self.report_tracing])

        with open(save_dir + '/ranks.txt', 'w+') as rank_file:
            rank_file.write(report_str)
            rank_file.flush()


def set_lib(pkg):
    global dvclib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    elif pkg == 'torch':
        devlib = importlib.import_module('cohere_core.lib.torchlib').torchlib
    calc.set_lib(devlib)


def set_ga_defaults(pars):
    if 'reconstructions' not in pars:
        pars['reconstructions'] = 1

    if 'ga_generations' not in pars:
        pars['ga_generations'] = 1

    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'

    # check if pc feature is on
    if 'pc' in pars['algorithm_sequence'] and 'pc_interval' in pars:
        if not 'ga_gen_pc_start' in pars:
            pars['ga_gen_pc_start'] = 0
            pars['ga_gen_pc_start'] = min(pars['ga_gen_pc_start'], pars['ga_generations']-1)

    if 'ga_fast' not in pars:
        pars['ga_fast'] = False

    if 'ga_metrics' not in pars:
        metrics = ['chi'] * pars['ga_generations']
    else:
        metrics = pars['ga_metrics']
        if len(metrics) == 1:
            metrics = metrics * pars['ga_generations']
        elif len(metrics) < pars['ga_generations']:
            metrics = metrics + ['chi'] * (pars['ga_generations'] - len(metrics))
    pars['ga_metrics'] = metrics

    ga_reconstructions = []
    if 'ga_cullings' in pars:
        worst_remove_no = pars['ga_cullings']
        if len(worst_remove_no) < pars['ga_generations']:
            worst_remove_no = worst_remove_no + [0] * (pars['ga_generations'] - len(worst_remove_no))
    else:
        worst_remove_no = [0] * pars['ga_generations']
    pars['worst_remove_no'] = worst_remove_no
    # calculate how many reconstructions should continue
    reconstructions = pars['reconstructions']
    for culling in worst_remove_no:
        reconstructions = reconstructions - culling
        if reconstructions <= 0:
            return 'culled down to 0 reconstructions, check configuration'
        ga_reconstructions.append(reconstructions)
    pars['ga_reconstructions'] = ga_reconstructions

    if 'shrink_wrap_threshold' in pars:
        sw_threshold = pars['shrink_wrap_threshold']
    else:
        sw_threshold = .1
    if 'ga_sw_thresholds' in pars:
        ga_sw_thresholds = pars['ga_sw_thresholds']
        if len(ga_sw_thresholds) == 1:
            ga_sw_thresholds = ga_sw_thresholds * pars['ga_generations']
        elif len(ga_sw_thresholds) < pars['ga_generations']:
            ga_sw_thresholds = ga_sw_thresholds + [sw_threshold] * (pars['ga_generations'] - len(ga_sw_thresholds))
    else:
        ga_sw_thresholds = [sw_threshold] * pars['ga_generations']
    pars['ga_sw_thresholds'] = ga_sw_thresholds

    if 'sw_gauss_sigma' in pars:
        sw_gauss_sigma = pars['sw_gauss_sigma']
    else:
        sw_gauss_sigma = .1
    if 'ga_sw_gauss_sigmas' in pars:
        ga_sw_gauss_sigmas = pars['ga_sw_gauss_sigmas']
        if len(ga_sw_gauss_sigmas) == 1:
            ga_sw_gauss_sigmas = ga_sw_gauss_sigmas * pars['ga_generations']
        elif len(pars['ga_sw_gauss_sigmas']) < pars['ga_generations']:
            ga_sw_gauss_sigmas = ga_sw_gauss_sigmas + [sw_gauss_sigma] * (pars['ga_generations'] - len(ga_sw_gauss_sigmas))
    else:
        ga_sw_gauss_sigmas = [sw_gauss_sigma] * pars['ga_generations']
    pars['ga_sw_gauss_sigmas'] = ga_sw_gauss_sigmas

    if 'ga_breed_modes' not in pars:
        ga_breed_modes = ['sqrt_ab'] * pars['ga_generations']
    else:
        ga_breed_modes = pars['ga_breed_modes']
        if len(ga_breed_modes) == 1:
            ga_breed_modes = ga_breed_modes * pars['ga_generations']
        elif len(ga_breed_modes) < pars['ga_generations']:
            ga_breed_modes = ga_breed_modes + ['sqrt_ab'] * (pars['ga_generations'] - len(ga_breed_modes))
    pars['ga_breed_modes'] = ga_breed_modes

    if 'ga_lpf_sigmas' in pars:
        pars['low_resolution_generations'] = len(pars['ga_lpf_sigmas'])
    else:
        pars['low_resolution_generations'] = 0

    if pars['low_resolution_generations'] > 0:
        if 'low_resolution_alg' not in pars:
            pars['low_resolution_alg'] = 'GAUSS'

    print()

    return pars


def order_dirs(tracing, dirs, evals, metric_type):
    """
    Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

    Parameters
    ----------
    tracing : object
        Tracing object that keeps fields related to tracing
    dirs : list
        list of directories where the reconstruction results files are saved
    evals : list
        list of evaluations of the results saved in the directories matching dirs list. The evaluations are dict
        <metric type> : <eval result>
    metric_type : metric type to be applied for evaluation

    Returns
    -------
    list :
        a list of directories where results are saved ordered from best to worst
    dict :
        evaluations of the best results
    """
    # ranks keeps indexes of reconstructions from best to worst
    # for most of the metric types the minimum of the metric is best, but for
    # 'summed_phase' and 'area' it is opposite, so reversing the order
    metric_evals = [e[metric_type] for e in evals]
    ranks = np.argsort(metric_evals).tolist()
    if metric_type == 'summed_phase' or metric_type == 'area':
        ranks.reverse()
    best_metrics = evals[ranks[0]]

    # Add tracing for the generation results
    gen_ranks = {}
    for i in range(len(evals)):
        gen_ranks[ranks[i]] = (i, evals[ranks[i]])
    tracing.append_gen(gen_ranks)

    # find how the directories based on ranking, map to the initial start
    prev_map = tracing.map
    map = {}
    for i in range(len(ranks)):
        prev_seq = int(os.path.basename(dirs[ranks[i]]))
        inx = prev_map[prev_seq]
        map[i] = inx
    prev_map.clear()
    tracing.set_map(map)

    # order directories by rank
    rank_dirs = []
    # append "_<rank>" to each result directory name
    for i in range(len(ranks)):
        dest = dirs[ranks[i]] + '_' + str(i)
        src = dirs[ranks[i]]
        shutil.move(src, dest)
        rank_dirs.append(dest)

    # remove the number preceding rank from each directory name, so the directories are numbered
    # according to rank
    current_dirs = []
    for dir in rank_dirs:
        last_sub = os.path.basename(dir)
        parent_dir = os.path.dirname(dir).replace(os.sep, '/')
        dest = parent_dir + '/' + last_sub.split('_')[-1]
        shutil.move(dir, dest)
        current_dirs.append(dest)
    return current_dirs, best_metrics


def order_processes(tracing, proc_metrics, metric_type):
    """
    Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

    Parameters
    ----------
    tracing : object
        Tracing object that keeps fields related to tracing
    proc_metrics : dict
        dictionary of <pid> : <evaluations> ; evaluations of the results saved in the directories matching dirs list. The evaluations are dict
        <metric type> : <eval result>
    metric_type : metric type to be applied for evaluation

    Returns
    -------
    list :
        a list of processes ids ordered from best to worst by the results the processes delivered
    dict :
        evaluations of the best results
    """
    proc_eval = [(key, proc_metrics[key][metric_type]) for key in proc_metrics.keys()]
    ranked_proc = sorted(proc_eval, key=lambda x: x[1])

    # ranks keeps indexes of reconstructions from best to worstpro
    # for most of the metric types the minimum of the metric is best, but for
    # 'summed_phase' and 'area' it is oposite, so reversing the order
    if metric_type == 'summed_phase' or metric_type == 'area':
        ranked_proc.reverse()
    gen_ranks = {}
    for i in range(len(ranked_proc)):
        pid = ranked_proc[i][0]
        gen_ranks[pid] = (i, proc_metrics[pid])
    tracing.append_gen(gen_ranks)
    return ranked_proc, proc_metrics[ranked_proc[0][0]]


def cull(lst, no_left):
    if len(lst) <= no_left:
        return lst
    else:
        return lst[0:no_left]


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    Controls reconstruction that employs genetic algorith (GA).

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the configuration file defines whether it is a random guess, AI algorithm determined (one reconstruction, the rest random), or starting from some saved state. It will set the initial guess accordingly and start GA algorithm. It will run multiple reconstructions for each generation in a loop. After each generation the best reconstruction, alpha is identified, and used for breeding. For each generation the results will be saved in g_x subdirectory, where x is the generation number, in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    """
    pars = ut.read_config(conf_file)
    pars = set_ga_defaults(pars)

    if pars['ga_generations'] < 2:
        print("number of generations must be greater than 1")
        return

    if pars['reconstructions'] < 2:
        print("GA not implemented for a single reconstruction")
        return

    if pars['ga_fast']:
        reconstructions = min(pars['reconstructions'], int(len(devices)/2))
    else:
        reconstructions = pars['reconstructions']
    print('GA starting', reconstructions, 'reconstructions')
    if 'ga_cullings' in pars:
        cull_sum = sum(pars['ga_cullings'])
        if reconstructions - cull_sum < 2:
            print("At least two reconstructions should be left after culling. Number of starting reconstructions is", reconstructions, "but ga_cullings adds to", cull_sum)
            return

    if pars['init_guess'] == 'AI_guess':
        # run AI part first
        import cohere_core.controller.AI_guess as ai
        ai_dir = ai.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = dir + '/' + filename.replace('config_rec', 'results_phasing')

    # create alpha dir and placeholder for the alpha's metrics
    alpha_dir = save_dir + '/alpha'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(alpha_dir):
        os.mkdir(alpha_dir)

    tracing = Tracing(reconstructions, pars, dir)

    if pars['ga_fast']:  # the number of processes is the same as available GPUs (can be same GPU if can fit more recs)
        set_lib(lib)

        workers = []
        processes = {}
        tracing_map = {}
        init_dirs = {}

        for i in range(reconstructions):
            workers.append(calc.Rec(pars, datafile))
            worker_qin = Queue()
            worker_qout = Queue()
            process = Process(target=workers[i].fast_ga, args=(worker_qin, worker_qout))
            process.start()
            processes[process.pid] = [worker_qin, worker_qout]
            init_dirs[process.pid] = tracing.init_dirs[i]
            tracing_map[process.pid] = i
        tracing.set_map(tracing_map)

        def handle_cmd():
            bad_processes = []
            for pid in processes:
                worker_qout = processes[pid][1]
                ret = worker_qout.get()
                if ret < 0:
                    worker_qin = processes[pid][0]
                    worker_qin.put('done')
                    bad_processes.append(pid)
            # remove bad processes from dict (in the future we may reuse them)
            for pid in bad_processes:
                processes.pop(pid)

        for g in range(pars['ga_generations']):
            print ('starting generation', g)
            if g == 0:
                for pid in processes:
                    worker_qin = processes[pid][0]
                    worker_qin.put(('init_dev', devices.pop()))
                handle_cmd()
                for pid in processes:
                    worker_qin = processes[pid][0]
                    worker_qin.put(('init', init_dirs[pid], alpha_dir, g))
                handle_cmd()
            else:
                for pid in processes:
                    worker_qin = processes[pid][0]
                    worker_qin.put(('init', None, alpha_dir, g))
                handle_cmd()
            if g > 0:
                for pid in processes:
                    worker_qin = processes[pid][0]
                    worker_qin.put('breed')
                handle_cmd()

            for pid in processes:
                worker_qin = processes[pid][0]
                worker_qin.put('iterate')
            handle_cmd()
            # get metric, i.e the goodness of reconstruction from each run
            proc_metrics = {}
            metric_type = pars['ga_metrics'][g]
            for pid in processes:
                worker_qin = processes[pid][0]
                worker_qin.put(('get_metric', metric_type))
            for pid in processes:
                worker_qout = processes[pid][1]
                metric = worker_qout.get()
                proc_metrics[pid] = metric
            # order processes by metric
            proc_ranks, best_metrics = order_processes(tracing, proc_metrics, metric_type)
            # cull
            culled_proc_ranks = cull(proc_ranks, pars['ga_reconstructions'][g])
            # remove culled processes from list (in the future we may reuse them)
            for i in range(len(culled_proc_ranks), len(proc_ranks)):
                pid = proc_ranks[i][0]
                worker_qin = processes[pid][0]
                worker_qin.put('done')
                processes.pop(pid)

            # compare current alpha and previous. If previous is better, set it as alpha.
            if (g == 0
                    or
                    best_metrics[metric_type] >= alpha_metrics[metric_type] and
                    (metric_type == 'summed_phase' or metric_type == 'area')
                    or
                    best_metrics[metric_type] < alpha_metrics[metric_type] and
                    (metric_type == 'chi' or metric_type == 'sharpness')):
                pid = culled_proc_ranks[0][0]
                worker_qin = processes[pid][0]
                worker_qin.put(('save_res', alpha_dir, True))
                worker_qout = processes[pid][1]
                ret = worker_qout.get()
                alpha_metrics = best_metrics

            # save results, we may modify it later to save only some
            gen_save_dir = save_dir + '/g_' + str(g)
            if g == pars['ga_generations'] -1:
                for i in range(len(culled_proc_ranks)):
                    pid = culled_proc_ranks[i][0]
                    worker_qin = processes[pid][0]
                    worker_qin.put(('save_res', gen_save_dir + '/' + str(i)))
                for pid in processes:
                    worker_qout = processes[pid][1]
                    ret = worker_qout.get()
            if len(processes) == 0:
                break
        for pid in processes:
            worker_qin = processes[pid][0]
            worker_qin.put('done')
    else:   # not fast GA
        q = Queue()
        prev_dirs = tracing.init_dirs
        tracing.set_map({i:i for i in range(len(prev_dirs))})
        rec = multi
        for g in range(pars['ga_generations']):
            # delete previous-previous generation
            if g > 1:
                shutil.rmtree(save_dir + '/g_' + str(g-2))
            print ('starting generation', g)
            gen_save_dir = save_dir + '/g_' + str(g)
            metric_type = pars['ga_metrics'][g]
            reconstructions = len(prev_dirs)
            p = Process(target=rec.multi_rec, args=(lib, gen_save_dir, devices, reconstructions, pars, datafile, prev_dirs, metric_type, g, alpha_dir, q))
            p.start()
            p.join()

            results = q.get()
            evals = []
            temp_dirs = []
            for r in results:
                eval = r[0]
                if eval is not None:
                    evals.append(eval)
                    temp_dirs.append(r[1])

            # results are saved in a list of directories - save_dir
            # it will be ranked, and moved to temporary ranked directories
            current_dirs, best_metrics = order_dirs(tracing, temp_dirs, evals, metric_type)
            reconstructions = pars['ga_reconstructions'][g]
            current_dirs = cull(current_dirs, reconstructions)
            prev_dirs = current_dirs

            # compare current alpha and previous. If previous is better, set it as alpha.
            # no need toset alpha  for last generation
            if (g == 0
                    or
                    best_metrics[metric_type] >= alpha_metrics[metric_type] and
                    (metric_type == 'summed_phase' or metric_type == 'area')
                    or
                    best_metrics[metric_type] < alpha_metrics[metric_type] and
                    (metric_type == 'chi' or metric_type == 'sharpness')):
                shutil.copyfile(current_dirs[0] + '/image.npy', alpha_dir + '/image.npy')
                alpha_metrics = best_metrics
        # remove the previous gen
        shutil.rmtree(save_dir + '/g_' + str(pars['ga_generations'] - 2))

    tracing.save(save_dir)

    print('done gen')