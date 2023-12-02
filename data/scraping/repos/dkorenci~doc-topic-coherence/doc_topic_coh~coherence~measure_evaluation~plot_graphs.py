from matplotlib import pyplot as plt
import pickle
from os import path

graphs2Test = [
    ('eval_graph_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'GRAPH-CNT'),
    ('eval_graph_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'GRAPH-EMBD'),
    ('eval_distance_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DISTANCE-CNT'),
    ('eval_distance_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DISTANCE-EMBD'),
    ('eval_density_params_corpus_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DENSITY-CNT'),
    ('eval_density_params_world_vectors_test_topic_split_[devSize=120]_[seed=9984].pickle', 'DENSITY-EMBD'),
]

graphs2TestCro = [
    ('eval_graph_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'GRAPH-CNT'),
    ('eval_graph_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'GRAPH-EMBD'),
    ('eval_distance_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DISTANCE-CNT'),
    ('eval_distance_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DISTANCE-EMBD'),
    ('eval_density_params_corpus_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DENSITY-CNT'),
    ('eval_density_params_world_vectors_croelect_topics_croelect_model1_croelect_model2_croelect_model3_croelect_model4.pickle', 'DENSITY-EMBD'),
]

from doc_topic_coh.coherence.measure_evaluation.model_selection import test
from doc_topic_coh.settings import experiments_folder as expFolder
from doc_topic_coh.coherence.measure_evaluation.evaluations import docCohBaseline, bestParamsDoc
from doc_topic_coh.coherence.measure_evaluation.evaluations_croelect import croelectTopics, bestParamsDocCroelect
import numpy as np

def algoClassBoxplots(lfiles, folder='/datafast/doc_topic_coherence/experiments/iter5_coherence/',
                      select=None, saveFile=None):
    results = []
    for lf in lfiles:
        r = pickle.load(open(path.join(folder, lf[0]), 'rb'))
        print '%15s num params: %d' % (lf[1], len(r))
        ordered = sorted(r, reverse=True)
        print ' '.join(('%.4f'%v) for v in ordered)
        selected = r[0]; best = ordered[0]
        print 'selected %.4f, best %.4f, diff %.4f' % (selected, best, abs(best-selected))
        results.append(r)
    fig, axes = plt.subplots(1, 1)
    axes.set_ylim([0.6, 0.85])
    axes.boxplot(results, showfliers=False)
    xcoord = range(1, len(results)+1) # x coordinates of boxes
    for i, res in enumerate(results):
        axes.scatter([xcoord[i]] * len(res), res, alpha=0.4)
        # plot the average of the top quartile
        q75 = np.percentile(res, 75)
        resq75 = [r for r in res if r >= q75]
        avgq75 = np.average(resq75)
        axes.plot(xcoord[i], avgq75, mec='blue', marker='o', mew=4, markersize=25, mfc="None")
        # plot marker on the top-dev result
        axes.plot(xcoord[i], res[0], color='r', marker='x', mew=6, markersize=30)
        if select:
            linex = xcoord[i]+0.35
            best = max(res)
            brackw = 0.03
            axes.plot([linex-brackw, linex, linex, linex, linex, linex-brackw ],
                      [select[i], select[i], select[i], best, best, best],
                      'r', linewidth=2)

    #axes.xaxis.tick_top()
    # Set the labels
    labels = [lf[1] for lf in lfiles]
    axes.set_xticklabels(labels, minor=False)
    for tick in axes.xaxis.get_major_ticks(): tick.label.set_fontsize(27)
    for tick in axes.yaxis.get_major_ticks(): tick.label.set_fontsize(27)
    #plt.xticks(rotation=45)
    axes.yaxis.grid(True)
    # Turn off x ticks
    for t in axes.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.tight_layout(pad=0)
    if saveFile: fig.savefig(filename=saveFile, box_inches='tight')
    else: plt.show()

def plotAucCurves(measureParams, ltopics, axLabels=None,
                  posClass=['theme', 'theme_noise'], grid=None,
                  repeat=None, baseline=None):
    from matplotlib import pyplot as plt
    from doc_topic_coh.coherence.tools import labelMatch
    from doc_topic_coh.coherence.coherence_builder import CoherenceFunctionBuilder
    from sklearn.metrics import roc_curve, precision_recall_curve
    # create measures from params
    cacheFolder = path.join(expFolder, 'function_cache')
    measures = []
    for p in measureParams:
        p['cache'] = cacheFolder
        measures.append(CoherenceFunctionBuilder(**p)())
    if baseline is not None:
        baseline = CoherenceFunctionBuilder(**baseline)()
    # init grid and plot
    if not grid: grid = (len(measures), 1)
    row, col = 0, 0
    fig, axes = plt.subplots(grid[0], grid[1])
    # render plots
    print len(measures)
    if repeat is not None:
        m = measures[repeat]
        cohrep = [ m(t) for t, tlabel in ltopics ]
        clasrep = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
    if baseline is not None:
        cohbase = [ baseline(t) for t, tlabel in ltopics ]
        clasbase = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
    for i, m in enumerate(measures):
        print row, col
        ax = axes[row, col]
        ax.yaxis.grid(True); ax.xaxis.grid(True)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        cohvals = [ m(t) for t, tlabel in ltopics ]
        classes = [ labelMatch(tlabel, posClass) for t, tlabel in ltopics ]
        pltpar = { 'linewidth': 2 }
        # ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off',
        #                 labelright='off', labelbottom='off')
        bott, top, left, right = 'off', 'off', 'off', 'off'
        # if col > 0 and row < grid[0]-1:
        #     ax.tick_params(axis='both', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        if col == 0: left = 'on'
        if col == grid[1]-1: right = 'on'
        if row == 0: top = 'on'
        if row == grid[0]-1: bott = 'on'
        ax.tick_params(axis='both',
                       left='off', bottom=bott, top='off', right=right,
                       labelleft='off', labelbottom=bott, labeltop='off', labelright=right)
        if top == 'on':
            ax.set_xlabel(axLabels[1][col], fontsize=25)
            ax.xaxis.set_label_position('top')
        if left == 'on':
            ax.set_ylabel(axLabels[0][row], fontsize=25)
            ax.yaxis.set_label_position('left')

        fpr, tpr, thresh = roc_curve(classes, cohvals, pos_label=1)
        print 'tpr: ', ','.join('%.3f'%v for v in tpr)
        print 'fpr: ', ','.join('%.3f'%v for v in fpr)
        ax.plot(fpr, tpr, **pltpar)
        if repeat is not None and repeat != i:
            fpr, tpr, thresh = roc_curve(clasrep, cohrep, pos_label=1)
            ax.plot(fpr, tpr, color='red', linewidth=2, linestyle=':')
        if baseline is not None:
            fpr, tpr, thresh = roc_curve(clasbase, cohbase, pos_label=1)
            ax.plot(fpr, tpr, color='green', linewidth=2, linestyle=':')

        ax.plot([0, 1], [0, 1], color='red', linewidth=0.5, linestyle='--')
        #for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(20)
        #for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(20)
        col += 1
        if col == grid[1]: col = 0; row += 1
    if axLabels:
        rows, cols = grid
        #for r in range(rows):
        #    axes[r, 0].
    plt.tight_layout(pad=0)
    plt.show()

def applyCoh(cohMeasure, labeledTopics):
    return [ cohMeasure(t) for t, tlabel in labeledTopics ]

def bestDocCohMeasuresAucCurves(grid = (2, 3), repeat=None, baseline=None):
    plotAucCurves(bestParamsDoc(False), test, grid=grid,
                    axLabels=(['CNT', 'EMB'], ['GRAPH', 'DISTANCE', 'DENSITY']),
                    repeat=repeat, baseline=baseline)

if __name__ == '__main__':
    algoClassBoxplots(graphs2Test, folder=expFolder, select=None)
    #algoClassBoxplots(graphs2TestCro, folder=expFolder, select=None)
    #bestDocCohMeasuresAucCurves((2,2), repeat=0, baseline=docCohBaseline)
    pass