import numpy as np
from scipy.signal import coherence as coherence_
from multiprocessing import Pool
from itertools import combinations
from functools import partial


N_JOBS = 3


def mutual_information(X, Y, bins=100):

    minX, maxX = X.min(), X.max()
    minY, maxY = X.min(), X.max()
    range1D = (min(minX, minY), max(maxX, maxY))
    range2D = (range1D, range1D)

    c_XY = np.histogram2d(X, Y, bins, range=range2D)[0]
    c_X = np.histogram(X, bins, range=range1D)[0]
    c_Y = np.histogram(Y, bins, range=range1D)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY

    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def correntropy(x, y, sigma=1):
    s = np.exp((-((x - y) ** 2)) / (2 * sigma ** 2)) / (sigma * np.sqrt(np.pi * 2))
    return s.mean()


def coherence(x, y):
    NPERSEG = 300
    return coherence_(
        x,
        y,
        fs=250.0,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NPERSEG // 2,
        nfft=NPERSEG,
    )


def apply_fn(args):
    x, y, func = args
    return func(x, y)


def apply_pairwise_parallel(arr, func=mutual_information):
    n = arr.shape[0]
    res_arr = []
    args = []
    for i1, i2 in combinations(range(n), 2):
        args.append((arr[i1, :], arr[i2, :], func))

    with Pool(N_JOBS) as pool:
        res_arr = pool.map(apply_fn, args)

    return np.array(res_arr).mean()


def apply_pairwise(arr, func=mutual_information):
    n = arr.shape[0]
    scores = list()
    for i1, i2 in combinations(range(n), 2):
        scores.append(
            func(arr[i1, :], arr[i2, :])
        )
    return np.array(scores).mean()


def promethee(metrics, w=None):
    final_score = 0
    n_rows, n_cols = metrics.shape
    if w is None:
        w = np.ones(n_cols)
    for c in range(n_cols):
        metric_col = metrics[:, [c]]
        score = (metric_col > metric_col.T).astype(int)
        final_score += score * w[c]
    
    final_score = final_score.sum(axis=1)
    return final_score / (n_cols * n_rows)


SCORING_FN_DICT = {
    "coherence": coherence,
    "correntropy_05": partial(correntropy, sigma=.5),
    "correntropy_1": correntropy,
    "correntropy_2": partial(correntropy, sigma=2.),
    "mutual_information": mutual_information,
}
