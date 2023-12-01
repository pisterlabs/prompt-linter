import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def online_variance(data):
    '''
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    '''
    n = 0
    mean = M2 = 0.0

    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2

    if n < 2:
        return float('nan')
    else:
        return M2 / n


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def explained_variance(ypred, y):
    """
    *** copied from openai/baselines ***
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def add_weight_decay(nets, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for net in nets:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if "bias" in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


def model_diff_lr(nets, lr_l, lr_s):
    large_lr, small_lr = [], []
    for net in nets:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if 'embedding.weight' in name:
                large_lr.append(param)
            else:
                small_lr.append(param)
    return [{'params': large_lr, 'lr': lr_l},
            {'params': small_lr, 'lr': lr_s}]


def print_red(skk):
    print("\033[91m {}\033[00m".format(skk))


def print_green(skk):
    print("\033[92m {}\033[00m".format(skk))


def print_yellow(skk):
    print("\033[93m {}\033[00m".format(skk))


def print_blue(skk):
    print("\033[94m {}\033[00m".format(skk))


def print_purple(skk):
    print("\033[95m {}\033[00m".format(skk))


def print_cyan(skk):
    print("\033[96m {}\033[00m".format(skk))
