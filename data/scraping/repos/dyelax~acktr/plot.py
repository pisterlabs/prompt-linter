# Adapted from OpenAI Baselines

import argparse
import numpy as np
from glob import glob
import matplotlib
import os
matplotlib.use('TkAgg') # Can change to 'Agg' for non-interactive mode

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from monitor import load_results
from utils import get_dir

EPISODES_WINDOW = 10
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window-1:], yw_func


def ts2xy(ts):
    x = np.cumsum(ts.l.values)
    y = np.array(ts.r.values)
    return x, y


def plot_curve(xs, ys, color, label, zorder):
    ymeans = np.mean(ys, axis=0)
    ystds = np.std(ys, axis=0)

    xs, ymeans_rolling = window_func(xs, ymeans, EPISODES_WINDOW, np.mean)
    _, ystds_rolling = window_func(xs, ystds, EPISODES_WINDOW, np.mean)

    plt.plot(xs, ymeans_rolling, color=color, label=label, zorder=zorder)
    plt.fill_between(xs, ymeans_rolling - ystds_rolling, ymeans_rolling + ystds_rolling,
                     facecolor=color, color=color, alpha=0.5, interpolate=True, zorder=zorder)

    # Without rolling average
    # plt.plot(xs, ymeans)
    # plt.fill_between(xs, ymeans - ystds, ymeans + ystds,
    #                  facecolor=color, color=color, alpha=0.5, interpolate=True)


def plot_results(dirs, num_timesteps, color, label, zorder=0):
    tslist = []
    for dir in dirs:
        ts = load_results(dir)
        ts = ts[ts.l.cumsum() <= num_timesteps * 1.1]
        tslist.append(ts)

    runs = [ts2xy(ts) for ts in tslist]

    # Linearly interpolate between data points so we can compare multiple runs
    # with different episode steps.
    xs = np.linspace(1, num_timesteps, 1000)
    ys = np.array([np.interp(xs, run[0], run[1]) for run in runs])

    plot_curve(xs, ys, color, label, zorder)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_path',
                        help='The path to save the figure to',
                        default='./save/presentation/plots/pong.png')
                        # default='./save/presentation/plots/breakout.png')
    parser.add_argument('--ours_dirs',
                        help='List of log directories for our ACKTR implementation',
                        nargs = '*',
                        default=['./save/ours/pong/softmax/*/results'])
                        # default=['./save/ours/breakout/final-old/*/results'])
    parser.add_argument('--acktr_dirs',
                        help='List of log directories for the OpenAI Baselines ACKTR implementation',
                        nargs = '*',
                        default=['./save/baselines/pong/acktr/*'])
                        # default=['./save/baselines/breakout/acktr/*'])
    parser.add_argument('--trpo_dirs',
                        help='List of log directories for the OpenAI Baselines TRPO implementation',
                        nargs = '*',
                        default=['./save/baselines/pong/trpo/*'])
                        # default=['./save/baselines/breakout/trpo/*'])
    parser.add_argument('--a2c_dirs',
                        help='List of log directories for the OpenAI Baselines A2C implementation',
                        nargs = '*',
                        default=['./save/baselines/pong/a2c/*'])
                        # default=['./save/baselines/breakout/a2c/*'])
    parser.add_argument('--ours_color',
                        help='The color with which to plot our ACKTR results',
                        default='magenta')
    parser.add_argument('--acktr_color',
                        help='The color with which to plot the OpenAI Baselines ACKTR results',
                        default='blue')
    parser.add_argument('--trpo_color',
                        help='The color with which to plot the OpenAI Baselines TRPO results',
                        default='green')
    parser.add_argument('--a2c_color',
                        help='The color with which to plot the OpenAI Baselines A2C results',
                        default='orange')
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--title', help = 'Title of plot', default = 'Pong')
    # parser.add_argument('--title', help = 'Title of plot', default = 'Breakout')
    args = parser.parse_args()

    get_dir(os.path.dirname(args.save_path))

    plt.figure(figsize=(16, 9))
    plt.xlim(0, args.num_timesteps)
    plt.xticks(range(1000000, 11000000 , 1000000), [str(i) + 'M' for i in range(1, 11)])
    plt.title(args.title)
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Rewards")

    ours_dirs = []
    for dir in args.ours_dirs: ours_dirs += glob(dir)
    plot_results(ours_dirs, args.num_timesteps, args.ours_color, 'ACKTR (Ours)', zorder=1)

    acktr_dirs = []
    for dir in args.acktr_dirs: acktr_dirs += glob(dir)
    plot_results(acktr_dirs, args.num_timesteps, args.acktr_color, 'ACKTR')

    trpo_dirs = []
    for dir in args.trpo_dirs: trpo_dirs += glob(dir)
    plot_results(trpo_dirs, args.num_timesteps, args.trpo_color, 'TRPO')

    a2c_dirs = []
    for dir in args.a2c_dirs: a2c_dirs += glob(dir)
    plot_results(a2c_dirs, args.num_timesteps, args.a2c_color, 'A2C')

    legend = plt.legend(fancybox=True, loc='upper left')
    legend.get_frame().set_alpha(0.5)

    plt.tight_layout()
    # plt.show()

    plt.savefig(args.save_path)

if __name__ == '__main__':
    main()