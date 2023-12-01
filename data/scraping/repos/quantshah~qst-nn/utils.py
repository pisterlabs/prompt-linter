import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np

import numpy as np
import os

import h5py

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


import tensorflow as tf

from scipy.special import factorial, binom
from scipy.sparse import lil_matrix, csr_matrix
from scipy.integrate import ode


from qutip.wigner import qfunc
from qutip.visualization import plot_fock_distribution
from qutip import Qobj
from qutip import coherent_dm, destroy, mesolve, Options, fock_dm, coherent, expect
from qutip.visualization import plot_wigner_fock_distribution, plot_fock_distribution
from qutip.wigner import qfunc, wigner
from qutip import Qobj, qeye
from qutip.states import enr_state_dictionaries
from qutip.superoperator import liouvillian, spre, spost
from qutip import liouvillian, mat2vec, state_number_enumerate
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result, Stats
from qutip import mesolve


from math import sqrt


from qutip import Qobj
from qutip.states import fock_dm, thermal_dm, coherent_dm, coherent, basis, fock
from qutip.operators import displace
from qutip import Qobj
from qutip.random_objects import rand_dm



import matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation

np.random.seed(42)




fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {# 'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.labelpad': 1,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
figpath = "figures/"

# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, cmap=None, ax=None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])

    ax.fill(xcorners, ycorners,
             color=cmap(int((w + w_max) * 256 / (2 * w_max))))



def show_image_grid(images, labels, title=None,
                    xvec=None, yvec=None):
  """
  Show a set of images in a grid
  
  
  Args:
      images (list/array): Image list or array with shape (n, k, k, c).
                           where the n should be the square of some number to
                           fit into the grid.
      labels (array [str]): A set of label strings.
      title (str, optional): Title of the plot.
      xvec, yvec (array, optional): The x(y) vectors for pcolor to plot.
                                    It will be set to `np.linspace(-5, 5, 32)`
  """
  if xvec is None:
    xvec = np.linspace(-5, 5, 32)
  if yvec is None:
    yvec = np.linspace(-5, 5, 32)

  image_grid_size = int(np.sqrt(len(images)))
  num_images = image_grid_size**2
  fig = plt.figure(figsize=(10, 6), constrained_layout=False)
  image_grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(image_grid_size, image_grid_size),
                         cbar_mode="edge",
                         cbar_pad = 0.1,
                         axes_pad=0.3,)
                   

  for i in range(num_images):
      ax, im, label = image_grid[i], images[i], labels[i]

      im = ax.pcolor(xvec, yvec, im/np.max(im), vmin=0, vmax=1, cmap="hot")
      ax.set_title(label)
      ax.cax.colorbar(im, ticks=[0, 0.5, 1])
      ax.set_ylabel(r"Im($\beta$)")
      ax.set_xlabel(r"Re($\beta$)")
  if title is not None:
    plt.suptitle(title)
  plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fig = None,
                          ax=None,
                          cax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
#     # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    if fig == None:
        fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if cax == None:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
    else:
        cbar = fig.colorbar(im, cax=cax, pad=0.03, ticks=[0, 0.5, 1])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_title(title, loc='center')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if (cm[i, j]  == 0.):
                l = str(0)
            else:
                l = format(cm[i, j], fmt)
            if float(l) < 1e-3:    
                l = str(0)

            if float(l) == 1.:    
                l = str(1)
            ax.text(j, i, l, fontsize=6,
                    ha="center", va="center",
                    color="w" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_three_husimi(d1, d2, d3, title="", subtitles=None,
  cmap=None, xvec=None, yvec=None, norm=None, normalize=True,
  cbar_ticks=None, cbar_ticklabels = None):
    """
    Plots three Husimi Q side by side
    """
    if xvec is None:
      xvec = np.linspace(-5, 5, 32)
    if yvec is None:
      yvec = np.linspace(-5, 5, 32)

    fig, ax = plt.subplots(1, 3, figsize=(fig_width, 0.35*fig_width), sharey=True, sharex=True)

    if norm is None:
      norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    if cmap is None:
      colors1 = colors.LinearSegmentedColormap.from_list("", [(0, "white"),
                                                        (1, "red"),
                                                       ])(np.linspace(0, 1, 256))

      cmap = colors.LinearSegmentedColormap.from_list('my_colormap', colors1)
    if normalize:
      d1 = d1/np.max(d1)
      d2 = d2/np.max(d2)
      d3 = d3/np.max(d3)

    im = ax[0].pcolor(xvec, yvec, d1,
                      norm=norm,
                      cmap=cmap)
    
    im = ax[1].pcolor(xvec, yvec, d2,
                      norm=norm,
                      cmap=cmap)
    
    im = ax[2].pcolor(xvec, yvec, d3,
                      norm=norm,
                      cmap=cmap)
    
    # ax[0].set_yticklabels(["-5", "", "5"])

    for axis in ax:
        axis.set_xticks([-xvec[-1], 0,  xvec[-1]])
        axis.set_yticks([-yvec[-1], 0, yvec[-1]])
        axis.set_xlabel(r"Re($\beta$)", labelpad=-6)    
        axis.set_aspect("equal")
    ax[0].set_xticklabels(["{:.0f}".format(-xvec[-1]), "", "{:.0f}".format(xvec[-1])])
    ax[0].set_yticklabels(["{:.0f}".format(-yvec[-1]), "", "{:.0f}".format(yvec[-1])])

    for i in range(0, 3):
        # ax[i].set_xticklabels(["-5", "", "5"])
        # ax[i].set_xticklabels(["-5", "", "5"])

        if subtitles != None:
            ax[i].set_title(subtitles[i])

    ax[0].set_ylabel(r"Im($\beta$)", labelpad=-9)
    # ax[0].set_yticklabels(["-5", "", "5"])

    fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)

    cax = fig.add_axes([0.864, 0.19, 0.01, 0.63])

    if cbar_ticks is None:
      cbar_ticks = [0, 0.5, 1]
      cbar_ticklabels = ["0", 0.5, "1"] 
    fig.colorbar(im, cax=cax, ticks=cbar_ticks)
    cax.set_yticklabels(cbar_ticklabels)
    plt.subplots_adjust(wspace=0.15)

    if subtitles != None:
        if title:
            plt.suptitle(title)
    else:
        plt.suptitle(title, y=.98)
    return fig, ax


def plot_three_fock(rho1, rho2, rho3, hinton_limit = 20,
  title="", subtitles=None, ylim=0.5):
    """Draws a Hinton diagram for visualizing a density matrix or superoperator.

    Parameters
    ----------
    rho : qobj
        Input density matrix or superoperator.

    xlabels : list of strings or False
        list of x labels

    ylabels : list of strings or False
        list of y labels

    title : string
        title of the plot (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    cmap : a matplotlib colormap instance
        Color map to use when plotting.

    label_top : bool
        If True, x-axis labels will be placed on top, otherwise
        they will appear below the plot.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    """
    params = {# 'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.labelpad': -1,
          'text.usetex': False,
          'figure.figsize': fig_size}
    plt.rcParams.update(params)
    
    
    
    fig, ax = plt.subplots(1, 3, figsize=(fig_width, 0.35*fig_width), sharey=True)

    rhos = [rho1, rho2, rho3]

    for i, axis in enumerate(ax):
        rho = Qobj(rhos[i][:hinton_limit, :hinton_limit])
        N = rho.shape[0]
        axis.bar(np.arange(0, N), np.real(rho.diag()),
               color="green", alpha=0.6, width=0.8)
        axis.set_xlim(-.5, N)

    ax[0].set_ylim(0, ylim)        
    
    for i in range(0, 3):
        ax[i].set_yticks([0, ylim/2, ylim])
        ax[i].set_xticks([0, int(hinton_limit/2), hinton_limit])
        ax[i].set_xticklabels([0, "", hinton_limit])
        ax[i].set_xlabel(r"|$n\rangle$", labelpad=-6, fontsize=8)
        ax[i].set_aspect(40)
        if subtitles != None:
            ax[i].set_title(subtitles[i])



    ax[0].set_ylabel(r"p($n$)", labelpad=-16, fontsize=8)
    ax[0].set_yticklabels(["0", "", ylim])


    if subtitles != None:
        if title:
            plt.suptitle(title)
    else:
        plt.suptitle(title, x=0.5, y=.98)
    fig.subplots_adjust(right=0.85, wspace=0.15, hspace=-7.6)

    return fig, ax


def plot_husimi_directly(x, title="",
  cmap=None,
  xvec=None,
  yvec=None,
  norm=None,
  normalize=True,
  cbar_ticks=None, cbar_ticklabels = None):
    """
    """
    if xvec is None:
      xvec = np.linspace(-5, 5, 32)
    if yvec is None:
      yvec = np.linspace(-5, 5, 32)
    params = {# 'backend': 'ps',
          'axes.labelsize': 8,
          'font.size': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'axes.labelpad': -1,
          'text.usetex': False,
          'figure.figsize': fig_size}
    plt.rcParams.update(params)
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.27, fig_width/3.27))
    if cmap is None:
      colors1 = colors.LinearSegmentedColormap.from_list("", [(0, "white"),
                                                        (1, "red"),
                                                       ])(np.linspace(0, 1, 256))

      cmap = colors.LinearSegmentedColormap.from_list('my_colormap', colors1)
    if normalize:
      x = x/np.max(x)

    if norm is None:
      norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    if cbar_ticks is None:
      cbar_ticks = [0., 0.5, 1.]
      cbar_ticklabels = ["0", 0.5, "1"]

    im = ax.pcolor(xvec, yvec, x, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_xticks([-xvec[-1], 0,  xvec[-1]])
    ax.set_yticks([-yvec[-1], 0, yvec[-1]])

    ax.set_xticklabels(["{:.0f}".format(-xvec[-1]), "", "{:.0f}".format(xvec[-1])])
    ax.set_yticklabels(["{:.0f}".format(-yvec[-1]), "", "{:.0f}".format(yvec[-1])])

    ax.set_xlabel(r"Re($\beta$)", labelpad=-6)
    ax.set_ylabel(r"Im$(\beta)$", labelpad=-9)

    fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)
    cax = fig.add_axes([0.9, 0.14, 0.0315, 0.73])

    fig.colorbar(im, cax=cax, fraction=0.0455, ticks=cbar_ticks)
    cax.set_yticklabels(cbar_ticklabels)
    # plt.subplots_adjust(wspace=0.15)


    # cbar = plt.colorbar(im, fraction=0.0455, ticks=cbar_ticks)
    # cbar.solids.set_edgecolor("face")
    # cbar.ax.set_yticklabels(cbar_tick_labels)
    ax.set_title(title)


    return fig, ax


def plot_hinton(rho, title=None, cbar_ticks=None, cmap=None, dm_cut=16):
  fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
  inches_per_pt = 1.0/72.27               # Convert pt to inch
  golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
  fig_width = fig_width_pt*inches_per_pt  # width in inches
  fig_height = fig_width*golden_mean      # height in inches
  fig_size =  [fig_width,fig_height]
  params = {# 'backend': 'ps',
            'axes.labelsize': 8,
            'font.size': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelpad': 1,
            'text.usetex': False,
            'figure.figsize': fig_size}
  plt.rcParams.update(params)
  dm = rho

  fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.27, fig_width/3.27))
  ax.set_xticks([dm_cut, int(dm_cut/2), 0])
  ax.set_yticks([dm_cut, int(dm_cut/2), 0])

  ax.set_yticklabels([0, "", int(dm_cut)])
  ax.set_xticklabels([int(dm_cut), "", 0])

  plt.subplots_adjust(hspace=-.69)

  if cmap is None:
    cmap = matplotlib.cm.RdBu
  # Extract plotting data W from the input.
  W = dm.full()[:dm_cut, :dm_cut]
  ax.set_aspect('equal')
  ax.set_frame_on(True)

  height, width = W.shape

  w_max = 1.25 * max(abs(np.diag(np.matrix(W))))
  if w_max <= 0.0:
      w_max = 1.0

  ax.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]),
          color=cmap(128))

  for x in range(width):
      for y in range(height):
          _x = x + 1
          _y = y + 1
          if np.real(W[x, y]) > 0.0:
              _blob(_x - 0.5, height - _y + 0.5, abs(W[x,
                    y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap, ax=ax)
          else:
              _blob(_x - 0.5, height - _y + 0.5, -abs(W[
                    x, y]), w_max, min(1, abs(W[x, y]) / w_max), cmap=cmap, ax=ax)

  # ax.xaxis.tick_top()
  # ax.xaxis.set_label_position('top') 

  ax.set_ylabel(r"$\langle n|$", labelpad=-12)
  ax.set_xlabel(r"$|n\rangle$", labelpad=-6)

  fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)
  cax = fig.add_axes([0.9, 0.14, 0.0315, 0.73])

  # fig.colorbar(im, cax=cax, fraction=0.0455, ticks=cbar_ticks)

  norm = matplotlib.colors.DivergingNorm(vmin = -abs(W).max(), vcenter=0, vmax=abs(W).max())
  if cbar_ticks is None:
    cbar_ticks = [-abs(W).max(), 0., abs(W).max()]
    cbar_tick_labels = ["{:.1f}".format(-abs(W).max()), "0", "{:.1f}".format(abs(W).max())] 
  
  matplotlib.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap,
    ticks=cbar_ticks)

  cax.set_yticklabels(cbar_tick_labels)

  if title is not None:
    ax.set_title(title)

  return fig, ax



def plot_fock(rho, title=None, dm_cut=16):
  fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
  inches_per_pt = 1.0/72.27               # Convert pt to inch
  golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
  fig_width = fig_width_pt*inches_per_pt  # width in inches
  fig_height = fig_width*golden_mean      # height in inches
  fig_size =  [fig_width,fig_height]
  params = {# 'backend': 'ps',
            'axes.labelsize': 8,
            'font.size': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelpad': 1,
            'text.usetex': False,
            'figure.figsize': fig_size}
  plt.rcParams.update(params)


  fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.27, fig_width/3.4))
  ax.set_xticks([int(dm_cut), int(dm_cut/2), 0])
  ax.set_xticklabels([int(dm_cut), "", 0])

  plt.subplots_adjust(hspace=-.69)

  ax.set_frame_on(True)
  N = rho.shape[0]
  ax.bar(np.arange(0, N), np.real(rho.diag()),
    color="green", alpha=0.6, width=0.8)

  ax.set_xlim(-.5, int(dm_cut))

  ymax = np.max(np.real(rho.diag()))

  ax.set_yticks([0, ymax/2, ymax])

  ax.set_yticklabels([0, '', '{:.1f}'.format(ymax)])

  ax.set_ylabel(r"p(n)", labelpad=-9)
  ax.set_xlabel(r"$|n\rangle$", labelpad=-6)

  fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)
  ax.set_ylim(0, ymax)
  if title is not None:
    ax.set_title(title, x=0.35)

  # ax.set_aspect('equal')
  return fig, ax




def extend_array(flist, max_iter=None):
    """
    Extends the array with repeating the last element
    """
    if max_iter == None:
        max_iter = np.max([len(f) for f in flist])

    arr = np.zeros((len(flist), max_iter))

    for i in range(len(flist)):
        last_element = flist[i][-1]
        length_flist = len(flist[i])

        arr[i][:length_flist] = flist[i]
        arr[i][length_flist:] = np.nan

    return arr


def get_mean_std_array(arr):
    """
    Obtains the mean of an array over second axis
    """
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)



def plot_shaded_fidelities(fidelities, x, title="", color="r", fig=None, ax=None, label="",
    grid=False):
    """
    """
    if fig == None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    extended_array = extend_array(fidelities, len(x))
    mean, std = get_mean_std_array(extended_array)
    line, = plt.plot(x, mean, color=color, label=label, linewidth=0.8, alpha=1)
    ax.fill_between(x, mean-std, mean+std, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlim([0.8, 11000])

    ax.set_ylabel("Fidelity")
    ax.set_xlabel("Iterations")

    ax.set_ylim([0, 1.02])

    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_yticklabels([0, "", "", "", "", "0.5", "", "", "", "", 1])

    plt.title(title)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
   #  ax.grid(which='minor', alpha=0.2)
    if grid==True:
        ax.grid(which='major', alpha=0.2)
    return fig, ax


def plot_all_fidelities(fidelities, x, title="", color="r",
  alpha=1, linewidth = 0.8, grid=False,
  fig=None, ax=None, label="", show_mean=False):
    """
    """
    if fig == None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if len(fidelities) == 1:
        ax.plot(fidelities[0], color=color, linewidth=linewidth, alpha=alpha)
    else:
        for i in range(len(fidelities)):
            ax.plot(fidelities[i], c=color, linewidth=linewidth, alpha=alpha)

    extended_array = extend_array(fidelities)
    mean, std = get_mean_std_array(extended_array)

    if show_mean:
      ax.plot(mean, "--", color=color, label=label, linewidth=linewidth,
        alpha=alpha)

    ax.set_xscale('log')
    ax.set_xlim([0.8, 11000])

    ax = ax
    ax.set_ylabel("Fidelity")
    ax.set_xlabel("Iterations")

    ax.set_ylim([0, 1.02])

    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_yticklabels([0, "", "", "", "", "0.5", "", "", "", "", 1])

    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax.set_xlim([0.8, 11000])
    plt.title(title)
   #  ax.grid(which='minor', alpha=0.2)
    if grid==True:
        ax.grid(which='major', alpha=0.2)
    # ax.legend(loc="lower right")
    return fig, ax



def add_photon_noise(rho0, gamma, tlist):
    """
    """
    n = rho0.shape[0]
    a = destroy(n)
    c_ops = [gamma*a,]
    H = -0*(a.dag() + a)
    opts = Options(atol=1e-20, store_states=True, nsteps=1500)
    L = liouvillian(H, c_ops=c_ops)
    states = mesolve(H, rho0, tlist, c_ops=c_ops)

    return states.states



def solve(L, rho0, tlist, options=None, e=0.8):
        """
        Solve the Lindblad equation given initial
        density matrix and time.
        """
        if options is None:
            options = Options()

        states = []
        states.append(rho0)
        
        n = rho0.shape[0]
        a = destroy(n)

        mean_photon_number = expect(a.dag()*a, rho0)

        dt = np.diff(tlist)
        rho = rho0.full().ravel("F")
        rho = rho.flatten()

        L = csr_matrix(L.full())
        r = ode(cy_ode_rhs)
        
        r.set_f_params(L.data, L.indices, L.indptr)

        r.set_integrator(
            "zvode",
            method=options.method,
            order=options.order,
            atol=options.atol,
            rtol=options.rtol,
            nsteps=options.nsteps,
            first_step=options.first_step,
            min_step=options.min_step,
            max_step=options.max_step,
        )

        r.set_initial_value(rho, tlist[0])

        n_tsteps = len(tlist)


        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                r1 = r.y.reshape((n, n))
                r1_q = Qobj(r1)
                states.append(r1_q)

                mphoton_number = np.real(expect(a.dag()*a, r1_q))

                if mphoton_number < e*mean_photon_number:
                    break

        return states




"""
Generates various classes of states.
"""

def cat(hilbert_size, alpha, S=0, mu=0):
    """
    Generates a cat state.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_

    
    Args:
    -----
        hilbert_size (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
                 default: 0
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state using the cat state.
                  default: 0


    Returns:
    -------
        cat (:class:`qutip.Qobj`): Cat state ket.
    """
    kend = 2 * S + 1
    cstates = 0 * (coherent(hilbert_size, 0))

    for k in range(0, int((kend + 1) / 2)):
        sign = 1

        if k >= S:
            sign = (-1) ** int(mu > 0.5)

        prefactor = np.exp(1j * (np.pi / (S + 1)) * k)

        cstates += sign * coherent(hilbert_size, prefactor * alpha * (-((1j) ** mu)))
        cstates += sign * coherent(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

    ket = cstates.unit()
    return ket


def _get_num_prob(idx):
    """Selects a random probability vector from the list of number states"""
    states17 = [
        [
            (np.sqrt(7 - np.sqrt(17))) / np.sqrt(6),
            0,
            0,
            (np.sqrt(np.sqrt(17) - 1) / np.sqrt(6)),
            0,
        ],
        [
            0,
            (np.sqrt(9 - np.sqrt(17)) / np.sqrt(6)),
            0,
            0,
            (np.sqrt(np.sqrt(17) - 3) / np.sqrt(6)),
        ],
    ]

    statesM = [
        [
            0.5458351325482939,
            -3.7726009161224436e-9,
            4.849511177634774e-8,
            -0.7114411727633639,
            -7.48481181758003e-8,
            -1.3146003192319789e-8,
            0.44172510726665587,
            1.1545802803733896e-8,
            1.0609402576342428e-8,
            -0.028182506843720707,
            -6.0233214626778965e-9,
            -6.392041552216322e-9,
            0.00037641909140801935,
            -6.9186916801058116e-9,
        ],
        [
            2.48926815257019e-9,
            -0.7446851186077535,
            -8.040831059521339e-9,
            6.01942995399906e-8,
            -0.5706020908811399,
            -3.151900508005823e-8,
            -7.384935824733578e-10,
            -0.3460030551087218,
            -8.485651303145757e-9,
            -1.2114327561832047e-8,
            0.011798401879159238,
            -4.660460771433317e-9,
            -5.090374160706911e-9,
            -0.00010758601713550998,
        ],
    ]

    statesP = [
        [
            0.0,
            0.7562859301326029,
            0.0,
            0.0,
            -0.5151947804474741,
            -0.20807866860791188,
            0.12704803323656158,
            0.05101928893751686,
            0.3171198939841734,
        ],
        [
            -0.5583217426728544,
            -0.0020589109231194413,
            0.0,
            -0.7014041964402703,
            -0.05583041652626998,
            0.0005664728465725445,
            -0.2755044401850055,
            -0.3333309025086189,
            0.0785824556163142,
        ],
    ]

    statesP2 = [
        [
            -0.5046617350158988,
            0.08380989527942606,
            -0.225295417417812,
            0.0,
            -0.45359477373452817,
            -0.5236866813756252,
            0.2523308675079494,
            0.0,
            0.09562538828178244,
            0.2172849136874009,
            0.0,
            0.0,
            0.0,
            -0.2793663175980869,
            -0.08280858231312467,
            -0.05106696128137072,
        ],
        [
            -0.0014249418817930378,
            0.5018692341095683,
            0.4839749920101922,
            -0.3874886488913531,
            0.055390715144453026,
            -0.25780190053922486,
            -0.08970154713375252,
            -0.1892386424818236,
            0.10840637100094529,
            -0.19963901508324772,
            -0.41852779130900664,
            -0.05747247660559087,
            0.0,
            -0.0007888071131354318,
            -0.1424131123943283,
            -0.0001441905475623907,
        ],
    ]

    statesM2 = [
        [
            -0.45717455741713664,
            np.complex(-1.0856965103853774e-6, 1.3239037829080093e-6),
            np.complex(-0.35772784377291084, -0.048007740168066144),
            np.complex(-3.5459165445315755e-6, 0.000012571453643232864),
            np.complex(-0.5383420820794502, -0.24179040513272307),
            np.complex(9.675641330014822e-7, 4.569566899500361e-6),
            np.complex(0.2587482691377581, 0.313044506480362),
            np.complex(4.1979351791851435e-6, -1.122460690803522e-6),
            np.complex(-0.11094500303308243, 0.20905585817734396),
            np.complex(-1.1837814323046472e-6, 3.8758497675466054e-7),
            np.complex(0.1275629945870373, -0.1177987279989385),
            np.complex(-2.690647673469878e-6, -3.6519804939862998e-6),
            np.complex(0.12095531973074151, -0.19588735180644176),
            np.complex(-2.6588791126371675e-6, -6.058292629669095e-7),
            np.complex(0.052905370429015865, -0.0626791930782206),
            np.complex(-1.6615538648519722e-7, 6.756126951837809e-8),
            np.complex(0.016378329200891946, -0.034743342821208854),
            np.complex(4.408946495377283e-8, 2.2826415255126898e-8),
            np.complex(0.002765352838800482, -0.010624191776867055),
            6.429253878486627e-8,
            np.complex(0.00027095836439738105, -0.002684435917226972),
            np.complex(1.1081202749445256e-8, -2.938812506852636e-8),
            np.complex(-0.000055767533641099717, -0.000525444354381421),
            np.complex(-1.0776974926155464e-8, -2.497769263148397e-8),
            np.complex(-0.000024992489351114305, -0.00008178444317382933),
            np.complex(-1.5079116121444066e-8, -2.0513760149701907e-8),
            np.complex(-5.64035228941742e-6, -0.000010297667130821428),
            np.complex(-1.488452012610573e-8, -1.7358623165948514e-8),
            np.complex(-8.909884885392901e-7, -1.04267002748775e-6),
            np.complex(-1.2056784102984098e-8, -1.2210951690230782e-8),
        ],
        [
            0,
            0.5871298855433338,
            np.complex(-3.3729618710801137e-6, 2.4152360811650373e-6),
            np.complex(-0.5233926069798007, -0.13655786303346068),
            np.complex(-4.623380373113224e-6, 0.000010362902695259763),
            np.complex(-0.17909656013941788, -0.11916639160269833),
            np.complex(-3.399720873431807e-6, -7.125008373682292e-7),
            np.complex(0.04072119358712736, -0.3719310475303641),
            np.complex(-7.536125619789242e-6, 1.885248226837573e-6),
            np.complex(-0.11393851510585044, -0.3456924286310791),
            np.complex(-2.3915763815197452e-6, -4.2406689395594674e-7),
            np.complex(0.12820184730203607, 0.0935942533049232),
            np.complex(-1.5407293261691393e-6, -2.4673669087089514e-6),
            np.complex(-0.012272903377715643, -0.13317144020065683),
            np.complex(-1.1260776123106269e-6, -1.6865728072273087e-7),
            np.complex(-0.01013345155253134, -0.0240812705564227),
            np.complex(0.0, -1.4163391111474348e-7),
            np.complex(-0.003213070562510137, -0.012363639898516247),
            np.complex(-1.0619280312362908e-8, -1.2021213613319027e-7),
            np.complex(-0.002006756716685063, -0.0026636832583059812),
            np.complex(0.0, -4.509035934797572e-8),
            np.complex(-0.00048585160444833446, -0.0005014735884977489),
            np.complex(-1.2286988061034212e-8, -2.1199721851825594e-8),
            np.complex(-0.00010897007463988193, -0.00007018240288615613),
            np.complex(-1.2811279935244964e-8, -1.160553871672415e-8),
            np.complex(-0.00001785800494916693, -6.603027186486886e-6),
            -1.1639448324793031e-8,
            np.complex(-2.4097385882316104e-6, -3.5223103057306496e-7),
            -1.0792272866841885e-8,
            np.complex(-2.597671478115077e-7, 2.622928060603902e-8),
        ],
    ]
    all_num_codes = [states17, statesM, statesM2, statesP, statesP2]
    probs = all_num_codes[idx]
    return probs


def num(hilbert_size, probs=None, mu=0):
    """
    Generates the number states.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_

    Args:
        hilbert_size (int): Hilbert space dimension (cutoff). For the well defined
                            number states that we use here, the Hilbert space size 
                            should not be less than 32. If the probabilities are not
                            supplied then we will randomly select a set.
        probs (None, optional): Probabilitiy vector for the number state. If not supplied then a
                                random vector is selected from the five different sets from the function
                                `_get_num_prob`.
        mu (int, optional): Logical encoding (0/1)
                            default: 0
    
    Returns:
        :class:`qutip.Qobj`: Number state ket.
    
    """
    if (probs == None) and (hilbert_size < 32):
        err = "Specify a larger Hilbert size for default\n"
        err += "num state if probabilities are not specified\n"
        raise ValueError(err)

    state = fock(hilbert_size, 0) * 0

    if probs == None:
        probs = _get_num_prob(0)

    for n, p in enumerate(probs[mu]):
        state += p * fock(hilbert_size, n)
    ket = state.unit()
    return ket


def binomial(hilbert_size, S, N=None, mu=0):
    """
    Generates a binomial state.

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
        hilbert_size (int): Hilbert space size (cutoff).
        S (int): An integer parameter specifying 
        N (None, optional): A non-negative integer which specifies the order to which we can
                            correct dephasing errors and is similar to the ´alpha´ parameter
                            for cat states.
        mu (int, optional): Logical encoding (0/1)
                            default: 0
    
    Returns:
        :class:`qutip.Qobj`: Binomial state ket.
    """
    if N == None:
        Nmax = int((hilbert_size) / (S + 1)) - 1
        try:
            N = np.random.randint(0, Nmax)
        except:
            N = Nmax

    c = 1 / sqrt(2 ** (N + 1))

    psi = 0 * fock(hilbert_size, 0)

    for m in range(N):
        psi += (
            c
            * ((-1) ** (mu * m))
            * np.sqrt(binom(N + 1, m))
            * fock(hilbert_size, (S + 1) * m)
        )
    ket = psi.unit()
    return ket


def gkp(hilbert_size, delta, mu=0, zrange=20):
    """Generates a GKP state. 

    For a detailed discussion on the definition see
    `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
    and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
    Args:
        hilbert_size (int): Hilbert space size (cutoff).
        delta (float): 
        mu (int, optional): Logical encoding (0/1)
                            default: 0
        zrange (int, optional): The number of lattice points to loop over to construct
                                the grid of states. This depends on the Hilbert space
                                size and the delta value.
                                default: 20
    
    Returns:
        :class:`qutip.Qobj`: GKP state.
    """
    gkp = 0 * coherent(hilbert_size, 0)

    c = np.sqrt(np.pi / 2)

    zrange = range(-20, 20)

    for n1 in zrange:
        for n2 in zrange:
            a = c * (2 * n1 + mu + 1j * n2)
            alpha = coherent(hilbert_size, a)
            gkp += (
                np.exp(-(delta ** 2) * np.abs(a) ** 2)
                * np.exp(-1j * c ** 2 * 2 * n1 * n2)
                * alpha
            )

    ket = gkp.unit()
    return ket


def gaus2d(x=0, y=0, n0=1):
    return 1. / (np.pi * n0) * np.exp(-((x**2 + y**2.0)/n0))


class GaussianConv(tf.keras.layers.Layer):
    """
    Expectation layer that calculates expectation values for a set of operators on a batch of rhos.
    You can specify different sets of operators for each density matrix in the batch.
    """
    def __init__(self, kernel = None, **kwargs):
        super(GaussianConv, self).__init__(**kwargs)
        self.kernel = kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, x):
        """Expectation function call
        """
        return tf.nn.conv2d(x, self.kernel, strides=[1, 1, 1, 1], padding="SAME")