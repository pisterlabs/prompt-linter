from rlhfutils.eval_utils import oai_kwargs, load_alldfs, annotate_apfarm, apf_format, load_wgpt, filter_and_sort_df
import pandas as pd
from statistics import mean, stdev
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer
from datasets import load_dataset
import openai
from rlhfutils.data import qaform
from transformers import AutoTokenizer
import pandas as pd
from rlhfutils.eval_utils import getapfsft, tok_dist
import matplotlib.pyplot as plt
from rlhfutils.debug_utils import load_rm, progress_rm
import argparse
import nltk
from nltk.tokenize import sent_tokenize
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# given a dataframe, use rm to get rmscos if it doesn't already have them. 
# then bin / mean at intervals of 10 tokens and return lists for a scatter
def score_rm(indf, rm):
    indf['qstrs'] = [qaform(r['question'], r['response']) for _, r in indf.iterrows()]
    rewards = progress_rm(list(indf['qstrs']), rm, kwargs)
    scos = [a[0]['score'] for a in rewards]
    indf['rewards'] = scos

binsize = 20
def binscatter(indf):
    mvals = []
    cntvals = []
    for i in range(0, 260, binsize):
        tmp = indf[(indf.rcnt>i)&(indf.rcnt<i+binsize)].copy()
        if len(tmp)>0:
            mvals.append(tmp.rewards.mean())
            cntvals.append(len(tmp))
        else:
            mvals.append(None)
            cntvals.append(None)
    return mvals, cntvals

def compscatter(k1, k2):
    xrange = list(range(0, 260, binsize))
    ysft = binscatter(adfs[k1])
    yppoorig = binscatter(adfs[k2])
    diffdist = []
    for i in range(len(ysft)):
        if ysft[i]!=None and yppoorig[i]!=None:
            diffdist.append(yppoorig[i]-ysft[i])
    plt.scatter(xrange, ysft, c='blue')
    plt.scatter(xrange, yppoorig, c='red')
    return mean(diffdist), stdev(diffdist)

# def overlap_vis(x, ry, by, intlist, col):   
    
#     # plt.scatter(x, ry, color='red', label='Red Points')
#     plt.scatter(x, by, color=col)
    
#     for i in range(len(x)):
#         alpha_intensity = intlist[i]
        
#         # Calculate arrow properties
#         dx = 0  # no horizontal movement
#         dy = ry[i] - by[i]  # vertical distance between blue and red
        
#         # Draw the arrow
#         #plt.arrow(x[i], by[i], dx, dy, head_width=2, head_length=0.05, fc=arrow_color, ec=arrow_color, alpha=0.2)
#         plt.arrow(x[i], by[i], dx, dy, head_width=2, head_length=0.05, fc=col, ec=col, alpha=min(1, alpha_intensity*5/sum(intlist)))
    
#     #plt.legend()
#     #plt.title("WGPT")
#     #plt.xlabel("Length")
#     #plt.ylabel("Rewarrd")
#     #plt.show()

def overlap_vis(x, ry, by, intlist, col, ax):   
    
    # Normalize intlist to get suitable sizes for plotting
    sizes = (np.array(intlist) / max(intlist)) * 60  # scaling factor of 100, can be adjusted
    
    ax.scatter(x, by, s=sizes, color=col)
    
    for i in range(len(x)):
        alpha_intensity = intlist[i]
        
        # Calculate arrow properties
        dx = 0  # no horizontal movement
        dy = ry[i] - by[i]  # vertical distance between blue and red
        
        # Draw the arrow
        ax.arrow(x[i], by[i], dx, dy, head_width=2, head_length=0.05, fc=col, ec=col, alpha=min(1, alpha_intensity*5/sum(intlist)))

def set_style(ax):
    # Set Seaborn deep color palette
    hexs = sns.color_palette("deep", 8).as_hex()
    color_1, color2 = hexs[0], hexs[1]
    
    # Gridlines
    ax.xaxis.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
    ax.yaxis.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgray')
    
    # Fonts
    font_title = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 13}
    font_labels = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 13}
    
    ax.set_title(ax.get_title(), fontdict=font_title)
    ax.set_xlabel(ax.get_xlabel(), fontdict=font_labels)
    ax.set_ylabel(ax.get_ylabel(), fontdict=font_labels)
    
    # Tick labels
    ax.tick_params(axis='both', labelsize=11, labelcolor='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')

def fancyscatter(names, title, dfs, ax):
    
    k1, k2, k3 = names[0], names[1], names[2]
    # Generate three datasets:
    # 1. For the scatter plot
    x_scatter = list(range(0, 260, binsize))
    sfthist = dfs[k1].rcnt
    ppohist = dfs[k2].rcnt

    
    sftscatter, scv = binscatter(dfs[k1])
    pposcatter, pcv = binscatter(dfs[k2])
    lsscatter, lcv = binscatter(dfs[k3])
    sp = dfs[k2].rewards.mean() - dfs[k1].rewards.mean()
    hp = dfs[k3].rewards.mean() - dfs[k1].rewards.mean()
    print(sp)
    print(hp)
    
    diffs = []
    ldiffs = []
    tval = sum([s for s in scv if s is not None])+sum([s for s in pcv if s is not None])
    for i in range(len(sftscatter)):
        if sftscatter[i]!=None and pposcatter[i]!=None:
            diffs.append((pposcatter[i]-sftscatter[i])*((scv[i]+pcv[i])/tval))
        if sftscatter[i]!=None and lsscatter[i]!=None:
            ldiffs.append((lsscatter[i]-sftscatter[i])*((scv[i]+lcv[i])/tval))
    print("tots")
    print(sum(diffs))
    print(sum(ldiffs))
    print('rats')
    print(sum(diffs)/sp)
    print(sum(ldiffs)/hp)
    nsft = []
    nppo = []
    lsft = []
    lppo = []
    lx = []
    nx = []
    ints = []
    lints = []
    for i in range(len(pposcatter)):
        if pposcatter[i]!=None and sftscatter[i]!=None:
            nsft.append(sftscatter[i])
            nppo.append(pposcatter[i])
            nx.append(x_scatter[i])
            ints.append((scv[i]+pcv[i]))
        if sftscatter[i]!=None and lsscatter[i]!=None:
            lsft.append(sftscatter[i])
            lppo.append(lsscatter[i])
            lx.append(x_scatter[i]+6)
            lints.append((scv[i]+lcv[i]))
    #plt.figure(figsize=(10, 6))
    if k2!=k3:
        overlap_vis(nx, nppo, nsft, ints, 'red', ax)
    
    overlap_vis(lx, lppo, lsft, lints, 'black', ax)
    ax.set_title(title)
    ax.set_xlabel("Length")
    ax.set_ylabel("Reward")
    set_style(ax)
    #b = dfs[k1].rewards.mean()
    #c = dfs[k2].rewards.mean()
    #d =  dfs[k3].rewards.mean()
    #ax.vlines(max(lx+nx)+20, b, c, colors='red', linestyles='dotted', lw=2)
    #ax.scatter([max(lx+nx)+20, max(lx+nx)+20], [b, c], color='red', marker='_', s=50)  # Adding horizontal caps using scatter
    #ax.vlines(max(lx+nx)+23, b, d, colors='black', linestyles='dotted', lw=2)
    #ax.scatter([max(lx+nx)+23, max(lx+nx)+23], [b, d], color='black', marker='_', s=50)  # Adding horizontal caps using scatter
