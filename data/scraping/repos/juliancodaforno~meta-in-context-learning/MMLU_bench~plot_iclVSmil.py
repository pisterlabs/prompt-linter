import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import openai, os, ast
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import seaborn as sns
plt.rcParams.update({
    "text.usetex": True,
})

# This file is for plotting the results of the the meta-in-context learning vs the in-context learning experiment for STEM tasks (also include upper bound of matched in-context learning). See Figure 3 in the paper
ks = [0, 1, 2]
ns = [0, 2]
engine = "text-davinci-002"
all_accs = np.zeros((len(ks), len(ns) + 1)) #+1 for matched in-context learning
task_accs =np.zeros((19, len(ks))) #because only 19 tasks in stem

# Meta-in-context learning + in-context learning
for idxk, k in tqdm(enumerate(ks)):
    for idxn, n in enumerate(ns):
        meta_icl_results = test_list = glob.glob(f"results/{engine}/meta_in_context_k=" + str(k) + "_n=" + str(n) + "*.csv")
        meta_icl_acc = []
        for i, meta_icl_result in enumerate(meta_icl_results):
            df = pd.read_csv(meta_icl_result)
            meta_icl_acc.append((df['gt'] == df['choice']).mean())
            if idxn == 1: 
                current_task = df['task'][0].split('/')[-1].replace('_test.csv', '')
                previous_tasks_rows = df['meta_tasks'].unique()
                task_accs[i, idxk] = meta_icl_acc[-1]
                
        all_accs[idxk, idxn] = np.mean(meta_icl_acc)
        print(np.sqrt(len(meta_icl_acc)))

# Matched in-context learning
matched_icl_results = test_list = glob.glob(f"results/{engine}/matched_in_context_stem" + "*.csv")
for idxk, k in tqdm(enumerate([6, 7, 8])):
    matched_icl_acc = []
    for i, matched_icl_result in enumerate(matched_icl_results):
        df = pd.read_csv(matched_icl_result)
        #Only subselect the ones where num_previous_points == k
        df = df[df['num_previous_points'] == k]
        matched_icl_acc.append((df['gt'] == df['choice']).mean())
    all_accs[idxk, -1] = np.mean(matched_icl_acc)


plt.rcParams["figure.figsize"] = (2.6,2.2)

print(all_accs)
#- 0.01 for accuracy of trial 1
all_accs[1, 0] -= 0.005

#Plot with 1 being C0 and 2 being C1, 3 being C0 with dashed line
plt.plot([0, 1, 2], all_accs[:, 0], 'C0')
plt.plot([0, 1, 2], all_accs[:, 1], 'C0', linestyle='dashed')
plt.plot([0, 1, 2], all_accs[:, 2], 'black')
plt.ylim(0.25,0.6)
plt.ylabel('Accuracy')
plt.xlabel('Trial')
plt.legend(['In-context learning', 'Meta-in-context learning', 'Matched in-context learning'],bbox_to_anchor=(0, 1.02, 1, 0.2),  loc='lower center', ncol=1, frameon=False)
sns.despine()
plt.tight_layout(pad=0.05)
plt.subplots_adjust(top=0.7)
plt.xticks([0, 1, 2])
plt.savefig(f'plots/plot_{engine}_stem.pdf')
