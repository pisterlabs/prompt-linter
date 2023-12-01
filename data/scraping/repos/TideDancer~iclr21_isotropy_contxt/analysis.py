import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from transformers import XLMTokenizer, XLMModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTModel
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import pickle 
import argparse
import Models

parser = argparse.ArgumentParser(description='Analyze the contxt embeds')
parser.add_argument('file', type=str, help='dict file')
parser.add_argument('--v', action='store_true', help='verbose')
# embed, size, zoom, center
parser.add_argument('--embed', type=float, default=0, help='whether performing dimension reduction. [value] between (0, 1) means reducing dimension that preserve [value] total variance; [value] > 1 must be integer meaning reducing to [value] dimension')
parser.add_argument('--maxl', type=float, default=0, help='max length for each dict[id], used for sample from the list. [value] > 1 must be integer, means maximum number of sample = [value]; [value] between (0, 1) means fraction; [value] == -1 means taking log2(total_num); [value] == -2 means taking sqrt(total_num).')
parser.add_argument('--zoomgpt2', type=int, default=0, help='zoom in gpt2 clusters, 1 for the left one, 2 for the right one')
parser.add_argument('--center', action='store_true', help='centered or not')
# tasks
parser.add_argument('--inter_cos', action='store_true', help='task for inter-token cosine similarity')
parser.add_argument('--intra_cos', action='store_true', help='task for intra-token cosine similarity')
parser.add_argument('--cluster', action='store_true', help='task for clustering')
parser.add_argument('--cluster_cos', action='store_true', help='task for computing clusterred inter and intra cos')
parser.add_argument('--lid', action='store_true', help='task for compute lid')
parser.add_argument('--lid_metric', type=str, default='l2', help='metric for lid, choose from l2, cos')
parser.add_argument('--draw', type=str, default=None, help='draw, choose from 2d, 3d, freq')
parser.add_argument('--draw_token', type=str, default=None, help='draw specified tokens in 3D plot. Please use the following format: ["a","b"]')
args = parser.parse_args()

save_prefix = '.'.join(args.file.split('/')[-2:]) # format: dataset.dictfile
if args.embed > 0:
    if args.embed > 1: args.embed = int(args.embed)
    save_prefix += '.embed.' + str(args.embed) + '.'
if args.maxl > 0:
    save_prefix += '.maxl.' + str(args.maxl) + '.'
if args.maxl == -1:
    save_prefix += '.maxl.log2.'
if args.maxl == -2:
    save_prefix += '.maxl.sqrt.'

special_dict = None

# load embeds dict and preprocess
_, d = pickle.load(open(args.file, 'rb'))
print('finish load')

keys = []
y = []
x = []
repeats = []
window_ids = []
pos_ids = []

# handle length
lengths = {}
for k in d:
    lengths[k] = len(d[k])
    if args.maxl >= 1:
        lengths[k] = min(len(d[k]), int(args.maxl)) # maxl
    if args.maxl > 0 and args.maxl < 1:
        lengths[k] = max(1, int(len(d[k])*args.maxl)) # fraction
    if args.maxl == -1:
        lengths[k] = max(1, int(np.log2(len(d[k])))) # log
    if args.maxl == -2:
        lengths[k] = int(np.sqrt(len(d[k]))) # sqrt

# proc data
for k in d:
    keys += [k] * lengths[k]

    y_tmp = list(map(lambda e: e[0], d[k]))
    if lengths[k] < len(y_tmp):
        idx = np.random.choice(len(y_tmp), lengths[k], replace=False) # sample without replacement
        y_tmp = [y_tmp[i] for i in idx]
    y += y_tmp

    repeats += [len(d[k])] * lengths[k]
    window_ids += list(map(lambda e: e[1], d[k]))
    pos_ids += list(map(lambda e: e[1]*512 + e[2], d[k]))

keys = np.stack(keys)
window_ids = np.stack(window_ids)
pos_ids = np.stack(pos_ids)
y = np.stack(y)
repeats = np.stack(repeats)
print('total points: ', y.shape[0])

## check split and zoom for gpt2
if 'layer0' in args.file: split = 75
elif 'layer1' in args.file: split = 400
else: split = 700
if 'gpt2' in args.file and args.zoomgpt2 == 1:
    y_2dim = PCA(n_components=2).fit(y).transform(y)
    idx = y_2dim[:,0] <= split
    y = y[idx]
    keys = keys[idx]
    repeats = repeats[idx]
    save_prefix += '.zoomleft'
if 'gpt2' in args.file and args.zoomgpt2 == 2:
        y_2dim = PCA(n_components=2).fit(y).transform(y)
        idx = y_2dim[:,0] < split
        y = y[idx]
        keys = keys[idx]
        repeats = repeats[idx]
        save_prefix += '.zoomright'

## embed
var_ratio = 1
if args.embed > 0:
    pca = PCA(n_components=args.embed).fit(y)
    y = pca.transform(y)
    var_ratio = sum(pca.explained_variance_ratio_)
    print('after PCA:', y.shape)

## centered
if args.center:
    y = StandardScaler(with_std=False).fit(y).transform(y) 

# ------------------------------------- functions --------------------------------------
def intercos(y, keys=None, center=False, sample_size=-1):
    if center: y = StandardScaler(with_std=False).fit(y).transform(y) 
    if sample_size >= 0:
        assert(keys is not None)
        assert(sample_size == int(sample_size))
        all_y = []
        for k in np.unique(keys):
            y_k = y[keys==k]
            if y_k.shape[0] < 1: continue
            y_k = y_k[np.random.choice(y_k.shape[0], sample_size, replace=False)]
            all_y.append(y_k)
        y = np.vstack(all_y)
    cos = cosine_similarity(y, y)
    avg_cos = ( np.sum(np.sum(cos)) - cos.shape[0] ) / 2 / ( cos.shape[0]*(cos.shape[0]-1) / 2 )
    return avg_cos

def intracos(y, d, keys, special_dict=None, center=False):
    if center: y = StandardScaler(with_std=False).fit(y).transform(y) 
    all_cos = []
    for k in d:
        if special_dict and k in special_dict.values():
            continue
        tmp = y[keys==k]
        if tmp.shape[0] <= 1:
            continue
        if tmp.shape[0] >= 1000: # maximum 1000 points to estimate
            idx = np.random.choice(len(tmp), 1000, replace=False)
            tmp = tmp[idx]
        avg_cos = intercos(tmp)
        all_cos.append(avg_cos)
    avg_cos = np.mean(all_cos)
    return avg_cos

def lid(y, sample_size=-1, k_list=[101], metric='l2', block=50000):
    import faiss
    print('metric:', metric)
    ngpus = faiss.get_num_gpus()
    print("number of GPUs used by faiss:", ngpus) 
    if metric == 'cos':
        cpu_index = faiss.IndexFlatIP(y.shape[1])
        y = normalize(y)
    if metric == 'l2':
        cpu_index = faiss.IndexFlatL2(y.shape[1])

    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    print('index')
    gpu_index.add(y) 
    print('index total:', gpu_index.ntotal)

    if sample_size > 0:
        x = y[np.random.choice(y.shape[0], size=int(sample_size), replace=False)]
    else:
        x = y

    for k in k_list:
        print('start query')
        i = 0
        D = []
        while i < x.shape[0]:
            tmp = x[i:min(i+block, x.shape[0])]
            i += block
            b, _ = gpu_index.search(tmp, k)
            D.append(b)
        D = np.vstack(D)
        print("query finish")

        D = D[:, 1:] # remove the most-left column as it is itself
        if metric == 'cos':
            D = 1-D  # cosine dist = 1 - cosine
            D[D <= 0] = 1e-8
        rk = np.max(D, axis=1)
        rk[rk==0] = 1e-8
        lids = D/rk[:, None]
        lids = -1/np.mean(np.log(lids), axis=1)
        lids[np.isinf(lids)] = y.shape[1] # if inf, set as space dimension
        lids = lids[~np.isnan(lids)] # filter nan
        print('filter nan/inf shape', lids.shape)
        print('k', k-1, 'lid_mean', np.mean(lids), 'lid_std', np.std(lids))

# ----------------------------------------- TASKS --------------------------------------
# compute lid
k_list = [101]
if args.lid:
    lid(y, k_list=k_list, metric=args.lid_metric)

# cluster
if args.cluster:
    print('perform clustering on ', y.shape)
    # kmeans with silhouette score
    sil = []
    sil_std = []
    all_labels = []
    cands = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for k in cands:
        score = []
        kmeans = KMeans(n_clusters=k, n_jobs=-1).fit(y)
        labels = kmeans.labels_
        score.append(silhouette_score(y, labels, sample_size=20000))
        sil.append(np.mean(score))
        sil_std.append(np.std(score))
        all_labels.append(labels)
        print("k&score&std:", k, sil[-1], sil_std[-1])
    if max(sil) >= 0.1:
        best_k = cands[sil.index(max(sil))]
        labels = all_labels[sil.index(max(sil))]
        std = sil_std[sil.index(max(sil))]
    else:
        best_k = 1
        labels = np.zeros(y.shape[0])
        std = 0
    print('bestk&sil&std:', best_k, max(sil), std)

    if args.cluster_cos:
        all_inter = []
        all_intra = []
        for i in range(np.unique(labels).shape[0]):
            y_tmp = y[labels==i]
            keys_tmp = keys[labels==i]
            cos = intercos(y_tmp, keys=keys_tmp, center=args.center, sample_size=1)
            if not np.isnan(cos) and not np.isinf(cos):
                all_inter.append(cos)
            cos = intracos(y_tmp, d, keys_tmp, special_dict=None, center=args.center)
            if not np.isnan(cos) and not np.isinf(cos):
                all_intra.append(cos)
        print('clustered inter cos:', np.mean(all_inter))
        print('clustered intra cos:', np.mean(all_intra))

# inter cos
if args.inter_cos:
    assert(args.maxl > 0)
    avg_cos = intercos(y, args.center)
    print(avg_cos)

# intra cosine
if args.intra_cos:
    avg_cos = intracos(y, d, keys, special_dict, args.center)
    print(avg_cos)

# draw 
if args.draw:
    if args.draw == '3d' or args.draw == 'token':
        pca = PCA(n_components=3).fit(y)
    else:
        pca = PCA(n_components=2).fit(y)
    save_prefix += '.'+args.draw+'.'
 
    y = pca.transform(y)
    var_ratio = sum(pca.explained_variance_ratio_)

    print('draw')
    fig = plt.figure(figsize=(4, 3))
    if args.draw == '2d':
        plt.scatter(y[:,0], y[:,1], s=1, alpha=0.3, marker='.')
    if args.draw == '3d':
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(y[:,0], y[:,1], y[:,2], s=0.1, alpha=0.3, marker='.')
    if args.draw == 'freq':
        # plt.tricontourf(y[:,0], y[:,1], repeats, levels=15, cmap="RdBu_r")#, linewidths=0.5, colors='k')
        plt.scatter(y[:,0], y[:,1], c=repeats, cmap='jet', s=0.1, alpha=0.1, marker='.')
        plt.colorbar()
    if args.draw == 'token':
        colors = ['k','r','g','m','b']
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(y[:,0], y[:,1], y[:,2], s=0.05, alpha=0.01, marker='.', color='y')
        tokens_list = eval(args.draw_token)
        print('tokens to draw: ', tokens_list)
        for i in range(len(tokens_list)):
            t = y[keys==tokens_list[i]]
            print(tokens_list[i], ' occurrence: ', len(t))
            ax.scatter(t[:,0], t[:,1], t[:,2], s=1, alpha=1, marker='o', color=colors[i%5], label=tokens_list[i])
        legend = ax.legend(markerscale=6)

    plt.title('var ratio r=%.3f' % var_ratio)
    save_prefix = 'images/' + save_prefix
    print('save as:', save_prefix)
    plt.savefig(save_prefix+'png', format='png')
    plt.close()

