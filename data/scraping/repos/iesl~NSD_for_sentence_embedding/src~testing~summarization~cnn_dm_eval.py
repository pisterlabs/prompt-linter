import argparse
import os
import numpy as np
import random
import torch
import sys
import json
import time
sys.path.insert(0, sys.path[0]+'/../..')
from collections import Counter
#from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, MiniBatchKMeans
import warnings
warnings.filterwarnings("ignore")

#import torch.nn as nn
#import torch.utils.data
#import coherency_eval

from utils import seed_all_randomness, load_word_dict, load_emb_from_path, load_idx2word_freq, loading_all_models, load_testing_article_summ, str2bool, Logger
import utils_testing
from pythonrouge.pythonrouge import Pythonrouge

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
#parser.add_argument('--input', type=str, default='/iesl/canvas/hschang/language_modeling/cnn-dailymail/data/finished_files_subset',
parser.add_argument('--input', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--dict', type=str, default='./data/processed/wiki2016_min100/dictionary_index',
                    help='location of the dictionary corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='eval_log/summ_scores.txt',
                    help='output file for generated text')
parser.add_argument('--outf_vis', type=str, default='eval_log/summ_vis/summ.json',
                    help='output file for generated text')

###system
#parser.add_argument('--baseline_only', default=False, action='store_true',
#parser.add_argument('--method_set', type=str, default='ours+embs+cluster',
#parser.add_argument('--method_set', type=str, default='ours+embs',
#parser.add_argument('--method_set', type=str, default='ours+cluster',
parser.add_argument('--method_set', type=str, default='ours',
#parser.add_argument('--method_set', type=str, default='cluster',
#parser.add_argument('--method_set', type=str, default='embs',
                    help='If we want to run all methods, use ours+embs+cluster. Notice that we will turn off the sentence preprocessing if ours is not included, so the baseline scores might be slightly different.')
parser.add_argument('--top_k_max', type=int, default=10,
                    help='How many sentences we want to select at most')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
#parser.add_argument('--max_batch_num', type=int, default=100,
#                    help='number of batches for evaluation')

###data_format
parser.add_argument('--max_sent_len', type=int, default=50,
                    help='max sentence length for input features')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()

#args.outf_vis = '{}-{}.json'.format(args.outf_vis, time.strftime("%Y%m%d-%H%M%S"))

logger = Logger(args.outf)

logger.logging(str(args))
logger.logging(args.outf_vis)

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint,"target_emb.pt")

seed_all_randomness(args.seed, args.cuda)

device = torch.device("cuda" if args.cuda else "cpu")

########################
print("Loading Models")
########################
with open(args.dict) as f_in:
    idx2word_freq = load_idx2word_freq(f_in)

if 'ours' in args.method_set:
    parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, args.max_sent_len)

    encoder.eval()
    decoder.eval()
else:
    word_emb, output_emb_size = load_emb_from_path(args.emb_file, device, idx2word_freq)
    word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
    word_norm_emb[0,:] = 0

with open(args.dict) as f_in:
    word_d2_idx_freq, max_ind = load_word_dict(f_in)

utils_testing.compute_freq_prob(word_d2_idx_freq)

#all_clustering_models = [MiniBatchKMeans(n_clusters=i, max_iter=10, n_init=1, max_no_improvement=3, compute_labels=False) for i in range(1,args.top_k_max+1)]
all_clustering_models = [KMeans(n_clusters=i, max_iter=100, n_init=1, precompute_distances=True, n_jobs = 5) for i in range(1,args.top_k_max+1)]

########################
print("Processing data")
########################

def load_tokenized_story(f_in):
    article = []
    abstract = []

    next_is_highlight = False
    for line in f_in:
        line = line.rstrip()
        if line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            abstract.append([line])
        else:
            article.append(line)
    #@highlight
    return article, abstract

##convert the set of word embedding we want to reconstruct into tensor
def article_to_embs(article, word_norm_emb, word_d2_idx_freq, device):
    emb_size = word_norm_emb.size(1)
    num_sent = len(article)
    sent_embs_tensor = torch.zeros(num_sent, emb_size, device = device)
    sent_embs_w_tensor = torch.zeros(num_sent, emb_size, device = device)
    w_emb_tensors_list = []
    sent_lens = torch.empty(num_sent, device = device)
    alpha = 0.0001
    for i_sent, sent in enumerate(article):
        w_emb_list = []
        #print(sent)
        w_list = sent.split()
        for w in w_list:
            if w in word_d2_idx_freq:
                w_idx, freq, freq_prob = word_d2_idx_freq[w]
                sent_embs_tensor[i_sent,:] += word_norm_emb[w_idx,:]
                sent_embs_w_tensor[i_sent,:] += word_norm_emb[w_idx,:] * (alpha / (alpha + freq_prob) )
                w_emb_list.append(word_norm_emb[w_idx,:].view(1,-1))
        w_emb_tensors_list.append( torch.cat(w_emb_list, dim = 0) )
        sent_lens[i_sent] = len(w_list) - 1 #remove the final <eos>
    sent_embs_tensor = sent_embs_tensor / (0.000000000001 + sent_embs_tensor.norm(dim = 1, keepdim=True) )
    sent_embs_w_tensor = sent_embs_w_tensor / (0.000000000001 + sent_embs_w_tensor.norm(dim = 1, keepdim=True) )

    #flatten_article = [ w for sent in article for w in sent.split() ]
    #num_word = len(flatten_article)
    artical_counter = Counter([ w for sent in article for w in sent.split() ])
    num_word = len(artical_counter)
    all_words_tensor = torch.zeros(num_word, emb_size, device = device)
    freq_prob_tensor = torch.zeros( num_word, device = device)
    w_freq_tensor = torch.zeros( num_word, device = device)
    #for i_w, w in enumerate(flatten_article):
    for i_w, (w, w_freq) in enumerate(artical_counter.items()):
        w_freq_tensor[i_w] = w_freq
        if w in word_d2_idx_freq:
            w_idx, freq, freq_prob = word_d2_idx_freq[w]
            freq_prob_tensor[i_w] = freq_prob
            all_words_tensor[i_w,:] = word_norm_emb[w_idx,:]

    return sent_embs_tensor, sent_embs_w_tensor, all_words_tensor, w_emb_tensors_list, sent_lens, freq_prob_tensor.view(1,-1), w_freq_tensor.view(1,-1)

def greedy_selection(sent_words_sim, top_k_max, sent_lens = None):
    num_words = sent_words_sim.size(1)
    max_sim = -10000 * torch.ones( (1,num_words), device = device )
    max_sent_idx_list = []
    for i in range(top_k_max):
        sent_sim_improvement = sent_words_sim - max_sim
        sent_sim_improvement[sent_sim_improvement<0] = 0
        sim_improve_sum = sent_sim_improvement.sum(dim = 1)
        if sent_lens is not None:
            sim_improve_sum = sim_improve_sum / sent_lens
        selected_sent = torch.argmax(sim_improve_sum)

        max_sim = max_sim + sent_sim_improvement[selected_sent,:]
        max_sent_idx_list.append(selected_sent.item())
    
    return max_sent_idx_list

def select_by_avg_dist_boost( sent_embs_tensor, all_words_tensor, w_freq_tensor, top_k_max, device, freq_w_tensor = None):
    sent_words_sim = torch.mm( sent_embs_tensor, all_words_tensor.t() )
    sent_words_sim *= w_freq_tensor
    if freq_w_tensor is not None:
        sent_words_sim *= freq_w_tensor
    max_sent_idx_list = greedy_selection(sent_words_sim, top_k_max)
    return max_sent_idx_list

def select_by_topics(basis_coeff_list, all_words_tensor, w_freq_tensor, top_k_max, device, sent_lens = None, freq_w_tensor = None):
    sent_num = len(basis_coeff_list)
    num_words = all_words_tensor.size(0)
    sent_word_sim = torch.empty( (sent_num, num_words), device = device )
    for i in range(sent_num):
        if sent_lens is None:
            basis, topic_w = basis_coeff_list[i]
        else:
            basis= basis_coeff_list[i]
        max_v, max_idx = torch.mm( basis, all_words_tensor.t() ).max(dim = 0)
        sent_word_sim[i,:] = max_v
    sent_word_sim *= w_freq_tensor
    if freq_w_tensor is not None:
        sent_word_sim *= freq_w_tensor
    max_sent_idx_list = greedy_selection(sent_word_sim, top_k_max, sent_lens)
    return max_sent_idx_list

#def select_by_clustering_sents(sent_embs_tensor, top_k_max, device, sent_lens = None):
#    sent_embs_numpy = sent_embs_tensor.cpu().numpy()
#    max_sent_idx_list = []
#    for i in range(top_k_max):
#        kmeans_model = KMeans(n_clusters=i)
#        kmeans_model.fit(sent_embs_numpy, sample_weight = sent_lens)
#        center_tensor = torch.tensor(kmeans_model.cluster_centers_, device = device)
#        sent_center_sim = torch.mm(sent_embs_tensor, center_tensor.t()) #assume sentence embeddings are normalized
#        max_v, max_idx = torch.max(sent_center_sim, dim = 0)
#        max_sent_idx_list.append( max_idx.cpu().tolist() )
#
#    return max_sent_idx_list

def select_by_clustering_words(sent_embs_tensor, all_words_tensor, top_k_max, device, sample_w = None):
    all_words_numpy = all_words_tensor.cpu().numpy()
    if sample_w is not None:
        sample_w_numpy = sample_w.cpu().numpy().reshape(-1)
    else:
        sample_w_numpy = None
    max_sent_idx_list = []
    #for i in range(1,top_k_max+1):
    #    print(i)
    #    kmeans_model = MiniBatchKMeans(n_clusters=i)
        #if sample_w is None:
        #    #print(all_words_numpy.tolist())
        #    kmeans_model.fit(all_words_numpy)
        #else:
        #    kmeans_model.fit(all_words_numpy, sample_weight = sample_w_numpy)
    for i in range(top_k_max):
        if all_words_tensor.size(0) < i+1:
            break
        kmeans_model = all_clustering_models[i]
        kmeans_model.fit(all_words_numpy, sample_weight = sample_w_numpy)
        center_tensor = torch.tensor(kmeans_model.cluster_centers_, device = device)
        sent_center_sim = torch.mm(sent_embs_tensor, center_tensor.t()) #assume sentence embeddings are normalized
        max_v, max_idx = torch.max(sent_center_sim, dim = 0)
        max_sent_idx_list.append( max_idx.cpu().tolist() )

    return max_sent_idx_list

def rank_sents(basis_coeff_list, article, word_norm_emb, word_d2_idx_freq, top_k_max, device):
    sent_embs_tensor, sent_embs_w_tensor, all_words_tensor, w_emb_tensors_list, sent_lens, freq_prob_tensor, w_freq_tensor = article_to_embs(article, word_norm_emb, word_d2_idx_freq, device)
    
    alpha = 0.0001
    freq_w_4_tensor = alpha / (alpha + freq_prob_tensor)
    m_d2_sent_ranks = {} 
    if 'embs' in args.method_set:
        m_d2_sent_ranks['sent_emb_dist_avg'] = select_by_avg_dist_boost( sent_embs_tensor, all_words_tensor, w_freq_tensor, top_k_max, device )
        m_d2_sent_ranks['sent_emb_dist_avg_freq_4'] = select_by_avg_dist_boost( sent_embs_tensor, all_words_tensor, w_freq_tensor, top_k_max, device, freq_w_4_tensor )
        m_d2_sent_ranks['sent_emb_freq_4_dist_avg_freq_4'] = select_by_avg_dist_boost( sent_embs_w_tensor, all_words_tensor, w_freq_tensor, top_k_max, device, freq_w_4_tensor )
        m_d2_sent_ranks['norm_w_in_sent'] = select_by_topics( w_emb_tensors_list, all_words_tensor, w_freq_tensor, top_k_max, device, sent_lens)
        m_d2_sent_ranks['norm_w_in_sent_freq_4'] = select_by_topics( w_emb_tensors_list, all_words_tensor, w_freq_tensor, top_k_max, device, sent_lens, freq_w_tensor = freq_w_4_tensor)
    if 'cluster' in args.method_set:
        #m_d2_sent_ranks['sent_emb_cluster_sent'] = select_by_clustering_sents(sent_embs_tensor, top_k_max, device)
        #m_d2_sent_ranks['sent_emb_cluster_sent_len'] = select_by_clustering_sents(sent_embs_tensor, top_k_max, device, sent_lens)
        
        #m_d2_sent_ranks['sent_emb_cluster_sent'] = select_by_clustering_words(sent_embs_tensor, sent_embs_tensor, top_k_max, device)
        #m_d2_sent_ranks['sent_emb_cluster_sent_len'] = select_by_clustering_words(sent_embs_tensor, sent_embs_tensor, top_k_max, device, sent_lens)
        m_d2_sent_ranks['sent_emb_cluster_word'] = select_by_clustering_words(sent_embs_tensor, all_words_tensor, top_k_max, device, w_freq_tensor)
        m_d2_sent_ranks['sent_emb_cluster_word_freq_4'] = select_by_clustering_words(sent_embs_tensor, all_words_tensor, top_k_max, device, w_freq_tensor*freq_w_4_tensor)
        m_d2_sent_ranks['sent_emb_freq_4_cluster_sent'] = select_by_clustering_words(sent_embs_w_tensor, sent_embs_w_tensor, top_k_max, device)
        m_d2_sent_ranks['sent_emb_freq_4_cluster_sent_len'] = select_by_clustering_words(sent_embs_w_tensor, sent_embs_w_tensor, top_k_max, device, sent_lens)
        #m_d2_sent_ranks['sent_emb_freq_4_cluster_word'] = select_by_clustering_words(sent_embs_w_tensor, all_words_tensor, top_k_max, device, w_freq_tensor)
        #m_d2_sent_ranks['sent_emb_freq_4_cluster_word_freq_4'] = select_by_clustering_words(sent_embs_w_tensor, all_words_tensor, top_k_max, device, w_freq_tensor*freq_w_4_tensor)
    if basis_coeff_list is not None:
        assert len(basis_coeff_list) == len(article)
        m_d2_sent_ranks['ours'] = select_by_topics(basis_coeff_list, all_words_tensor, w_freq_tensor, top_k_max, device)
        m_d2_sent_ranks['ours_freq_4'] = select_by_topics(basis_coeff_list, all_words_tensor, w_freq_tensor, top_k_max, device, sent_lens = None, freq_w_tensor = freq_w_4_tensor)
    
    return m_d2_sent_ranks


fname_d2_sent_rank = {}
fname_list = []
article_list = []
abstract_list = []

stories = os.listdir(args.input)
for file_name in stories:
    print("Processing "+ file_name)
    sys.stdout.flush()
    #base_name = os.path.basename(file_name)
    with open(args.input + '/' + file_name) as f_in:
        
        fname_list.append(file_name)
        article, abstract = load_tokenized_story(f_in)
        article_list.append(article)
        abstract_list.append(abstract)
        with torch.no_grad():
            if 'ours' in args.method_set:
                dataloader_test = load_testing_article_summ(word_d2_idx_freq, article, args.max_sent_len, args.batch_size, device)
                #sent_d2_basis, article_proc = utils_testing.output_sent_basis_summ(dataloader_test, org_sent_list, parallel_encoder, parallel_decoder, args.n_basis, idx2word_freq)
                basis_coeff_list, article_proc = utils_testing.output_sent_basis_summ(dataloader_test, parallel_encoder, parallel_decoder, args.n_basis, idx2word_freq)
            else:
                basis_coeff_list = None
                article_proc = article
            fname_d2_sent_rank[file_name] = rank_sents(basis_coeff_list, article_proc, word_norm_emb, word_d2_idx_freq, args.top_k_max, device)


with open(args.outf_vis, 'w') as f_out:
    json.dump(fname_d2_sent_rank, f_out, indent = 1)

########################
print("Scoring all methods")
########################

all_method_list = []

if 'ours' in args.method_set:
    all_method_list += ['ours', 'ours_freq_4']
if 'embs' in args.method_set:
    all_method_list += ['sent_emb_dist_avg', 'sent_emb_dist_avg_freq_4', 'sent_emb_freq_4_dist_avg_freq_4', 'norm_w_in_sent', 'norm_w_in_sent_freq_4']
if 'cluster' in args.method_set:
    all_method_list += ['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4']
    #all_method_list += ['sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4']
    #all_method_list += ['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_freq_4_cluster_word','sent_emb_freq_4_cluster_word_freq_4']

all_method_list += ['first','rnd']
#all_method_list = ['ours', 'ours_freq_4', 'sent_emb_dist_avg', 'sent_emb_dist_avg_freq_4', 'norm_w_in_sent', 'norm_w_in_sent_freq_4', 'sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4', 'first']

not_inclusive_methods = set(['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_freq_4_cluster_word','sent_emb_freq_4_cluster_word_freq_4', 'sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4'])

pretty_header = ['method_name', 'sent_num', 'avg_sent_len', 'ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-SU4-F']

m_d2_output_list = {}
for m in all_method_list:
    m_d2_output_list[m] = []

for top_k in range(1,args.top_k_max+1):
    logger.logging("top k: "+str(top_k))
    for method in all_method_list:
        logger.logging(method)
        selected_sent_all = []
        summ_len_sum = 0
        effective_doc_count = 0
        for i in range(len(fname_list)):
            file_name = fname_list[i]
            article = article_list[i]
            if len(article) <= top_k:
                continue
            if method == 'first':
                sent_rank = list(range(top_k))
            elif method == 'rnd':
                sent_rank = np.random.choice(len(article), top_k).tolist()
            else:
                sent_rank = fname_d2_sent_rank[file_name][method]
            if method in not_inclusive_methods:
                selected_sent = [article[s] for s in sent_rank[top_k-1]]
            else:
                selected_sent = [article[s] for s in sent_rank[:top_k]]
            summ_len = sum([len(sent.split()) for sent in selected_sent])
            summ_len_sum += summ_len
            effective_doc_count += 1
            selected_sent_all.append(selected_sent)
        rouge = Pythonrouge(summary_file_exist=False,
                    summary=selected_sent_all, reference=abstract_list,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    #recall_only=True, stemming=True, stopwords=True,
                    #recall_only=False, stemming=True, stopwords=True,
                    recall_only=False, stemming=True, stopwords=False,
                    word_level=True, length_limit=False, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        avg_summ_len = summ_len_sum / float(effective_doc_count)
        logger.logging(str(score))
        logger.logging("summarization length "+str(avg_summ_len))
        m_d2_output_list[method].append( [method, str(top_k), str(avg_summ_len), str(score[pretty_header[3]]), str(score[pretty_header[4]]), str(score[pretty_header[5]])])

logger.logging(','.join(pretty_header))
for method in m_d2_output_list:
    for fields in m_d2_output_list[method]:
        logger.logging(','.join(fields))
