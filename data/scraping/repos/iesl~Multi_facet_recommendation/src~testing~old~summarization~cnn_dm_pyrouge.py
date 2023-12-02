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

bert_dir = '/mnt/nfs/scratch1/hschang/language_modeling/pytorch-pretrained-BERT'
sys.path.insert(1, bert_dir)
from pytorch_pretrained_bert import BertTokenizer, BertModel

#import torch.nn as nn
#import torch.utils.data
#import coherency_eval

from utils import seed_all_randomness, load_word_dict, load_emb_from_path, load_idx2word_freq, loading_all_models, load_testing_article_summ, str2bool, Logger
import utils_testing
from pythonrouge.pythonrouge import Pythonrouge
from pyrouge import Rouge155
import uuid
import string
import logging
import tempfile

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
temp_file_dir = "./temp/"
assert os.path.isdir(temp_file_dir)

#pyrouge will write temp file to /tmp/, but it won't delete it, so we need to manually redirect the temp files and delete them
unique_dir_name = str(uuid.uuid4())
tempfile.tempdir = "./temp_pyrouge/" + unique_dir_name + '/'

os.mkdir(tempfile.tempdir)

idx2word_freq = None
if args.dict != 'None':
    with open(args.dict) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)

word_norm_emb = None
if 'ours' in args.method_set:
    parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, args.max_sent_len)

    encoder.eval()
    decoder.eval()
elif args.method_set != 'bert':
    word_emb, output_emb_size = load_emb_from_path(args.emb_file, device, idx2word_freq)
    word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
    word_norm_emb[0,:] = 0

if 'bert' in args.method_set:
    #BERT_model_path = 'bert-base-cased'
    BERT_model_path = 'bert-large-cased'
    lower_case = False
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_model_path, cache_dir = bert_dir + '/cache_dir/', do_lower_case = lower_case)
    bert_max_len = 2 + 500
    bert_batch_size = 100
    bert_model = BertModel.from_pretrained(BERT_model_path)
    bert_model.eval()
    bert_model.to(device)

def run_bert(input_sents, device, bert_tokenizer, bert_model, word_d2_idx_freq):
    freq_prob_list = []
    sent_emb_list = []
    w_emb_list = []
    idx_list = []
    sent_lens = []
    for sent in input_sents:
        tokenized_text = bert_tokenizer.tokenize('[CLS] ' + sent + ' [SEP]')
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        idx_list.append(indexed_tokens)
        sent_lens.append( min(len(indexed_tokens), bert_max_len) )
        if word_d2_idx_freq is not None:
            for w in tokenized_text:
                if w in word_d2_idx_freq:
                    w_idx, freq, freq_prob = word_d2_idx_freq[w]
                    freq_prob_list.append(freq_prob)
                else:
                    freq_prob_list.append(0) 
    if word_d2_idx_freq is None:
        freq_prob_tensor = None
    else:
        freq_prob_tensor = torch.tensor(freq_prob_list,device = device)

    for i in range(len(idx_list)):
        storing_idx = i % bert_batch_size
        if storing_idx == 0:
            sys.stdout.write(str(i)+' ')
            sys.stdout.flush()
            tokens_tensor = torch.zeros(bert_batch_size, bert_max_len, device = device, dtype = torch.long)
            mask_tensor = torch.zeros(bert_batch_size, bert_max_len, device = device, dtype = torch.long)
            length_list = []
        proc_sent = idx_list[i]
        #sent_len = len(proc_sent)
        sent_len = sent_lens[i]
        tokens_tensor[storing_idx, :sent_len] = torch.tensor(proc_sent[:sent_len], device = device)
        mask_tensor[storing_idx, :sent_len] = 1
        length_list.append(sent_len)
        #proc_sent_inner.append(idx_list[i])

        if storing_idx == bert_batch_size - 1 or i == len(idx_list)-1:
            with torch.no_grad():
                encoded_layers_batch, cls_emb_batch = bert_model(tokens_tensor, attention_mask=mask_tensor, output_all_encoded_layers=False)
            for inner_i in range(len(length_list)):
                sent_len = length_list[inner_i]
                avg_emb = torch.mean(encoded_layers_batch[inner_i, :sent_len, :], dim = 0, keepdim=True)
                w_emb = encoded_layers_batch[inner_i, :sent_len, :]
                w_emb_list.append(w_emb)
                sent_emb_list.append(avg_emb)
    
    return sent_emb_list, w_emb_list, sent_lens, freq_prob_tensor

word_d2_idx_freq = None
if args.dict != 'None':
    with open(args.dict) as f_in:
        word_d2_idx_freq, max_ind = load_word_dict(f_in)

    utils_testing.compute_freq_prob(word_d2_idx_freq)

#print(word_d2_idx_freq)

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


def article_to_embs_bert(article, bert_tokenizer, bert_model, word_d2_idx_freq, device): 
    sent_emb_list, w_emb_tensors_list, sent_lens_list, freq_prob_tensor = run_bert(article, device, bert_tokenizer, bert_model, word_d2_idx_freq)
    sent_embs_tensor = torch.cat(sent_emb_list, dim=0)
    all_words_tensor = torch.cat(w_emb_tensors_list, dim=0)
    sent_embs_tensor = sent_embs_tensor / (0.000000000001 + sent_embs_tensor.norm(dim = 1, keepdim=True) )
    all_words_tensor = all_words_tensor / (0.000000000001 + all_words_tensor.norm(dim = 1, keepdim=True) )

    sent_lens = torch.tensor(sent_lens_list, dtype=torch.float32 , device = device)
    
    num_word = all_words_tensor.size(0)
    w_freq_tensor = torch.ones(1,num_word,device = device)
        
    #emb_size = sent_emb_list[0].size(1)
    #num_sent = len(article)
    #sent_embs_tensor = torch.zeros(num_sent, emb_size, device = device)
    #w_emb_tensors_list = []
    #sent_lens = torch.empty(num_sent, device = device)
    return sent_embs_tensor, all_words_tensor, w_emb_tensors_list, sent_lens, freq_prob_tensor, w_freq_tensor

##convert the set of word embedding we want to reconstruct into tensor
def article_to_embs(article, word_norm_emb, word_d2_idx_freq, device):
    emb_size = word_norm_emb.size(1)
    num_sent = len(article)
    sent_embs_tensor = torch.zeros(num_sent, emb_size, device = device)
    sent_embs_w_tensor = torch.zeros(num_sent, emb_size, device = device)
    w_emb_tensors_list = []
    sent_lens = torch.empty(num_sent, device = device)
    alpha = 0.0001
    #print(article)
    for i_sent, sent in enumerate(article):
        w_emb_list = []
        #print(sent)
        w_list = sent.split()
        for w in w_list:
            #sys.stdout.write(w + ' ')
            if w in word_d2_idx_freq:
                w_idx, freq, freq_prob = word_d2_idx_freq[w]
                sent_embs_tensor[i_sent,:] += word_norm_emb[w_idx,:]
                sent_embs_w_tensor[i_sent,:] += word_norm_emb[w_idx,:] * (alpha / (alpha + freq_prob) )
                w_emb_list.append(word_norm_emb[w_idx,:].view(1,-1))
        #print(len(w_emb_list))
        if len(w_emb_list) > 0:
            w_emb_tensors_list.append( torch.cat(w_emb_list, dim = 0) )
        else:
            w_emb_tensors_list.append( torch.zeros(1, emb_size, device = device) )
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
    #max_sim = -10000 * torch.ones( (1,num_words), device = device )
    max_sim = -1 * torch.ones( (1,num_words), device = device )
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
    alpha = 0.0001
    m_d2_sent_ranks = {} 
    if args.method_set != 'bert':
        sent_embs_tensor, sent_embs_w_tensor, all_words_tensor, w_emb_tensors_list, sent_lens, freq_prob_tensor, w_freq_tensor = article_to_embs(article, word_norm_emb, word_d2_idx_freq, device)
        freq_w_4_tensor = alpha / (alpha + freq_prob_tensor)
    if 'bert' in args.method_set:
        sent_embs_tensor_bert, all_words_tensor_bert, w_emb_tensors_list_bert, sent_lens_bert, freq_prob_tensor_bert, w_freq_tensor_bert = article_to_embs_bert(article, bert_tokenizer, bert_model, word_d2_idx_freq, device)

        #m_d2_sent_ranks['bert_sent_emb_dist_avg'] = select_by_avg_dist_boost( sent_embs_tensor_bert, all_words_tensor_bert, w_freq_tensor_bert, top_k_max, device )
        #m_d2_sent_ranks['bert_norm_w_in_sent'] = select_by_topics( w_emb_tensors_list_bert, all_words_tensor_bert, w_freq_tensor_bert, top_k_max, device, sent_lens_bert)
        if freq_prob_tensor_bert is not None:
            freq_w_4_tensor_bert = alpha / (alpha + freq_prob_tensor_bert)
            m_d2_sent_ranks['bert_sent_emb_dist_avg_freq_4'] = select_by_avg_dist_boost( sent_embs_tensor_bert, all_words_tensor_bert, w_freq_tensor_bert, top_k_max, device, freq_w_4_tensor_bert )
            m_d2_sent_ranks['bert_norm_w_in_sent_freq_4'] = select_by_topics( w_emb_tensors_list_bert, all_words_tensor_bert, w_freq_tensor_bert, top_k_max, device, sent_lens_bert, freq_w_tensor = freq_w_4_tensor_bert)


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

def eval_by_pyrouge(selected_sent_all, abstract_list, temp_file_prefix, temp_dir_for_pyrouge):
    r = Rouge155(log_level=logging.ERROR)

    #r.model_dir = temp_file_dir
    #r.system_dir = temp_file_dir
    r.model_dir = temp_file_prefix
    r.system_dir = temp_file_prefix
    #file_basename = os.path.basename(temp_file_prefix)
    #r.system_filename_pattern = file_basename + '.(\d+).txt'
    #r.model_filename_pattern = file_basename + '.[A-Z].#ID#.txt'
    r.system_filename_pattern = 'sys.(\d+).txt'
    r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
    assert len(abstract_list) < 10000000
    assert len(abstract_list) == len(selected_sent_all)
    #alphabet = string.ascii_uppercase
    #cmd = 'rm '+ temp_file_prefix +'*'
    #os.system(cmd)
    for i in range(len(abstract_list)):
        np.savetxt(temp_file_prefix + 'sys.{0:07d}.txt'.format(i), selected_sent_all[i], newline = '\n', fmt="%s")
        abstract_list_i_flat = [ x[k] for x in abstract_list[i] for k in range(len(x))]
        np.savetxt(temp_file_prefix + 'model.A.{:07d}.txt'.format(i), abstract_list_i_flat, newline = '\n', fmt="%s")
        #for j in range(len(abstract_list[i])):
        #    np.savetxt(temp_file_prefix + '.{}.{:07d}.txt'.format(alphabet[j],i), abstract_list[i][j], newline = '\n', fmt="%s")
    output = r.convert_and_evaluate()
    output = r.output_to_dict(output)
    cmd = 'rm '+ temp_file_prefix +'*'
    os.system(cmd)
    cmd = 'rm -r '+ temp_dir_for_pyrouge +'*'
    os.system(cmd)
    return output
            
    

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
if 'bert' in args.method_set:
    all_method_list += ['bert_sent_emb_dist_avg', 'bert_sent_emb_dist_avg_freq_4', 'bert_norm_w_in_sent', 'bert_norm_w_in_sent_freq_4']
if 'cluster' in args.method_set:
    all_method_list += ['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4']
    #all_method_list += ['sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4']
    #all_method_list += ['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_freq_4_cluster_word','sent_emb_freq_4_cluster_word_freq_4']

#all_method_list += ['first','rnd']
#all_method_list = ['ours', 'ours_freq_4', 'sent_emb_dist_avg', 'sent_emb_dist_avg_freq_4', 'norm_w_in_sent', 'norm_w_in_sent_freq_4', 'sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4', 'first']

not_inclusive_methods = set(['sent_emb_freq_4_cluster_sent','sent_emb_freq_4_cluster_sent_len','sent_emb_freq_4_cluster_word','sent_emb_freq_4_cluster_word_freq_4', 'sent_emb_cluster_sent','sent_emb_cluster_sent_len','sent_emb_cluster_word','sent_emb_cluster_word_freq_4'])

#pretty_header = ['method_name', 'sent_num', 'avg_sent_len', 'ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-SU4-F']
pretty_header = ['method_name', 'sent_num', 'avg_sent_len', 'rouge_1_f_score', 'rouge_2_f_score', 'rouge_su*_f_score', 'rouge_l_f_score']

unique_dir_name = str(uuid.uuid4())
temp_file_prefix = temp_file_dir + unique_dir_name + '/'

os.mkdir(temp_file_prefix)

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
            #if len(article) <= top_k:
            #    continue
            top_k_art_len = min(len(article), top_k)
            if method == 'first':
                sent_rank = list(range(top_k_art_len))
            elif method == 'rnd':
                sent_rank = np.random.choice(len(article), top_k_art_len).tolist()
            else:
                if method not in fname_d2_sent_rank[file_name]:
                    continue
                sent_rank = fname_d2_sent_rank[file_name][method]
            if method in not_inclusive_methods:
                selected_sent = [article[s] for s in sent_rank[top_k_art_len-1]]
            else:
                selected_sent = [article[s] for s in sent_rank[:top_k_art_len]]
            summ_len = sum([len(sent.split()) for sent in set(selected_sent)])
            summ_len_sum += summ_len
            effective_doc_count += 1
            selected_sent_all.append(selected_sent)
        #selected_sent_all
        if len(selected_sent_all) != len(abstract_list):
            logger.logging(str(len(selected_sent_all)))
            logger.logging(str(len(abstract_list)))
            logger.logging("do not run " + method)
            continue
        score = eval_by_pyrouge(selected_sent_all, abstract_list, temp_file_prefix, tempfile.tempdir)
        #rouge = Pythonrouge(summary_file_exist=False, summary=selected_sent_all, reference=abstract_list, n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
        #            #recall_only=True, stemming=True, stopwords=True,
        #            #recall_only=False, stemming=True, stopwords=True,
        #            recall_only=False, stemming=False, stopwords=False,
        #            word_level=True, length_limit=False, length=50,
        #            use_cf=False, cf=95, scoring_formula='average',
        #            resampling=True, samples=1000, favor=True, p=0.5)
        #score = rouge.calc_score()
        avg_summ_len = summ_len_sum / float(effective_doc_count)
        logger.logging(str(score))
        logger.logging("summarization length "+str(avg_summ_len))
        m_d2_output_list[method].append( [method, str(top_k), str(avg_summ_len), str(score[pretty_header[3]]), str(score[pretty_header[4]]), str(score[pretty_header[5]]), str(score[pretty_header[6]])])

cmd = 'rm -r '+ tempfile.tempdir
os.system(cmd)
cmd = 'rm -r '+ temp_file_prefix
os.system(cmd)

logger.logging(','.join(pretty_header))
for method in m_d2_output_list:
    for fields in m_d2_output_list[method]:
        logger.logging(','.join(fields))
