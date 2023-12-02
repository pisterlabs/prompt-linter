import argparse
import os
import numpy as np
import random
import torch
#import torch.nn as nn
#import torch.utils.data
#import coherency_eval
import sys
sys.path.insert(0, sys.path[0]+'/..')
from utils import seed_all_randomness, load_corpus, loading_all_models, str2bool
import utils_testing
from gpt2_model.tokenization_gpt2 import GPT2Tokenizer

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_10000000_min100',
                    help='location of the data corpus')
parser.add_argument('--checkpoint_topics', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
#parser.add_argument('--gpt2_vocab_file', type=str, default='./resources/distilgpt2-vocab.json',
parser.add_argument('--gpt2_vocab_file', type=str, default='',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda_topics', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=512,
                    help='sequence length')
parser.add_argument('--dilated_head_span', type=int, default=10,
                    help='The span of each head which generates the topics')
parser.add_argument('--max_batch_num', type=int, default=100, 
                    help='number of batches for evaluation')
parser.add_argument('--topic_models', type=str, default='NSD_vis',
                    help='The topic models will be tested or visualized. Could be NSD_vis or NSD+kmeans_cluster+SC_cluster+LDA_org+random_word+random_vocab+global_centers')
parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
                    help='path to the file of a stop word list')
parser.add_argument('--LDA_model_path', type=str, default='',
                    help='path to the file of a LDA mdoel')
parser.add_argument('--word_emb_center_path', type=str, default='',
                    help='path to the file of a clustering results in a word embedding space')
parser.add_argument('--readable_context', type=str2bool, nargs='?', default=False,
                    help='use CUDA for conditional LM')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint_topics,"target_emb.pt")

#if args.nhidlast2 < 0:
#    args.nhidlast2 = args.emsize

#if args.trans_nhid < 0:
#    args.trans_nhid = args.emsize

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda_topics)


########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda_topics else "cpu")

#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device )
#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = False )
random_start = True
#random_start = False
#if args.topic_models == "NSD_vis":
#    random_start = True
#idx2word_freq, dataloader_train_arr, dataloader_val = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, -1, args.dilated_head_span, device, args.tensor_folder, skip_training = True, want_to_shuffle_val = True, random_start = random_start )
idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, -1, args.dilated_head_span, device, args.tensor_folder, skip_training = True, want_to_shuffle_val = True, load_testing = True, random_start = random_start )
#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, -1, args.dilated_head_span, device, args.tensor_folder, skip_training = True, want_to_shuffle_val = False, load_testing = True, random_start = random_start )
dataloader_train = dataloader_train_arr[0]

utils_testing.compute_freq_prob_idx2word(idx2word_freq)

########################
print("Loading Models")
########################

parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device)

if len(args.gpt2_vocab_file) == 0:
    tokenizer_GPT2 = GPT2Tokenizer.from_pretrained('distilgpt2')
else:
    with open(args.gpt2_vocab_file) as f_in:
        idx_l2_w_gpt2 = utils_testing.load_gpt2_vocab(f_in)

if args.topic_models != 'NSD_vis':
    def convert_stop_to_ind_lower(f_in, idx2word_freq):
        stop_word_org_set = set()
        for line in f_in:
            w = line.rstrip()
            stop_word_org_set.add(w)
        stop_word_set = set()
        for idx, (w, freq, prob) in enumerate(idx2word_freq):
            if w.lower() in stop_word_org_set:
                stop_word_set.add(idx)
        return stop_word_set
    with open(args.stop_word_file) as f_in:
        stop_word_set = convert_stop_to_ind_lower(f_in, idx2word_freq)

encoder.eval()
decoder.eval()

with open(args.outf, 'w') as outf:
    if args.topic_models == 'NSD_vis':
        outf.write('Validation Topics:\n\n')
        #utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, idx_l2_w_gpt2)
        utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, tokenizer_GPT2)
        if dataloader_train:
            outf.write('Training Topics:\n\n')
            #utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, idx_l2_w_gpt2 )
            utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, args.de_en_connection, tokenizer_GPT2 )

    #test_batch_size = 1
    #test_data = batchify(corpus.test, test_batch_size, args)
    else:
        #utils_testing.testing_all_topic_baselines(dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, tokenizer_GPT2, stop_word_set, args.topic_models, args.de_en_connection, args.LDA_model_path, args.word_emb_center_path)
        utils_testing.testing_all_topic_baselines(dataloader_test, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num, tokenizer_GPT2, stop_word_set, args.topic_models, args.de_en_connection, args.LDA_model_path, args.word_emb_center_path, args.readable_context)
