import argparse
import os
import numpy as np
import random
import torch
import sys
import json
sys.path.insert(0, sys.path[0]+'/../..')


#import torch.nn as nn
#import torch.utils.data
#import coherency_eval

from utils import seed_all_randomness, load_corpus, load_testing_sent, loading_all_models, str2bool
import utils_testing

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--input', type=str, default='',
                    help='location of the data corpus')
#parser.add_argument('--dict', type=str, default='./data/processed/wackypedia/dictionary_index',
parser.add_argument('--dict', type=str, default='./data/processed/wiki2016_min100/dictionary_index',
                    help='location of the dictionary corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.json',
                    help='output file for generated text')
parser.add_argument('--outf_vis', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
#parser.add_argument('--max_batch_num', type=int, default=100,
#                    help='number of batches for evaluation')

###data_format
parser.add_argument('--max_sent_len', type=int, default=50,
                    help='max sentence length for input features')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()
print(args)

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint,"target_emb.pt")

#if args.nhidlast2 < 0:
#    args.nhidlast2 = args.emsize
#if args.linear_mapping_dim < 0:
#    args.linear_mapping_dim = args.nhid

#if args.trans_nhid < 0:
#    args.trans_nhid = args.emsize

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed, args.cuda)

########################
print("Preprocessing data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

dataloader_test, org_sent_list, idx2word_freq = load_testing_sent(args.dict, args.input, args.max_sent_len, args.batch_size, device)

########################
print("Loading Models")
########################

#parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, use_position_emb = True)
#parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, use_position_emb = False)
#parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, args.linear_mapping_dim)
parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, args.max_sent_len)

encoder.eval()
decoder.eval()

with open(args.outf_vis, 'w') as outf_vis:
    basis_json = utils_testing.output_sent_basis(dataloader_test, org_sent_list, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, args.n_basis, outf_vis)
with open(args.outf, 'w') as outf:
    json.dump(basis_json, outf, indent = 1)

