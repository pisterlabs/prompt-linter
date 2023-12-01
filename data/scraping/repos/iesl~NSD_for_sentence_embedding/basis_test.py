import argparse
import os
import numpy as np
import random
import torch
#import torch.nn as nn
#import torch.utils.data
#import coherency_eval

from utils import seed_all_randomness, load_corpus, loading_all_models, str2bool
import utils_testing

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/wackypedia/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
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
parser.add_argument('--max_batch_num', type=int, default=100, 
                    help='number of batches for evaluation')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint,"target_emb.pt")

#if args.nhidlast2 < 0:
#    args.nhidlast2 = args.emsize

#if args.trans_nhid < 0:
#    args.trans_nhid = args.emsize

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)


########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device )
#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = False )
#idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = True, tensor_folder = args.tensor_folder )
idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = False, tensor_folder = args.tensor_folder )
dataloader_train = dataloader_train_arr[0]

########################
print("Loading Models")
########################


parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, max_sent_len)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameters(model, excluding_prefix = None):
    parameter_sum = 0
    for name, param in model.named_parameters():
        if excluding_prefix is not None and excluding_prefix == name[:len(excluding_prefix)]:
            print("Skipping ", name, param.numel())
            continue
        if param.requires_grad:
            print(name, param.numel())
            parameter_sum += param.numel()
    return parameter_sum

#print("encoder after filtering ", print_parameters(encoder, 'encoder'))
#print("decoder after filtering ", print_parameters(decoder, 'coeff'))
#print(count_parameters(encoder))
#print(count_parameters(decoder))


encoder.eval()
decoder.eval()

with open(args.outf, 'w') as outf:
    utils_testing.test_time(dataloader_val_shuffled, encoder, decoder, args.max_batch_num)
    #outf.write('Shuffled Validation Topics:\n\n')
    #utils_testing.visualize_topics_val(dataloader_val_shuffled, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num)
    #outf.write('Validation Topics:\n\n')
    #utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num)
    if dataloader_train:
        outf.write('Training Topics:\n\n')
        utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num)

#test_batch_size = 1
#test_data = batchify(corpus.test, test_batch_size, args)
