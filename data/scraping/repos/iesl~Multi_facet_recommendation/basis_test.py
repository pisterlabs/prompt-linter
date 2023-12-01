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
parser.add_argument('--data', type=str, default='./data/processed/citeulike-a_lower/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_cold_0',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--tag_emb_file', type=str, default='tag_emb.pt',
                    help='path to the file of an user embedding file')
parser.add_argument('--user_emb_file', type=str, default='user_emb.pt',
                    help='path to the file of a tag embedding file')
parser.add_argument('--word_emb_file', type=str, default='',
                    help='path to the file of a word embedding file')
parser.add_argument('--testing_target', type=str, default='tag',
                    help='testing tag or user or auto')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')
parser.add_argument('--vis_map', type=str, default='data/raw/gorc/id_title_full_org_cut',
                    help='Load either a mapping from author to titles of the paper he/she wrote or a mapping from paper id to its title')


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

if args.testing_target == 'auto':
    assert len(args.word_emb_file) > 0
else:
    assert len(args.word_emb_file) == 0

#if args.tag_emb_file == "tag_emb.pt":
if args.tag_emb_file[:7] == "tag_emb":
    args.tag_emb_file =  os.path.join(args.checkpoint,args.tag_emb_file)
if args.user_emb_file[:8] == "user_emb":
    args.user_emb_file =  os.path.join(args.checkpoint,args.user_emb_file)

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
#idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, dataloader_val, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = True, tensor_folder = args.tensor_folder )
idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_test, max_sent_len = load_corpus(args.data, args.batch_size, args.batch_size, device, skip_training = True, want_to_shuffle_val = False, tensor_folder = args.tensor_folder, load_test = True )
dataloader_train = dataloader_train_arr[0]

########################
print("Loading Models")
########################

load_auto = False
if args.testing_target == 'auto':
    load_auto = True

output_fields = loading_all_models(args, idx2word_freq, user_idx2word_freq, tag_idx2word_freq, device, max_sent_len, load_auto = load_auto)

if args.testing_target == 'auto':
    parallel_encoder, parallel_decoder, encoder, decoder, user_norm_emb, tag_norm_emb, word_norm_emb_trans = output_fields
else:
    parallel_encoder, parallel_decoder, encoder, decoder, user_norm_emb, tag_norm_emb = output_fields

encoder.eval()
decoder.eval()

word_d2_vis = {}
if len(args.vis_map) > 0:
    with open(args.vis_map) as f_in:
        for line in f_in:
            word, vis = line.rstrip().split('\t')
            word = word.replace(' ','_')
            word_d2_vis[word] = vis

with open(args.outf, 'w') as outf:
    #outf.write('Shuffled Validation Topics:\n\n')
    #utils_testing.visualize_topics_val(dataloader_val_shuffled, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, args.n_basis, args.max_batch_num)
    if args.testing_target == 'tag':
        outf.write('Validation Topics:\n\n')
        #utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, tag_norm_emb, idx2word_freq, tag_idx2word_freq, outf, args.max_batch_num, word_d2_vis)
        utils_testing.visualize_topics_val(dataloader_test, parallel_encoder, parallel_decoder, tag_norm_emb, idx2word_freq, tag_idx2word_freq, outf, args.max_batch_num, word_d2_vis)
        if dataloader_train:
            outf.write('Training Topics:\n\n')
            utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, tag_norm_emb, idx2word_freq, tag_idx2word_freq, outf, args.max_batch_num, word_d2_vis)
    elif args.testing_target == 'user':
        outf.write('Validation Topics:\n\n')
        utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, user_norm_emb, idx2word_freq, user_idx2word_freq, outf, args.max_batch_num, word_d2_vis)
        if dataloader_train:
            outf.write('Training Topics:\n\n')
            utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, user_norm_emb, idx2word_freq, user_idx2word_freq, outf, args.max_batch_num, word_d2_vis)
    elif args.testing_target == 'auto':
        outf.write('Validation Topics:\n\n')
        utils_testing.visualize_topics_val(dataloader_val, parallel_encoder, parallel_decoder, word_norm_emb_trans, idx2word_freq, idx2word_freq, outf, args.max_batch_num)
        if dataloader_train:
            outf.write('Training Topics:\n\n')
            utils_testing.visualize_topics_val(dataloader_train, parallel_encoder, parallel_decoder, word_norm_emb_trans, idx2word_freq, idx2word_freq, outf, args.max_batch_num)

#test_batch_size = 1
#test_data = batchify(corpus.test, test_batch_size, args)
