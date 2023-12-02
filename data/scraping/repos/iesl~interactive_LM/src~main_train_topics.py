import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.cuda as cutorch

import gc

#import data
#import random
import model as model_code
import nsd_loss
#import coherency_eval
from gpt2_model.modeling_gpt2 import GPT2Model
from gpt2_model.configuration_gpt2 import GPT2Config

from utils import seed_all_randomness, create_exp_dir, save_checkpoint, load_idx2word_freq, load_emb_file_to_tensor, load_corpus, output_parallel_models, str2bool

parser = argparse.ArgumentParser(description='PyTorch Train Future Topic Prediction')
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_gpt2/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_all_min100',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,  default='./models/',
                    help='path to save the final model')
parser.add_argument('--emb_file', type=str, default='./resources/glove.840B.300d.txt',
                    help='path to the file of a word embedding file')

parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=512,
                    help='sequence length')
parser.add_argument('--dilated_head_span', type=int, default=10,
                    help='The span of each head which generates the topics')


parser.add_argument('--optimizer', type=str, default="SGD",
                    help='optimization algorithm. Could be SGD or Adam')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--nonmono', type=int, default=2,
                    help='random seed')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--update_target_emb', default=False, action='store_true',
                    help='Whether to update target embedding')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--coeff_opt', type=str, default='lc',
                    help='Could be max or lc')
parser.add_argument('--coeff_opt_algo', type=str, default='rmsprop',
                    help='Could be sgd_bmm, sgd, asgd, adagrad, rmsprop, and adam')
parser.add_argument('--training_split_num', type=int, default=2,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')
parser.add_argument('--start_training_split', type=int, default=0,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')
parser.add_argument('--valid_per_epoch', type=int, default=2,
                    help='Number of times we want to run through validation data and save model within an epoch')


###decoder
#both
parser.add_argument('--de_model', type=str, default='TRANS',
                    help='type of decoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')
parser.add_argument('--n_basis', type=int, default=10,
                    help='number of basis we want to predict')
parser.add_argument('--n_further', type=int, default=50,
                    help='number of words we want to look ahead')
parser.add_argument('--L1_losss_B', type=float, default=0.2,
                    help='L1 loss for the coefficient matrix')
parser.add_argument('--positional_option', type=str, default='linear',
                    help='options of encode positional embedding into models (linear, cat, add)')
parser.add_argument('--dropoutp', type=float, default=0.5,
                    help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')
parser.add_argument('--nhidlast2', type=int, default=300,
                    help='hidden embedding size of the second LSTM/TRANS')
#LSTM only
#TRANS only
parser.add_argument('--trans_layers', type=int, default=5,
                    help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
parser.add_argument('--de_en_connection', type=str2bool, nargs='?', default=True, 
                    help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
parser.add_argument('--dropout_prob_trans', type=float, default=0.1,
                    help='hidden_dropout_prob and attention_probs_dropout_prob in Transformer')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
#parser.add_argument('--cuda', action='store_false',
#parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--single_gpu', default=False, action='store_true', 
                    help='use single GPU')


#print("start")
args = parser.parse_args()

########################
print("Set up environment")
########################

assert args.training_split_num >= args.valid_per_epoch

if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size
assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

if args.coeff_opt == 'maxlc':
    current_coeff_opt = 'max'
else:
    current_coeff_opt = args.coeff_opt

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=['./src/main_train_topics.py', './src/model.py', './src/nsd_loss.py'])


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)

logging('Args: {}'.format(args))

###############################################################################
# Load data
###############################################################################

device = torch.device("cuda" if args.cuda else "cpu")

idx2word_freq, dataloader_train_arr, dataloader_val = load_corpus(args.data, args.batch_size, args.batch_size, args.bptt, args.n_further, args.dilated_head_span, device, args.tensor_folder, args.training_split_num)

#corpus = data.Corpus(args.data)

#eval_batch_size = 10
#test_batch_size = 1
#train_data = batchify(corpus.train, args.batch_size, args)
#val_data = batchify(corpus.valid, eval_batch_size, args)
#test_data = batchify(corpus.test, test_batch_size, args)

def counter_to_tensor(idx2word_freq,device):
    total = len(idx2word_freq)
    w_freq = torch.zeros(total, dtype=torch.float, device = device, requires_grad = False)
    for i in range(total):
        w_freq[i] = 1
        #w_freq[i] = math.sqrt(idx2word_freq[x][1])
        #w_freq[i] = idx2word_freq[x][1]
    w_freq[0] = -1
    return w_freq

external_emb, output_emb_size, extra_init_idx = load_emb_file_to_tensor(args.emb_file, device, idx2word_freq)
external_emb = external_emb / (0.000000000001 + external_emb.norm(dim = 1, keepdim=True))
external_emb.requires_grad = args.update_target_emb
print("loading ", args.emb_file)

w_freq = counter_to_tensor(idx2word_freq,device)

#print("Memory ",str(cutorch.memory_allocated(0)) , ' ', str(cutorch.max_memory_allocated(0)) , ' ' , str(cutorch.get_device_properties(0).total_memory))
###############################################################################
# Build the model
###############################################################################

model_name = 'distilgpt2'

ntokens = len(idx2word_freq)
gpt2_config = GPT2Config.from_pretrained(model_name)

decoder = model_code.EMB2SEQ(args.de_model.split('+'), gpt2_config.n_embd, args.nhidlast2, output_emb_size, 1, args.n_basis, positional_option = args.positional_option, dropoutp= args.dropoutp, trans_layers = args.trans_layers, using_memory =  args.de_en_connection, dropout_prob_trans = args.dropout_prob_trans)

if not args.continue_train:
    encoder = GPT2Model.from_pretrained(model_name)
else:
    encoder_state_dict = torch.load(os.path.join(args.save, 'encoder.pt'), map_location=device)
    encoder = GPT2Model.from_pretrained(model_name, state_dict = encoder_state_dict)
    decoder.load_state_dict(torch.load(os.path.join(args.save, 'decoder.pt'), map_location=device))


#model.config_class.n_embd
#print(model.config_class.__dict__)
#if args.continue_train:
#    #model = torch.load(os.path.join(args.save, 'model.pt'))
#    model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
#    model_further.load_state_dict(torch.load(os.path.join(args.save, 'model_further.pt')))

parallel_encoder, parallel_decoder = output_parallel_models(args.cuda, args.single_gpu, encoder, decoder)

#print("Memory ",str(cutorch.memory_allocated(0)) , ' ', str(cutorch.max_memory_allocated(0)) , ' ' , str(cutorch.get_device_properties(0).total_memory))

total_params = sum(x.data.nelement() for x in encoder.parameters())
logging('Encoder total parameters: {}'.format(total_params))
total_params = sum(x.data.nelement() for x in decoder.parameters())
logging('Decoder total parameters: {}'.format(total_params))


###############################################################################
# Training code
###############################################################################


def evaluate(dataloader, external_emb, current_coeff_opt):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_loss_set = 0
    total_loss_set_reg = 0
    total_loss_set_div = 0
    total_loss_set_neg = 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target_unfold, inner_idx_tensor, future_mask = sample_batched
            target = target_unfold.view(-1, target_unfold.size(2) )
            #print(target)
            output_emb, past = parallel_encoder(feature)
            batch_size, num_head, seq_len = future_mask.size()
            hidden_size = output_emb.size(2)
            output_emb_head = output_emb.gather(dim=1, index=inner_idx_tensor.unsqueeze(dim=-1).expand(batch_size,num_head,hidden_size))
            output_emb_last = output_emb_head.view(-1, output_emb_head.size(2))
            
            #output_emb_masked = output_emb.unsqueeze(dim=1).expand(batch_size,num_head,seq_len,hidden_size)
            #output_emb_masked[future_mask.unsqueeze(dim=-1).expand( batch_size,num_head,seq_len,hidden_size)] = 0
            #output_emb_masked = output_emb_masked.view(-1, seq_len, hidden_size)
            if args.de_en_connection:
                output_emb_masked = output_emb.unsqueeze(dim=1).expand(batch_size,num_head,seq_len,hidden_size)
                #output_emb_masked = output_emb.unsqueeze(dim=0).expand(num_head,batch_size,seq_len,hidden_size)
                #future_mask_expanded = future_mask.unsqueeze(dim=-1).expand( batch_size,num_head,seq_len,hidden_size)
                #output_emb_masked[future_mask_expanded] = 0
                #output_emb_masked = output_emb_masked.view(-1, seq_len, hidden_size)
                output_emb_masked = output_emb_masked.reshape(-1, seq_len, hidden_size)
                future_mask = future_mask.view(-1, seq_len)

                basis_pred = parallel_decoder(output_emb_last, output_emb_masked, memory_attention_mask = future_mask)
            else:
                basis_pred = parallel_decoder(output_emb_last)
            
            #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
            #output_emb_last, output_emb = parallel_encoder(feature)
            #basis_pred=  parallel_decoder(output_emb_last, output_emb)

            compute_target_grad = False
            loss_set, loss_set_reg, loss_set_div, loss_set_neg = nsd_loss.compute_loss_set(basis_pred, external_emb, target, args.L1_losss_B, device, w_freq, current_coeff_opt, compute_target_grad, args.coeff_opt_algo)
            #print(loss_set, loss_set_reg, loss_set_div, loss_set_neg)
            loss = loss_set + loss_set_neg 
            batch_size = feature.size(0)
            total_loss += loss * batch_size
            total_loss_set += loss_set * batch_size
            total_loss_set_reg += loss_set_reg * batch_size
            total_loss_set_div += loss_set_div * batch_size
            total_loss_set_neg += loss_set_neg * batch_size

    return total_loss.item() / len(dataloader.dataset), total_loss_set.item() / len(dataloader.dataset), \
           total_loss_set_neg.item() / len(dataloader.dataset), \
           total_loss_set_reg.item() / len(dataloader.dataset), total_loss_set_div.item() / len(dataloader.dataset)


def train_one_epoch(dataloader_train, external_emb, lr, current_coeff_opt, split_i):
    start_time = time.time()
    total_loss = 0.
    total_loss_set = 0.
    total_loss_set_reg = 0.
    total_loss_set_div = 0.
    total_loss_set_neg = 0.
    
    encoder.train()
    decoder.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        #init_head_posi = random.randint(0, args.dilated_head_span - 1)
        #dilated_heads_posi = list(range(init_head_posi, len(feature), args.dilated_head_span))
        #if dilated_heads_posi[-1] != len(feature)-1:
        #    dilated_heads_posi.append(len(feature)-1)
        feature, target_unfold, inner_idx_tensor, future_mask = sample_batched
        #feature.size(0) % args.dilated_head_span
        #target_idx = idx_feature_to_target[:,init_head_posi:].unfold(dim=1, args.n_further, args.dilated_head_span)
        #target_idx = torch.cat( (target_idx, idx_feature_to_target[:,-1]), dim=1)
        
        #print(feature.size())
        #print(target_unfold.size()) 
        #print(future_mask.size())
        #target should have size (batch, num_head, n_further)
        target = target_unfold.view(-1, target_unfold.size(2) )

        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
        #output_emb, hidden, output_emb_last = parallel_encoder(feature)
        #output_emb_last, output_emb = parallel_encoder(feature)
        output_emb, past = parallel_encoder(feature)
        #output_emb should have size (batch, seq_len, hidden_size)
        hidden_size = output_emb.size(2)
        batch_size, num_head, seq_len = future_mask.size()
        
        #print(inner_idx_tensor)
        output_emb_head = output_emb.gather(dim=1,index=inner_idx_tensor.unsqueeze(dim=-1).expand(batch_size,num_head,hidden_size))
        #output_emb_head should have size (batch, num_head, hidden_size)
        output_emb_last = output_emb_head.view(-1, output_emb_head.size(2))

        if args.de_en_connection:
            output_emb_masked = output_emb.unsqueeze(dim=1).expand(batch_size,num_head,seq_len,hidden_size)
            #output_emb_masked = output_emb.unsqueeze(dim=0).expand(num_head,batch_size,seq_len,hidden_size)
            #future_mask_expanded = future_mask.unsqueeze(dim=-1).expand( batch_size,num_head,seq_len,hidden_size)
            #output_emb_masked[future_mask_expanded] = 0
            #output_emb_masked = output_emb_masked.view(-1, seq_len, hidden_size)
            output_emb_masked = output_emb_masked.reshape(-1, seq_len, hidden_size)
            future_mask = future_mask.view(-1, seq_len)

            basis_pred = parallel_decoder(output_emb_last, output_emb_masked, memory_attention_mask = future_mask)
        else:
            basis_pred = parallel_decoder(output_emb_last)
        compute_target_grad = args.update_target_emb
        loss_set, loss_set_reg, loss_set_div, loss_set_neg = nsd_loss.compute_loss_set(basis_pred, external_emb, target, args.L1_losss_B, device, w_freq, current_coeff_opt, compute_target_grad, args.coeff_opt_algo)
        if torch.isnan(loss_set):
            sys.stdout.write('nan, ')
            continue
        total_loss_set += loss_set.item() * args.small_batch_size / args.batch_size
        total_loss_set_reg += loss_set_reg.item() * args.small_batch_size / args.batch_size
        total_loss_set_div += loss_set_div.item() * args.small_batch_size / args.batch_size
        total_loss_set_neg += loss_set_neg.item() * args.small_batch_size / args.batch_size
        
        #BT_nonneg = torch.max( torch.tensor([0.0], device=device), BT )
        #loss = loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = 9 * torch.max( torch.tensor([0.7], device=device), loss_set) +  loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred + 0.01 * loss_set_div
        #loss = 4 * torch.max( torch.tensor([0.7], device=device), loss_set) + 4 * torch.max( torch.tensor([0.7], device=device), -loss_set_neg) +  loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + 0.9 * loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + args.w_loss_coeff* loss_coeff_pred
        loss = loss_set 
        if -loss_set_neg > 1:
            loss -= loss_set_neg
        else:
            loss += loss_set_neg
        
        loss *= args.small_batch_size / args.batch_size
        total_loss += loss.item()

        loss.backward()

        gc.collect()
        
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        optimizer_e.step()
        optimizer_d.step()
        
        #del sample_batched
        #break
    
        if args.update_target_emb:
            #print(external_emb.requires_grad)
            #print(external_emb.grad)
            if args.optimizer == 'SGD':
                external_emb.data -= lr/10.0 * external_emb.grad.data
            else:
                #external_emb.data -= 0.1/10.0 * external_emb.grad.data
                external_emb.data -= 10/10.0 * external_emb.grad.data
            external_emb.data[0,:] = 0
            external_emb.grad.data.zero_()
            #with torch.no_grad():
            external_emb.data = external_emb.data / (0.000000000001 + external_emb.data.norm(dim = 1, keepdim=True))

        if i_batch % args.log_interval == 0 and i_batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_loss_set = total_loss_set / args.log_interval
            cur_loss_set_reg = total_loss_set_reg / args.log_interval
            cur_loss_set_div = total_loss_set_div / args.log_interval
            cur_loss_set_neg = total_loss_set_neg / args.log_interval
            elapsed = time.time() - start_time
            logging('| e {:3d} {:3d} | {:5d}/{:5d} b | lr {:02.2f} | ms/batch {:5.2f} | '
                    'l {:5.2f} | l_f {:5.4f} + {:5.4f} = {:5.4f} | reg {:5.2f} | div {:5.2f} '.format(
                epoch, split_i, i_batch, len(dataloader_train.dataset) // args.batch_size, optimizer_e.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss_set, cur_loss_set_neg, cur_loss_set + cur_loss_set_neg, cur_loss_set_reg, cur_loss_set_div))

            #if args.coeff_opt == 'maxlc' and current_coeff_opt == 'max' and cur_loss_set + cur_loss_set_neg < -0.02:
            if args.coeff_opt == 'maxlc' and current_coeff_opt == 'max' and cur_loss_set + cur_loss_set_neg < -0.02:
                current_coeff_opt = 'lc'
                print("switch to lc")
            total_loss = 0.
            total_loss_set = 0.
            total_loss_set_reg = 0.
            total_loss_set_div = 0.
            total_loss_set_neg = 0.
            start_time = time.time()
    return current_coeff_opt

if args.optimizer == 'SGD':
    optimizer_e = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer_d = torch.optim.SGD(decoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
else:
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.wdecay)

if args.continue_train:
    try:
        optimizer_e_state_dict = torch.load(os.path.join(args.save, 'optimizer_e.pt'), map_location=device)
        optimizer_e.load_state_dict(optimizer_e_state_dict)
    except:
        print('randomly initialize encoder optimizer')
    try:
        optimizer_d_state_dict = torch.load(os.path.join(args.save, 'optimizer_d.pt'), map_location=device)
        optimizer_d.load_state_dict(optimizer_d_state_dict)
    except:
        print('randomly initialize decoder optimizer')
        

lr = args.lr
best_val_loss = None
nonmono_count = 0
saving_freq = int(math.floor(args.training_split_num / args.valid_per_epoch))

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    for i in range(len(dataloader_train_arr)):
        if epoch == 1 and i < args.start_training_split:
            print("Skipping epoch "+str(epoch) + ' split '+str(i) )
            continue
        current_coeff_opt = train_one_epoch(dataloader_train_arr[i], external_emb, lr, current_coeff_opt, i)
        
        if i != args.training_split_num - 1 and (i + 1) % saving_freq != 0:
            continue

        val_loss_all, val_loss_set, val_loss_set_neg, val_loss_set_reg, val_loss_set_div = evaluate(dataloader_val, external_emb, current_coeff_opt)
        logging('-' * 89)
        logging('| end of epoch {:3d} split {:3d} | time: {:5.2f}s | lr {:5.2f} | valid loss {:5.2f} | l_f {:5.4f} + {:5.4f} = {:5.4f} | reg {:5.2f} | div {:5.2f} | '
                .format(epoch, i, (time.time() - epoch_start_time), lr,
                                           val_loss_all, val_loss_set, val_loss_set_neg, val_loss_set + val_loss_set_neg, val_loss_set_reg, val_loss_set_div))
        
        val_loss_important = val_loss_set + val_loss_set_neg
        
        if not best_val_loss or val_loss_important < best_val_loss:
            save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, external_emb, args.save)
            best_val_loss = val_loss_important
            logging('Models Saved')
        else:
            nonmono_count += 1
        
        if nonmono_count >= args.nonmono:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            nonmono_count = 0
            lr /= 4.0
            for param_group in optimizer_e.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = lr
