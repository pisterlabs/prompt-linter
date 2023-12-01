import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path
from sklearn.utils import shuffle
sys.path.insert(1, '../utils')
from datasets import roc_stories
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer_models import GPT2MemModel
from opt import OpenAIAdam
from utils import (encode_dataset2, iter_data, ResultLogger, make_path)
from loss import LossCompute
import pickle
from tensorboard_logger import configure, log_value

configure("./log", flush_secs=5)

def transform_story(X1, X2):
    n_batch = len(X1)
    n_ctx = 512
    end_token = [encoder["<|endoftext|>"]]
    xmb_kg = np.tile(np.array([encoder['<|PAD|>']] * n_ctx,dtype=np.int32),(n_batch,1)) 
    xmb_mk = np.ones((n_batch,n_ctx))
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1
        new_x2 = x2
        x12 = new_x1
        x13 = new_x2 + end_token
        x14 = new_x2 
        x15 = new_x1[-2:] + new_x1[:-2] + end_token 
        xmb_kg[i,:len(x12)] = x12
        xmb_kg[i,len(x12):len(x12)+len(x13)] = x13
        xmb_mk[i,len(x14)+len(x15):] = 1
    return xmb_kg, xmb_mk

def iter_apply(X1s, X2s, Ms):
    logits = []
    cost = 0
    losses = []
    with torch.no_grad():
        model.eval()
        for xmb1, xmb2, mem in iter_data(X1s, X2s, Ms, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb1)
            XMB_KG = torch.tensor(xmb1, dtype=torch.long).to(device)
            XMB_ST = torch.tensor(xmb2, dtype=torch.long).to(device) 
            if args.use_mem:
               mem = [handle_empty(mem[i][0][:args.max_mem_size*5]) for i in range(len(mem))]
               mem = [torch.LongTensor([pad_rels(r) for r in m]) for m in mem]
               mem = torch.stack(mem)
               lm_logits = model(XMB_KG, attention_mask=XMB_ST, update_mem=mem, clear_mem=True, mem_k=args.mem_k,size_mem=1)[0] 
               loss = compute_loss_fct(lm_logits=lm_logits, lm_labels=XMB_KG, encoder=text_encoder, only_return_losses=True)
            else:
               lm_logits = model(XMB_KG,attention_mask=XMB_ST)[0]
               loss = compute_loss_fct(lm_logits=lm_logits, lm_labels=XMB_KG, encoder=text_encoder, only_return_losses=True)
            losses.append(float(loss.sum()))
    return np.sum(losses), np.mean(losses)


def log(save_dir, desc='model', iter=0,save='',save_model=True):
    global best_score
    print("Logging")
    tr_sum_loss, tr_mean_loss = iter_apply(trX_kg[:n_valid],trX_st[:n_valid], trMem[:n_valid])
    va_sum_loss, va_mean_loss = iter_apply(vaX_kg[:n_valid],vaX_st[:n_valid], vaMem[:n_valid])
    log_value('va_sum_loss',va_sum_loss,n_updates)
    log_value('va_mean_loss',va_mean_loss,n_updates)
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=float(tr_sum_loss), va_cost=float(va_sum_loss), tr_acc=float(tr_mean_loss), va_acc=float(va_mean_loss))
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_sum_loss, va_sum_loss, tr_mean_loss, va_mean_loss))
    path = os.path.join(save_dir, desc, 'best_params_' + str(iter+1) + save)
    if save_model:
       torch.save(model.state_dict(), make_path(path))
    return va_mean_loss

def pad_rels(relation, pad_len=100):
    return relation[:100] + [encoder['<|PAD|>']] * (100-len(relation[:100]))

def handle_empty(list_of_rels):
    if len(list_of_rels) == 0:
       return [[] for i in range(args.max_mem_size*5)]
    if len(list_of_rels) < args.max_mem_size*5:
       list_of_rels.extend([[] for i in range(args.max_mem_size*5 - len(list_of_rels))])
    return list_of_rels 

def run_epoch(iter):
    losses = []
    i = 0
    for xmb_kg, xmb_st, mem in iter_data(*shuffle(trX_kg, trX_st, trMem, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        global past_loss
        if n_updates == 0:
           past_loss = math.inf 
        model.train()
        XMB_KG = torch.tensor(xmb_kg, dtype=torch.long).to(device)
        XMB_ST = torch.tensor(xmb_st, dtype=torch.long).to(device)
        if args.use_mem:
           mem = [handle_empty(mem[i][0][:args.max_mem_size*5]) for i in range(len(mem))] 
           mem = [torch.LongTensor([pad_rels(r) for r in m]) for m in mem]
           mem = torch.stack(mem)
           lm_logits = model(XMB_KG, attention_mask=XMB_ST, update_mem=mem, clear_mem=True, mem_k=args.mem_k,size_mem=1)[0] 
           loss = compute_loss_fct(lm_logits=lm_logits, lm_labels=XMB_KG, encoder=text_encoder, batch_num=n_updates, accum_steps=int(16/args.n_batch))
        else:
           lm_logits = model(XMB_KG,attention_mask=XMB_ST)[0]
           loss = compute_loss_fct(lm_logits=lm_logits, lm_labels=XMB_KG, encoder=text_encoder, batch_num=n_updates, accum_steps=int(16/args.n_batch))
        loss = float(loss)
        losses.append(loss)
        n_updates += 1
        if (n_updates + 1) % 20000 == 0:
           va_loss = log(save_dir,desc, iter,save='_'+str(n_updates),save_model=False)
        log_value('batch_train_loss',loss,n_updates)
        log_value('mean_train_loss',np.mean(losses),n_updates)
        log_value('total_train_loss',np.sum(losses),n_updates)
argmax = lambda x: np.argmax(x, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_mem_size',type=int,default=45)
    parser.add_argument('--use_multigpu',type=bool,default=True)
    parser.add_argument('--use_pretrain',type=bool,default=False)
    parser.add_argument('--use_filter',type=bool,default=True)
    parser.add_argument('--mem_k',type=int,default=1)
    parser.add_argument('--use_mem',type=bool,default=False)
    parser.add_argument('--desc',type=str,default='model',help="Description")
    parser.add_argument('--comet',type=bool,default=False)
    parser.add_argument('--kg_type',type=str,default='atomic')
    parser.add_argument('--log_dir', type=str, default='h_log_nopre/')
    parser.add_argument('--model_dir', type=str, default='h_models_nopre/')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=40)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Constants
    desc = args.desc
    n_ctx = args.n_ctx
    save_dir = args.model_dir
    data_dir = args.data_dir
    log_dir = args.log_dir

    Path(save_dir).mkdir(parents=True, exist_ok=True)  
    Path(log_dir).mkdir(parents=True, exist_ok=True)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
    encoder = text_encoder.encoder

    #sentence-level special tokens
    encoder['<|sent0|>'] = len(encoder)
    encoder['<|sent1|>'] = len(encoder)
    encoder['<|sent2|>'] = len(encoder)
    encoder['<|sent3|>'] = len(encoder)
    encoder['<|sent4|>'] = len(encoder)

    #ATOMIC special tokens
    if args.kg_type == 'atomic':
       encoder['<|xNeed|>'] = len(encoder)
       encoder['<|xIntent|>'] = len(encoder)
       encoder['<|xWant|>'] = len(encoder)
       encoder['<|oEffect|>'] = len(encoder)
       encoder['<|xReact|>'] = len(encoder)
       encoder['<|oWant|>'] = len(encoder)
       encoder['<|oReact|>'] = len(encoder)
       encoder['<|xEffect|>'] = len(encoder)
       encoder['<|xAttr|>'] = len(encoder)

    #padding special tokens
    encoder['<|PAD|>'] = len(encoder) 
    text_encoder.encoder = encoder
    n_vocab = len(text_encoder.encoder)
    print("Encoding dataset...")

    try:
       trX_kg, trX_st, trMem, vaX_kg, vaX_st, vaMem =  pickle.load(open(data_dir + '/' + 't_' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'data.pkl','rb'))
    except:
       try:
          data_dump = pickle.load(open(data_dir + '/' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'data.pkl','rb'))
          trX1, trX2, trMem, trIds = data_dump[0]
          vaX1, vaX2, vaMem, vaIds = data_dump[1]
       except:
        ((trX1, trX2, trMem, trIds),
         (vaX1, vaX2, vaMem, vaIds)) = encode_dataset2(*roc_stories(data_dir, args.comet, args.kg_type),encoder=text_encoder)
        pickle.dump([(trX1,trX2, trMem, trIds), (vaX1, vaX2, vaMem, vaIds)], open(data_dir + '/' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' +  args.kg_type + '_' + 'data.pkl','wb'))
       trX_kg, trX_st = transform_story(trX1, trX2)
       vaX_kg, vaX_st = transform_story(vaX1, vaX2)
       pickle.dump((trX_kg, trX_st, trMem, vaX_kg, vaX_st, vaMem), open(data_dir + '/t_' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'data.pkl','wb'))

    n_train = len(trX_kg)
    n_valid = len(vaX_kg)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter
    if args.use_mem:
       model = GPT2MemModel.from_pretrained('gpt2')
    else:
       model = GPT2LMHeadModel.from_pretrained('gpt2')
    if not args.use_pretrain:
       model.init_weights()
    model.resize_token_embeddings(len(encoder))
    if args.use_multigpu:
       model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=encoder['<|PAD|>'])
    model_opt = OpenAIAdam(model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = LossCompute(criterion,model_opt)


    n_updates = 0
    n_epochs = 0
    best_score = 0
    for i in range(n_epochs, args.n_iter):
        print("running epoch", i)
        run_epoch(n_epochs)
        n_epochs += 1
        log(save_dir,iter=i)
