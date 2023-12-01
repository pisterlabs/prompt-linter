""" train the abstractor"""
from coherence_interface.coherence_inference import graph_init
from coherence_interface.getAppositionReward import getAppositionReward
from coherence_interface.getAnaphoraReward import getAnaphoraReward
from coherence_interface.getLocalReward import getLocalReward
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose, concat

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import get_basic_grad_fn, rl_validate
from training import AbsSelfCriticalPipeline, BasicTrainer

from data.data import CnnDmDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy_rl, batchify_fn_copy_rl
from data.batcher import BucketedGenerater

from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding
import re


# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')

class MatchDataset_paulus(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents = (
            js_data['article'], js_data['abstract'])
        art_sents = [' '.join(art_sents)]
        abs_sents = [' '.join(abs_sents)]
        return art_sents, abs_sents

class MatchDataset_all2all(CnnDmDataset):
    """ single article sentence -> single abstract sentence
    (dataset created by greedily matching ROUGE)129.10.112.80
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (
            js_data['article'], js_data['abstract'], js_data['extracted_combine'])
        extracts = sorted(extracts)
        matched_arts = [art_sents[i] for i in extracts]
        matched_arts = [' '.join(matched_arts)]
        abs_sents = [' '.join(abs_sents)]
        return matched_arts, abs_sents

def load_best_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward/score"""
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    print('loading checkpoint {}...'.format(ckpts[0]))
    ckpt = torch.load(
        join(model_dir, 'ckpt/{}'.format(ckpts[0])), map_location=lambda storage, loc: storage
    )['state_dict']
    return ckpt

def configure_net(abs_dir):
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    abs_meta = json.load(open(join(abs_dir, 'meta.json')))
    assert abs_meta['net'] == 'base_abstractor'
    net_args = abs_meta['net_args']

    abs_ckpt = load_best_ckpt(abs_dir)
    net = CopySumm(**net_args)
    net.load_state_dict(abs_ckpt)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam', 'adagrad']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    if opt == 'adagrad':
        opt_kwargs['initial_accumulator_value'] = 0.1
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)
    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params

def build_batchers(word2id, cuda, debug):
    prepro = prepro_fn(args.max_art, args.max_abs)
    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))
    batchify = compose(
        batchify_fn_copy_rl(PAD, START, END, cuda=cuda),
        convert_batch_copy_rl(UNK, word2id)
    )

    train_loader = DataLoader(
        MatchDataset_paulus('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)
    val_loader = DataLoader(
        MatchDataset_paulus('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)

    return train_batcher, val_batcher

def main(args):
    # create data batcher, vocabulary
    # batcher
    word2id = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
    train_batcher, val_batcher = build_batchers(word2id,
                                                args.cuda, args.debug)

    # make net
    net, net_args = configure_net(args.abs_dir)

    # configure training setting
    criterion, train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'base_abstractor'
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    if args.coherence or args.all_local:
        print('use coref score')
        word_dict, model, sess = graph_init(model="cnndm")
        coh_func = (model, sess, word_dict)
        print('finish load coref model')
    else:
        coh_func = None
    assert not (args.anaphora and args.apposition)
    if args.anaphora:
        print('use anaphora')
        local_coh_fun = getAnaphoraReward
    elif args.apposition:
        print('use apposition')
        local_coh_fun = getAppositionReward
    elif args.all_local:
        local_coh_fun = getLocalReward
    else:
        local_coh_fun = None
    # prepare trainer
    if args.cuda:
        net = net.cuda()
    val_fn = rl_validate(net, coherence_func=coh_func, coh_coef=args.coh_coef, local_coh_func=local_coh_fun, local_coh_coef=args.local_coh_coef)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    #optimizer = optim.Adagrad(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    pipeline = AbsSelfCriticalPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn, optimizer, grad_fn, coh_func, args.coh_coef, local_coh_fun, args.local_coh_coef)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler, val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--path', required=True, help='root of saving the model')

    parser.add_argument('--abs_dir', required=True, help='root of the abs model')
    parser.add_argument('--coherence', action='store_true', help='coherence as reward')
    parser.add_argument('--anaphora', action='store_true', help='anaphora')
    parser.add_argument('--apposition', action='store_true', help='apposition reward')
    parser.add_argument('--all_local', action='store_true', help='entity repetition reward')
    parser.add_argument('--coh_coef', type=float, default=0.01, help='coherent score coefficient')
    parser.add_argument('--local_coh_coef', type=float, default=0.005, help='local coherent score coefficient')


    # length limit
    parser.add_argument('--max_art', type=int, action='store', default=400,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_abs', type=int, action='store', default=100,
                        help='maximun words in a single abstract sentence')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=2,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=50,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=1, help='gpu id')
    args = parser.parse_args()
    #args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    torch.cuda.set_device(args.gpu_id)

    main(args)
