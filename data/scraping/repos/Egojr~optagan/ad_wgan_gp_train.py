from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

from modules.gan import Generator, Critic

import glob
import os
import pickle
import random

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from func import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, BertConfig
from func import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForLatentConnector
from func import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from func import XLNetLMHeadModel, XLNetTokenizer
from func import TransfoXLLMHeadModel, TransfoXLTokenizer
from func import BertForLatentConnector, BertTokenizer

from collections import defaultdict
from utils import (TextDataset_Split, TextDataset_2Tokenizers, BucketingDataLoader)
import pdb
from modules.utils import (calc_blue_parallel_func, pad_seq, rollout, rollout_test)
from transformers.modeling_utils import top_k_top_p_filtering

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer)
}

def load_and_cache_examples(args, tokenizer):
    if isinstance(tokenizer, list):
        dataset = TextDataset_2Tokenizers(tokenizer, args, args.train_data_file, block_size=args.block_size)
    else:
        dataset = TextDataset_Split(tokenizer, args, args.train_data_file, block_size=args.block_size)
    return dataset

def build_dataload_and_cache_examples(args, tokenizer):
    if isinstance(tokenizer, list):
        args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        file_path=args.train_data_file
        dataloader = BucketingDataLoader(file_path, args.batch_size, args.max_seq_length, tokenizer, args, bucket=100, shuffle=True)
    else:
        pass 
    return dataloader

def compute_grad_penalty(critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    if args.cuda:
        alpha = alpha.cuda()
    sample = alpha*real_data + (1-alpha)*fake_data
    sample.requires_grad_(True)
    score = critic(sample)

    outputs = torch.FloatTensor(B, 1).fill_(1.0) #args.latent_size
    outputs.requires_grad_(False)
    if args.cuda:
        outputs = outputs.cuda()
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    #grads = grads.view(B, -1)
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()
    return grad_penalty

def train(epoch):
    model_encoder.eval()
    model_decoder.eval()
    generator.train()
    critic.train()
    c_train_loss = 0.
    g_train_loss = 0.
    g_batches = 0
    c_batches = 0
    c_loss_0 = 1
    g_loss_0 = 1
    for i, x in enumerate(train_loader):
        x = x[0]
        if args.cuda:
            x = x.cuda()
        # Generate noise
        B = args.per_gpu_train_batch_size
        noise = torch.from_numpy(np.random.normal(0, 1, (B,
                                 args.latent_size))).float()
        if args.cuda:
            noise = noise.cuda()
        # Get original text latent embeddings
        with torch.no_grad(): 
            pooled_hidden_fea = model_encoder(x, attention_mask=(x > 0).float())[1]
            mean, logvar = model_encoder.linear(pooled_hidden_fea).chunk(2, -1)
            z_real = mean.squeeze(1) 

        # Evaluate and get losses
        z_fake = generator(noise)
        real_score = critic(z_real)
        fake_score = critic(z_fake)
        grad_penalty = compute_grad_penalty(critic, z_real.data, z_fake.data)
        c_loss = -torch.mean(real_score) + torch.mean(fake_score) + \
                 args.gp_lambda*grad_penalty

        fake_score = critic(generator(noise))
        g_loss = -torch.mean(fake_score)
        
        r_g = abs(((g_loss.item() - g_loss_0) / (g_loss_0 + 0.001))) 
        r_c = abs(((c_loss.item() - c_loss_0) / (c_loss_0 + 0.001))) 
        
        # Update critic or generator
        if ((2 + epoch) / epoch) * r_c > r_g:
            c_optimizer.zero_grad()
            c_batches += 1
            c_train_loss += c_loss.item()
            c_loss.backward()
            c_optimizer.step()
        else:
            g_optimizer.zero_grad()
            g_batches += 1
            g_train_loss += g_loss.item()
            g_loss.backward()
            g_optimizer.step()

        c_loss_0 = c_loss.item()
        g_loss_0 = g_loss.item()

        if args.interval > 0 and i % args.interval == 0:

            logger.info('Epoch: {} | Batch: {}/{} ({:.0f}%) | G Loss: {:.6f} | C Loss: {:.6f}'.format(
                epoch, args.batch_size*i, len(train_loader.dataset),
                100.*(args.batch_size*i)/len(train_loader.dataset),
                g_loss.item(), c_loss.item()
            ))
            test_noise = torch.Tensor(np.random.normal(0, 1, (1, args.latent_size))).to(args.device)
            test_new_z = generator(test_noise).data
            # create new sent
            test_z = rollout_test(model_decoder, test_new_z, tokenizer_decoder, args.max_seq_length, 1, 0, 1)
            logger.info("Text: {}".format(test_z))

    c_train_loss /= c_batches + 1
    g_train_loss /= g_batches + 1
    logger.info('* (Train) Epoch: {} | G Loss: {:.4f} | C Loss: {:.4f} | Updates G: {} | Updates C: {}'.format(
        epoch, g_train_loss, c_train_loss, g_batches, c_batches
    ))
    return (g_train_loss, c_train_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gp_lambda', type=int, default=10)
    parser.add_argument('--n_layers', type=int, default=20, help="Number of layers of generator and critic")
    parser.add_argument('--block_dim', type=int, default=100)
    parser.add_argument('--interval', type=int, default=10, help="Steps before logging output")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    
    # Optimus parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--valid_data_file", default=None, type=str, required=True,
                        help="The input validation data file (a text file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=True,
                        help="The directory where checkpoints are saved.")
    parser.add_argument('--generator_dir', default=None, type=str, help="Directory where GAN models are saved")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset", default='Snli', type=str, help="The dataset.")    
    parser.add_argument("--latent_size", default=32, type=int, help="Latent space dimension.")
    ## Encoder options
    parser.add_argument("--encoder_model_type", default="bert", type=str,
                        help="The encoder model architecture to be fine-tuned.")
    parser.add_argument("--encoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The encoder model checkpoint for weights initialization.")
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--encoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    ## Decoder options
    parser.add_argument("--decoder_model_type", default="gpt2", type=str,
                        help="The decoder model architecture to be fine-tuned.")
    parser.add_argument("--decoder_model_name_or_path", default="bert-base-cased", type=str,
                        help="The decoder model checkpoint for weights initialization.")
    parser.add_argument("--decoder_config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--decoder_tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length")

    ## Variational auto-encoder(check this)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--use_philly", action='store_true',
                        help="Use Philly for computing.")
    parser.add_argument('--gloabl_step_eval', type=int, default=661,
                        help="Evaluate the results at the given global step")

    # Load a trained Encoder model and vocabulary that you have fine-tuned
    args = parser.parse_args()
    global_step = args.gloabl_step_eval

    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)       
    
    args.encoder_model_type = args.encoder_model_type.lower()
    args.decoder_model_type = args.decoder_model_type.lower()

    output_encoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-encoder-{}'.format(global_step))
    output_decoder_dir = os.path.join(args.checkpoint_dir, 'checkpoint-decoder-{}'.format(global_step)) 
    checkpoints = [ [output_encoder_dir, output_decoder_dir] ]

    # Load a trained Encoder model and vocabulary that you have fine-tuned
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[args.encoder_model_type]
    model_encoder = encoder_model_class.from_pretrained(output_encoder_dir, latent_size=args.latent_size)
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(args.encoder_tokenizer_name if args.encoder_tokenizer_name else args.encoder_model_name_or_path, do_lower_case=args.do_lower_case)

    model_encoder.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_encoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    # Load a trained Decoder model and vocabulary that you have fine-tuned
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[args.decoder_model_type]
    model_decoder = decoder_model_class.from_pretrained(output_decoder_dir, latent_size=args.latent_size)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(args.decoder_tokenizer_name if args.decoder_tokenizer_name else args.decoder_model_name_or_path, do_lower_case=args.do_lower_case)
    model_decoder.to(args.device)
    if args.block_size <= 0:
        args.block_size = tokenizer_decoder.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    logger.info('We have added {} tokens to GPT2'.format(num_added_toks))
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == '<PAD>'
    
    train_loader = build_dataload_and_cache_examples(args, [tokenizer_encoder, tokenizer_decoder]) 
    generator = Generator(args.n_layers, args.block_dim,args.latent_size)
    critic = Critic(args.n_layers, args.block_dim,args.latent_size)

    if args.generator_dir!=None:
        generator.load_state_dict(torch.load(args.generator_dir+'/generator_'+str(args.gloabl_step_eval)+'.th'))
        critic.load_state_dict(torch.load(args.generator_dir+'/critic_'+str(args.gloabl_step_eval)+'.th'))

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    c_optimizer = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    if args.cuda:
        generator = generator.cuda()
        critic = critic.cuda()
    
    logger.info('G Parameters:{}'.format(sum([p.numel() for p in generator.parameters() if \
                                p.requires_grad])))
    logger.info('C Parameters:{}'.format(sum([p.numel() for p in critic.parameters() if \
                                p.requires_grad])))
    
    best_bleu = 0
    reference = list()
    with(open(args.valid_data_file,"r")) as valid:
        for sents in valid:
            reference.append(sents.replace("\n", ""))
            
    for epoch in range(1, args.epochs + 1):
        g_loss, c_loss = train(epoch)

        data_test = list()
        for i in range(2):
            test_noise = torch.Tensor(np.random.normal(0, 1, (250, args.latent_size))).to(args.device)
            test_z = generator(test_noise).data
            new_sent = rollout_test(model_decoder, test_z, tokenizer_decoder, args.max_seq_length, 250, 0, 1)
            data_test.extend(new_sent)       

        p_reference = random.sample(reference, 500)
        bleu = calc_blue_parallel_func(p_reference, data_test, 2, 500)
        b_bleu = calc_blue_parallel_func(data_test, p_reference, 2, 500)
        logger.info("Bleu-2:{:0.3f} | B-Bleu-2:{:0.3f}".format(bleu, b_bleu))

        if (bleu+b_bleu) > best_bleu:
            best_bleu = bleu + b_bleu
            logger.info('* Saving. Best Score:{:0.3f} | Bleu-2:{:0.3f} | B-Bleu-2:{:0.3f}'.format(best_bleu, bleu, b_bleu))
            torch.save(generator.state_dict(), args.output_dir+'/generator_'+str(args.gloabl_step_eval)+'.th')
            torch.save(critic.state_dict(), args.output_dir+'/critic_'+str(args.gloabl_step_eval)+'.th')