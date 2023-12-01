"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import os
import argparse
import netifaces
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import torch
import re
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules import Module
import torch.utils.data.distributed 
import torch.distributed as dist
from torch.autograd import Variable
from torch.utils.data import Dataset
from datetime import timedelta
from data_loader import get_data_loader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pytorch_pretrained_bert import OpenAIAdam, BertTokenizer, cached_path
import sys
import signal
from model_sampler import print_samples
logger = logging.getLogger(__name__)

class DistributedDataParallel(Module):
  def __init__(self, module):
    super(DistributedDataParallel, self).__init__()
    self.module = module
    self.first_call = True

    def allreduce_params():
      if self.needs_reduction:
        self.needs_reduction = False  # pylint: disable = attribute-defined-outside-init
        buckets = {}
        for param in self.module.parameters():
          if param.requires_grad and param.grad is not None:
            tp = type(param.data)
            if tp not in buckets:
              buckets[tp] = []
            buckets[tp].append(param)
        for tp in buckets:
          bucket = buckets[tp]
          grads = [param.grad.data for param in bucket]
          coalesced = _flatten_dense_tensors(grads)
          dist.all_reduce(coalesced)
          coalesced /= dist.get_world_size()
          for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    for param in list(self.module.parameters()):
      def allreduce_hook(*unused):  # pylint: disable = unused-argument
        Variable._execution_engine.queue_callback(allreduce_params)  # pylint: disable = protected-access

      if param.requires_grad:
        param.register_hook(allreduce_hook)

  def weight_broadcast(self):
    for param in self.module.parameters():
      dist.broadcast(param.data, 0)

  def forward(self, *inputs, **kwargs):  # pylint: disable = arguments-differ
    if self.first_call:
      logging.info("first broadcast start")
      self.weight_broadcast()
      self.first_call = False
      logging.info("first broadcast done")
    self.needs_reduction = True  # pylint: disable = attribute-defined-outside-init
    return self.module(*inputs, **kwargs)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = './data/checkpoint.pt'
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class ModelBuffer(object):
    def __init__(self, network):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        super(ModelBuffer, self).__init__()
        self.recv_buf = []
        self.layer_cur_step = []
        self.layer_shape = []
        '''
        initialize space to receive model from parameter server
        '''
        # consider we don't want to update the param of `BatchNorm` layer right now
        # we temporirially deprecate the foregoing version and only update the model
        # parameters
        for param_idx, param in enumerate(network.parameters()):
            self.recv_buf.append(torch.zeros(param.size()))


class Trainer:

    def __init__(self, model,path, train_dataset, test_dataset, config, rank, master, port, time, device,vocab):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.rank = rank
        self.port = port
        self.master =  master
        self.time = time
        self.path = path
        self.vocab = vocab
        backend = 'tcp://'
        masterurl = backend+self.master+':'+self.port
        print(torch.cuda.get_device_name(0))
        logger.info("connecting to master %s", masterurl)
         
        dist.init_process_group(backend='gloo',init_method=masterurl,world_size=2,rank=self.rank,timeout=timedelta(seconds=self.time))
        # take over whatever gpus are on the system
        self.device = device
        if torch.cuda.is_available():
            #self.device = torch.device('cuda:0')
            self.device = device
            self.model = self.model.to(self.device)
            #self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0], find_unused_parameters=True)        
            #self.model = torch.nn.DataParallel(self.model).to(self.device)
    def init_recv_buf(self):
        self.model_recv_buf = ModelBuffer(self.model)

    def _fetch_weight(self):
        for layer_idx, layer in enumerate(self.model_recv_buf.recv_buf):
            dist.broadcast(self.model_recv_buf.recv_buf[layer_idx], src=0)
        self.model_update(self.model_recv_buf.recv_buf)
        # Note that at here we update the global step
        self.cur_step += 1
    
    def update_step(self):
        '''update local (global) step on worker'''
        changed = (self.cur_step != self.next_step)
        self.cur_step = self.next_step
        return changed

    def model_update(self, weights_to_update):
        """write model fetched from parameter server to local model"""
        new_state_dict = {}
        model_counter_ = 0
        for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
            assert param.size() == weights_to_update[model_counter_].size()
            tmp_dict = {key_name: weights_to_update[model_counter_].to(self._device)}
            model_counter_ += 1
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
    

    def train(self):
        model, config = self.model, self.config
        logger.info("training started")
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = self.vocab * self.config.max_epochs // self.config.batch_size
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=self.config.learning_rate,
                               warmup=0.002,
                               max_grad_norm=self.config.grad_norm_clip,
                               weight_decay=self.config.weight_decay,
                               t_total=num_train_optimization_steps)
        enc = GPT2Tokenizer.from_pretrained('gpt2')
        #optimizer = optim.AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
        def resetmodel(model):
            for param in model.parameters():
                torch.distributed.barrier()
            print("master model reset")
        def average_gradients(model):
            """ Gradient averaging. """
            
            size = float(dist.get_world_size())
            #group = dist.new_group([0])
            print("send tensor to master")
            req = None
            for param in model.parameters():
                #print(param)
                torch.distributed.barrier()
                dist.reduce(param.data, dst=0, op=dist.reduce_op.SUM)
                #gather(param.data, dst=0)
                #req.wait()
                #dist.reduce(tensor, 0, op=dist.reduce_op.SUM, group=group)
                #dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
                #param.grad.data /= size
                torch.distributed.barrier()
                dist.broadcast(param.data, src=0)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            
            loader = get_data_loader(data, enc, config.batch_size, 128, self.path)
            #loader = DataLoader(data, shuffle=True, pin_memory=True,batch_size=config.batch_size,num_workers=config.num_workers)

            losses = []
            tr_loss = 0
            nb_tr_steps = 0
            exp_average_loss = None
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                batch = batch.to(self.device)
                # place data on the correct device
                #x, y = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
                #x = x.to(self.device)
                #y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    output = model(batch, labels=batch)
                    loss =  output[0]
                    #loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    tr_loss += loss.item()
                    exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (batch >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    #perplexity  = torch.exp(exp_average_loss)
                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e} perplexity {math.exp(exp_average_loss)}")
            
            if not is_train:
                #test_loss = float(np.mean(losses))
                print(loss)
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                #nb_steps += 1
                #pbar.set_description = f"Eval loss: {exp_average_loss:.2e} ppl: {math.exp(exp_average_loss):.2e}"
                logger.info("test loss: %f", exp_average_loss)
                logger.info("perplexity: %f", math.exp(exp_average_loss))
                return exp_average_loss
            #sample = print_samples(model, enc, self.device,context_tokens=next(iter(loader)),batch_size=1, length=200, nsamples=1,temperature=1, top_k=40)
            logging.info("done")
            #torch.distributed.barrier()
            #average_gradients(model)

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            resetmodel(model)
            run_epoch('train')
            torch.distributed.barrier()
            average_gradients(model)
            #self.save_checkpoint()
            #torch.distributed.barrier()
            if self.test_dataset is not None:
                run_epoch('test')
            self.save_checkpoint()
            # supports early stopping based on the test loss, or just save always if no test set is provided
            #good_model = self.test_dataset is None or test_loss < best_loss
            #if self.config.ckpt_path is not None and good_model:
                #best_loss = test_loss
                #self.save_checkpoint()
        
        #torch.distributed.barrier()

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def signal_handler(sig, frame):
    print('fantastic exit!')
    sys.exit(0)
    
class WordDataset(Dataset):

    def __init__(self, data, block_size):
        words = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(words)
        print('data has %d words, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(words) }
        self.itos = { i:ch for i,ch in enumerate(words) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data_size = data_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every word to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        # See https://github.com/karpathy/minGPT/blob/master/play_char.ipynb for
        # explainer of Dataset construction
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y    


def main():
    train = []
    test = []
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    block_size = 128 
    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--master_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--master_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--log_file", type=str, default="", help="For distant debugging.")
    parser.add_argument("--timeout", type=str, default="", help="For distant debugging.")
    parser.add_argument("--test_data", type=str, default="", help="For distant debugging.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache_gist', help="Path or url of the dataset cache")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()
    logging.basicConfig(filename=sys.argv[12],
                            filemode='a',
                            format='%(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    net_inf = netifaces.gateways()['default'][netifaces.AF_INET][1]
    logger.info("using interface %s", net_inf)
    os.environ['TF_SOCKET_IFNAME'] = net_inf
    os.environ['TP_SOCKET_IFNAME'] = net_inf
    os.environ['GLOO_SOCKET_IFNAME'] = net_inf
    
    devicename = sys.argv[20]
    logging.info('{"status": "READY"}')
    #print(f"GPU : {devicename}")
    # initialize a trainer instance and kick off training
    #mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
    #              n_layer=12, n_head=12, n_embd=768)
    #model = GPT(mconf)
    with open(sys.argv[2], "r") as f:
        for line in f:
            test.extend(line.split())
    #f = lambda x: x.strip().replace("\n"," ")+" #EOS"
    #test = [f(x) for x in test]
    # seperate all words and punctuation
    test = [re.findall(r"[\w']+|[.,!?;]", x) for x in test]
    # turn list of lists in to single list
    test = [j for i in test for j in i]
    test_str='.'.join(test)
    with open("pt.txt", "w") as valid_file:
        valid_file.write(test_str)
    #print(abstract)
    train_dataset = WordDataset(test, block_size) 
    #print(f"GPU : {devicename}")
    # initialize a trainer instance and kick off training
    #mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
    #              n_layer=12, n_head=12, n_embd=768)
    #model = GPT(mconf)
    logging.info('vocab size : %s',str(train_dataset.vocab_size))
    vocab_size = train_dataset.vocab_size
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    #model.to(device)
    out = sys.argv[4] + 'checkpoint.pt'
    nepoch = int(sys.argv[22])
    try:
        tconf = TrainerConfig(max_epochs=nepoch, batch_size=int(sys.argv[18]), learning_rate=2.5e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=nepoch*vocab_size*block_size,
                          num_workers=16, ckpt_path = out)
        trainer = Trainer(model, sys.argv[4], sys.argv[2], sys.argv[16], tconf,int(sys.argv[6]), sys.argv[8], sys.argv[10], int(sys.argv[14]),devicename, train_dataset.data_size)
        trainer.train()
    except BrokenPipeError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)
    except RuntimeError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)
    except AttributeError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)
    except TypeError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)    
    except ValueError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)
    except AssertionError as err:
        logging.info('{"status": "FAILED", "error":"%s"}', err)
    except :
        logging.info('{"status": "FAILED", "error":"%s"}', sys.exc_info()[0])
    print('training done')

if __name__ == "__main__":
   
    main()
