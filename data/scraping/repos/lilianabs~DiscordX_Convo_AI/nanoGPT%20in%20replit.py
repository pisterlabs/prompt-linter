"""
# nanoGPT demo as of Jan 17th, 2023
# Modified from Andrej Karpathy's nanoGPT project: https://github.com/karpathy/nanoGPT/
# See Karpathy's nanoGPT codebase for a description of the training itself

############################################################
Required steps before running:
############################################################

# 1. Installations needed
pip install transformers datasets tiktoken wandb tqdm protobuf==3.20

# 2. You also need to first prepare the dataset, in the Shell run:
cd nanoGPT && python data/shakespeare/prepare.py

# 3. wandb on Replit currently requires adding this line to your code (already added in this script):
os.environ["WANDB_CONSOLE"]="wrap"

############################################################
Other notes for this repl
############################################################

# Storage management - hard disk storage on Replit is currently limited to 1 GB, 
# meaning you may have to delete unused data or model files during or after training

# Tip 1: Delete the downloaded GPT2 model weights once training has started.
# To delete your hugging face model weights run the following in the Shell:
rm -rf /home/runner/.cache/huggingface/hub

# Tip 2: Do not keep more than 1 trained model checkpoint in your out_dir,
# "out-shakespeare" in this case. Remove any existing checkpoints before saving another

# Tip 3: Do not save optimizer.state_dict() in the model checkpoint,
# (this is already implemented in this script)

# Tip 4: Check your available GPU memory and flush it if needed.
# This can be useful if you want to start a new training run but the GPU
# memory is stil full from the previous training run
# i.Check your available memory by running `nvidia-smi` in the Shell, or `nvidia-smi -l 1` to monitor it continuously
# ii.Get the PID of the service by running `fuser -v /dev/nvidia*` in the Shell
# iii. Kill the process by running `kill -9 "PID"`

For more info how how nanoGPT is trained and run please see
# the original codebase here: https://github.com/karpathy/nanoGPT/
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanoGPT.model import GPTConfig, GPT
import tiktoken

os.environ["WANDB_CONSOLE"]="wrap"

# -----------------------------------------------------------------------------
# I/O
log_interval = 1
eval_iters = 50
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval

dataset = 'shakespeare'
init_from = 'gpt2'
batch_size = 4
block_size = 512
gradient_accumulation_steps = 2 # used to simulate larger batch sizes

learning_rate = 1e-5
max_iters = 1000
decay_lr = False

out_dir = 'out-shakespeare'
abs_dir = 'home/runner/nanoGPT-in-Replit/nanoGPT'
eval_interval = 50

# Weights & Biases logging options
wandb_log = True # feel free to turn on
wandb_project = 'shakespeare-on-replit'
wandb_run_name = 'ft-' + str(time.time())

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
weight_decay = 1e-2
learning_rate = 1e-5
max_iters = 1000

beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate

warmup_iters = 50 # how many steps to warm up for
lr_decay_iters = 1000 # should be ~= max_iters per Chinchilla
min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float32' # 'float32' or 'bfloat16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]

exec(open('nanoGPT/configurator.py').read()) # overrides from command line or config file

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 would require us to change the code to use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[dtype]
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
ctx = nullcontext() if device_type == 'cpu' else torch.autocast("cuda", dtype=ptdtype)
# poor man's data loader, TODO evaluate need for actual DataLoader
data_dir = os.path.join('nanoGPT/data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (from {meta_path})")
else:
    print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
    vocab_size = 50257

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout, vocab_size = vocab_size)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch\n")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k, v in model_args.items():
        assert checkpoint_model_args[k] == v, "for now"
        # TODO: think through how passed in params should interact with checkpoint params
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}\n")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off and override the GPT sizing model args from the model config
    model_args['n_layer'] = model.config.n_layer
    model_args['n_head'] = model.config.n_head
    model_args['n_embd'] = model.config.n_embd
# crop down the model block size if desired
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
model.to(device)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    # print(f"debug: model device cuda: {next(model.parameters()).is_cuda}")
    print('estimating train and val loss...')
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def sample():
  """
  Sample from a trained model
  """
  
  # -----------------------------------------------------------------------------
  # start = "\n" # or "<|endoftext|>" or whatever you like
  start = "\nTo be or not to be" # or "<|endoftext|>" or whatever you like
  num_samples = 3 # number of samples to draw
  max_new_tokens = 250 # number of tokens generated in each sample
  temperature = 0.8 # higher temperature (up to 1) is more random, lower (down to 0) means more greedy
  top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

  print("generating samples...")
  model.eval()
  
  # ok let's assume gpt-2 encodings by default
  print("No meta.pkl found, assuming GPT-2 encodings...")
  enc = tiktoken.get_encoding("gpt2")
  encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
  decode = lambda l: enc.decode(l)
  
  # encode the beginning of the prompt
  start_ids = encode(start)
  x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
  
  # run generation
  with torch.no_grad():
      gen_list = []
      with ctx:
          for k in range(num_samples):
              y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
              g = decode(y[0].tolist())
              gen_list.append(g)
              print(g)
              print('---------------')
  model.train()
  return gen_list, start


  
# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if iter > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    # clear wandb logs from this public demo
    os.system("rm -rf ~/nanoGPT/wandb")
  
    os.environ["WANDB_CONSOLE"]="wrap"
    import wandb
  
    wandb.init(project=wandb_project, name=wandb_run_name, config=config, anonymous="allow")
    wandb.run._label(repo="replit_nanoGPT")

# training loop
t0 = time.time()
while True:
    if iter_num==0: print("starting training...")

    # determine the learning rate for this iteration
    if decay_lr:
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # print('debug: est loss')
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            gen_list, start = sample()

            wandb_table = wandb.Table(columns=["run_id", "iter_num", "start_text", "gen_1","gen_2","gen_3"])
            wandb_table.add_data(wandb_run_name, iter_num, start, gen_list[0], gen_list[1], gen_list[2])              
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/lr": lr,
                f"table/gen_{iter_num}": wandb_table
            })
          
        if (losses['val'] < best_val_loss or always_save_checkpoint) and iter_num != 0:
            # delete old model weights to increase storage
            os.system("rm -rf /home/runner/.cache/huggingface/hub")
          
            best_val_loss = losses['val']
            raw_model = model.module if ddp else model
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    # do not save the optimizer, means that resuming training won't work so well
                    # 'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                # clear any existing checkpoints, for storage management
                os.system(f"rm -rf ~/nanoGPT/{out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print("Saving model weights to Weights & Biases Artifacts")
                wandb.log_artifact(os.path.join(out_dir, 'ckpt.pt'), name='nanoGPT-shakespeare', type='model', 
                                   aliases=[f'iter_num-{iter_num}', f'val_loss-{best_val_loss:.4f}']) 
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
        loss.backward()
    optimizer.step()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() # loss as float. TODO note CPU-GPU sync! profile, make sure not too slow
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        # wandb logging frequency to every iteration
        if iter_num % eval_interval != 0 and master_process:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/lr": lr
            })
          
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        # clear wandb logs from this public demo
        os.system("rm -rf ~/nanoGPT/wandb")
        break

if ddp:
    destroy_process_group()


