"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig
from model import MemoryGPT as GPT
from my_configuration_roberta import MemoryRobertaConfig
from my_modeling_roberta import MemoryRobertaModel


# -----------------------------------------------------------------------------
# default config values designed to train a evolver (roberta)
evolver_n_layer = 6
evolver_n_head = 12
evolver_n_embd = 768
evolver_n_intermediate = 3072
evolver_n_mem = 50

######### no use
evolver_pad_token_id = 0
evolver_gpt2_token_id_offset = 20 # the token id produced by gpt2 tokenizer should added by this offset
#################

segment_num = 1 # if > 1, train memory
num_target_model_layer = 12

remember_prob = 95 # 大于这个的话，minibatch的记忆都会被删除
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
seed=1337
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
min_block_size = 50
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    
    # 多此一举
    # # world_size number of processes will be training simultaneously, so we can scale
    # # down the desired gradient accumulation iterations per process proportionally
    # assert gradient_accumulation_steps % ddp_world_size == 0
    # gradient_accumulation_steps //= ddp_world_size
else:
    ddp_rank = 0
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# get initial data pointer for each batch in this rank: this_rank_batch_train_data_pointer
this_rank_train_data_start = ddp_rank * (len(train_data) // ddp_world_size)

if ddp_rank == (ddp_world_size-1):
    this_rank_train_data_end = len(train_data)
else:
    this_rank_train_data_end = (ddp_rank + 1) * (len(train_data) // ddp_world_size)

this_rank_data_num = this_rank_train_data_end - this_rank_train_data_start
this_rank_batch_train_data_pointer = []
actual_batch_size = batch_size * gradient_accumulation_steps
for bi in range(0, actual_batch_size):
    this_rank_batch_train_data_pointer.append(this_rank_train_data_start + bi * (this_rank_data_num // actual_batch_size))


# get data for a minibatch
def get_seq_train_batch(this_batch_seg_num, plus_one=False):
    data = train_data

    x_list = []
    y_list = []

    fetch_seg_num = this_batch_seg_num
    if plus_one:
        fetch_seg_num += 1

    for seg_index in range(fetch_seg_num):
        random_length = torch.randint(block_size - min_block_size, (actual_batch_size,)) # random segment len for each batch item

        for bi in range(actual_batch_size):
            this_end = random_length[bi] + min_block_size + this_rank_batch_train_data_pointer[bi] # end index
            random_length[bi] = this_end if this_end < len(data) else len(data) - 1

        segment_ends = random_length

        # (batch size, xxx)
        x = [torch.from_numpy((data[this_rank_batch_train_data_pointer[bi]:segment_ends[bi]]).astype(np.int64)) for bi in range(actual_batch_size)]
        y = [torch.from_numpy((data[this_rank_batch_train_data_pointer[bi] + 1:segment_ends[bi] + 1]).astype(np.int64)) for bi in range(actual_batch_size)]

        # padding to (batch size, block size)

        padding_x = x[0].new_full((actual_batch_size, block_size), fill_value=0)
        padding_y = y[0].new_full((actual_batch_size, block_size), fill_value=-1)

        for bi in range(actual_batch_size):
            padding_x[bi][:len(x[bi])] = x[bi] + 1
            padding_y[bi][:len(y[bi])] = y[bi]

            # update this_rank_batch_train_data_pointer
            if seg_index == this_batch_seg_num:
                pass
            else:
                this_rank_batch_train_data_pointer[bi] = segment_ends[bi] if segment_ends[bi] < len(data) - min_block_size else 0

        # return_offset_x.append(offset_padding_x)
        x_list.append(padding_x)
        y_list.append(padding_y)
    
    # (batch size, segment num, block size)
    x = torch.stack(x_list, dim=1)
    attention_mask = x.ne(0).int()
    x = x + x.new_full(attention_mask.size(), fill_value=-1) * attention_mask # pad_token_id is 0

    y = torch.stack(y_list, dim=1)
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, attention_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)

    return x, y, attention_mask


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
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
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

# backbone forzen
for p in model.parameters():
    p.requires_grad_(False)

# --------------------------------------------------------------------------
# create my evolver 
evolver_config = MemoryRobertaConfig(vocab_size=model_args['vocab_size'] + evolver_gpt2_token_id_offset, num_hidden_layers=evolver_n_layer,
                                     num_attention_heads=evolver_n_head, hidden_size=evolver_n_embd, max_position_embeddings=block_size, intermediate_size=evolver_n_intermediate,
                                     pad_token_id=evolver_pad_token_id, gpt2_token_id_offset=evolver_gpt2_token_id_offset, num_memory=evolver_n_mem,
                                     num_target_model_layer=num_target_model_layer, no_embeddings=True)
evolver_model = MemoryRobertaModel(evolver_config)
evolver_model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = evolver_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    evolver_model = torch.compile(evolver_model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.
    # model = DDP(model, device_ids=[ddp_local_rank]) 
    evolver_model = DDP(evolver_model, device_ids=[ddp_local_rank], broadcast_buffers=False, find_unused_parameters=True) # https://github.com/pytorch/pytorch/issues/22095

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, attention_mask = get_seq_train_batch(segment_num, True) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model = model.module if ddp else model # unwrap DDP container if needed
raw_model = model
raw_evolver_model = evolver_model.module if ddp else evolver_model # unwrap DDP container if needed
running_mfu = -1.0

input_memory_list = [None for _ in range(gradient_accumulation_steps)]

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # if wandb_log:
        #     wandb.log({
        #         "iter": iter_num,
        #         "train/loss": losses['train'],
        #         "val/loss": losses['val'],
        #         "lr": lr,
        #         "mfu": running_mfu*100, # convert to percentage
        #     })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'evolver_model': raw_evolver_model.state_dict(),
                    'evolver_config': evolver_config,
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
    lossf = 0.0
    revise_lossf = 0.0
    predict_lossf = 0.0

    gpt_lossf = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        all_loss = torch.tensor(0.0, device=device, requires_grad=True)
        predict_loss = torch.tensor(0.0, device=device)
        revise_loss = torch.tensor(0.0, device=device)
        gpt_loss = torch.tensor(0.0, device=device)
        
        input_memory = input_memory_list[micro_step]

        this_micro_X = X[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_Y = Y[batch_size*micro_step : batch_size*(1+micro_step)]
        this_micro_attention_mask = attention_mask[batch_size*micro_step : batch_size*(1+micro_step)]

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            evolver_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

            past_segments_x = []
            past_segments_y = []

            sampled_segments_index = set(random.sample(range(segment_num), max(int(segment_num * 0.5), 1)))
            sampled_segments_index = []
            trained_seg_num = segment_num + len(sampled_segments_index)

            # get data for first segment
            this_x = this_micro_X[:, 0, :]
            this_y = this_micro_Y[:, 0, :]
            this_attention_mask = this_micro_attention_mask[:, 0, :]

            for si in range(segment_num):

                # 保存数据用于复习
                if si in sampled_segments_index:
                    past_segments_x.append(this_x)
                    past_segments_y.append(this_y)

                # generate embeddings by pretrained model
                with torch.no_grad():
                    output_embeds = model(idx=this_x, input_parameter=target_model_parameter, output_embeds=True)

                # X -> memory
                input_memory = evolver_model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]

                # last memory -> X
                target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

                # _, loss = model(this_x, this_y, target_model_parameter)
                # all_loss = loss + all_loss

                # with torch.no_grad():
                #     _, loss = model(this_x, this_y)
                #     gpt_loss = loss + gpt_loss

                # 参照RLHF，搞一个KL散度，防止与预训练模型偏离太严重。
                # todo
                
                # get data for next segment
                this_x = this_micro_X[:, si + 1, :]
                this_y = this_micro_Y[:, si + 1, :]
                this_attention_mask = this_micro_attention_mask[:, si + 1, :]

                _, loss = model(this_x, this_y, target_model_parameter)
                all_loss = loss + all_loss
                predict_loss = loss + predict_loss

                with torch.no_grad():
                    _, loss = model(this_x, this_y)
                    gpt_loss = loss + gpt_loss

            # 复习一下past_segments
            # target_model_parameter = evolver_model(input_memory=input_memory, produce_parameter_flag=True)

            for (this_x, this_y) in zip(past_segments_x, past_segments_y):
                _, loss = model(this_x, this_y, target_model_parameter)
                all_loss = loss + all_loss
                revise_loss = loss + revise_loss

                with torch.no_grad():
                    _, loss = model(this_x, this_y)
                    gpt_loss = loss + gpt_loss

            ###
            all_loss = all_loss / (gradient_accumulation_steps * trained_seg_num) # scale the loss to account for gradient accumulation
            revise_loss = revise_loss / (gradient_accumulation_steps * len(past_segments_x))
            predict_loss = predict_loss / (gradient_accumulation_steps * segment_num)
            gpt_loss = gpt_loss / (gradient_accumulation_steps * trained_seg_num)

            lossf += all_loss.item()
            revise_lossf += revise_loss.item()
            predict_lossf += predict_loss.item()
            gpt_lossf += gpt_loss.item()

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, attention_mask = get_seq_train_batch(segment_num, True)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(all_loss).backward()

        # input_memory_list[micro_step] = input_memory.detach()
        input_memory_list[micro_step] = None

        # # 重启记忆
        # for bi in range(input_memory.shape[0]):
        #     if random.randint(1, 100) > remember_prob:
        #         input_memory_list[micro_step][bi] = raw_evolver_model.initial_memory

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, gpt_loss {gpt_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/predict_loss": predict_lossf,
                "train/revise_loss": revise_lossf,
                "train/gpt_loss": gpt_lossf,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
