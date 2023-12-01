"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 

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
import glob
import socket

from contextlib import nullcontext

import numpy as np
import random
import torch
import wandb
import config.train_shakespeare_char as params
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from model import GPTConfig, GPT

#requires PyTorch >=2.0 #TODO

# https://github.com/pytorch/kineto/issues/726
#os.environ.update({'KINETO_LOG_LEVEL' : '3'})  #TODO

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
backend =  'nccl'  # 'nccl', 'gloo', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
wandb_config = {k: globals()[k] for k in config_keys} # will be useful for logging

#print('\n wandb_config', wandb_config)
#print('\n wandb.__file__',wandb.__file__)
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#print('\n check device',torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.device(0), torch.cuda.get_device_name(0))

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def main():

    #we will use ddp even with a single GPU
    if torch.cuda.is_available():
        print('\n Using Torch DDP \n ')
        ddp = True
        if torch.cuda.device_count() > 1:
            ddp_local_rank = int(os.environ['RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])#ngpus_per_node = torch.cuda.device_count() #=local world_size
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(params.random_port) #has to be the same as server_socket so wandb will access GPU
            ddp_local_rank = 0
            ddp_world_size = 1

        init_process_group(backend=backend, rank=ddp_local_rank, world_size=ddp_world_size) 
        master_process = ddp_local_rank  == 0
        seed_offset = ddp_local_rank # each process gets a different seed
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device) 
        torch.manual_seed(1337 + seed_offset)
        
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert params.gradient_accumulation_steps % ddp_world_size == 0
        params.gradient_accumulation_steps //= ddp_world_size

    else:
        ddp = False
        device = 'cpu'
        print('\n using CPU, this will be slow')
        master_process = True
        ddp_world_size = 1

      
    tokens_per_iter = params.gradient_accumulation_steps * ddp_world_size * params.batch_size * params.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    
        
    if master_process:
        os.makedirs(params.out_dir, exist_ok=True)

    # use distributed sampler? https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py#L87
    data_dir = os.path.join('/home/nanoGPT/data', params.dataset) # TODO adapt os.path.abspath(os.path.join('data', params.dataset))
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - params.block_size, (params.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+params.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+params.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

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
    model_args = dict(n_layer=params.n_layer, n_head=params.n_head, n_embd=params.n_embd, block_size=params.block_size,
                    bias=params.bias, vocab_size=None, dropout=params.dropout, flash=params.flash) # start with model_args from command line

    if params.init_from == 'scratch':
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif params.init_from == 'resume':
        print(f"Resuming training from {params.out_dir}")
        ckpt_path = os.path.join(params.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'flash']:
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
        print('\n warm start from iter_num \n ',iter_num)
        best_val_loss = checkpoint['best_val_loss']
    elif params.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {params.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=params.dropout)
        model = GPT.from_pretrained(params.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'flash']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if params.block_size < model.config.block_size:
        model.crop_block_size(params.block_size)
        model_args['block_size'] = params.block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(params.weight_decay, 
                                        params.learning_rate, 
                                        (params.beta1, params.beta2), 
                                        device_type)
    if params.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        print('model.module')

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(params.eval_iters)
            for k in range(params.eval_iters):
                X, Y = get_batch(split)
                with ctx: #MM: ctx = autocast(device_type, dtype=ptdtype)
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < params.warmup_iters:
            return params.learning_rate * it / params.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > params.lr_decay_iters:
            return params.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - params.warmup_iters) / (params.lr_decay_iters - params.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return params.min_lr + coeff * (params.learning_rate - params.min_lr)



    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0


    #create a separate logs folder for each attention run
    attention_type = 'flash' if params.flash else 'slow'
    parameters = "-".join([f"{key}{value}" for key, value in params.__dict__.items() if key in ['max_iters', 'n_head', 'n_embd', 'block_size']])

    log_folder_name = attention_type+'-'+parameters+"-".join([f"{key}{value}" for key, value in params.profiler_schedule_args.items()])
    log_folder_name = f"logs_{log_folder_name}"
    if master_process:
        os.makedirs(params.out_dir+'/'+log_folder_name, exist_ok=True)

    if params.wandb_log and master_process:
        # wandb.tensorboard.patch(root_logdir=out_dir) #+'/'+log_folder_name)
        wandb_run_name = attention_type+'-'+parameters
        wandb.init(project=params.wandb_project, name=wandb_run_name, config=wandb_config, sync_tensorboard=True) 

    # Wrap train loop in the profiler context manager:
    #print('\n torch.profiler.itt.is_available()',torch.profiler.itt.is_available())
    print("\n \n profiler_schedule_args:  {} \n \n".format(params.profiler_schedule_args))
    with torch.profiler.profile(

        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        schedule = torch.profiler.schedule(**params.profiler_schedule_args),

        on_trace_ready = torch.profiler.tensorboard_trace_handler(params.out_dir+'/'+log_folder_name, worker_name="gpu0"),#callable that is called at each step when schedule returns ProfilerAction.RECORD_AND_SAVE during the profiling.
        with_stack = True,
        with_flops = True,
        with_modules = True   
    ) as profiler:
        
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if params.decay_lr else params.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % params.eval_interval == 0 and master_process:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if params.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # model flops utilization - convert to percentage
                    })
                if losses['val'] < best_val_loss or params.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': wandb_config,
                        }
                        print(f"saving checkpoint to {params.out_dir}")
                        torch.save(checkpoint, os.path.join(params.out_dir, 'ckpt.pt'))
            if iter_num == 0 and params.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(params.gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (micro_step == params.gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / params.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = get_batch('train')
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
                
            # clip the gradient
            if params.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % params.log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * params.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(params.batch_size * params.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > params.max_iters:
                break

            profiler.step()
        profiler.stop()
        for cycle in glob.glob(f"{params.out_dir+'/'+log_folder_name}/*.pt.trace.json", recursive=True):
            wandb.save(cycle, base_path=f"{params.out_dir+'/'+log_folder_name}", policy="now") 
        wandb.finish(quiet=True) #TODO move this to boilerplate? #quiet:true to minimize log output
                
    if ddp:
        destroy_process_group()



if __name__ == '__main__':
    host = '127.0.0.1'
    port = params.random_port
    print('\n \n \n socket params.random_port', params.random_port)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
    except Exception as e:
        print(f"Failed to bind: {e}")
    else:
        server_socket.listen(1)
        print(f"Listening on {host}:{port}")
    
        try:
            main()
        finally:
            server_socket.close()


