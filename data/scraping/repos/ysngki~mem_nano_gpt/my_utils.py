import numpy as np
import math
import random
import torch
from peft import prepare_model_for_int8_training

from my_modeling_llama import LlamaForCausalLM
from my_modeling_opt import OPTForCausalLM
from model import MemoryGPT as GPT


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, config):
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> pretrained model has {total_params / 1e6} Million trainable params\n")


# get data for a minibatch
def get_seq_train_batch(data, data_pointer, this_batch_seg_num, block_size, min_block_size, device, device_type, plus_one=False):
    x_list = []
    y_list = []
    seg_length_list = []

    this_batch_size = len(data_pointer)

    def get_x_y_tensor_list(batch_start_point, no_update=False):
        # random segment len for each batch item
        random_length = torch.randint(block_size - min_block_size, (this_batch_size,)) + min_block_size
        segment_ends = random_length.clone()

        for bi in range(this_batch_size):
            this_end = random_length[bi] + batch_start_point[bi] # end index
            segment_ends[bi] = this_end if this_end < len(data) else len(data) - 1
            random_length[bi] = segment_ends[bi] - batch_start_point[bi] # actual length

        # (batch size, xxx)
        x = [torch.from_numpy((data[batch_start_point[bi]:segment_ends[bi]]).astype(np.int64)) for bi in range(this_batch_size)]
        y = [torch.from_numpy((data[batch_start_point[bi] + 1:segment_ends[bi] + 1]).astype(np.int64)) for bi in range(this_batch_size)]

        # update batch_start_point
        if no_update:
            pass
        else:
            for bi in range(this_batch_size):
                batch_start_point[bi] = segment_ends[bi] if segment_ends[bi] < len(data) - min_block_size * this_batch_seg_num else 0

        return x, y, batch_start_point, random_length
    

    fetch_seg_num = this_batch_seg_num + 1 if plus_one else this_batch_seg_num

    for seg_index in range(fetch_seg_num):
        # get data for this segment
        if seg_index == this_batch_seg_num: # plus one segment for prediction
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer, True)
        else:
            this_x, this_y, data_pointer, this_seg_length = get_x_y_tensor_list(data_pointer)

        seg_length_list.append(this_seg_length)

        # padding to (batch size, block size)
        padding_x = this_x[0].new_full((this_batch_size, block_size), fill_value=0)
        padding_y = this_y[0].new_full((this_batch_size, block_size), fill_value=-1)

        for bi in range(this_batch_size):
            padding_x[bi][:len(this_x[bi])] = this_x[bi]
            padding_y[bi][:len(this_y[bi])] = this_y[bi]

        # return_offset_x.append(offset_padding_x)
        x_list.append(padding_x)
        y_list.append(padding_y)
    
    # (actual batch size, segment num, block size)
    x = torch.stack(x_list, dim=1)
    y = torch.stack(y_list, dim=1)
    attention_mask = y.ne(-1).int()
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, attention_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, attention_mask = x.to(device), y.to(device), attention_mask.to(device)
    
    # (segment num, actual batch size)
    seg_length_list = torch.stack(seg_length_list, dim=0)

    # seg_length_list: (segment num, actual batch size); x,y,attention_mask shape: (actual batch size, segment num, block size)
    return data_pointer, x, y, attention_mask, seg_length_list


def load_pretrained_model(config):
    if "gpt" in config.pretrained_model_name:
        print(f"Initializing from OpenAI GPT-2 weights: {config.pretrained_model_name}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=config.dropout)
        pretrained_model = GPT.from_pretrained(config.pretrained_model_name, override_args)
        pretrained_model.to(config.device)

        pretrained_model_config = pretrained_model.config
        pretrained_model_config.num_hidden_layers = 12
        pretrained_model_config.hidden_size = 768
        pretrained_model_config.max_position_embeddings = 1024
    elif "llama" in config.pretrained_model_name:
        print(f"Initializing from llama weights: {config.pretrained_model_name}")
        # pretrained_model = LlamaForCausalLM.from_pretrained(config.pretrained_model_name, device_map="auto", cache_dir=config.cache_dir)
        pretrained_model = LlamaForCausalLM.from_pretrained(config.pretrained_model_name, load_in_8bit=True, device_map={'': config.device}, torch_dtype=torch.float16, cache_dir=config.cache_dir)
        # pretrained_model = LlamaForCausalLM.from_pretrained(config.pretrained_model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, cache_dir=config.cache_dir)
        pretrained_model = prepare_model_for_int8_training(pretrained_model)

        pretrained_model_config = pretrained_model.config
    elif "opt" in config.pretrained_model_name:
        print(f"Initializing from opt weights: {config.pretrained_model_name}")
        pretrained_model = OPTForCausalLM.from_pretrained(config.pretrained_model_name, load_in_8bit=True, device_map={'': config.device}, torch_dtype=torch.float16, cache_dir=config.cache_dir)
        pretrained_model = prepare_model_for_int8_training(pretrained_model)
        pretrained_model_config = pretrained_model.config
    else:
        raise Exception(f"Unrecognized pretrained model {config.pretrained_model_name}")

    return pretrained_model, pretrained_model_config


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def accelerate_estimate_predict_loss(accelerator, model, pretrained_model, val_dataloader, config):
    out = {}
    model.eval()

    split = "val"

    losses = torch.zeros(config.eval_iters, device=config.device)
    gpt_losses = torch.zeros(config.eval_iters, device=config.device)
    segment_losses = torch.zeros((config.segment_num, config.eval_iters), device=config.device)
    context_losses = torch.zeros(config.eval_iters, device=config.device)
    
    actual_batch_count = 0
    for val_bi, batch in enumerate(val_dataloader):
        if actual_batch_count < config.eval_iters:

            this_iter_loss = torch.tensor(0.0, device=config.device)
            this_iter_gpt_loss = torch.tensor(0.0, device=config.device)
            this_iter_segment_loss = torch.zeros((config.segment_num, ), device=config.device)
            this_iter_context_loss = torch.tensor(0.0, device=config.device)

            # empty memory
            input_memory = None

            for micro_step in range(config.gradient_accumulation_steps):
                input_ids, labels, attention_mask, segment_lengths = batch

                # get data for first segment
                this_x = input_ids[:, 0, :]
                this_y = labels[:, 0, :]
                this_attention_mask = attention_mask[:, 0, :]
                this_seg_len = segment_lengths[:, 0]

                target_model_parameter = model(input_memory=input_memory, produce_parameter_flag=True)
                
                for si in range(config.segment_num):
                    output_embeds = pretrained_model(input_ids=this_x, output_embeds=True, return_dict=False)
                    if hasattr(model, "dtype"):
                        output_embeds = output_embeds.to(model.dtype)
                    # X -> memory
                    input_memory = model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]
                    # last memory -> X
                    last_target_model_parameter = target_model_parameter
                    target_model_parameter = model(input_memory=input_memory, produce_parameter_flag=True)

                    # get data for next segment
                    next_x = input_ids[:, si + 1, :]
                    next_y = labels[:, si + 1, :]
                    next_attention_mask = attention_mask[:, si + 1, :]
                    next_seg_len = segment_lengths[:, si+1]

                    # predict next segment with memory
                    _, loss = pretrained_model(input_ids=next_x, labels=next_y, input_parameter=target_model_parameter, peft=config.peft_method, return_dict=False)
                    this_iter_loss = loss + this_iter_loss
                    this_iter_segment_loss[si] = loss + this_iter_segment_loss[si]

                    _, gpt_loss  = pretrained_model(input_ids=next_x, labels=next_y, return_dict=False)
                    this_iter_gpt_loss = gpt_loss + this_iter_gpt_loss

                    # predict with context (teacher) and last memory
                    with torch.no_grad():
                        bsz = this_x.shape[0]
                        two_seg_block_size = this_x.shape[1] + next_x.shape[1]

                        # concatenate two segments to get context
                        x_container = this_x.new_full((bsz, two_seg_block_size), fill_value=0)
                        for bi in range(bsz):
                            x_container[bi, :this_seg_len[bi]] = this_x[bi, :this_seg_len[bi]]
                            x_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_x[bi, :next_seg_len[bi]]
                        
                        y_container = this_y.new_full(x_container.size(), fill_value=-1)
                        for bi in range(bsz):
                            y_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_y[bi, :next_seg_len[bi]]

                        # predict
                        _, loss = pretrained_model(input_ids=x_container, labels=y_container, input_parameter=last_target_model_parameter, peft=config.peft_method, return_dict=False) # shape of logits_with_context: (batch_size, two_seg_block_size, vocab_size)
                        this_iter_context_loss = loss + this_iter_context_loss

                    # assignment
                    this_x = next_x
                    this_y = next_y
                    this_attention_mask = next_attention_mask
                    this_seg_len = next_seg_len

            this_iter_loss = this_iter_loss / (config.gradient_accumulation_steps * config.segment_num)
            this_iter_gpt_loss = this_iter_gpt_loss / (config.gradient_accumulation_steps * config.segment_num)
            this_iter_segment_loss = this_iter_segment_loss / config.gradient_accumulation_steps
            this_iter_context_loss = this_iter_context_loss / (config.gradient_accumulation_steps * config.segment_num)

            losses[val_bi] = this_iter_loss.item()
            gpt_losses[val_bi] = this_iter_gpt_loss.item()
            segment_losses[:, val_bi] = this_iter_segment_loss
            context_losses[val_bi] = this_iter_context_loss.item()

            actual_batch_count += 1
        else:
            break

        out[split] = accelerator.reduce(losses.mean(), reduction="mean")
        out[split + "_gpt"] = accelerator.reduce(gpt_losses.mean(), reduction="mean")
        out[split + "_segment"] = accelerator.reduce(segment_losses.mean(dim=1), reduction="mean")
        out[split + "_context"] = accelerator.reduce(context_losses.mean(), reduction="mean")
    model.train()
    return out


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_predict_loss(model, pretrained_model, train_data, val_data, actual_batch_size, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            data = train_data
        elif split == 'val':
            data = val_data
        else:
            raise NotImplementedError
        
        losses = torch.zeros(config.eval_iters)
        gpt_losses = torch.zeros(config.eval_iters)
        segment_losses = torch.zeros((config.segment_num, config.eval_iters))
        context_losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            this_iter_loss = torch.tensor(0.0, device=config.device)
            this_iter_gpt_loss = torch.tensor(0.0, device=config.device)
            this_iter_segment_loss = torch.zeros((config.segment_num, ), device=config.device)
            this_iter_context_loss = torch.tensor(0.0, device=config.device)

            # random start
            random_data_start_pointer = []
            for _ in range(0, actual_batch_size):
                random_data_start_pointer.append(random.randint(0, len(data) - config.block_size * config.segment_num - 1))

            # fetch data for this batch
            device_type = 'cuda' if 'cuda' in config.device else 'cpu'
            random_data_start_pointer, X, Y, attention_mask, seg_length_list = get_seq_train_batch(data, random_data_start_pointer, config.segment_num, config.block_size, config.min_block_size, config.device, device_type, True) # fetch the very first batch

            # empty memory
            input_memory_list = [None for _ in range(config.gradient_accumulation_steps)]

            for micro_step in range(config.gradient_accumulation_steps):
                this_micro_X = X[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
                this_micro_Y = Y[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
                this_micro_attention_mask = attention_mask[config.batch_size*micro_step : config.batch_size*(1+micro_step)]
                this_micro_seg_length_list = seg_length_list[:, config.batch_size*micro_step : config.batch_size*(1+micro_step)] # (seg num, micro batch size)

                # get data for first segment
                this_x = this_micro_X[:, 0, :]
                this_y = this_micro_Y[:, 0, :]
                this_attention_mask = this_micro_attention_mask[:, 0, :]
                this_seg_len = this_micro_seg_length_list[0]

                # get memory of last step
                input_memory = input_memory_list[micro_step]

                target_model_parameter = model(input_memory=input_memory, produce_parameter_flag=True)
                
                for si in range(config.segment_num):
                    output_embeds = pretrained_model(input_ids=this_x, output_embeds=True, return_dict=False)
                    # X -> memory
                    input_memory = model(inputs_embeds=output_embeds, attention_mask=this_attention_mask, input_memory=input_memory)["memory_output"]
                    # last memory -> X
                    last_target_model_parameter = target_model_parameter
                    target_model_parameter = model(input_memory=input_memory, produce_parameter_flag=True)

                    # get data for next segment
                    next_x = this_micro_X[:, si + 1, :]
                    next_y = this_micro_Y[:, si + 1, :]
                    next_attention_mask = this_micro_attention_mask[:, si + 1, :]
                    next_seg_len = this_micro_seg_length_list[si+1]

                    # predict next segment with memory
                    _, loss = pretrained_model(input_ids=next_x, labels=next_y, input_parameter=target_model_parameter, peft=config.peft_method, return_dict=False)
                    this_iter_loss = loss + this_iter_loss
                    this_iter_segment_loss[si] = loss + this_iter_segment_loss[si]

                    _, gpt_loss  = pretrained_model(input_ids=next_x, labels=next_y, return_dict=False)
                    this_iter_gpt_loss = gpt_loss + this_iter_gpt_loss

                    # predict with context (teacher) and last memory
                    with torch.no_grad():
                        bsz = this_x.shape[0]
                        two_seg_block_size = this_x.shape[1] + next_x.shape[1]

                        # concatenate two segments to get context
                        x_container = this_x.new_full((bsz, two_seg_block_size), fill_value=0)
                        for bi in range(bsz):
                            x_container[bi, :this_seg_len[bi]] = this_x[bi, :this_seg_len[bi]]
                            x_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_x[bi, :next_seg_len[bi]]
                        
                        y_container = this_y.new_full(x_container.size(), fill_value=-1)
                        for bi in range(bsz):
                            y_container[bi, this_seg_len[bi]:this_seg_len[bi] + next_seg_len[bi]] = next_y[bi, :next_seg_len[bi]]

                        # predict
                        _, loss = pretrained_model(input_ids=x_container, labels=y_container, input_parameter=last_target_model_parameter, peft=config.peft_method, return_dict=False) # shape of logits_with_context: (batch_size, two_seg_block_size, vocab_size)
                        this_iter_context_loss = loss + this_iter_context_loss

                    # assignment
                    this_x = next_x
                    this_y = next_y
                    this_attention_mask = next_attention_mask
                    this_seg_len = next_seg_len

            this_iter_loss = this_iter_loss / (config.gradient_accumulation_steps * config.segment_num)
            this_iter_gpt_loss = this_iter_gpt_loss / (config.gradient_accumulation_steps * config.segment_num)
            this_iter_segment_loss = this_iter_segment_loss / config.gradient_accumulation_steps
            this_iter_context_loss = this_iter_context_loss / (config.gradient_accumulation_steps * config.segment_num)

            losses[k] = this_iter_loss.item()
            gpt_losses[k] = this_iter_gpt_loss.item()
            segment_losses[:, k] = this_iter_segment_loss
            context_losses[k] = this_iter_context_loss.item()

        out[split] = losses.mean()
        out[split + "_gpt"] = gpt_losses.mean()
        out[split + "_segment"] = segment_losses.mean(dim=1).tolist()
        out[split + "_context"] = context_losses.mean()
    model.train()
    return out