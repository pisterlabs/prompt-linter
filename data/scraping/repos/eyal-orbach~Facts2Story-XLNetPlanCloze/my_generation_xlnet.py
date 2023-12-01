#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
from typing import List

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import numpy as np

from MaskedPlotDataset import MaskedPlotDataset
from custom_model.CXLNetModel import CXLNetLMHeadModel
from run_xlnet_finetuning import get_target_mapping, get_perm_masks
from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer

import examples.my_text_parser as my_parser

NAIVE_MASK_EXPOSED_SIZE = 20

RESULTS_OUT_PATH = "/home/nlp/eyalo/tmp/pycharm_project_711/customxlnet/examples/generated_text/full_plots_genres/"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig)), ())

MODEL_CLASSES = {
    'cxlnet': (CXLNetLMHeadModel, XLNetTokenizer),
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


global_device = "2"

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = 0
        logits = logits/logits.sum()

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = 0
        logits = logits/logits.sum()
    return logits


def get_perm_mask(tokenized_parts_data, perm_list_idx, permutation_list, samples_num):
    mask = []
    tokens_len = 0
    for part in tokenized_parts_data:
        part_len = len(part["tokens"])
        tokens_len += part_len
        part_mask = torch.ones(part_len, dtype=torch.float, device=global_device)
        if not part["hidden"]:
            part_mask *= 0
        mask.append(part_mask)
    mask = torch.cat(mask)
    relative_indices_to_uncover = torch.tensor(permutation_list[:perm_list_idx], dtype=torch.long, device=global_device)
    indices_to_uncover = mask.nonzero().take(relative_indices_to_uncover)
    mask.index_put_([indices_to_uncover], torch.tensor(0, device=global_device).float())
    return mask.expand(samples_num, tokens_len, -1)


# def get_target_mapping(tokenized_parts_data, perm_list_idx, permutation_list, num_samples):
#     idx, length = index_to_predict(perm_list_idx, permutation_list, tokenized_parts_data)
#     target = torch.zeros(length, device=global_device)
#     target[idx] = 1
#     return target.expand(num_samples, 1, -1)


def append_to_generated(generated, tokens, tokenized_parts_data, perm_list_idx, permutation_list):
    for sample_idx, generated_token in enumerate(tokens):
        gen = generated[sample_idx]
        index_to_fill, _ = index_to_predict(perm_list_idx, permutation_list, tokenized_parts_data)
        gen.index_put_([index_to_fill], generated_token)
        generated[sample_idx] = gen

    return generated


def index_to_predict(perm_list_idx, permutation_list, tokenized_parts_data):
    mask = []
    tokens_len = 0
    for part in tokenized_parts_data:
        part_len = len(part["tokens"])
        tokens_len += part_len
        part_mask = torch.ones(part_len, device=global_device)
        if not part["hidden"]:
            part_mask *= 0
        mask.append(part_mask)
    mask = torch.cat(mask)
    relative_index_to_fill = torch.tensor(permutation_list[perm_list_idx], dtype=torch.long, device=global_device)
    index_to_fill = mask.nonzero().take(relative_index_to_fill)
    return index_to_fill, tokens_len


def sample_sequence(model, context, perm_masks, padding_masks, target_mappings, temperature=1, top_k=0, top_p=0.0,
                    device='cpu'):
    global global_device
    global_device = device
    generated = context.masked_fill(target_mappings[0].diag().bool() ,0)
    with torch.no_grad():
        for perm_list_idx in range(perm_masks.size(-1)):
            input_ids = generated
            curr_perm_mask = perm_masks[:,perm_list_idx,:].expand(perm_masks.size())
            curr_target_mapping = target_mappings[:,perm_list_idx,:].unsqueeze(1)
            inputs = {'input_ids': input_ids, 'perm_mask': curr_perm_mask, 'target_mapping': curr_target_mapping, }
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            for i in range(outputs[0].size(0)):
                if target_mappings[i,perm_list_idx,perm_list_idx] > 0:
                    logits = outputs[0][i][0]
                    distrbution = F.softmax(logits/ temperature, dim=-1)
                    token = generate_token(distrbution, top_k, top_p)
                    generated[i, perm_list_idx] = token.clone()

    return generated


def generate_token(token_logits, top_k, top_p):
    filtered_logits = top_k_top_p_filtering(token_logits, top_k=top_k, top_p=top_p)
    next_token = torch.multinomial(filtered_logits, num_samples=1)
    return next_token

tokenizer = None


def get_text_with_blanks(inputs, target_map):
    mask_token = torch.tensor(tokenizer._convert_token_to_id("_"))
    masked_input = inputs[0].masked_fill(target_map[0].diag().bool() ,mask_token)
    return tokenizer.decode(masked_input)



def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--test_batch_size', type=int, default=1)

    parser.add_argument('--test_file_path', type=str, default=None,
                        help="path of parsed plots to generate completion")

    args = parser.parse_args()

    args.device = torch.device("cuda:2" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("device is " + str(args.device))
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    args.length = model.config.max_position_embeddings  # No generation bigger than model size
    if args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        seqs, masks = zip(*examples)
        pad_seqs =pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
        pad_masks = pad_sequence(masks, batch_first=True, padding_value=tokenizer.pad_token_id)
        return list(zip(pad_seqs, pad_masks))

    test_dataset=MaskedPlotDataset(tokenizer, args, args.test_file_path, block_size=512)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, collate_fn=collate
    )

    batch_counter=0
    spltarr =args.model_name_or_path.split("/")
    model_name = spltarr[-1] if spltarr[-1] != "" else spltarr[-2]
    out_path = os.path.join(RESULTS_OUT_PATH, model_name)
    for batch in test_dataloader:
        batch_counter+=1
        with torch.no_grad():
            inputs_raw = torch.stack([seq.to(args.device) for seq, mask in batch])
            masks_raw = torch.stack([mask.to(args.device) for seq, mask in batch])

            # naive_input_text = tokenizer.encode("please continue from here",add_special_tokens=False, return_tensors="pt")
            # NAIVE_MASK_EXPOSED_SIZE = naive_input_text.size(-1)
            # naive_input= torch.cat([naive_input_text, torch.zeros(inputs_raw.size(1)-NAIVE_MASK_EXPOSED_SIZE, dtype=torch.long).unsqueeze(0)], dim = -1) \
            #     .to(args.device).expand(inputs_raw.size())


            # naive_masks = torch.cat([torch.ones(NAIVE_MASK_EXPOSED_SIZE), torch.zeros(inputs_raw.size(1)-NAIVE_MASK_EXPOSED_SIZE)])\
            #     .to(args.device).unsqueeze(0).expand(inputs_raw.size())
            # masks_raw = naive_masks
            # # inputs_raw = naive_input


            prefix_tensor = tokenizer.encode(PADDING_TEXT,add_special_tokens=False, return_tensors="pt").to(args.device).long()
            prefix_mask = torch.ones(prefix_tensor.size()).to(args.device)
            prefix_tensor =prefix_tensor.expand((inputs_raw.size(0),prefix_tensor.size(-1)))
            prefix_mask =prefix_mask.expand((masks_raw.size(0),prefix_mask.size(-1)))
            inputs = torch.cat([prefix_tensor, inputs_raw], dim=1)
            masks = torch.cat([prefix_mask, masks_raw], dim=1)

            padding_masks = torch.where(masks == tokenizer.pad_token_id,
                                        torch.ones_like(masks), torch.zeros_like(masks))

            perm_masks = get_perm_masks(masks, order="L2R")
            target_map = get_target_mapping(masks, args.device)

            out = sample_sequence(
                model=model,
                context=inputs,
                perm_masks = perm_masks,
                padding_masks=padding_masks,
                target_mappings=target_map,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )


            text = tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=True)[len(PADDING_TEXT):]
            print(text)
            original_text = tokenizer.decode(inputs[0])[len(PADDING_TEXT):]
            masked_text = get_text_with_blanks(inputs, target_map)[len(PADDING_TEXT):]
            os.makedirs(out_path, exist_ok=True)
            with open(out_path + "/result" + str(batch_counter), "w") as f:

                f.writelines("\ntext:\n\n")
                f.writelines(text)
                f.writelines("\n--------masked-------\n")
                f.writelines(masked_text)
                f.writelines("\n--------original-------\n")
                f.writelines(original_text)


def custom_print(out):
    padding_len = len(PADDING_TEXT)
    split1 = out[:padding_len]
    split2 = out[padding_len:]
    text1= tokenizer.decode(split1, clean_up_tokenization_spaces=True)

    text2 = tokenizer.decode(split2, clean_up_tokenization_spaces=True)
    print(text1)
    print("\n")
    print(text2)
    return text1 + text2

def get_context_tokens():
    raw_text = " This is a small test with some text to start a language model"
    print("raw text=" + raw_text)
    small_padding =PADDING_TEXT
    raw_text = small_padding + raw_text
    context_tokens = tokenizer.encode(raw_text)
    return context_tokens


def get_permutation_list(tokenized_parts, do_shuffle):
    from random import shuffle
    plist = get_inorder_list(tokenized_parts)
    if do_shuffle:
        shuffle(plist)

    return plist



def get_inorder_list(tokenized_parts):
    list_len = 0
    for part in tokenized_parts:
        if part["hidden"]:
            list_len += len(part["tokens"])
    return list(range(list_len))



#create equivelent for exsiting inputs
def get_masked_context_data(prefix_text="", shuffle=False):
    parts, _ = my_parser.get_parts(prefix_text=prefix_text)
    tokenized_parts = []
    all_tokenized = []
    for tup in parts:
        part_text = tup[1]
        part_text_tokenized = tokenizer.encode(" ".join(part_text))
        part_obj = {"hidden":tup[0], "tokens": part_text_tokenized}
        tokenized_parts.append(part_obj)
        all_tokenized += part_text_tokenized

    permutation_list = get_permutation_list(tokenized_parts, shuffle)
    return all_tokenized, tokenized_parts, permutation_list


if __name__ == '__main__':
    PADDING_TEXT = ""
    main()
