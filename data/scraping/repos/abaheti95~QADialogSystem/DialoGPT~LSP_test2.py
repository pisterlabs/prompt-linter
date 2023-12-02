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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
import re
import time

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
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

def print_list(l):
    for e in l:
        print(e)
    print()

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def beam_search(model, tokenizer, beam_size, n_best, max_length, context, answer_string, device='cpu'):
    # Encode and decode the answer_string
    answer_string_tokens = tokenizer.encode(answer_string, add_special_tokens=False)
    answer_string = tokenizer.decode(answer_string_tokens)
    
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    with torch.no_grad():
        generated = context
        hypotheses = [None] * beam_size
        scores = [0.0] * beam_size
        # print("MAX LENGTH:", max_length)
        possible_responses_with_scores = list()
        impossible_responses_with_scores = list()
        possible_responses_scores = list()
        for i in range(max_length):
            if i == 0:
                # expand once
                inputs = {'input_ids': generated.to(device)}
                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :]
                # We will convert logits to softmax probs
                next_token_logits = torch.nn.functional.log_softmax(next_token_logits, dim=1)
                filtered_logits, top_indices = torch.topk(next_token_logits, beam_size)
                for j in range(beam_size):
                    next_token = top_indices[:,j]
                    next_token_score = filtered_logits[:,j]
                    # print(tokenizer.decode(next_token.tolist()), "\t", (next_token_score))
                    beam_candiate = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                    hypotheses[j] = beam_candiate
                    scores[j] = next_token_score
                # print("\n\n\n")
            else:
                all_possibilities = list()
                for j, hypothesis in enumerate(hypotheses):
                    # expand every hypothesis
                    inputs = {'input_ids': hypothesis.to(device)}
                    outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                    next_token_logits = outputs[0][:, -1, :]
                    next_token_logits = torch.nn.functional.log_softmax(next_token_logits, dim=1)
                    filtered_logits, top_indices = torch.topk(next_token_logits, beam_size)
                    hyp_indices_list = hypothesis.squeeze().tolist()
                    # print(tokenizer.decode(hyp_indices_list))
                    # print(scores[j])
                    # print the next tokens for own reference
                    for k in range(beam_size):
                        next_token = top_indices[:,k]
                        next_token_score = filtered_logits[:,k]
                        all_possibilities.append((scores[j] + next_token_score, next_token, j))
                        # print(tokenizer.decode(next_token.tolist()), "\t", (scores[j] + next_token_score))
                        # all_possibilities.append((next_token_score, next_token, j))
                        # print(tokenizer.decode(next_token.tolist()), "\t", (next_token_score))
                # Sort all_possibilities and take top beam_size candidates
                all_possibilities = sorted(all_possibilities, key=lambda tup: tup[0], reverse=True)
                # Update all the hypothesis and scores
                new_hypotheses = list()
                new_scores = list()
                k = 0
                current_beam_best_score = None
                for score, next_token, j in all_possibilities:
                    END_TOKEN_ID = 50256
                    # 198 in the vocab is \n
                    # 628 in the vocab is \n\n
                    NEW_LINE_TOKEN_IDS = [198, 628]
                    if int(next_token) in NEW_LINE_TOKEN_IDS:
                        continue
                    # text = tokenizer.decode(next_token.tolist()).replace(" ", "")
                    # if re.search("\s",text):
                    #     print("AYAYAYAYA")
                    #     print(next_token)

                    #     print(";",[ord(c) for c in text],";")
                    #     exit()
                    if next_token == END_TOKEN_ID:
                        # Save this to the global list
                        # Don't add it again as hypothesis
                        finished_hyp = tokenizer.decode(hypotheses[j].squeeze().tolist())
                        if answer_string not in finished_hyp.split("<|endoftext|>")[1]:
                            impossible_responses_with_scores.append(((score/(i+1)), finished_hyp))
                            impossible_responses_with_scores = sorted(impossible_responses_with_scores, key=lambda tup: tup[0], reverse=True)
                            continue
                        # check if this finished_hyp is already present in the list
                        hyp_already_present_in_the_list_flag = False
                        for _, possible_response in possible_responses_with_scores:
                            if finished_hyp == possible_response:
                                hyp_already_present_in_the_list_flag = True
                                break
                        if hyp_already_present_in_the_list_flag:
                            continue
                        # possible_responses_with_scores.append((score, finished_hyp))
                        # Note:Save with length normalized scores for testing
                        possible_responses_with_scores.append(((score/(i+1)), finished_hyp))
                        possible_responses_with_scores = sorted(possible_responses_with_scores, key=lambda tup: tup[0], reverse=True)
                        continue
                    if k == 0:
                        # best score for this beam step
                        current_beam_best_score = score
                    new_hypotheses.append(torch.cat((hypotheses[j], next_token.unsqueeze(0)), dim=1)) 
                    new_scores.append(score)
                    # print(new_scores[k], "\t", tokenizer.decode(new_hypotheses[k].squeeze().tolist()).strip())
                    k += 1
                    if k == beam_size:
                        break
                hypotheses = new_hypotheses
                scores = new_scores

                # print("Iteration number:", i)
                # print("Hypotheses size:", len(hypotheses))
                # print("scores size:", len(scores))
                # print("Possible responses len:", len(possible_responses_with_scores))
                # print("\n\n\n")

                # if len(possible_responses_with_scores)>n_best and current_beam_best_score < possible_responses_with_scores[n_best-1][0]:
                if len(possible_responses_with_scores)>n_best and (current_beam_best_score/(i+1)) < possible_responses_with_scores[n_best-1][0]:
                    # if the current beam best score is greater than the n_best'th score then there is no chance that any of the future beam candidates will be in the n_best. Therefore we can terminate the beam_search
                    # pass
                    break
        # Sort the possible responses by score and print them
        possible_responses_with_scores = sorted(possible_responses_with_scores, key=lambda tup: tup[0], reverse=True)
        if len(possible_responses_with_scores) == 0:
            if len(impossible_responses_with_scores) == 0:
                # Send the current hypotheses and scores
                impossible_responses_with_scores = [(float(score)/25.0, tokenizer.decode(hyp.squeeze().tolist())) for score, hyp in zip(scores, hypotheses)]
            return impossible_responses_with_scores
        
        return possible_responses_with_scores





def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--state_dict", default=None, type=str, required=True,
                        help="Path to trained model state dict")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="File that needs to be tested")
    parser.add_argument("--out_file", default=None, type=str, required=True,
                        help="File to store the model predictions")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # args.device = torch.device("cpu")
    # args.n_gpu = 0

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    state_dict = torch.load(args.state_dict)
    state_dict["lm_head.weight"] = state_dict["lm_head.decoder.weight"]
    del state_dict["lm_head.decoder.weight"]
    model.load_state_dict(state_dict)
    logger.info("Loaded state dict from:{}".format(args.state_dict))
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    with open(args.test_file, "r") as reader, open(args.out_file, "w") as writer:
        for q_no, line in enumerate(reader):
            line = line.strip()
            xlm_lang = None
            # XLM Language usage detailed in the issues #1414
            if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
                    and model.config.use_lang_emb:
                if args.xlm_lang:
                    language = args.xlm_lang
                else:
                    language = None
                    while language not in tokenizer.lang2id.keys():
                        language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
                xlm_lang = tokenizer.lang2id[language]

            # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
            is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
            if is_xlm_mlm:
                xlm_mask_token = tokenizer.mask_token_id
            else:
                xlm_mask_token = None

            answer_string = line.split(" ||| ")[1]
            raw_text = line + "<|endoftext|>"
            if args.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
            context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
            if args.model_type == "ctrl":
                if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                    logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
            # out = sample_sequence(
            #     model=model,
            #     context=context_tokens,
            #     num_samples=args.num_samples,
            #     length=args.length,
            #     temperature=args.temperature,
            #     top_k=args.top_k,
            #     top_p=args.top_p,
            #     repetition_penalty=args.repetition_penalty,
            #     is_xlnet=bool(args.model_type == "xlnet"),
            #     is_xlm_mlm=is_xlm_mlm,
            #     xlm_mask_token=xlm_mask_token,
            #     xlm_lang=xlm_lang,
            #     device=args.device,
            # )
            start_time = time.time()
            out = beam_search(model, tokenizer, args.beam_size, args.n_best, args.length, context_tokens, answer_string, device=args.device)
            print(q_no, ":Question:", line)
            writer.write("{}:{}\n".format(q_no, line))
            for r_no, (score, response_text) in enumerate(out):
                score = float(score)
                response_text = response_text.split("<|endoftext|>")[1]
                print(r_no, "\t", score, "\t", response_text)
                # Save to out_file
                writer.write("{0}\t:{1:.4f}:\t{2}\n".format(r_no, score, response_text))
                # NOTE: temporarily print only the first response
                # break
            print("")
            writer.write("\n")
            print("Processing time:", time.time() - start_time)


if __name__ == '__main__':
    main()

# export CUDA_VISIBLE_DEVICES=3
# python LSP_test2.py --model_type gpt2 --model_name_or_path ./models/small --test_file ./data/src_squad_seq2seq_dev_moses_test_squad_model_predictions.txt --length 25 --n_best 5 --beam_size 20 --stop_token "<|endoftext|>" --out_file dialoGPT_ss_finetuned_predictions_on_squad_dev_test_with_squad_model_predictions.txt