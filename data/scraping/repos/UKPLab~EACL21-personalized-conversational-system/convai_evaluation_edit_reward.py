# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
import logging
from pprint import pformat
from collections import defaultdict
from functools import partial
from tqdm import trange
import math

import torch
import torch.nn.functional as F
import numpy as np
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from projects.convai2.eval_hits import eval_hits, setup_args as setup_args_hits
from projects.convai2.eval_f1 import eval_f1, setup_args as setup_args_f1
from projects.convai2.eval_ppl import eval_ppl, setup_args as setup_args_ppl
from projects.convai2.build_dict import build_dict
from pytorch_pretrained_bert import OpenAIGPTTokenizer
from modeling_openai import OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTConfig

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling_openai import OpenAIGPTConfig

from train import build_input_from_segments, pad_dataset, SPECIAL_TOKENS
from utils import download_pretrained_model, AttrDict
from interact import sample_sequence
from rep_utils import get_ngrams, intrep_frac

from lib.bert_cls.nli_task import main as nli_engine
from rl_utils import reset_seed, bleu_rewarder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"   # TODO: now evaluating valid data!!!

class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument("--model_checkpoint", type=str, default="./runs/Sep10_22-10-31_krusty/", help="Path, url or short name of the model")   # "./runs/Jun03_00-25-57_krusty/"   All empty model: Aug17_00-03-04_krusty
        agent_args.add_argument("--eval_type", type=str, default="f1", help="hits@1, ppl or f1")   # please don't change this parameter
        # agent_args.add_argument("--model", type=str, default="openai-gpt", help="Model type (gpt or gpt2)")
        agent_args.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        agent_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        agent_args.add_argument("--no_sample", action='store_true')
        agent_args.add_argument("--max_length", type=int, default=20)
        agent_args.add_argument("--min_length", type=int, default=1)
        agent_args.add_argument("--seed", type=int, default=42)   # 0
        agent_args.add_argument("--temperature", type=int, default=0.7)
        agent_args.add_argument("--top_k", type=int, default=0)   # 20
        agent_args.add_argument("--top_p", type=float, default=0.9)  # del

        # NLI
        agent_args.add_argument("--do_lower_case", type=bool, default=True,
                            help="Set this flag if you are using an uncased model.")
        agent_args.add_argument("--output_dir", default='nli_output/', type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
        agent_args.add_argument("--bert_model", default='bert-base-uncased', type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                 "bert-base-multilingual-cased, bert-base-chinese.")
        # LM
        agent_args.add_argument("--lm_model_path", type=str, default='openai-gpt', help="Path of language model.")
        agent_args.add_argument("--lm_output_dir", type=str, default='lm_models/gpt_output', help="Output dir of language model.")

        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        args = AttrDict(opt)  # to keep most commands identical to the interact.py script
        self.args = args

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(args))

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if shared is None:
            self.logger.info("Get pretrained model and tokenizer")
            if args.model_checkpoint == "":
                args.model_checkpoint = download_pretrained_model()

            self.tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
            if self.args.eval_type == "hits@1":
                self.model_checkpoint = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_checkpoint)
            else:
                self.model_checkpoint = OpenAIGPTLMHeadModel.from_pretrained(args.model_checkpoint)
            self.model_checkpoint.to(args.device)
            self.model_checkpoint.eval()

            self.logger.info("Build BPE prefix dictionary")
            convai_dict = build_dict()
            assert len(convai_dict) == 19304
            self.prefix2words = self.get_prefix2words(convai_dict)
        else:
            self.model_checkpoint = shared['model']
            self.tokenizer = shared['tokenizer']
            self.prefix2words = shared['prefix2words']

        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        self.persona = []
        self.history = []
        self.labels = []

        self.reward = []
        self.nli_scores = np.array([0, 0, 0])
        self.reward_scores = 0   # reward function
        self.c_scores = 0   # C score
        self.cnm = 0   # C_new
        self.sample_num = 0   # sample number
        self.con_en = np.array([0, 0, 0])   # if the persona contains a contradicted/entail profile (not applied)
        self.intrep_scores = 0   # internal repetition score
        self.lm_ppl_scores = 0   # fine-tuned GPT-based language model
        self.bleu_scores = 0   # BLEU-2 score

        # Loading NLI models
        reset_seed(args.seed)
        self.nli_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        # print('config_file:', output_config_file)
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # print('model_file:', output_model_file)

        nli_config = BertConfig(output_config_file)
        self.nli_model = BertForSequenceClassification(nli_config, num_labels=3)
        self.nli_model.load_state_dict(torch.load(output_model_file))
        self.nli_model.to(args.device)
        self.nli_model.eval()

        # Loading LM models
        reset_seed(args.seed)
        self.lm_special_tokens = ['_start_', '_delimiter_', '_classify_']   # special tokens for LM
        # Load pre-trained model (weights)
        with torch.no_grad():
            lm_output_config_file = os.path.join(args.lm_output_dir, CONFIG_NAME)
            lm_config = OpenAIGPTConfig(lm_output_config_file)
            print(type(lm_config))
            if not isinstance(lm_config, OpenAIGPTConfig):
                print('NOT')
            lm_output_model_file = os.path.join(args.lm_output_dir, WEIGHTS_NAME)
            lm_model_state_dict = torch.load(lm_output_model_file)
            self.lm_model = OpenAIGPTLMHeadModel(lm_config)
            self.lm_model.load_state_dict(lm_model_state_dict)

            # Load pre-trained model tokenizer (vocabulary)
            self.lm_tokenizer = OpenAIGPTTokenizer.from_pretrained(args.lm_model_path, special_tokens=self.lm_special_tokens)
        self.special_tokens_ids = list(self.lm_tokenizer.convert_tokens_to_ids(token) for token in self.lm_special_tokens)
        self.lm_model.to(args.device)
        self.lm_model.eval()

        reset_seed(args.seed)

        self.reset()

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if self.labels:
            # Add the previous response to the history
            self.history.append(self.labels)

        if 'labels' in observation or 'eval_labels' in observation:
            text = observation.get('labels', observation.get('eval_labels', [[]]))[0]
            self.labels = self.tokenizer.encode(text)

        if 'text' in observation:
            text = observation['text']
            for subtext in text.split('\n'):
                subtext = subtext.strip()
                if subtext.startswith('your persona:'):
                    subtext = subtext.replace('your persona:', '').strip()
                    self.persona.append(self.tokenizer.encode(subtext))
                else:
                    self.history.append(self.tokenizer.encode(subtext))

        self.history = self.history[-(2*self.args.max_history+1):]

        candidates = []
        if 'label_candidates' in observation:
            for candidate in observation['label_candidates']:
                candidates.append((self.tokenizer.encode(candidate), candidate))
        self.candidates = candidates

        self.episode_done = observation['episode_done']
        self.observation = observation
        return observation

    def act(self):
        reply = {}

        if self.args.eval_type == "hits@1" and len(self.candidates) > 0:
            instances = defaultdict(list)
            for candidate, _ in self.candidates:
                instance, _ = build_input_from_segments(self.persona, self.history, candidate, self.tokenizer)
                for input_name, input_array in instance.items():
                    instances[input_name].append(input_array)

            inputs = pad_dataset(instances, padding=self.special_tokens_ids[-1])

            tensor_inputs = {}
            for input_name in ["input_ids", "mc_token_ids", "token_type_ids"]:
                tensor = torch.tensor(inputs[input_name], device=self.args.device)
                tensor = tensor.view((-1, len(self.candidates)) + tensor.shape[1:])
                tensor_inputs[input_name] = tensor

            with torch.no_grad():
                _, mc_logits = self.model_checkpoint(**tensor_inputs)

            val, ind = torch.sort(mc_logits[0], descending=True)

            ypred = self.candidates[ind[0].item()][1] # match
            tc = []
            for j in range(len(self.candidates)):
                tc.append(self.candidates[ind[j].item()][1])
            reply = {'text': ypred, 'text_candidates': tc}
        else:
            # We are in interactive of f1 evaluation mode => just sample
            with torch.no_grad():
                out_ids = sample_sequence(self.persona, self.history, self.tokenizer, self.model_checkpoint, self.args)   # YW: TODO: out_ids, _?
            # Get a generated response
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
            out_text_org = out_text
            out_text = out_text.replace(' \' ', '\'')   # TODO: tbd
            out_text = out_text.replace(' \'', '\'')
            # persona NLI
            profiles = []
            for profile in self.persona:
                profile_text = self.tokenizer.decode(profile, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                profile_text = profile_text.replace(' \' ', '\'')   # TODO: tbd
                profile_text = profile_text.replace(' \'', '\'')
                profiles.append(profile_text)
            nli_score, reward_score, c_score, current_con_en = nli_engine(out_text, profiles, self.nli_tokenizer, self.nli_model, eval=True)
            self.nli_scores += nli_score   # persona NLI
            self.reward_scores += reward_score   # reward function
            self.c_scores += c_score   # C score
            self.sample_num += 1
            self.con_en += current_con_en   # if this persona contains a contradicted/entail profile or not (not applied)

            # internal repetition
            response_tok = out_text_org.split()
            intrep_1gram = intrep_frac(response_tok)
            # if 2-gram or 3-gram are going to be used:
            ''''
            # intrep_2gram
            response_tok_2gram = get_ngrams(out_text, 2)
            intrep_2gram = intrep_frac(response_tok_2gram)
            # intrep_3gram
            response_tok_3gram = get_ngrams(out_text, 3)
            intrep_3gram = intrep_frac(response_tok_3gram)
            '''
            intern_rep_reward = intrep_1gram
            self.intrep_scores += intern_rep_reward

            # bleu
            label_text = self.tokenizer.decode(self.labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            current_bleu = bleu_rewarder(out_text_org, label_text)
            self.bleu_scores += current_bleu

            # fine-tuned GPT-based language model
            lm_tokenize_input = self.lm_tokenizer.tokenize(out_text)
            # lm_tensor_input = torch.tensor([lm_tokenizer.convert_tokens_to_ids(lm_tokenize_input)]).to(args.device)
            lm_tensor_input = torch.tensor([[self.special_tokens_ids[0]] + self.lm_tokenizer.convert_tokens_to_ids(lm_tokenize_input) + [self.special_tokens_ids[-1]]]).to(self.args.device)
            lm_loss = self.lm_model(lm_tensor_input, lm_labels=lm_tensor_input)
            lm_ppl = math.exp(lm_loss.item())
            self.lm_ppl_scores += lm_ppl

            print('out_text:', out_text)
            print('current nli:', self.nli_scores)
            print('current score:', self.reward_scores / self.sample_num)
            print('current c_score_macro:', self.c_scores / self.sample_num)
            current_c_score_micro = (self.nli_scores[1] - self.nli_scores[0]) / sum(self.nli_scores)
            cn_res = nli_score[1] - nli_score[0]   # cn: C_new (persona level)
            # C_new calculation
            if cn_res > 0:
                current_cn = 1
            elif cn_res < 0:
                current_cn = -1
            else:
                current_cn = 0
            self.cnm += current_cn
            print('current c_new:', self.cnm / self.sample_num)
            print('current c_score_micro:', current_c_score_micro)
            print('current con_en:', self.con_en)
            print('current intrep score:', self.intrep_scores / self.sample_num)
            print('current BLEU:', self.bleu_scores / self.sample_num)
            print('current PPL:', self.lm_ppl_scores / self.sample_num)
            reply = {'text': out_text}

        return reply

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.
        """
        partial_out_ids = self.tokenizer.encode(' '.join(partial_out))
        instance, _ = build_input_from_segments(self.persona, self.history, partial_out_ids,
                                             self.tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=self.args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=self.args.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model_checkpoint(input_ids, token_type_ids=token_type_ids)

        probs = F.softmax(logits[0, -1], dim=0)

        dist = {}
        for prefix_id, words in self.prefix2words.items():
            for word, ratio in words.items():
                dist[word] = probs[prefix_id].item() * ratio
        return dist

    def get_prefix2words(self, convai_dict, smoothing_freq=5):
        """ map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts) """
        prefix2words = defaultdict(dict)
        for i in trange(len(convai_dict)):
            word = convai_dict[i]
            freq = convai_dict.freq[word] + smoothing_freq
            bpe_tokens = self.tokenizer.bpe(word).split(' ')
            prefix_id = self.tokenizer.convert_tokens_to_ids(bpe_tokens[0])
            prefix2words[prefix_id].update(dict([(word, freq)]))

        for prefix_id, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix_id] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model_checkpoint
        shared['prefix2words'] = self.prefix2words
        return shared

    def reset(self):
        self.persona = []
        self.history = []
        self.labels = []
        self.candidates = []
        self.episode_done = True
        self.observation = None


if __name__ == '__main__':
    parser = base_setup_args(None)
    parser.set_params(
        model='convai_evaluation_edit_reward:TransformerAgent')
    opt = parser.parse_args(print_args=False)

    if opt['eval_type'] == "hits@1":
        setup_args = setup_args_hits(None)
        eval_fct = partial(eval_hits, print_parser=setup_args)
    elif opt['eval_type'] == "ppl":
        setup_args = setup_args_ppl(None)
        eval_fct = eval_ppl
    elif opt['eval_type'] == "f1":
        setup_args = setup_args_f1(None)
        eval_fct = partial(eval_f1, print_parser=setup_args)
    else:
        raise ValueError

    setup_args.set_params(
        model='convai_evaluation_edit_reward:TransformerAgent')
    opt = setup_args.parse_args(print_args=False)

    eval_fct(opt)
