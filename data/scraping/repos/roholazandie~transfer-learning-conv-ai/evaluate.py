# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments
from utils import download_pretrained_model, get_dataset, _bleu, _f1_score


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids: #todo rooh: to remove special tokens
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def get_emotions(dataset):


    for data in tqdm(dataset['valid']):
        utterances = data['utterances']

        for utterance in utterances:
            true_emotion = utterance["emotion"]


def calculate_metrics(args, model, tokenizer, dataset, special_tokens):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    all_blues = []
    all_f1_scores = []
    all_true_sentences = []
    all_predicted_sentences = []
    for data in tqdm(dataset['valid']):
        personality = data['personality']
        utterances = data['utterances']

        #utterance = utterances[-1] #only the longest conversaion
        for utterance in utterances:
            true_label = utterance['candidates'][-1]
            history = utterance['history']
            predicted_output = []
            for i in range(args.max_length):
                instance, _ = build_input_from_segments(personality, history, predicted_output, tokenizer, special_tokens, with_eos=False)

                try:

                    if len(instance["input_ids"]) > 310:
                        truncated_history = [hist[:5] for hist in history]
                        instance, _ = build_input_from_segments(personality, truncated_history, predicted_output, tokenizer, special_tokens, with_eos=False)

                    input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
                    token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

                    logits = model(input_ids, token_type_ids=token_type_ids)
                except:
                    print("exception")
                    continue

                if "gpt2" == args.model:
                    logits = logits[0]
                logits = logits[0, -1, :] / args.temperature
                logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                probs = F.softmax(logits, dim=-1)

                prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
                # if i < args.min_length and prev.item() in special_tokens_ids:
                #     k=0
                #     while prev.item() in special_tokens_ids and k < 100:
                #         prev = torch.multinomial(probs, num_samples=1)
                #         k+=1

                if i < args.min_length:
                    prev = torch.multinomial(probs, num_samples=1)

                # if prev.item() in special_tokens_ids:
                #     break
                predicted_output.append(prev.item())

            predicted_sentence = tokenizer.decode(predicted_output, skip_special_tokens=True)
            true_sentence = tokenizer.decode(true_label, skip_special_tokens=True)
            #looks like zero gives the best results

            all_predicted_sentences.append(predicted_sentence)
            all_true_sentences.append(true_sentence)

            bleus = [_bleu(predicted_sentence, [true_sentence], method="method"+str(i)) for i in [0,1,2,3,5]]
            #bleu = _bleu(predicted_sentence, [true_sentence])
            f1_score = _f1_score(predicted_sentence, [true_sentence])
            #print(f1_score)
            all_blues.append(bleus)
            all_f1_scores.append(f1_score)
            #compare predicted and label with bleu


    with open("/home/rohola/codes/transfer-learning-conv-ai/out/emotion_correlation_input.txt", 'w') as fw:
        for predicted_sentence, true_sentence in zip(all_predicted_sentences, all_true_sentences):
            fw.write(predicted_sentence + "\t" + true_sentence + "\n")

    print("avg bleu", np.array(all_blues).mean(axis=0))
    print("avg f1 score", np.mean(all_f1_scores))
    print("max bleu", np.array(all_blues).max(axis=0))


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/rohola/data/daily_dialog_full/daily_dialog.json",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--dataset_cache", type=str, default='./caches/dataset_cache_OpenAIGPTTokenizer', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="/home/rohola/codes/transfer-learning-conv-ai/logs/emotion_recognition_log", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    special_tokens = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
    calculate_metrics(args, model, tokenizer, dataset, special_tokens)
    #get_emotions(dataset)


if __name__ == "__main__":
    run()
