import pdb
import time
from argparse import ArgumentParser
from tools import read_jsonlines, define_logger
import jsonlines
import tqdm
import os
import openai
import backoff
import json
import pdb


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, json.decoder.JSONDecodeError, openai.error.APIError))
def get_response(hps, pmp):
    responds = openai.ChatCompletion.create(
        model=hps.model,
        messages=pmp,
        max_tokens=hps.max_tokens,
        temperature=hps.temperature,
        api_key=openai.api_key,
        frequency_penalty=0,
        presence_penalty=0,
        n=hps.num_sequences
    )
    return responds


def hyper_parameters():
    parser = ArgumentParser('Few-Shot')

    parser.add_argument('--data_dir', type=str, default='./data/jsonlines')
    parser.add_argument('--dataset', type=str, default='train_full')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--mode', type=str, default='zero_shot_chatgpt')

    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./output/chatgpt')
    parser.add_argument('--num_sequences', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    hps = hyper_parameters()
    hps.log_name = '{}_{}.txt'.format(hps.mode, hps.dataset)

    logger = define_logger(hps)

    logger.info('[HPS]: {}'.format(hps))
    logger.info('[API Key]: {}'.format(hps.api))
    logger.info('[Date]: {}'.format(hps.dataset))
    logger.info('[Model]: {}'.format(hps.model))

    data = read_jsonlines(os.path.join(hps.data_dir, '{}.jsonl'.format(hps.dataset)))

    openai.api_key = hps.api
    if not os.path.exists(os.path.join(hps.output_dir)):
        os.makedirs(os.path.join(hps.output_dir))

    output_file = os.path.join(hps.output_dir, '{}_sc.jsonl'.format(hps.dataset))

    if os.path.exists(output_file):
        fi = jsonlines.open(output_file, 'r')
        number = len([d for d in fi])
        data = [d[number:] for d in data]
        fi.close()
        fo = jsonlines.open(output_file, 'a')
    else:
        fo = jsonlines.open(output_file, 'w')
    
    bar = tqdm.trange(len(data[0]))
    for q, o, a, r, _ in zip(*data, bar):
        if hps.dataset != 'strategyqa':
            prompt = [{"role": "user", "content": "Question: {} Choices: {}\nPlease answer the above question by choosing a more plausible answer.You can choose only one answer from choices and give a short explanation. Please use the format like 'Answer: (A|B).\nExplanation: _'".format(q, o)}]
        else:
            prompt = [{"role": "user", "content": "Question: {}. Please answer yes or no for this question and give a short explanation. Please use the format like 'Answer: _.\nExplanation: _'".format(q)}]
        rs = get_response(hps, prompt)
        # pdb.set_trace()
        # choices = rs['choices'][0]
        # fo.write({'Q': q, 'A': a, 'O': o, 'R': {'text': choices['message']['content']}})
        fo.write({'Q': q, 'A': a, 'O': o, 'R': rs['choices']})
    fo.close()
