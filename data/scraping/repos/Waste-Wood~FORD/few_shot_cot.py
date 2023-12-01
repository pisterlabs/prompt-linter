from argparse import ArgumentParser
from tools import read_jsonlines, define_logger, DynamicDataset
import jsonlines
import tqdm
import os
import time
import openai
from torch.utils.data import DataLoader
import backoff
import torch
import pdb


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError))
def get_response(hps, pmp, batch):
    qs, os, _, _ = batch
    if os[0] != 'None' and "yes" not in os[0]:
        overall_prompt = ["{}\n\nQ: {} Answer Choices: {}\nA: ".format(pmp, q, o) for q, o in zip(qs, os)]
    else:
        overall_prompt = ["{}\n\nQ: {}\nA: ".format(pmp, q) for q in qs]
    responds = openai.Completion.create(
        model=hps.model,
        prompt=overall_prompt,
        temperature=hps.temperature,
        max_tokens=hps.max_tokens,
        top_p=1.0,
        api_key=openai.api_key,
        n=hps.num_sequences,
        
    )
    return responds


def hyper_parameters():
    parser = ArgumentParser('Few-Shot')

    parser.add_argument('--data_dir', type=str, default='./data/jsonlines')
    parser.add_argument('--dataset', type=str, default='aqua')
    parser.add_argument('--prompt_dir', type=str, default='./data/prompts')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='code-davinci-002')
    parser.add_argument('--mode', type=str, default='manual_cot')

    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./output/few_shot_cot')
    parser.add_argument('--num_sequences', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=150)
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

    if hps.dataset in ['addsub', 'gsm8k', 'multiarith', 'singleop', 'singleeq', 'svamp']:
        prompt = open(os.path.join(hps.prompt_dir, '{}_{}.txt'.format(hps.mode, 'mwp')), 'r').read()
    else:
        prompt = open(os.path.join(hps.prompt_dir, '{}_{}.txt'.format(hps.mode, hps.dataset)), 'r').read()

    openai.api_key = hps.api
    if not os.path.exists(os.path.join(hps.output_dir)):
        os.makedirs(os.path.join(hps.output_dir))

    output_file = os.path.join(hps.output_dir, '{}_sc.jsonl'.format(hps.dataset))

    if os.path.exists(output_file):
        number = len([d for d in open(output_file, 'r')])
        data = [d[number:] for d in data]
        fo = jsonlines.open(output_file, 'a')
    else:
        fo = jsonlines.open(output_file, 'w')

    DATA = DynamicDataset(*data)
    dataloader = DataLoader(DATA, batch_size=hps.batch_size)
    bar = tqdm.trange(len(dataloader))

    for batch, _ in zip(dataloader, bar):
        if batch[-1][0] == 'None':
            rs = get_response(hps, prompt, batch)
            # pdb.set_trace()
            # for r, q, o, a, _ in zip(rs['choices'], *batch):
            #     if isinstance(a, torch.Tensor):
            #         fo.write({'Q': q, 'A': a.item(), 'O': o, 'R': r})
            #     else:
            #         fo.write({'Q': q, 'A': a, 'O': o, 'R': r})
            for i, q, o, a, _ in zip(range(len(batch[0])), *batch):
                if isinstance(a, torch.Tensor):
                    fo.write({'Q': q, 'A': a.item(), 'O': o, 'R': rs['choices'][i*3:(i+1)*3]})
                else:
                    fo.write({'Q': q, 'A': a, 'O': o, 'R': rs['choices'][i*3:(i+1)*3]})
        else:
            for i, q, o, a, r in zip(range(len(batch[0])), *batch):
                fo.write({'Q': q, 'A': a, 'O': o, 'R': rs['choices'][i*3:(i+1)*3]})
        # time.sleep(5)
    fo.close()
