import pdb
import time
from argparse import ArgumentParser
from tools import read_jsonlines, define_logger
import jsonlines
from tqdm import tqdm
import os
import openai
import backoff
import re
import json


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError, json.decoder.JSONDecodeError))
def get_response(hps, pmp):
    if 'gpt' in hps.model:
        responds = openai.ChatCompletion.create(
            model=hps.model,
            messages=pmp,
            max_tokens=hps.max_tokens,
            temperature=hps.temperature,
            api_key=openai.api_key,
            frequency_penalty=0,
            presence_penalty=0,
        )
    else:
        responds = openai.Completion.create(
            model=hps.model,
            prompt=pmp,
            max_tokens=hps.max_tokens,
            temperature=hps.temperature,
            api_key=openai.api_key,
            frequency_penalty=0,
            presence_penalty=0,
        )
    return responds


def process_prediction_chatgpt(task, response):
    text = response['choices'][0]['message']['content'].strip()
    try:
        text = text.split('Answer:')[1]
        prediction, explanation = text.split('Explanation: ')
        predictions = re.findall(pattern, prediction.replace("Answer", 'answer'))
        if len(predictions) >= 1:
            prediction = predictions[0]
        else:
            prediction = "None"
    except:
        explanation = text.strip()
        prediction = "None"
    return prediction, explanation


def process_prediction_davinci(task, response):
    text = response['choices'][0]['text'].strip()
    try:
        prediction, explanation = text.split('Explanation: ')
        predictions = re.findall(pattern, prediction.replace("Answer", 'answer'))
        if len(prediction) >= 1:
            prediction = predictions[0]
        else:
            prediction = "None"
    except:
        explanation = text.strip()
        prediction = "None"
    return prediction, explanation


def get_debate_process(example, model):
    rounds = min(len(example['R'][2]), len(example['R'][3]), len(example['R'][4]))
    chatgpt_arguments = example['R'][2][-rounds:]
    davinci_arguments = example['R'][3][-rounds:]
    gpt4_arguments = example['R'][4][-rounds:]

    if model == 'gpt-3.5-turbo':
        debate_procedure = [{"role": "system", "content": example['R'][0]['content']+f"\n\nQuestion: {example['Q']}\nOptions: {example['O']}"},
                            {"role": "user", "content": "\n".join([f"user1: {chatgpt_arguments[i]['content']}\nuser2: {davinci_arguments[i]['content']}\nuser3: {gpt4_arguments[i]['content']}" for i in range(rounds)])}]
        debate_procedure[-1]['content'] += f"\n\n{instruction}"

    elif model == 'gpt-4':
        debate_procedure = [{"role": "system", "content": example['R'][0]['content'].replace('user1', 'user3')+f"\n\nQuestion: {example['Q']}\nOptions: {example['O']}"},
                            {"role": "user", "content": "\n".join([f"user3: {gpt4_arguments[i]['content']}\nuser1: {chatgpt_arguments[i]['content']}\nuser2: {davinci_arguments[i]['content']}" for i in range(rounds)])}]
        debate_procedure[-1]['content'] += f"\n\n{instruction}"
    else:
        debate_procedure = f"{example['R'][0]['content'].replace('user1', 'user2')}\n\nQuestion: {example['Q']}\nOptions: {example['O']}\n\nDebate: \n" + "\n".join([f"user2: {davinci_arguments[i]['content']}\nuser3: {gpt4_arguments[i]['content']}\nuser1: {chatgpt_arguments[i]['content']}" for i in range(rounds)]) + f"\n\n{instruction}"

    return debate_procedure


def hyper_parameters():
    parser = ArgumentParser('Few-Shot')

    parser.add_argument('--candidate', type=str, default='./output/debate/round0')
    parser.add_argument('--dataset', type=str, default='strategyqa')
    parser.add_argument('--api', type=str, default='')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--mode', type=str, default='debate')

    parser.add_argument('--log_dir', type=str, default='./output/logger')
    parser.add_argument('--log_name', type=str, default='')

    parser.add_argument('--output_dir', type=str, default='./output/debate/round1_chatgpt')
    parser.add_argument('--num_sequences', type=int, default=1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--run', type=int, default=2)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    hps = hyper_parameters()
    hps.log_name = '{}_{}.txt'.format(hps.mode, hps.dataset)

    logger = define_logger(hps)

    logger.info('[HPS]: {}'.format(hps))
    logger.info('[API Key]: {}'.format(hps.api))
    logger.info('[Data]: {}'.format(hps.dataset))
    logger.info('[Model]: {}'.format(hps.model))

    if 'round0' not in hps.candidate:
        candidates = jsonlines.open(os.path.join(hps.candidate, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'r')
    else:
        candidates = jsonlines.open(os.path.join(hps.candidate, '{}_candidate.jsonl'.format(hps.dataset)), 'r')
    candidates = [c for c in candidates]

    openai.api_key = hps.api
    if not os.path.exists(os.path.join(hps.output_dir)):
        os.makedirs(os.path.join(hps.output_dir))

    questions = []
    if os.path.exists(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run))):
        fi = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'r')
        fi2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'r')
        questions += [d['Q'] for d in fi]
        questions += [d['Q'] for d in fi2]
        fi.close()
        fi2.close()
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'a')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'a')
    else:
        fo1 = jsonlines.open(os.path.join(hps.output_dir, '{}_agreed{}.jsonl'.format(hps.dataset, hps.run)), 'w')
        fo2 = jsonlines.open(os.path.join(hps.output_dir, '{}_candidate{}.jsonl'.format(hps.dataset, hps.run)), 'w')
    
    if hps.model == 'gpt-3.5-turbo':
        instruction = "Remember you are user1. What do you think about the opinions of user2 and user3? more reasonable? or more unreasonable? Please give your final answer choice of the Question starting with \"Answer: (A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one option."
    elif hps.model == 'text-davinci-003':
        instruction = "Remember you are user2. What do you think about the opinions of user1 and user3? more reasonable? or more unreasonable? Please give your final answer choice of the Question starting with \"Answer: (A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one option."
    else:
        # assert hps.model == 'gpt-3.5-turbo-0301'
        instruction = "Remember you are user3. What do you think about the opinions of user1 and user2? more reasonable? or more unreasonable? Please give your final answer choice of the Question starting with \"Answer: (A|B) is more plausible.\" and explain very shortly starting with \"Explanation: \". You should choose only one option."

    pattern = re.compile('\(?(A|B)\)?')
    pattern_davinci = re.compile("[So, |Therefore, ]?[t|T]he answer is \(?(A|B)\)?")

    candidate, summary = [], []
    agreed_r, agreed_w, disagreed = 0, 0, 0

    for c in tqdm(candidates):
        if c['Q'] in questions:
            continue

        prompt = get_debate_process(c, hps.model)

        if agreed_r + agreed_w + disagreed == 0:
            print(json.dumps(prompt, indent=1))

        rs = get_response(hps, prompt)
        if hps.model.startswith('gpt'):
            prediction, explanation = process_prediction_chatgpt(hps.dataset, rs)
        else:
            prediction, explanation = process_prediction_davinci(hps.dataset, rs)
        
        if hps.model == 'gpt-3.5-turbo':
            c['R'][2].append({"role": "assistant", "content": explanation.strip()})
            c['M'].append('chatgpt')
        elif hps.model == 'text-davinci-003':
            c['R'][3].append({"role": "assistant", "content": explanation.strip()})
            c['M'].append('davinci')
        else:
            c['R'][4].append({"role": "assistant", "content": explanation.strip()})
            c['M'].append('chatgpt0301')

        c['P'].append(prediction)

        answer = re.findall(pattern, c['A'])[0]
        
        if prediction == c['P'][-2] == c['P'][-3] == answer:
            summary.append(c)
            fo1.write(c)
            agreed_r += 1
        elif prediction == c['P'][-2] == c['P'][-3]:
            summary.append(c)
            fo1.write(c)
            agreed_w += 1
        else:
            candidate.append(c)
            fo2.write(c)
            disagreed += 1

    print('[Count]: {}'.format(agreed_r))
    print('[Summary]: {}'.format(len(summary)))
    print('[Candidate]: {}'.format(len(candidate)))
    print(f'[R-W-C]: {agreed_r}-{agreed_w}-{disagreed}')
    fo1.close()
    fo2.close()
