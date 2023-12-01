import os
import sys
import json
import time
import math
import argparse

import dotenv
import openai
import requests
import jsonlines as jsl
from tqdm import tqdm

from dataset import TextualDatasetFromYAML, TextualDatasetFromJsonLines


PROMPT_TEMPLATE = "Skriv {} läsförståelsefrågor med alternativ (a, b, c, d, o.s.v.) och ge varje fråga en unik nummer (1, 2, 3, o.s.v.). Första alternativet (a) ska alltid vara rätt, medan de andra alternativen (b, c, d, o.s.v.) ska vara felaktiga, men troliga. Alla frågor måste kunna besvaras av den följande texten."


def generate_mcq(text, num_mcqs, use_chat_gpt=False):
    trials, max_trials = 0, 10
    while trials < max_trials:
        try:
            gen_params = {
                'top_p': 0.9,
                'max_tokens': 2048
            }
            prompt = '{}\n\n{}\n\n'.format(
                PROMPT_TEMPLATE.format(num_mcqs), text
            )

            if use_chat_gpt:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **gen_params
                )
            else:
                gen_params['prompt'] = prompt
                completion = openai.Completion.create(
                model='text-davinci-003',
                    **gen_params
                )
            break
        except openai.error.RateLimitError:
            trials += 1
            print("Tried {} time(s) [rate limit]".format(trials))
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            trials += 1
            print("Tried {} time(s) [service unavailable]".format(trials))
            time.sleep(60)

    if trials == max_trials:
        sys.exit(1)
    else:
        return gen_params, completion


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loader', type=str, help="Which dataset to load")
    parser.add_argument('-d', '--dataset', type=str, help="Dataset path")
    parser.add_argument('-p', '--prompt', type=str, help="Prompt (optional)")
    parser.add_argument('-o', '--output', type=str, help="Suffix for the output file")
    parser.add_argument('-n', '--nsamples', default=-1, type=int)
    parser.add_argument('--chat', action='store_true')
    args = parser.parse_args()

    dotenv.load_dotenv()
    openai.api_key = os.getenv('SECRET_KEY')

    if args.prompt:
        print("Prompt: {}".format(args.prompt))
        for i in range(args.nsamples):
            print(generate_mcq(args.prompt))
    else:
        # Dataset
        if args.loader == 'swequad-mc':
            ds = SweQuadMCDataset(args.dataset.split(","))
        elif args.loader == 'plugga':
            ds = TextualDatasetFromJsonLines(args.dataset.split(","))
        else:
            ds = TextualDatasetFromYAML(args.dataset.split(","))
        
        ts = int(time.time())
        model = 'chatgpt' if args.chat else 'gpt3'
        if args.output:
            out_fname = "{}_{}_{}.jsonl".format(model, ts, args.output)
        else:
            out_fname = "{}_{}.jsonl".format(model, ts)

        with jsl.open(out_fname, 'w') as writer:
            for record in tqdm(ds):
                text = record['context']
                if args.nsamples > 0:
                    num_mcqs = args.nsamples
                else:
                    num_mcqs = math.ceil(len(text) / (12.78 * 4.81))
                gen_params, completion = generate_mcq(text, num_mcqs, use_chat_gpt=args.chat)
                writer.write({
                    'text': text,
                    'requested_q': num_mcqs,
                    'params': gen_params,
                    'res': completion
                })
