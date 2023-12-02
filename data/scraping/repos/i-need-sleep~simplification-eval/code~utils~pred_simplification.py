import os
import time

import openai
import pandas as pd

import utils.keychain as keychain
import utils.globals as uglobals

openai.api_key = keychain.OPENAI_API_KEY

def gptturbo_inference(dataset, out_dir=uglobals.SIM_OPENWEBTEXT_DIR, save_interval=500):
    if not os.path.isdir(uglobals.SIM_OPENWEBTEXT_DIR):
        os.mkdir(uglobals.PROCESSED_DIR)
        os.mkdir(uglobals.SIM_OPENWEBTEXT_DIR)

    out_path = f'{out_dir}/gpt_turbo.csv'

    out = {
        'src': [],
        'pred': []
    }

    # Continue from the output file
    if os.path.isfile(out_path):
        data = pd.read_csv(out_path)
        out['src'] = list(data['src'])
        out['pred'] = list(data['pred'])
        print(f'Continuing from {out_path}')

    for line_idx, line in enumerate(dataset):
        if line_idx < len(out['pred']):
            continue

        src = line['src']
        prompt = line['prompt']

        pause_idx = 0 # Try for a maximum of 5 times. Skip the sequence if failed.
        while True:
            try:
                c = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=prompt,
                    max_tokens=1024,
                    temperature=0,
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                pause_idx += 1
                if pause_idx == 5:
                    break
                continue

        if pause_idx == 5:
            continue
        pred = c['choices'][0]['message']['content']


        out['src'].append(src)
        out['pred'].append(pred)

        if line_idx % save_interval == 0:
            df = pd.DataFrame(out)
            df.to_csv(out_path)

def gpt3_inference(engine, dataset, out_dir=uglobals.SIM_OPENWEBTEXT_DIR, save_interval=100):
    if not os.path.isdir(uglobals.SIM_OPENWEBTEXT_DIR):
        os.mkdir(uglobals.PROCESSED_DIR)
        os.mkdir(uglobals.SIM_OPENWEBTEXT_DIR)
    out_path = f'{out_dir}/gpt_{engine}.csv'

    out = {
        'src': [],
        'pred': []
    }

    # Continue from the output file
    if os.path.isfile(out_path):
        data = pd.read_csv(out_path)
        out['src'] = list(data['src'])
        out['pred'] = list(data['pred'])
        print(f'Continuing from {out_path}')

    for line_idx, line in enumerate(dataset):
        if line_idx < len(out['pred']):
            continue

        src = line['src']
        prompt = line['prompt'][1]['content']

        pause_idx = 0
        while True:
            try:
                c = openai.Completion.create(
                    model=engine,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.8,
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                pause_idx += 1
                if pause_idx == 5:
                    break
                continue
        
        if pause_idx == 5:
            continue

        pred = c['choices'][0]['text'].split('\n')[0]


        out['src'].append(src)
        out['pred'].append(pred)

        if line_idx % save_interval == 0:
            df = pd.DataFrame(out)
            df.to_csv(out_path)