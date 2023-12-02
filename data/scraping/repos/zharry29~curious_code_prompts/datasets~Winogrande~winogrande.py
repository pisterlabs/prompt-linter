import os 
import argparse
import ast
import pickle
import random
import time

import numpy as np
import openai
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import utils


class Winogrande():
    def __init__(self, templates):
        self.apply_template = templates

    def build_text_prompt(self, max_len):
        print('Building prompts...')
        text_prompt = ""
        total_token = max_len
        threshold = args.context_size 
        tolerance = 0
        counter = 0
        while total_token < threshold:
            example_index = random.sample(range(len(dataset['train'])), 1)
            example = dataset['train'][example_index]
            input_text, output_text = self.apply_template(example)
            candidate_prompt = input_text.replace('[', '').replace(']', '') + '\n\nAnswer: ' + output_text.replace('[', '').replace(']', '') + '\n\n\n'
            token_count = utils.gpt3_tokenizer(candidate_prompt)
            prev_total = total_token
            if total_token + token_count < threshold:
                text_prompt += candidate_prompt
                total_token += token_count
                counter += 1
            if  total_token - prev_total < 10:
                tolerance += 1
                if tolerance > 1:
                    break
        print(f'Total samples in prompt: {counter}')
        print(f'Average tokens per sample: {total_token / counter}')
        return text_prompt

    def build_code_prompt(self, max_len, prompt):
        print('Building prompts...')
        if prompt:
            code_prompt = prompt
            total_token = utils.gpt3_tokenizer(prompt)
        else:
            code_prompt = ""
            total_token = max_len
        threshold = args.context_size 
        tolerance = 0
        while total_token < threshold:
            example_index = random.sample(range(len(dataset['train'])), 1)
            example = dataset['train'][example_index]
            input_text, output_text = self.apply_template(example)
            candidate_prompt = input_text + output_text + '\n\n'
            token_count = utils.gpt3_tokenizer(candidate_prompt)
            prev_total = total_token
            if total_token + token_count < threshold:
                code_prompt += candidate_prompt
                total_token += token_count
            if  total_token - prev_total < 10:
                tolerance += 1
                if tolerance > 1:
                    break
        return code_prompt
        
    def parse_input(self, inp):
        return '- ' + "'" + inp.replace('-', '').strip() + "'"

    def run_llm(self, prompt, model, max_tokens, temperature=0.7, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "davinci-old": "davinci",
            "davinci003": "text-davinci-003",
            "curie": "text-curie-001",
            "codex": "code-davinci-002",
            "ada": "text-ada-001",
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop
                )
                break
            except Exception as e:
                print(e)
                print("Retrying in 10 seconds")
                time.sleep(10)

        gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
        return gen_text

    def predict(self, index=None):
        val_data = dataset['validation']
        if not index:
            val_idx = np.random.choice(np.arange(len(val_data['answer'])), 1000, replace=False)
            with open(f'./indices/{args.model}_val_idx.pkl', 'wb') as f:
                pickle.dump(val_idx, f)
            f.close()
        else:
            val_idx = pickle.load(open(f'./indices/{args.index}.pkl', 'rb'))

        max_len = compute_longest_prompt(val_idx, val_data, self.apply_template)

        if args.style == 'comment':
            prompt = "'''\nThis is a coference resolution task. There will be a '_' in a given sentence and options will be provided. You need to choose from given options and fill in the '_'.\n'''\n\n"
        elif args.style == 'class':
            with open(f'./code-prompts/class_header.py') as f:
                prompt = f.read()
            f.close()
        else:
            prompt = None
        
        if args.prompt == "text":
            prompt = self.build_text_prompt(max_len)
        elif args.prompt == "code":
            prompt = self.build_code_prompt(max_len, prompt)

        preds = []
        golds = []
        for idx in tqdm(val_idx):
            example = val_data[int(idx)]
            input_text, output_text = self.apply_template(example)
            if args.prompt == 'text':
                options = '[' + ','.join(['"' + e.replace('-', '').lower().strip() + '"' for e in input_text.split('\n')[-2:]]) + ']'
            elif args.style == 'class':
                options = input_text.split('\n')[-3].split('=')[-1].strip()
            else:
                options = input_text.split('\n')[-2].split('=')[-1].strip()
            
            options = ast.literal_eval(options)
            options = [e.strip().lower() for e in options]

            if args.prompt == 'text':
                input_text = input_text.split('\n')
                input_text[-2], input_text[-1] = self.parse_input(input_text[-2]), self.parse_input(input_text[-1])
                input_text = '\n'.join(input_text)

            if args.prompt == "text": 
                pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model, args.completion_size)
            else: 
                pred = self.run_llm(prompt + input_text, args.model, args.completion_size)
            pred = pred.replace("'", '').replace('"', '').lower().strip()
            if pred in options:
                pred = str(options.index(pred) + 1)
            else:
                pred = str(-1)
            
            gold = example['answer']
            preds.append(pred)
            golds.append(gold)

        pred_name, gold_name = get_fname() 
        
        with open(f'./result/{pred_name}.txt', 'w') as f:
            f.writelines([x + '\n' for x in preds])
        with open(f'./result/{gold_name}.txt', 'w') as f:
            f.writelines([x + '\n' for x in golds])

    def evaluate(self):

        pred_name, gold_name = get_fname() 

        with open(f'./result/{pred_name}.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open(f'./result/{gold_name}.txt', 'r') as f:
            golds = [x.strip() for x in f.readlines()]
        print("Accuracy", accuracy_score(golds, preds))
        print("F1 Score", f1_score(golds, preds, average='macro'))
        return accuracy_score(golds, preds)


def compute_longest_prompt(val_idx, val_data, apply_template):
    max_len = 0
    for idx in tqdm(val_idx):
        example = val_data[int(idx)]
        input_text, output_text = apply_template(example)
        cur_len = utils.gpt3_tokenizer(input_text + '\n\nAnswer:')
        if cur_len > max_len:
            max_len = cur_len
    return max_len


def apply_code_template(example):
    with open(f'./code-prompts/{args.style}' + '.py') as f:
        template = f.read()
    ret = []
    options = example['option1'] + example['option2']
    for t in template.split('$'):
        if type(example['sentence']) is list:
            ret.append(t.replace("{sentence}", example["sentence"][0]).replace("{option1}", example["option1"][0]).replace("{option2}", example["option2"][0]).replace("{answer}", options[int(example["answer"][0])-1]))
        else:
            ret.append(t.replace("{sentence}", example["sentence"]).replace("{option1}", example["option1"]).replace("{option2}", example["option2"]).replace("{answer}", options[int(example["answer"])-1]))
    return ret


def get_fname():
    if not args.style:
        pred_name = f'{args.prompt}_{args.model}_pred_{args.context_size}_{args.seed}'
    else:
        pred_name = f'{args.prompt}_{args.style}_{args.model}_pred_{args.context_size}'
    
    if not args.style:
        gold_name = f'{args.prompt}_{args.model}_gold_{args.context_size}_{args.seed}'
    else:
        gold_name = f'{args.prompt}_{args.style}_{args.model}_gold_{args.context_size}'
    
    return pred_name, gold_name


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='text',help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--style', type=str, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
parser.add_argument('--context_size', type=int, help='token threshold for GPT3 context prompt.')
parser.add_argument('--completion_size', type=int, help='token threshold for GPT3 completion.')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--index', type=str, help='file name of the saved indices')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    data_name = 'winogrande'
    dataset, templates = utils.load_data(data_name)

    if args.prompt == "text":
        apply_template = templates.apply
    elif args.prompt == "code":
        apply_template = apply_code_template

    inference_model = Winogrande(apply_template)

    inference_model.predict()
    inference_model.evaluate()
