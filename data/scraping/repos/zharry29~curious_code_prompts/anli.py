import time
import json
import utils
import pickle
import random
import openai
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score

random.seed(29)


class ANLI():
    def __init__(self, apply_template, idx):
        self.apply_template = apply_template
        self.idx = idx

    def build_text_prompt(self, max_len):
        print('Building prompts...')
        text_prompt = ""
        total_token = max_len
        threshold = args.context_size 
        tolerance = 0
        counter = 0
        while total_token < threshold:
            example_index = random.sample(range(len(dataset[f'train_r{self.idx}'])), 1)[0]
            example = dataset[f'train_r{self.idx}'][example_index]
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
        if prompt:
            code_prompt = prompt
            total_token = utils.gpt3_tokenizer(prompt)
        else:
            code_prompt = ""
            total_token = max_len
        threshold = args.context_size 
        tolerance = 0
        while total_token < threshold:
            example_index = random.sample(range(len(dataset[f'train_r{self.idx}'])), 1)
            example = dataset[f'train_r{self.idx}'][example_index]
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

    def run_llm(self, prompt, model, max_tokens, temperature=0, stop=['\n']):
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
    
    def parse_pred(self, pred):
        if pred.strip().lower() == 'true':
            return 0
        elif pred.strip().lower() == 'neither':
            return 1
        elif pred.strip().lower() == 'false':
            return 2
        else:
            return -1
    
    def predict(self, val_idx=None):
        val_data = dataset[f'dev_r{self.idx}']
        if not val_idx:
            val_idx = np.random.choice(np.arange(len(val_data)), 333, replace=False)

        max_len = compute_longest_prompt(val_idx, val_data, self.apply_template)

        if args.prompt == "text":
            prompt = self.build_text_prompt(max_len)
        elif args.prompt == "code":
            if args.style == 'comment':
                prompt = open('./code-prompts/comment_prefix.py').read()
                prompt = self.build_code_prompt(max_len, prompt)
            elif args.style == 'class':
                prompt = open('./code-prompts/class_prefix.py').read()
                prompt = self.build_code_prompt(max_len, prompt)
            else:
                prompt = None
                prompt = self.build_code_prompt(max_len, prompt)
        
        preds = []
        golds = []
        for idx in tqdm(val_idx):
            example = val_data[int(idx)]
            input_text, output_text = self.apply_template(example)

            if args.prompt == 'text':
                pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model, args.completion_size)
            elif args.prompt == 'code':
                pred = self.run_llm(prompt + input_text, args.model, args.completion_size)
            
            pred = self.parse_pred(pred.replace('"', ''))
            gold = example['label']
            preds.append(pred)
            golds.append(gold)
        return list(val_idx), preds, golds
    
    def evaluate(self):
        pred_name, gold_name = get_fname()
        with open(f'./result/{pred_name}.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open(f'./result/{gold_name}.txt', 'r') as f:
            golds = [x.strip() for x in f.readlines()]
        print("Accuracy", accuracy_score(golds, preds))
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
    label_to_text = {'0': "True", '1': "Neither", '2': "False"}
    premise = example['premise']
    hypothesis = example['hypothesis']
    label_idx = example['label']

    if type(premise) is list:
        premise = premise[0]

    if type(hypothesis) is list:
        hypothesis = hypothesis[0]

    if type(label_idx) is list:
        label_idx = label_idx[0]
    label_text = label_to_text[str(label_idx)]
    for t in template.split('$'):
        if hypothesis.strip()[-1] != '.':
            hypothesis += '.'
        ret.append(t.replace("{premise}", premise).replace("{hypothesis}", hypothesis).replace("{label}", label_text))
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
parser.add_argument('--prompt', type=str, help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--context_size', type=int, help='token threshold for GPT3 context prompt.')
parser.add_argument('--completion_size', type=int, help='token threshold for GPT3 completion.')
parser.add_argument('--style', type=str, default=None, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
parser.add_argument('--seed', type=int, default=None, help='set seed')
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
    
    data_name = 'anli'
    dataset, templates = utils.load_data(data_name)

    if args.prompt == "text":
        apply_template = templates.apply
    elif args.prompt == "code":
        apply_template = apply_code_template

    if not args.index:
        val_idx_dict = []
    else:
        val_idx_dict = pickle.load(open(f'./indices/{args.index}.pkl', 'rb'))

    preds, golds = [], []
    for i in range(1, 4):
        inference_model = ANLI(apply_template, i)
        if args.index:
            val_idx = val_idx_dict[i-1]
        else:
            val_idx = None
        val_idx, pred, gold = inference_model.predict(val_idx)
        if not args.index:
            val_idx_dict.append(val_idx)
        preds += pred
        golds += gold

    pred_name, gold_name = get_fname()
    
    with open(f'./result/{pred_name}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open(f'./result/{gold_name}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])

    with open(f'./indices/{args.model}_val_idx_{args.context_size}.pkl', 'wb') as f:
        pickle.dump(val_idx_dict, f)
    f.close()

    for i in range(1, 4):
        inference_model = ANLI(templates, i)
        inference_model.evaluate()










