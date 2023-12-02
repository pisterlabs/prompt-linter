import time
import json
import utils
import random
import pickle
import openai
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score


class HotpotQA():
    def __init__(self, apply_template):
        self.apply_template = apply_template
    
    def build_text_prompt(self, input_text):
        text_prompt = ""
        total_token = utils.gpt3_tokenizer(input_text)
        threshold = args.context_size
        counter = 0
        while total_token < threshold:
            example_index = random.sample(range(len(dataset['train'])), 1)[0]
            example = dataset['train'][example_index]
            input_text, output_text = self.apply_template(example)
            candidate_prompt = input_text.replace('[', '').replace(']', '') + '\n\nAnswer: ' + output_text.replace('[', '').replace(']', '') + '\n\n\n'
            token_count = utils.gpt3_tokenizer(candidate_prompt)
            if total_token + token_count < threshold:
                text_prompt += candidate_prompt
                total_token += token_count
                counter += 1
            else:
                if text_prompt:
                    break
        print(f'Total samples in prompt: {counter}')
        print(f'Average tokens per sample: {total_token / counter}')
        return text_prompt
    
    def build_code_prompt(self, input_text, prompt=None):
        if prompt:
            text_prompt = prompt
        else:
            text_prompt = ""
        total_token = utils.gpt3_tokenizer(input_text)
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
                text_prompt += candidate_prompt
                total_token += token_count
            if  total_token - prev_total < 10:
                tolerance += 1
                if tolerance > 1:
                    break
        return text_prompt
    
    def run_llm(self, prompt, model, max_tokens, temperature=0.7, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
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
        
        preds = {}
        golds = []
        preds['answer'] = {}
        for i, idx in enumerate(tqdm(val_idx)):
            example = val_data[int(idx)]
            input_text, output_text = self.apply_template(example)

            if args.style == 'comment':
                prompt = open('./code-prompts/comment_prefix.py').read()
            elif args.style == 'class':
                prompt = open('./code-prompts/class_prefix.py').read()
            else:
                prompt = None

            if args.prompt == "text":
                prompt = self.build_text_prompt(input_text)
            elif args.prompt == "code":
                prompt = self.build_code_prompt(input_text, prompt)
            
            if args.prompt == 'text':
                pred = self.run_llm(prompt + input_text + '\n\nAnswer:', args.model, args.completion_size)
            else:
                pred = self.run_llm(prompt + input_text, args.model, args.completion_size)
            
            gold = example['answer']

            preds['answer'][f'seacow-{i}'] = pred
            golds.append({'_id': f'seacow-{i}', 'answer': gold})
        
        pred_name, gold_name = get_fname()
        
        with open(f'./result/{pred_name}.json', 'w') as f:
            json.dump(preds, f, indent=4)
        f.close()
        with open(f'./result/{gold_name}.pkl', 'wb') as f:
            pickle.dump(golds, f)
        f.close()
    
    def evaluate():
        with open('pred.txt', 'r') as f:
            preds = [x.strip() for x in f.readlines()]
        with open('gold.txt', 'r') as f:
            golds = [x.strip() for x in f.readlines()]
        print("Accuracy", accuracy_score(golds, preds))
        return accuracy_score(golds, preds)


def apply_code_template(example):
    with open(f'./code-prompts/{args.style}' + '.py') as f:
        template = f.read()
    ret = []
    question = example['question']
    answer = example['answer']

    if type(question) is list:
        question = question[0]
    if type(answer) is list:
        answer = answer[0]

    try:
        supporting_facts = example['context']['sentences']
    except:
        supporting_facts = example['context'][0]['sentences']
    supporting_facts = [[s.replace('"', "'") for s in supporting_facts[i]] for i in range(len(supporting_facts))]

    if args.style == 'vanilla':
        supporting_facts_vars = [f'input{str(i+2)} = "{" ".join(supporting_facts[i])}"' for i in range(len(supporting_facts))]
    elif args.style in ['good_varname', 'comment']:
        supporting_facts_vars = [f'supporting_fact{str(i+1)} = "{" ".join(supporting_facts[i])}"' for i in range(len(supporting_facts))]
    elif args.style == 'class':
        supporting_facts_vars = [f'\t"{" ".join(supporting_facts[i])}"' for i in range(len(supporting_facts))]

    supporting_facts = '\n'.join(supporting_facts_vars)
    for t in template.split('$'):
        ret.append(t.replace('{question}', question.replace('"', "'")).replace('{supporting-documents}', supporting_facts).replace('{answer}', answer))
    
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
parser.add_argument('--context_size', type=int, help='Context window size of GPT3 model.')
parser.add_argument('--completion_size', type=int, help='completion (max_lens) size of GPT3 model.')
parser.add_argument('--style', type=str, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
parser.add_argument('--index', type=str, help='file name of the saved indices')
parser.add_argument('--seed', type=int, default=None, help='random seed')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


def compute_longest_prompt(val_idx, val_data, apply_template):
    max_len = 0
    for idx in tqdm(val_idx):
        example = val_data[int(idx)]
        input_text, output_text = apply_template(example)
        cur_len = utils.gpt3_tokenizer(input_text + '\n\nAnswer:')
        if cur_len > max_len:
            max_len = cur_len
    return max_len


if __name__ == '__main__':
    args = parser.parse_args()
    openai.api_key_path = f'../../_private/{args.key}.key'
    if args.seed:
        np.random.seed(args.seed)
        random.seed(args.seed)

    data_name = 'hotpot_qa'
    dataset, templates = utils.load_data(data_name)

    if args.prompt == "text":
        apply_template = templates.apply
    elif args.prompt == "code":
        apply_template = apply_code_template

    inference_model = HotpotQA(apply_template)
    inference_model.predict(args.index)
    inference_model.evaluate()










