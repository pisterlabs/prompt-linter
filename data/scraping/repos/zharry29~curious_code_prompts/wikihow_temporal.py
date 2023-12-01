import argparse
import openai
from datasets import load_dataset
import random
random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from sklearn.metrics import accuracy_score
import csv
import pickle
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
import backoff

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', default='text', type=str, help='Either text or code.')
parser.add_argument('--model', default='codex', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--max_prompt', type=int, default=4000, help='Maximum number of tokens in the prompt.')
parser.add_argument('--key', default='harry', type=str, help='The name of the OpenAI API key file.')
parser.add_argument('--seed', default='', type=str, help='Random seed.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()
if args.seed:
    args.seed = '_' + args.seed
    random.seed(int(args.seed[1:]))

def load_data():
    dataset = {'train': [], 'test': []}
    for split in ['train', 'test']:
        with open(f'order_{split}.csv', encoding = "ISO-8859-1") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset[split].append(row)
    return dataset

dataset = load_data()

def apply_text_template(example):
    input_prompt = f"You are trying to {example['goal'].lower()}. You need to do two things:\n(a) {example['step0'].strip('.')}\n(b) {example['step1'].strip('.')}\nThe first thing to do is"
    output_prompt = ['(a) ' + example['step0'].strip('.'), '(b) ' + example['step1'].strip('.')][int(example["label"])]
    return input_prompt, output_prompt

def apply_code_template(example):
    with open(args.prompt + '.py') as f:
        template = f.read()
    ret = []
    for t in template.split('$'):
        ret.append(t.replace("{step0}", example["step0"]).replace("{step1}", example["step1"]).replace("{goal}", example["goal"]).replace("{label_first}", str(example["label"])).replace("{label_after}", str(1-int(example["label"]))))
    return ret
if args.prompt == "text":
    apply_template = apply_text_template
elif args.prompt.startswith("code"):
    apply_template = apply_code_template
    
def predict():
    # Build prompt
    def build_text_prompt(example):
        inference_input_text = apply_template(example)[0]
        text_prompt = ""
        prev_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), 100)
        for example_index in example_indices:
            if len(tokenizer(text_prompt + inference_input_text)['input_ids']) > args.max_prompt - 20:
                break
            example = dataset['train'][example_index]
            input_text, output_text = apply_template(example)
            prev_prompt = text_prompt
            text_prompt += input_text + ' ' + output_text + '\n\n'

        return(prev_prompt + inference_input_text)

    def build_code_prompt(example):
        inference_input_text = apply_template(example)[0]
        text_prompt = ""
        prev_prompt = ""
        example_indices = random.sample(range(len(dataset['train'])), 100)
        for example_index in example_indices:
            if len(tokenizer(text_prompt + inference_input_text)['input_ids']) > args.max_prompt:
                break
            example = dataset['train'][example_index]
            input_text, output_text = apply_template(example)
            prev_prompt = text_prompt
            text_prompt += input_text + output_text + '\n\n\n'
        return(prev_prompt + inference_input_text)

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def run_llm(prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "davinci003": "text-davinci-003",
            "old_davinci": "davinci",
            "curie": "text-curie-001",
            "ada": "text-ada-001",
            "codex": "code-davinci-002",
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=20,
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

    preds = []
    golds = []
    print("Total examples: ", len(dataset['test']))
    count = 0
    with open("sampled_1000_indices.pkl", "rb") as f:
        indices = pickle.load(f)
    for index in indices:
        example = dataset['test'][index]
        count += 1
        print(count)
        if args.prompt == "text":
            prompt = build_text_prompt(example)
        elif "code" in args.prompt:
            prompt = build_code_prompt(example)
        pred_text = run_llm(prompt, args.model)
        #print(prompt)
        #print(pred_text)
        #print(len(tokenizer(prompt)['input_ids']))
        if args.prompt == "text":
            if '(b)' in pred_text:
                pred = 1
            else:
                pred = 0
        else:
            if 'input1, input0' in pred_text:
                pred = 1
            else:
                pred = 0
        #print(pred)
        #raise SystemExit
        gold = int(example['label'])
        preds.append(pred)
        golds.append(gold)

    with open(f'pred_{args.model}_{args.prompt}_{args.max_prompt}{args.seed}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open('gold.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])

def evaluate():
    with open(f'pred_{args.model}_{args.prompt}_{args.max_prompt}{args.seed}.txt', 'r') as f:
        preds = [x.strip() for x in f.readlines()]
    with open('gold.txt', 'r') as f:
        golds = [x.strip() for x in f.readlines()]
    print("Accuracy", accuracy_score(golds, preds))
    return "Accuracy", accuracy_score(golds, preds)

if __name__ == "__main__":
    predict()
    evaluate()