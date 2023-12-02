import argparse
import openai
from datasets import load_dataset
import random
random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from transformers import AutoTokenizer
from tqdm import tqdm
from rouge import FilesRouge

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
parser.add_argument('--model', required=True, type=str, help='Either davinci, curie or codex.')
parser.add_argument('--max_prompt', type=int, default=4000, help='Maximum number of tokens in the prompt.')
parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

SELECTED_PROMPT_NAME = "DOC_tldr"
MAX_RESPONSE_TOKENS = 50
rouge = FilesRouge()

template = DatasetTemplates("xsum")[SELECTED_PROMPT_NAME]
dataset = load_dataset("xsum", cache_dir="/nlp/data/huggingface_cache")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(args.max_prompt)

def apply_code_template(example):
    with open(args.prompt + '.py') as f:
        template = f.read()
    input = example["document"].replace('\n','')
    template = template.replace("{article}", input).replace("{highlights}", example["summary"])
    return template.split("<\split>")

def predict():
    # Build prompt
    def build_text_prompt(input_text):
        len_input = len(tokenizer(input_text)['input_ids'])
        text_prompt = ""
        max_len = args.max_prompt - MAX_RESPONSE_TOKENS - len_input
        sampled_indices = []
        while True:
            index = random.choice(range(len(dataset['train'])))
            while index in sampled_indices:
                index = random.choice(range(len(dataset['train'])))
            sampled_indices.append(index)
            example = dataset['train'][index]
            input_text, output_text = template.apply(example)
            new_prompt = text_prompt + input_text + ' ' + output_text + '\n\n\n'
            if len(tokenizer(new_prompt)['input_ids']) > max_len - 50:
                break
            text_prompt = new_prompt
        return text_prompt, sampled_indices

    def build_code_prompt(model, input_text):
        len_input = len(tokenizer(input_text)['input_ids'])
        code_prompt = ""
        max_len = args.max_prompt - MAX_RESPONSE_TOKENS - len_input
        sampled_indices = []
        while True:
            index = random.choice(range(len(dataset['train'])))
            while index in sampled_indices:
                index = random.choice(range(len(dataset['train'])))
            sampled_indices.append(index)
            example = dataset['train'][index]
            input_text, output_text = apply_code_template(example)
            new_prompt = code_prompt + input_text + output_text + '\n\n'
            if len(tokenizer(new_prompt)['input_ids']) > max_len - 50:
                break
            code_prompt = new_prompt
        return code_prompt, sampled_indices

    def run_llm(prompt, model, stop=['\n\n\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "davinciOld": "davinci",
            "davinci3": "text-davinci-003",
            "curie": "text-curie-001",
            "ada": "text-ada-001",
            "codex": "code-davinci-002",
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=MAX_RESPONSE_TOKENS,
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
    full_indices = []
    f = open("sampled_1000_indices.txt", "r")
    example_indices = [int(s.strip()) for s in f.readlines()]
    for index in tqdm(example_indices):
        example = dataset['validation'][index]
        if args.prompt == "text":
            input_text, _ = template.apply(example)
            prompt, indices = build_text_prompt(args.model, input_text)
            pred = run_llm(prompt + input_text, args.model)
        elif args.prompt.startswith("code"):
            input_text, _ = apply_code_template(example)
            if len(tokenizer(input_text)['input_ids']) > args.max_prompt - MAX_RESPONSE_TOKENS:
                preds.append("")
                golds.append(example["summary"])
                full_indices.append("")
                continue
            prompt, indices = build_code_prompt(args.model, input_text)
            pred = run_llm(prompt + input_text, args.model)

        gold = example["summary"]
        preds.append(pred)
        golds.append(gold)
        full_indices.append(indices)

    with open(f'pred-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([x.strip('\"').replace('\n', ' ') + '\n' for x in preds])
    with open(f'gold-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([x.replace('\n', ' ') + '\n' for x in golds])
    with open(f'indices-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in full_indices])

def evaluate():
    scores = rouge.get_scores(f'pred-{args.model}-{args.max_prompt}-{args.prompt}.txt',f'gold-{args.model}-{args.max_prompt}-{args.prompt}.txt',avg=True) #TODO: Double check that this is correct/makes sense
    print("Rouge Score", scores)
    return "Rouge", scores

if __name__ == "__main__":
    predict()
    evaluate()