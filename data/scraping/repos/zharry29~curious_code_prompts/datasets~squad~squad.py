import argparse
import openai
from datasets import load_dataset
import random
random.seed(29)
from promptsource.templates import DatasetTemplates
import time
from transformers import AutoTokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', required=True, type=str, help='Either text or code.')
parser.add_argument('--model', required=True, type=str, help='Either davinci, curie or codex.')
parser.add_argument('--max_prompt', type=int, default=4000, help='Maximum number of tokens in the prompt.')
parser.add_argument('--key', required=True, type=str, help='The name of the OpenAI API key file.')

args = parser.parse_args()
openai.api_key = open(f'../../_private/{args.key}.key').read()

SELECTED_PROMPT_NAME = "Questions with Context +unanswerable"
MAX_RESPONSE_TOKENS = 300

template = DatasetTemplates("squad_v2")[SELECTED_PROMPT_NAME]
dataset = load_dataset("squad_v2", cache_dir="/nlp/data/huggingface_cache")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def apply_code_template(example):
    with open(args.prompt + '.py') as f:
        template = f.read()
    answer = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "unanswerable"
    template = template.replace("{context}", example["context"]).replace("{question}", example["question"]).replace("{answer}", answer)
    return template.split("<\split>")

def predict():
    # Build prompt
    def build_text_prompt(model, input_text):
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
            new_prompt = text_prompt + input_text + ' ' + output_text + '\n\n'
            if len(tokenizer(new_prompt)['input_ids']) > max_len:
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
            if len(tokenizer(new_prompt)['input_ids']) > max_len - 20:
                break
            code_prompt = new_prompt
        return code_prompt, sampled_indices

    def run_llm(prompt, model, temperature=0, stop=['\n']):
        model_name = {
            "davinciOld": "davinci",
            "davinci": "text-davinci-002",
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
                    temperature=temperature,
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
            prompt, indices = build_code_prompt(args.model, input_text)
            pred = run_llm(prompt + input_text, args.model)
        
        gold = example["answers"]["text"]
        pred = normalize_text(pred)
        preds.append(pred if pred != 'unanswerable' else '')
        golds.append(gold if len(gold) > 0 else [''])
        full_indices.append(indices)

    with open(f'pred-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in preds])
    with open(f'gold-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in golds])
    with open(f'indices-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'w') as f:
        f.writelines([str(x) + '\n' for x in full_indices])

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate():
    with open(f'pred-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'r') as f:
        preds = [x.strip() for x in f.readlines()]
    with open(f'gold-{args.model}-{args.max_prompt}-{args.prompt}.txt', 'r') as f:
        golds = [eval(x.strip()) for x in f.readlines()]
    em_scores = []
    f1_scores = []
    for pred, gold in zip(preds,golds):
        em = max((compute_exact_match(pred, answer)) for answer in gold)
        f1 = max((compute_f1(pred, answer)) for answer in gold)
        em_scores.append(em)
        f1_scores.append(f1)
    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores) #TODO: Double check that this is the right way to report this
    print("EM", em)
    print("F1", f1)
    return em, f1

if __name__ == "__main__":
    predict()
    evaluate()