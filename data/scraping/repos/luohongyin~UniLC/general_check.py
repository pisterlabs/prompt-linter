import sys
import json
import time
import argparse

import openai

from collections import Counter
from nltk.tokenize import regexp_tokenize, sent_tokenize
from nltk.translate.bleu_score import sentence_bleu

from analysis import load_eval_data

try:
    from pyserini.search import FaissSearcher
except:
    pass

import warnings
warnings.filterwarnings("ignore")

openai.api_key = open('openai-key.txt').read().strip()

parser = argparse.ArgumentParser(
    prog='python general_check.py',
    description='Unified language safety checking with LLMs.',
    epilog='Submit issues on Github for addtional help.'
)

parser.add_argument('-t', '--task', type=str, help = 'Language safety checking task.')
parser.add_argument('-m', '--mode', type=str, help = 'Prompting mode [zero | fp | cot].')
parser.add_argument('-s', '--start-idx', type=int, help = 'Index of the first sample to process.')
parser.add_argument('-n', '--exp-name', type=int, help = 'Name tag for the experiment log file.')
parser.add_argument('-v', '--verbose', action='store_true')

def build_background(hits):
    bg_list = []
    for hit in hits:
        res_title = hit.docid
        ret_text = json.loads(searcher.doc(res_title).raw())['contents']
        ret_text = ret_text.split('\n')[1]
        bg_list.append(ret_text)
    return '\n\n'.join(bg_list)


def is_answerable(ans_str):
    ans_str = ans_str.lower()
    if 'however, ' in ans_str:
        return True

    signal_list = [
        'i\'m sorry', 'language model', 'i cannot',
        'not provide', 'provide more information'
    ]
    for signal in signal_list:
        if signal in ans_str:
            return False
    return False


def load_dataset(domain = None, split = 'dev'):

    def gather_evidence(evi_list):
        evi_title_list = []
        for evi_obj in evi_list:
            contents = evi_obj['content']
            title_list = [x.split('_')[0] for x in contents]
            evi_title_list += title_list
        return set(evi_title_list)
    
    def extract_climate_case(jsonl):
        case = json.loads(jsonl)
        evi_set = None
        return (case['claim'], case['claim_label'], evi_set)
    
    def extract_scifact_case(jsonl):
        case = json.loads(jsonl)
        evi_set = None
        return (case['claim'], case['label'], evi_set)
    
    def extract_covid_case(jsonl):
        case = json.loads(jsonl)
        evi_set = None
        label_dict = {
            'SUPPORTED': 'SUPPORTS',
            'REFUTED': 'REFUTES'
        }
        return (case['claim'], label_dict[case['label']], evi_set)
    
    fn = f'ulsc_data/{args.task}_claims.jsonl'
    
    if domain == 'climate':
        extract_case_func = extract_climate_case
    
    if domain == 'hsd':
        extract_case_func = extract_scifact_case
    
    if domain == 'health':
        extract_case_func = extract_scifact_case
    
    if domain == 'sbic':
        extract_case_func = extract_scifact_case
    
    data_list = open(fn).readlines()
    dataset = [extract_case_func(x) for x in data_list]
    dataset = [x for x in dataset if x[1] == 'SUPPORTS' or x[1] == 'REFUTES']
    
    print(f'Dataset size = {len(dataset)}')
    return dataset


def gen_response_chat(
        prompt, t, max_tok, n, system_info = '',
    ):

    complete = False
    while not complete:
        try:
            response = openai.ChatCompletion.create(
                model = 'gpt-3.5-turbo',
                messages = [
                    {
                        'role': 'system',
                        'content': system_info
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                temperature = t,
                max_tokens = max_tok,
                n = n
            )

            ans_list = [
                cand['message']['content'].strip() for cand in response['choices'] # \
                    # if condition_str in cand['message']['content']
            ]
            if n == 1:
                ans_txt = ans_list[0]
            else:
                ans_txt = ans_list
                # ans_txt, confidence = select_func(ans_list, keywords = keywords)
            complete = True
        except:
            print('Chat error')
            # print(prompt)
            # abort()
            time.sleep(2)
    return ans_txt


def think_twice(claim, verify_prompt, searcher, search_mode):
    prompt = f'{verify_prompt}\n\nSomeone said: {claim}\nQuestion: Is it fair to say that?\nAnswer:'

    system_info = 'Find the most suspicious information of the statement and check scientific or social fact as the examples.'
    
    qa_str = gen_response_chat(
        prompt, t = 0.1, max_tok = 128, n = 1,
        system_info = system_info
    ).strip()

    return qa_str


def verify_dataset(searcher, dataset, verify_prompt, args):

    def query_verify(verify_str):
        verify_str = verify_str.lower()
        return not ('no.' in verify_str or 'no,' in verify_str)
    
    def query_verify_neg(verify_str):
        verify_str = verify_str.lower()
        return not ('yes.' in verify_str or 'yes,' in verify_str)

    def cot_verify(verify_str):
        return 'accept' in verify_str.split('\n')[-1]

    num_case = len(dataset)
    crr = 0.

    log_list = []
    wrong_list = []

    if args.mode == 'cot':
        ent_prompt = open(f'general_prompts/ent_cot.txt').read()

    verify_str_list = []

    for i, case in enumerate(dataset):
        claim, label, _ = case
        
        if args.mode == 'zero':
            prompt = f'Someone said: {claim}\nQuestion: Is it fair to say that?\nAnswer:'
            ans_prompt = None
            qa_str = None
            ans_str = None
            system_info = 'Analysis if the claim is factual and fair.'
            verify_func = query_verify
        
        elif args.mode == 'cot':
            prompt = f'Someone said: {claim}\nQuestion: Is it fair to say that?\nAnswer:'
            ans_prompt = None
            qa_str = None
            ans_str = None
            system_info = 'Let\'s think step by step and answer the question as the examples according to scientific or social facts.'

            verify_func = query_verify
        
        elif args.mode == 'fp':
            try:
                qa_str = think_twice(
                    claim, verify_prompt, searcher, args.mode
                )
            except:
                print(f'Error occured at sample {i}\n')
                sys.exit()

            prompt = f"Someone said: {claim}\n{qa_str}\nYes/no question: Is it fair to say that?"
            
            system_info = 'Analyze if the statement is fair according to the fact and context. If the claim is unfactual or biased, the answer is no; if the claim is factual and respectful, the answer is yes.'
            verify_func = query_verify
        
        if args.mode == 'cot':
            prompt = f'{ent_prompt}\n\n{prompt}'
        
        verify_str = gen_response_chat(
            prompt, t = 0.1, max_tok = 128, n = 1, system_info = system_info
        )

        if args.mode == 'fp':
            verify_str_list.append(qa_str)
        else:
            verify_str_list.append(verify_str)

        if verify_func(verify_str):
            pred = 'SUPPORTS'
        else:
            pred = 'REFUTES'

        if args.verbose:
            print(claim)
            print(label)
            print('--')
            print(qa_str)
            print('--')
            print(verify_str)
            print(f'pred = {pred}')
            print('--')
            print(prompt)
            sys.exit()
        
        if pred == label:
            crr += 1.
        
        else:
            wrong_list.append(i)
        
        case_log = [
            f'---- {i} / {num_case} -- crr = {crr}, acc = {crr / (i + 1)}',
            f'---- Claim: {claim}',
            f'---- Label: {label}',
            f'---- QA_str: {qa_str}',
            f'---- Verifi_str: {verify_str}',
            f'---- Prediction: {pred}',
            '----------------------------------------------------\n'
        ]
        
        log_list.append('\n'.join(case_log))

        if i % 100 == 0:
            print(f'Processed {i + 1} claims.')
    
    open(f'log/{args.task}_{args.mode}_check_{args.exp_name}.log', 'w').write(
        '\n'.join(log_list)
    )
    json.dump(wrong_list, open(f'log/{args.task}_wrong_list_joint.json', 'w'))
    json.dump(verify_str_list, open(f'log/{args.task}_{args.mode}_verify_list_{args.exp_name}.json', 'w'))
    
    return crr / num_case


if __name__ == '__main__':
    # args.task: climate | hsd
    # args.verbose: true | false
    # args.mode: zero | fp | cot
    # args.start_idx: int >= 0
    # args.exp_name: example = "joint"

    args = parser.parse_args()

    if args.mode == 'search':
        searcher = FaissSearcher.from_prebuilt_index(
            'wikipedia-dpr-multi-bf',
            'facebook/dpr-question_encoder-multiset-base'
        )
    else:
        searcher = None

    dataset = load_dataset(args.task)[args.start_idx:]
    verify_prompt = open('general_prompts/verify_prompts.txt').read()
    
    acc = verify_dataset(
        searcher, dataset, verify_prompt, args
    )
    f1 = load_eval_data(task = args.task, mode = args.mode, exp_name = args.exp_name)
    print(f'\nAcc = {acc}\nF1 = {f1}\n')
