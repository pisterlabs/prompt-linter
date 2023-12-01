import os
import json
import openai
import concurrent.futures
from tqdm import tqdm
import time
import random
from openai.error import RateLimitError
from argparse import ArgumentParser

openai.api_key = ''

parser = ArgumentParser()
parser.add_argument('--prompt_type', required=True, choices=['weak', 'full', 'example', 'group'])
parser.add_argument('--weak_file')
parser.add_argument('--lbl_file')
parser.add_argument('--test_file')
# parser.add_argument('--pseudo_file')
# parser.add_argument('--shots_file')
parser.add_argument('--example_file')
parser.add_argument('--save_to', required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=1000000)
parser.add_argument('--dataset', required=True, choices=['amazon', 'wiki', 'eurlex'])
parser.add_argument('--hint_file', required=True)
args = parser.parse_args()

def get_completion_with_retries(prompt, return_text=True, reduce_length=False, tqdm=None):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30
            )
            break
        except Exception as e:
            print(str(e))
            if "This model's maximum context length is" in str(e):
                print('reduce_length')
                return 'ERROR::reduce_length'
            # self.key_id = (self.key_id + 1) % len(self.key)
            # openai.api_key = self.key[self.key_id]
            time.sleep(10)
    if return_text:
        completion = completion['choices'][0]['message']['content']
    return completion


def main():
    lbl, text = [], []
    # results = json.load(open('../xml/AmazonCat-13K.bow/tstfidf.json'))
    with open(args.lbl_file, encoding='latin-1') as f:
        for line in f.readlines():
            lbl.append(json.loads(line))
    with open(args.test_file) as f:
        for line in f.readlines():
            text.append(json.loads(line))

    random_prompt = ''
    with open('../preprocessing/random10prompt.txt') as f:
        for line in f.readlines():
            random_prompt += line
    
    hints = []
    with open(args.hint_file) as f:
        for line in f.readlines():
            hints.append(json.loads(line))

    # cands = {}
    # with open(args.pseudo_file) as f:
    #     for line in f.readlines():
    #         data = json.loads(line)
    #         cands[data['uid']] = data['cat']
    
    if args.example_file is not None:
        examples = []
        with open(args.example_file) as f:
            for line in f.readlines():
                # print(len(json.loads(line)['output']))
                examples.append(json.loads(line))

    prompts = []
    # for i in range(len(text)):
    for i in range(args.start, args.end):
        shots_prompt = ''
        if args.prompt_type == 'group':
            labels = hints[i]['output']
            inputs = examples[i - args.start]['output']
            # print(inputs)
            if len(inputs) == 0:
                prompts.append(random_prompt)
                continue
            if isinstance(inputs[0], list):
                x = []
                for input in inputs:
                    for ii in input:
                        if isinstance(ii, list):
                            for iii in ii:
                                if isinstance(iii, list):
                                    for iiii in iii:
                                        x.append(iiii)
                                else:
                                    x.append(iii)
                        else:
                            x.append(ii)
                inputs = x
            shot_cands_dict = {}
            for index, input in enumerate(inputs[:len(labels)]):
                if isinstance(input, list):
                    print(input)
                    input = input[0]
                input = input.encode('utf-8').decode('utf-8')
                if input not in shot_cands_dict.keys():
                    shot_cands_dict[input] = []
                if labels[index] != input:
                    shot_cands_dict[input].append(labels[index])
            for shot_title, _shot_cands in shot_cands_dict.items():
                if _shot_cands is None:
                    continue
                if args.dataset == 'amazon':
                    # true
                    shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + '\n'.join(_shot_cands) + '\n'
                    # label
                    # shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + '\n'.join([x['title'] for x in random.sample(lbl, len(_shot_cands))]) + '\n'
                    # input
                    # random_word = random.choice(temp)['title']
                    # shot_prompt = "Product title on Amazon: " + random_word + "\nSimilar product: " + '\n'.join(_shot_cands) + '\n'
                if args.dataset == 'wiki':
                    shot_prompt = "Passage title on Wikipedia: " + shot_title + "\nSee also passage: " + '\n'.join(_shot_cands) + '\n'
                if args.dataset == 'eurlex':
                    shot_prompt = "EU legislative document title: " + shot_title + "\nTagged labels: " + '\n'.join(_shot_cands) + '\n'
                shots_prompt += shot_prompt
        # else:
        #     shot_cands = []
        #     for j in range(min(len(shots[i]), 5)):
        #         shot_title = shots[i][j]['title']
        #         # shot_content = shots[i][j]['content']
        #         if args.prompt_type == 'full':
        #             shot_cands = '\n'.join([json.loads(item)['title'] for item in shots[i][j]['cat']])
        #         if args.prompt_type == 'example':
        #             shot_cands = '\n'.join(shots[i][j]['cat'])
        #         if args.prompt_type == 'weak':
        #             shot_cands = '\n'.join([item['text'] for item in cands[shots[i][j]['uid']][:5]])
        #         # shot_prompt = "Product title on Amazon: " + shot_title + ". Product content: " + shot_content + "\n Similar product: " + shot_cands + '\n'
        #         shot_prompt = "Product title on Amazon: " + shot_title + "\nSimilar product: " + shot_cands + '\n'
        #         shots_prompt += shot_prompt
        
        if shots_prompt == '':
            shots_prompt = random_prompt

        title = text[i]['title']
        content = text[i]['content']
        if args.prompt_type == 'example':
            cands = '\n'.join(hints[i]['output'][:30])
            if args.dataset == 'amazon':
                prompt1 = "For an Amazon product recommendation task, product title: " + title + "\nCandidate labels: " + cands + '\n'
                prompt = prompt1 + 'For each label, guess an input title. Format: ["title1", "title2", "title3", ...], each title is a guess based on a candidate label, title1 is a guess for first label, and so on. Only output one list and the list should be of size 30. do not explain or say anthing.\n'
            if args.dataset == 'wiki':
                # prompt1 = "For a Wikipedia page 'see also' suggestion task, wiki title: " + title + "\nCandidate labels: " + cands + '\n'
                # prompt = prompt1 + 'For each "See also" reference label, generate a plausible Wikipedia page title that could lead to a it. Format: ["title1", "title2", "title3", ...], each title is a generation based on a candidate label, title1 is generated for first label, and so on. Only output one list and the list should be of size 30. do not explain or say anthing.'
                prompt1 = "There's a list of Wikipedia page titles: " + cands + '\n'
                prompt = prompt1 + 'For each page, generate a "See also" page title. Format: ["title1", "title2", "title3", ...], each title is a generation based on a candidate label, title1 is generated for first label, and so on. Only output one list and the list should be of size 30. do not explain or say anthing.\n'
            # prompt = prompt1 + 'For each label, guess an input title. Format: title1\ntitle2\ntitle3\neach title is a guess based on a candidate label, title1 is a guess for first label, and so on. Only output 50 lines of titles. do not explain or say anthing.\n'
            if args.dataset == 'eurlex':
                prompt1 = "For a EU legislative document tagging task, document title: " + title + "\nCandidate tags: " + cands + '\n'
                prompt = prompt1 + 'For each tag, generate a corresponding EU legislative document title. Format: ["title1", "title2", "title3", ...], each title is a generation based on a candidate label, title1 is generated for first label, and so on. Only output one list and the list should be of size 30. do not explain or say anthing.\n'

        elif args.prompt_type == 'group':
        # prompt_side = "background knowledge: " + background[i]['background'] + '\n'
        # hint = '\n'.join(hints[i]['output'])
        # hint = '(Hint: The answer may near to: ' + hint + ')'
            if args.dataset == 'amazon':
                prompt = "You are now trying to predict at least 10 relevant products for a new Amazon product title: " + title + "\ndescription: " + content + '\n'
            if args.dataset == 'wiki':
                p_cut = 'Title: ' + title + '\nContent: ' + content[:10000] + '\n'
                prompt = p_cut + "Generate 'See also' suggestions related to the Wikipedia title: " + title + '\nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n...'
            if args.dataset == 'eurlex':
                p_cut = 'Title: ' + title + '\nHeader: ' + text[i]['header'] + '\nRecitals: ' + text[i]['recitals'] + '\n'
                prompt = p_cut + "Given the above EU legislative document, generate relevant labels. \nOnly output titles with line break, do not include anything else. example: title 1\ntitle2\n..."
                # prompt = ''
                shots_prompt = shots_prompt[:(13000 - len(prompt))]
            prompt = shots_prompt + prompt
        print(prompt)
        print(len(prompt))
        # if (len(prompt) > 10000):
        #     print(prompt)
        prompt = prompt[:13000]
        prompts.append(prompt)

        ### explore the effect of retrieved labels
        # out = []
        # for j in range(len(shots[i])):
        #     cands4out = [item['text'] for item in cands[shots[i][j]['uid']]]
        #     out.extend(cands4out)
        # out = list(set(out))
        # with open('../preprocessing/guess_weak_content/oracle_200.jsonl', "a") as outfile:
        #         outfile.write(
        #             json.dumps(
        #                 {
        #                     "id": text[i]['uid'],
        #                     "output": out
        #                 }
        #             ) + "\n"
        #         )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        completions = list(tqdm(executor.map(get_completion_with_retries, prompts), total=len(prompts)))
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     with tqdm(total=len(prompts)) as pbar:
    #         completions = list(executor.map(lambda prompt: get_completion_with_retries(prompt, tqdm=pbar), prompts))
    #         pbar.update()

    with open(args.save_to, 'w') as fw:
        for i, completion in enumerate(completions):
            gt = [lbl[j]['title'] for j in text[i + args.start]['target_ind']]
            result = text[i + args.start]
            result['gt'] = gt
            result['pred'] = completion
        
            json.dump(result, fw)
            fw.write('\n')





main()
