import openai
import time
import string
import json
import pickle
import os
from nltk import tokenize

def log_prediction(text, log_file_path):
    with open(log_file_path, 'a') as f:
        f.write(text)

def log_inconsistency(inconsistency_count, sub_folder_path, file_path, args):
    log_file_name = f'{args.world_tracker_mode}_{args.world_checker_mode}.txt'
    log_file_path = os.path.join(sub_folder_path, log_file_name)
    with open(log_file_path, 'a') as f:
        f.write(f'{file_path}: inconsistency_count {inconsistency_count}\n')

def create_folder(args):
    if not os.path.exists(args.folder_path):
        os.mkdir(args.folder_path)
    sub_folder = args.world_tracker_mode
    sub_folder_path = os.path.join(args.folder_path, sub_folder)
    if not os.path.exists(sub_folder_path):
        os.mkdir(sub_folder_path)
        # else:
        #     print('folder exists make sure you want to override')

# def write_to_file(output, story_idx, story_type, gen_count, args):
#     sub_folder = args.world_tracker_mode
#     if 'Codex' in args.world_tracker_mode:
#         file_name = f'story_{story_idx}_{story_type}_{gen_count}.py'
#     else:
#         file_name = f'story_{story_idx}_{story_type}_{gen_count}.txt'
#     output_path = os.path.join(args.folder_path, sub_folder, file_name)
#     with open(output_path, "w") as f:
#         f.write(output)

def write_to_file(output, sub_folder, subsubfolder, output_path):
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)
    if not os.path.exists(subsubfolder):
        os.mkdir(subsubfolder)
    with open(output_path, "w") as f:
        f.write(output)
    

def load_story(idx, args):
    print(idx)
    # if idx == 9 or idx == 76: ## These stories are used as examples
    #     return None, None, None, None
    
    altered_info_path = f"{args.story_folder_path}/altered/{idx}.pkl"
    original_info_path = f"{args.story_folder_path}/original/{idx}.pkl"
    altered_story_path = f"{args.story_folder_path}/altered_stories/{idx}.txt"
    original_story_path = f"{args.story_folder_path}/original_stories/{idx}.txt"

    try:
        with open(original_info_path, 'rb') as f:
            oringinal_info = pickle.load(f)
        with open(altered_info_path, 'rb') as f:
            altered_info = pickle.load(f)
        with open(original_story_path, 'r') as f:
            original_story = f.read()
        with open(altered_story_path, 'r') as f:
            altered_story = f.read()
    except: 
        import pdb; pdb.set_trace()
        return None, None, None, None

    if original_story == 'SKIPPED' or altered_story == 'SKIPPED' or "SKIPPED" in original_story or "SKIPPED" in altered_story:
        print('story skipped')
        return None, None, None, None
    
    return oringinal_info, altered_info, original_story, altered_story

def get_gpt3_response(prompt, model = 'text-davinci-002', max_tokens = 100, temperature = 0, top_p = 0.9, frequency_penalty = 0, presence_penalty = 0, stop = ['###', '===']) -> str:
    if model not in ['text-davinci-002', 'text-curie-001', 'code-davinci-001', 'code-davinci-002']:
        print("model name is wrong, please check you model name")

    try:
        res = openai.Completion.create(
            model = model,
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            stop = stop
        )
        return res['choices'][0]['text']
    except Exception as e:
        print(e)
        print('wait for 60 seconds')
        time.sleep(60)
        return get_codex_response(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop)

def get_codex_response(prompt, model = 'code-davinci-002', max_tokens = 3500, temperature = 0.5, top_p = 0.9, frequency_penalty = 0, presence_penalty = 0,  stop = ['###', '\n\n\n\n', '===', 'class character']) -> str:
    print('start Codex query')
    if model not in ['text-davinci-002', 'text-curie-001', 'code-davinci-002']:
        print("model name is wrong, please check you model name")

    try:
        res = openai.Completion.create(
            model = model,
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,
            stop = stop
        )
        return res['choices'][0]['text']
    except Exception as e:
        print(e)
        print('wait for 60 seconds')
        time.sleep(60)
        return get_codex_response(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop)

def check_GPT_res(res):
    # import pdb; pdb.set_trace()
    if 'yes' in res.lower():
        return True
    elif 'no' in res.lower():
        return False
    elif 'no contradiction' in res.lower():
        return False
    elif 'contradiction' in res.lower():
        return True
    else:
        print('error checking in GPT response')

def check_codex_res(res):
    if 'inconsistent' in res.lower():
        return True
    elif 'consistent' in res.lower() and 'inconsistent' not in res.lower():
        return False
    else:
        print('error in parsing Codex res: Yes/No')
        return 'NA'
        
def text_to_code(s, capitalize=False, lower=True):
    s = s.replace('\'t', ' not')
    for punctuation in string.punctuation:
        s = s.replace(punctuation, ' ')
    s = s.replace('”', ' ').replace('“', ' ').replace('’', ' ').replace('‘', ' ').replace('—', ' ').replace('.', ' ').replace('…',' ')
    s = s.strip().lower().replace(' ', '_')
    while ('__' in s):
        s = s.replace('__', '_')
    return s

def dump_to_json(dict, output_path):
    with open(output_path, 'a') as f:
        f.write(json.dumps(dict) + "\n")

def split_into_sentences(text):
    text = text.rstrip('\n').replace('\n', ' ')
    return tokenize.sent_tokenize(text)