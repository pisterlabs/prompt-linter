import os
import json
import openai
import copy
import random
from datetime import datetime
import tiktoken

try:
    import src.utils as utils
    import src.dialogue_manager as dialogue_manager
    import src.answer_parser as answer_parser
    import src.output_matching as output_matching
    from src.api_secrets import API_KEY
except ImportError:
    try:
        import utils
        import dialogue_manager
        import answer_parser
        import output_matching
        from api_secrets import API_KEY
    except ImportError as e:
        print('ERROR:')
        print(e)
        exit(-1)

def num_tokens_from_string(string, encoding_name = "cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    # print('checking length: ', num_tokens)
    return num_tokens

def keep_context(messages):
    large_s = ' '.join([msg['content'] for msg in messages])
    n_tokens = num_tokens_from_string(large_s)
    # print('\nchecking context length')
    # print('before: ', n_tokens)
    while n_tokens > 4000:
        messages = messages[:1] + messages[3:]
        large_s = ' '.join([msg['content'] for msg in messages])
        n_tokens = num_tokens_from_string(large_s)
    # print('after: ', n_tokens)
    return messages


def auto_user(question, config):
    with open('resources/user-profiles.json', 'r') as f:
        profiles = json.load(f)
    num_p = len(profiles)
    choice = random.randint(0,num_p-1)
    profile = profiles[choice]
    # create the prompt and call make_api_call
    with open(f'blueprints/user_{config["form"]}.txt', 'r') as f:
        prompt = f.read()
    prompt = prompt.replace('[Insert 1]', profile)
    prompt = prompt.replace('[Insert 2]', question)
    if config['form'] == 'epa':
        prompt = prompt.replace('[Insert 3]', "reporting an environmental violation")
    elif config['form'] == 'ss5':
        prompt = prompt.replace('[Insert 3]', "an application for a social security card")

    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]
    config['temperature'] = config['high temp']
    messages = keep_context(messages)
    out = utils.make_api_call(messages, config['model'], config)
    config['temperature'] = config['base temp']
    span, out_trim = output_matching.match_output(out, 'user')
    if span is None:
        messages.extend([
            {'role': 'assistant', 'content': out},
            {'role': 'user', 'content': ('\n\n The output needs to fit the following Regular Expression:\n ' 
                                    + output_matching.re_user.pattern)}
        ])
        config['temperature'] = config['high temp']
        messages = keep_context(messages)
        out = utils.make_api_call(messages, config['model'], config)
        config['temperature'] = config['base temp']
        span, out_trim = output_matching.match_output(out, 'user')
        if span is None:
            return ' '
        
    ans = json.loads(out_trim)
    return ans['answer']


def main(config):
    dialogue = []
    form = utils.load_form(config['form'])
    state = utils.init_cf_state(form)
    with open('blueprints/single_agent_qgen.txt', 'r') as f:
        prompt = f.read()
    prompt = prompt.replace('[Insert 1]', json.dumps(form))
    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]
    messages = keep_context(messages)
    response = utils.make_api_call(messages, config['model'], config)
    span, group = output_matching.match_output(response, 'fupq')
    i = 0
    while span is None:
        if i//10 >= 1:
            print("stuck ... exiting")
            exit(1)
        messages.extend(
            [
                {'role': 'assistant', 'content': response},
                {'role': 'user', 'content': 'Your output needs to have the following format: {"question": "<Your question>"}'}
            ]
        )
        messages = keep_context(messages)
        response = utils.make_api_call(messages, config['model'], config)
        span, group = output_matching.match_output(response, 'fupq')
        i += 1
    messages.append({'role': 'assistant', 'content': response})
    loaded = json.loads(group)
    print(loaded['question'])
    if config['autouser']:
        ans = auto_user(loaded['question'], config)
        print('\nANSWER:\n', ans)
    else:
        ans = input("\nANSWER:\n")
    print("\n")
    dialogue.append({"Assistant": loaded['question']})
    dialogue.append({"User": ans})
    with open('blueprints/single_agent_apar.txt', 'r') as f:
        apar_prompt = f.read()
    
    while True:
        # print(messages)
        messages.append({'role': 'user', 'content': apar_prompt.replace('[Insert 1]', ans)})
        messages = keep_context(messages)
        response = utils.make_api_call(messages, config['model'], config)
        span, group = output_matching.match_output(response, 'apar')
        i = 0
        while span is None:
            if i//10 >= 1:
                print("stuck ... exiting")
                exit(1)
            messages.extend(
                [
                    {'role': 'assistant', 'content': response},
                    {'role': 'user', 'content': 'Your output needs to have the following format: {"next action": ""}'}
                ]
            )
            messages = keep_context(messages)
            response = utils.make_api_call(messages, config['model'], config)
            span, group = output_matching.match_output(response, 'apar')
            i += 1
        loaded = json.loads(group)
        print(group)
        messages.append({'role': 'assistant', 'content': response})
        if loaded['next action'] == 'follow_up':
            messages.append({'role': 'user', 'content': 'Ask another question and return it in the following format: {"question": ""}'})
            messages = keep_context(messages)
            response = utils.make_api_call(messages, config['model'], config)
            span, group = output_matching.match_output(response, 'fupq')
            i = 0
            while span is None:
                if i//10 >= 1:
                    print("stuck ... exiting")
                    exit(1)
                messages.extend(
                    [
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': 'Your output needs to have the following format: {"question": ""}'}
                    ]
                )
                messages = keep_context(messages)
                response = utils.make_api_call(messages, config['model'], config)
                span, group = output_matching.match_output(response, 'fupq')
                i += 1
            messages.append({'role': 'assistant', 'content': response})
            loaded = json.loads(group)
            print(loaded['question'])
            if config['autouser']:
                ans = auto_user(loaded['question'], config)
                print('\nANSWER:\n', ans)
            else:
                ans = input("\nANSWER:\n")
            print("\n")
            dialogue.append({"Assistant": loaded['question']})
            dialogue.append({"User": ans})
            continue
        elif loaded['next action'] == 'fill_form':
            messages.append({'role': 'user', 'content': 'Fill out any fields for which you have gathered information, then return the entire FORM.'})
            messages = keep_context(messages)
            response = utils.make_api_call(messages, config['model'], config)
            span, group = output_matching.match_output(response, 'ffil')
            i = 0
            fail = False
            if span is None:
                fail = True
            elif not utils.check_likeness(form, json.loads(group)):
                fail = True
            while fail:
                if i//10 >= 1:
                    print("stuck ... exiting")
                    exit(1)
                messages.extend(
                    [
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': 'Be sure to return the entire FORM with only the appropriate answers filled out.'}
                    ]
                )
                messages = keep_context(messages)
                response = utils.make_api_call(messages, config['model'], config)
                span, group = output_matching.match_output(response, 'ffil')
                fail = False
                if span is None:
                    fail = True
                elif not utils.check_likeness(form, json.loads(group)):
                    fail = True
                i += 1
            messages.append({'role': 'assistant', 'content': response})
            loaded = json.loads(group)
            form = utils.fill_in_parts(form, loaded)
            state = utils.update_cf_state(state, form)
            still_open = utils.get_unanswered(form, state['fields'])
            msg = f"after disregarding fields that have been filled out, you are left with the following FORM:\n\n {json.dumps(still_open)}\n\n"
            msg += 'Ask another question and return it in the following format: {"question": ""}'
            messages.append({'role': 'user', 'content': msg})
            messages = keep_context(messages)
            response = utils.make_api_call(messages, config['model'], config)
            span, group = output_matching.match_output(response, 'fupq')
            i = 0
            while span is None:
                if i//10 >= 1:
                    print("stuck ... exiting")
                    exit(1)
                messages.extend(
                    [
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': 'Your output needs to have the following format: {"question": ""}'}
                    ]
                )
                messages = keep_context(messages)
                response = utils.make_api_call(messages, config['model'], config)
                span, group = output_matching.match_output(response, 'fupq')
                i += 1
            messages.append({'role': 'assistant', 'content': response})
            loaded = json.loads(group)
            print(loaded['question'])
            if config['autouser']:
                ans = auto_user(loaded['question'], config)
                print('\nANSWER:\n', ans)
            else:
                ans = input("\nANSWER:\n")
            print("\n")
            dialogue.append({"Assistant": loaded['question']})
            dialogue.append({"User": ans})
            continue
        elif loaded['next action'] == 'stop':
            break
    return dialogue, form



    # while not stop
    #   ask question
    #   parse results and fill out form
    #   validate 



if __name__ == '__main__':

    config = utils.load_config("config/single_agent_config.json") 
    # setting key
    openai.api_key = API_KEY
    out_dir = utils.get_out_dir()
    for i in range(config['num_dialogues']):
        dialogue, form = main(config)
        with open(os.path.join(out_dir, 'dialogue.json'), 'w') as f:
            json.dump(dialogue, f)
        with open(os.path.join(out_dir, 'filled-form.json'), 'w') as f:
            json.dump(form, f)