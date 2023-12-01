import json
import re
import os
import openai
import copy
import warnings
from collections import Counter
from utils.other_utils import get_gpt3_response, check_GPT_res, log_prediction, get_codex_response, check_codex_res
from utils.prompts import *
from utils.world_tracker_utils import check_bad_generation
warnings.filterwarnings("ignore")

skip_bad_generation = False

def import_codex_world_class(args, file_path, sub_folder):
    import_path = file_path.strip('.py')
    import pdb; pdb.set_trace()
    exec(f"from {args.folder_path.strip('/').split('/')[-1]}.{sub_folder}.{import_path} import World", globals())
    world = World()
    if has_method(world, 'story_setting'):
        world.story_setting()
    world.story()
    return world

def majority_vote(world_list):
    res = copy.deepcopy(world_list[0])
    for char_name, char in vars(res).items():
        vars(char)['relations'] = falten_relation_dict(vars(char)['relations'])
    
    for world in world_list[1:]:
        for char_name, char in vars(world).items():
            vars(char)['relations'] = falten_relation_dict(vars(char)['relations'])
            for attr_name, attr in vars(char).items():
                if attr_name != 'name':
                    char_res = getattr(res, char_name)
                    attr_res = getattr(char_res, attr_name)
                    attr_res += attr

    for char_name, char in vars(res).items():
        for attr_name, attr in vars(char).items():
            if attr_name != 'name':
                # attr = list(filter(lambda s: attr.count(s) > (2), attr))
                tmp = []
                for key, value in Counter(attr).items():
                    if value >= len(world_list) / 2:
                        # print(key, value)
                        tmp.append(key)
                # attr = tmp
                setattr(char, attr_name, tmp)
    return res

attr_to_sent_example = """
Jason Westfall's appearance is dark blue eyes.
Jason Westfall has dark blue eyes.
###
Jason Westfall's appearance is average height.
Jason Westfall is of average height.
###

"""

## input: world object
## output: {char_name: [char's attribute]}
def transform_to_description(world):
    res = {}
    for char_name, char in vars(world).items():
        char_name = char_name.replace('_', ' ')
        description = []
        for attr_name, attr in vars(char).items():
            if attr_name == 'relations' and vars(char)[attr_name] != []:
                for relation in attr:
                    description.append(f"{relation.replace('_', ' ')} {char_name}.")
            elif vars(char)[attr_name] != [] and attr_name != 'name':
                description.append(get_gpt3_response(attr_to_sent_example + f"{char_name}'s {attr_name} is {', '.join(list(set(attr)))}.\n"))
        res[char_name] = description
    return res
### A.relations['father'] = B --> B is the father of A            

# def transform_to_description_2(world):
#     res = {}
#     for char_name, char in vars(world).items():
#         char_name = char_name.replace('_', ' ')
#         description = {}
#         for attr_name, attr in vars(char).items():
#             if attr_name == 'relations' and vars(char)[attr_name] != []:
#                 description['relations'] = []
#                 for relation in attr:
#                     description['relations'].append(f"{relation.replace('_', ' ')} {char_name}.")
#             elif vars(char)[attr_name] != [] and attr_name != 'name':
#                 description[attr_name] = get_gpt3_response(attr_to_sent_example + f"{char_name}'s {attr_name} is {', '.join(list(set(attr)))}.\n")
#         res[char_name] = description
#     return res

def get_world_model(file_path, subsubfolder):
    folder_path, subfolder_name, story_idx = subsubfolder.split('/')[-3], subsubfolder.split('/')[-2], subsubfolder.split('/')[-1]
    exec(f"from {folder_path}.{subfolder_name}.{story_idx}.{file_path.replace('.py', '')} import World", globals())
    world = World()
    return world

def check_consistency_seperated(args, file_path, subsubfolder):
    log_file_path = os.path.join(subsubfolder, file_path.replace('.py', '.txt')) 
    
    import_path = file_path.strip('.py')
    ## re3_Nov_25/Codex_2/  0  /inconsistent_1_0.py

    folder_path, subfolder_name, story_idx = subsubfolder.split('/')[-3], subsubfolder.split('/')[-2], subsubfolder.split('/')[-1]
    exec(f"from {folder_path}.{subfolder_name}.{story_idx}.{import_path} import World", globals())
    world_setting = World()
    world_setting.story_setting()
    world = World()
    world.story()
    total_inconsistency = 0
    
    print(args.world_checker_mode)
    print(os.path.join(args.folder_path, subsubfolder , file_path))
    
    if args.world_checker_mode == 'GPT3_one_attr':
        for char_name in vars(world_setting).keys():
            char_setting = vars(world_setting)[char_name]
            char = vars(world)[char_name]
            for attr_name, attr in vars(char).items():
                if attr_name == 'relations' and vars(char)[attr_name] != {} and vars(char_setting)[attr_name] != {}:
                    relation_story = ' '.join([f"{char.name}'s {relation} is {name}." for relation, name in vars(char)['relations'].items()])
                    relation_setting = ' '.join([f"{char.name}'s {relation} is {name}." for relation, name in vars(char_setting)['relations'].items()])
                    res = check_consistency(relation_story, relation_setting, attr = attr_name, method = "GPT3")
                    total_inconsistency += res
                    text = 'STORY: ' + relation_story + ' ' + '\nSTORY_SETTING: ' + relation_setting + ' \n' + str(res) + '\n\n'
                    log_prediction(text, log_file_path)
                elif vars(char)[attr_name] != [] and vars(char_setting)[attr_name] != [] and  attr_name != 'name' and attr_name != 'relations':
                    attr_description_setting = transform_to_description(char.name + "'s " + attr_name + ':  ' + str(vars(char_setting)[attr_name]))
                    attr_description_story = transform_to_description(char.name + "'s " + attr_name + ':  ' + str(vars(char)[attr_name]))
                    res = check_consistency(attr_description_setting, attr_description_story, attr = attr_name, method = "GPT3")
                    total_inconsistency += res
                    text = 'STORY: ' + attr_description_setting + '\nSTORY_SETTING: ' + attr_description_story + ' \n' + str(res) + '\n\n'
                    log_prediction(text, log_file_path)
    
    elif args.world_checker_mode == 'GPT3_all_attr':
        for char_name in vars(world_setting).keys():
            char_setting = vars(world_setting)[char_name]
            char = vars(world)[char_name]
            vars(char)['relations'] = falten_relation_dict(vars(char)['relations'])
            vars(char_setting)['relations'] = falten_relation_dict(vars(char_setting)['relations'])
            question_setting, question_story = '', ''
            
            for attr_name, attr in vars(char).items():
                if vars(char)[attr_name] != [] and attr_name != 'name':
                    question_story += f"The {char_name}'s {attr_name} is {', '.join(list(set(attr)))}. "
                if vars(char_setting)[attr_name] != [] and attr_name != 'name':
                    question_setting += f"The {char_name}'s {attr_name} is {', '.join(list(set(vars(char_setting)[attr_name])))}. "
                
            prompt = f"Question: Are there any contradiction in the person's description? \nStory Setting: {question_setting}  \nStory: {question_story} \nAnswer:"
            res = get_gpt3_response(GPT3_consistency_prompt_3 + prompt, max_tokens = 200) 
            inconsistency_count = re.findall(r'\d+', res)
            total_inconsistency += int(inconsistency_count[0])
            log_prediction(prompt + res + '\n', log_file_path)
                
    return total_inconsistency

def check_world_consistency(mode, world, file_path, sub_folder, args):
    # if check_bad_generation(os.path.join(args.folder_path, sub_folder, file_path)) and skip_bad_generation:
    #     return 'NA'
    if mode in ['TE', 'GPT3_one_attr', 'GPT3_all_attr', 'Codex_one_attr']:
        total_inconsistency = 0
        for char_name, char in vars(world).items():
            try:
                total_inconsistency += check_char_consistency(mode, char, file_path, sub_folder, args)
            except:
                print('unexpected character object')
        return total_inconsistency
    else:
        print("check your world checker mode, it\'s not in 'TE', 'GPT3_one_attr', 'GPT3_all_attr'")
        return 'NA'

def check_char_consistency(mode, char, file_path, sub_folder, args):
    inconsistency_count = 0
    if not os.path.exists(os.path.join(args.folder_path, sub_folder, args.world_checker_mode)):
        os.mkdir(os.path.join(args.folder_path, sub_folder, args.world_checker_mode))
    log_file_path = os.path.join(args.folder_path, sub_folder, args.world_checker_mode, file_path.replace('.py', '.txt')) 
    
    if mode == 'TE':
        for attr_name, attr in vars(char).items():
            if attr_name == 'relations':
                attr = falten_relation_dict(attr)
                
            if attr_name != 'name' and len(attr) > 1 and attr_name != 'status':
                tmp = 'has' if attr_name == 'appearance' else 'is'
                premise = f"The person {tmp} {attr[0]}"
                hypothesis= f"The person {tmp} {attr[-1]}"
                prediction = TE_model.predict(premise, hypothesis)
                if max(prediction['label_probs'])==prediction['label_probs'][1]: ## check['label_probs']=[entailment, contradition, irrelevant]
                    inconsistency_count += 1
                
                text = f"premise {premise}\nhypothesis{hypothesis}\nprediction{prediction}"
                log_prediction(text, log_file_path)
                
        return inconsistency_count

    elif mode == 'Codex_one_attr':
        for attr_name, attr in vars(char).items():
            if attr != [] and len(attr) > 1 and attr_name != 'name' and attr_name != 'status':
                if isinstance(attr, list):
                    question_body = f"{char.name.replace(' ', '_')}.{attr_name}={list(set(attr))}"
                else:
                    question_body = f"{char.name.replace(' ', '_')}.{attr_name}={attr}"
                prompt = f"Check whether there are inconsistencies for the following character.\n{question_body}\nAnswer: "
                print(prompt)
                res = get_codex_response(Codex_consistency_prompt_1 + prompt, max_tokens = 5)
                if check_codex_res(res): ## True if contradition detected
                    inconsistency_count += 1
                
                text = f"{question_body}\nAnswer: {res}\n"
                log_prediction(text, log_file_path)
        return inconsistency_count
                
                
    elif mode == 'GPT3_one_attr':
        for attr_name, attr in vars(char).items():
            if attr_name == 'relations':
                attr = falten_relation_dict(attr)
            if attr_name != 'name' and len(attr) > 1 and attr_name != 'status':
                question_body = f"{attr_name}: {', '.join(list(set(attr)))}?"
                prompt = f"Question: Are there any contradiction in the person's {question_body}"
                res = get_gpt3_response(GPT3_consistency_prompt_1 + prompt, max_tokens = 5) 
                if check_GPT_res(res): ## if contradition detected
                    inconsistency_count += 1
                
                text = f"{char}: {prompt}\n{res}\n"
                log_prediction(text, log_file_path)
                
        return inconsistency_count
        
    elif mode == 'GPT3_all_attr':
        question_body = ''
        for attr_name, attr in vars(char).items():
            if attr_name == 'relations':
                attr = falten_relation_dict(attr)
            if attr_name != 'name' and len(attr) > 1:
                question_body += f"The person's {attr_name} is {', '.join(list(set(attr)))}."
                
        prompt = f"Question: Are there any contradiction in the person's description? {question_body}"
        res = get_gpt3_response(GPT3_consistency_prompt_2 + prompt, max_tokens = 200) 
        inconsistency_count = re.findall(r'\d+', res)
        
        if len(inconsistency_count) != 1:
            print('prompt', prompt)
            print('\nres', res)
            print('error in parsing GPT3 res: number')
            ## TO-DO: in the future should return NA and change some code above
            return 0

        text =f"{char}: {prompt}\n{res}\ninconsistency_count{inconsistency_count}\n"
        log_prediction(text, log_file_path)
        return int(inconsistency_count[0])

# def transform_to_description(text):
#     res = get_gpt3_response(transform_to_description_example + text)
#     return res

## if contradiction returns True
def check_consistency(attr_description_setting, attr_description_story, attr, method):
    if method == 'TE':
        prediction = TE_model.predict(attr_description_setting, attr_description_story)
        if max(prediction['label_probs'])==prediction['label_probs'][1]: ## check['label_probs']=[entailment, contradition, irrelevant]
            res = True
        else:
            res = False
            
    if method == 'GPT3':
        res = get_gpt3_response(check_consistency_GPT3_example + "\nStory Setting: " + attr_description_setting + "\nStory: " + attr_description_story)
        res = check_GPT_res(res)
    return res

def falten_relation_dict(dict):
    return [f"{name} is the {relation} of" for relation, name in dict.items()]

### A.relations['father'] = B --> B is the father of A     
## the father of B

def has_method(o, name):
    return callable(getattr(o, name, None))
