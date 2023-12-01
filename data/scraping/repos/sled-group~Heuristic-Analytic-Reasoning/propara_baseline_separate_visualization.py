import sys
sys.path.append("../")

import json
import numpy as np
from numpy.linalg import norm
import re
from argparse import ArgumentParser
import os
import datetime
from pprint import pprint
from tqdm import tqdm
import time
import openai
import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from data.propara import load_propara_dataset, plausibility_demo_full, conflict_demo_full, physical_states_demo_full, story_pair_prompt_generator
from eval.propara import check_response, response_extractor
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_chat_gpt_with_caching, VICUNA13B_PATH, prompt_llama_with_caching, get_65b_device_mapping
from utils import get_output_dir_name
from visualization import separation_generator

def remove_empty_strings(my_list):
    return [item for item in my_list if item != ""]

# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.

def generate_heatmap_upon_text(text_list, attention_list, latex_file, color='red'):
    assert(len(text_list) == len(attention_list))
    word_num = len(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
            \special{papersize=210mm,297mm}
            \usepackage{color}
            \usepackage{tcolorbox}
            \usepackage{CJK}
            \usepackage{adjustbox}
            \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
            \begin{document}
            \begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            if text_list[idx] == ' ':
                formatted_number = "{:.10f}".format(attention_list[idx])
                string += "\\colorbox{%s!%s}{"%(color, formatted_number)+"\\strut \hspace{0.05em}" + text_list[idx]+"}"
            elif text_list[idx] == '\n':
                string += r"\newline"
            else:
                formatted_number = "{:.10f}".format(attention_list[idx])
                string += "\\colorbox{%s!%s}{"%(color, formatted_number)+"\\strut " + text_list[idx]+"}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}\end{document}''')

def slice_4d_list(lst, d2_idx):
    """
    Slice a 4D list with inhomogeneous 4th dimension (list of lists of lists of lists).

    Parameters:
    lst (list): the list to slice.
    d2_idx (int): the index for the second dimension.

    Returns:
    list: the sliced list.
    """
    return [[[sublist3 for sublist3 in sublist2[d2_idx]] for sublist2 in sublist1] for sublist1 in lst]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lm_backbone", default="gpt4", choices=["gpt3", "gpt4", "chatgpt", "vicuna", 'llama7b', 'llama13b', 'llama30b', 'llama65b'])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--cache_only", action="store_true", default=False)

    parser.add_argument("--action_type", type=str, default="conversion", choices=["conversion"])
    parser.add_argument("--output_attn", action="store_true", default=False) 

    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LETTER_TO_IDX = {'A': 0, 'B': 1}

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo', 'alpaca13b': 'alpaca13b'}

    action_types = [args.action_type]
    dataset = load_propara_dataset(action_types=action_types)
    train_dataset = dataset[args.action_type]["train"]
    test_dataset = dataset[args.action_type]["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

    CACHE_FNAME = f'cache_files/propara_lm_cache_{args.lm_backbone}_{args.demo_choice}_baseline-separate.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
    if os.path.exists(CACHE_FNAME):
        lm_cache = pickle.load(open(CACHE_FNAME, 'rb'))
        print(len(lm_cache))
    else:
        lm_cache = {}
    args.cache_fname = CACHE_FNAME

    TOKEN_COUNTS = {'prompt': 0, 'gen': 0}


    if args.lm_backbone in ["gpt4", "chatgpt"]:
        preamble = "You are a smart chatbot capable of physical commonsense reasoning, and understanding the physical state of the world implied in written texts. "
        preamble += "You will read stories that involve physical actions, and will be asked to "
        
        # Generate preamble messages for chat models - one for each separated task
        preamble_plausibility = preamble + (
            "determine which of two stories is more plausible by comparing possible conflicts within them. "
        )
        preamble_conflict = preamble + (
            "use your understanding of the physical state changes in each story to locate a plausibility conflict. "
        )
        preamble_physical_states = preamble + (
            "You will report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
            "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
            "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
        )
    
    if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
        print("Setting up LLaMA...")
        if not args.skip_prompting and not args.cache_only:
            if args.model_path is None and args.local_only:
                raise ValueError("Must specify --model_path if --local_only is set.")
            model_path = args.model_path if args.model_path is not None else f"decapoda-research/llama-{args.lm_backbone[-2:]}-hf"
            local_only = args.local_only
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_only,
            )
            print("Loaded tokenizer.")
            config = AutoConfig.from_pretrained(model_path)
            # reconfigure to output attentions
            config.output_attentions = args.output_attn
            config.return_dict_in_generate = args.output_attn
            if args.lm_backbone == 'llama65b' and args.output_attn:
                device_map = get_65b_device_mapping(config, torch.cuda.device_count(), config.num_hidden_layers//(torch.cuda.device_count()-1)+1)
            else:
                device_map = "auto"
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            model.tie_weights()
            model = load_checkpoint_and_dispatch(
                model, 
                model_path, 
                device_map=device_map, 
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float16,
            )
            print("Loaded model.")
        else:
            model, tokenizer = None, None

    if args.demo_choice == "stories-4":
        selected_story_example_ids = [200, 68, 107, 311]
    # elif args.demo_choice == "balanced-6":
    #     # sample 2 stories for 3 types of conflict
    #     selected_story_example_ids = balanced_sample_story_ids(train_dataset, select_n_stories_for_each_category=2) # If encounters any error, change a random seed!
    else:
        raise Exception("demo_choice out of options")
    
    print("selected story IDs:", selected_story_example_ids)
    
    # selected_story_conflict_reasons = {
    #     588: "the donut was put in the trash in sentence 4, but Mary ate the donut in sentence 5",
    #     107: "in sentence 1 Ann realizes that all of her tools are missing yet in sentence 2 she is taking out an axe, a pair of scissors, rope and a few other things",
    #     0: "Tom put the soup in the microwave and then ate it cold",
    #     51: "Sentence 3 states that the hair dryer would not turn on, while Sentence 4 states that Ann used the hair dryer to dry her hair",
    #     311: "you cannot turn something on that is not plugged in",
    # }

    # Generate some demos for story plausibility
    plausibility_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        plausibility_demos += plausibility_demo_full(story_pair) + "\n"

    # Generate some demos for conflict detection
    conflict_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        conflict_demos += conflict_demo_full(story_pair) + "\n"

    # # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    # app_demos = generate_app_demos(train_dataset)
    # aep_demos = generate_aep_demos(train_dataset)

    # Generate some demos for physical states
    # physical_states_demos = app_demos + "\n" + aep_demos
    physical_states_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        physical_states_demos += physical_states_demo_full(story_pair) + "\n"

    for i in tqdm(range(0, len(test_dataset))):

        example_id = str(test_dataset[i]['example_id'])

        prediction_obj = {'example_id': example_id}
        story_pair = test_dataset[i]
        if story_pair['story_converted'] == 'A':
            story_converted = story_pair["story_A_sentences"]
        else:
            story_converted = story_pair["story_B_sentences"]
        story_prompt = story_pair_prompt_generator(story_pair)

        # STEP 1: Prompt LM for which story is plausible
        if args.lm_backbone in ['gpt3']:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
            prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
            plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['gpt4', 'chatgpt']:
            plausibility_prompt = []
            plausibility_prompt.append(get_chat_message('system', preamble_plausibility))
            plausibility_prompt.append(get_chat_message('system', "Here are some examples:\n\n" + plausibility_demos))
            plausibility_prompt.append(get_chat_message('user', story_pair_prompt_generator(story_pair)))
            prediction_obj['input_prompt_plausibility'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in plausibility_prompt]
            plausibility_generated_text = prompt_chat_gpt_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            plausibility_prompt = plausibility_demos
            plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
            if args.output_attn:
                separations = separation_generator(plausibility_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer)
                separations[0][1] = separations[0][1][:-1]
                separations[1][1][-1] = separations[0][1][-1]
            else:
                separations = None
            plausibility_generated_text = prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=128, output_attn=args.output_attn, separations=separations)
        if args.lm_backbone in ['gpt3', 'gpt4', 'chatgpt']:
            predicted_plausible_story_letter = plausibility_generated_text.split(' ')[1][:1] # Just take second token as the plausible story
        elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            response = response_extractor(plausibility_generated_text, type='plausibility')
            predicted_plausible_story_letter = response['plausibility']['story_converted']
        prediction_obj['plausible_story'] = predicted_plausible_story_letter
        prediction_obj['generated_text_plausibility'] = plausibility_generated_text.split('\n')

        # If LM was correct, separately ask for the evidence for the end task prediction
        prediction_obj['confl_pairs'] = []
        prediction_obj['physical_states'] = []
        acc, cons, ver = 0, 0, 0
        if check_response(plausibility_generated_text, story_pair, demo_type="plausibility"):
            # Picked the correct story!
            acc = 1

            # STEP 2: Prompt LM for which sentences are conflicting
            if args.lm_backbone in ['gpt3']:
                conflict_prompt = conflict_demos
                conflict_prompt += "\n" + story_pair_prompt_generator(story_pair)
                prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
                conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['gpt4', 'chatgpt']:
                conflict_prompt = []
                conflict_prompt.append(get_chat_message("system", preamble_conflict))
                conflict_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + conflict_demos))
                conflict_prompt.append(get_chat_message("user", story_pair_prompt_generator(story_pair)))
                prediction_obj['input_prompt_conflict'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in conflict_prompt]
                conflict_generated_text = prompt_chat_gpt_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                conflict_prompt = conflict_demos
                conflict_prompt += "\n" + story_pair_prompt_generator(story_pair)
                if args.output_attn:
                    separations = separation_generator(conflict_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer) 
                    separations[0][1] = separations[0][1][:-1]
                    separations[1][1][-1] = separations[0][1][-1]
                else:
                    separations = None
                conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=128, output_attn=args.output_attn, separations=separations)
            response = response_extractor(conflict_generated_text, type='conflict')
            prediction_obj['conv_sentence'] = response['conflict']['state_converted_to']
            prediction_obj['generated_text_conflict'] = conflict_generated_text

            if check_response(conflict_generated_text, story_pair, demo_type="conflict"):
                # Picked the correct conflicting sentences!
                cons = 1
                print("Cons Selected ID:", i)

                # STEP 3: Prompt LM for physical states in conflicting sentences
                if args.lm_backbone in ['gpt3']:
                    physical_states_prompt = physical_states_demos
                    physical_states_prompt += "\n" + story_pair_prompt_generator(story_pair)
                    prediction_obj['input_prompt_physical_states'] = physical_states_prompt.split('\n')
                    physical_states_generated_text = prompt_gpt3_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ['gpt4', 'chatgpt']:
                    physical_states_prompt = []
                    physical_states_prompt.append(get_chat_message("system", preamble_physical_states))
                    physical_states_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + physical_states_demos))
                    physical_states_prompt.append(get_chat_message("user", story_pair_prompt_generator(story_pair)))
                    prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in physical_states_prompt]
                    physical_states_generated_text = prompt_chat_gpt_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)    
                elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                    physical_states_prompt = physical_states_demos
                    physical_states_prompt += "\n" + story_pair_prompt_generator(story_pair)
                    if args.output_attn:
                        separations = separation_generator(physical_states_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer)
                        separations[0][1] = separations[0][1][:-1]
                        separations[1][1][-1] = separations[0][1][-1]
                    else:
                        separations = None
                    physical_states_generated_text = prompt_llama_with_caching(model, tokenizer, physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=128, output_attn=args.output_attn, separations=separations)  
                response = response_extractor(physical_states_generated_text, type='physical_states')
                prediction_obj['participant_converted_to'] = response['physical_states']['participant_converted_to']
                prediction_obj['generated_text_physical_states'] = physical_states_generated_text

                if check_response(physical_states_generated_text, story_pair, demo_type="physical_states"):
                    # Picked the correct physical states!
                    ver = 1
                    print("Verif Selected ID:", i)
                    # print("Correct!")
    

    # Start testing
    predictions = []
    plausibility_score, conflict_score, physical_states_score = 0, 0, 0

    correct_story_total_attn = 0
    correct_sentence_total_attn = 0
    incorrect_story_total_attn = 0
    incorrect_sentence_total_attn = 0
    num_correct_story = 0
    num_correct_sentence = 0

    story_attn_threshold = 0.1
    num_story_attn_exceed_threshold = 0
    sentence_attn_threshold = 0.1
    num_sentence_attn_exceed_threshold = 0

    num_true_positive_second_level = 0
    num_true_negative_second_level = 0
    num_false_positive_second_level = 0
    num_false_negative_second_level = 0
    num_true_positive_third_level = 0
    num_true_negative_third_level = 0
    num_false_positive_third_level = 0
    num_false_negative_third_level = 0

    # i = 32 # pick one (second level)
    i = 56 # pick one (third level)

    print("participant converted:", test_dataset[i]['participant_converted'])
    print("conversion:", test_dataset[i]['conversions'])

    example_id = str(test_dataset[i]['example_id'])

    prediction_obj = {'example_id': example_id}
    story_pair = test_dataset[i]
    if story_pair['story_converted'] == 'A':
        story_converted = story_pair["story_A_sentences"]
    else:
        story_converted = story_pair["story_B_sentences"]
    story_prompt = story_pair_prompt_generator(story_pair)

    # STEP 1: Prompt LM for which story is plausible
    if args.lm_backbone in ['gpt3']:
        plausibility_prompt = plausibility_demos
        plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
        prediction_obj['input_prompt_plausibility'] = plausibility_prompt.split('\n')
        plausibility_generated_text = prompt_gpt3_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
    elif args.lm_backbone in ['gpt4', 'chatgpt']:
        plausibility_prompt = []
        plausibility_prompt.append(get_chat_message('system', preamble_plausibility))
        plausibility_prompt.append(get_chat_message('system', "Here are some examples:\n\n" + plausibility_demos))
        plausibility_prompt.append(get_chat_message('user', story_pair_prompt_generator(story_pair)))
        prediction_obj['input_prompt_plausibility'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in plausibility_prompt]
        plausibility_generated_text = prompt_chat_gpt_with_caching(plausibility_prompt, args, lm_cache, example_id + '_plausibility', token_counts=TOKEN_COUNTS)
    elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
        plausibility_prompt = plausibility_demos
        plausibility_prompt += "\n" + story_pair_prompt_generator(story_pair)
        if args.output_attn:
            separations = separation_generator(plausibility_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer)
            separations[0][1] = separations[0][1][:-1]
            separations[1][1][-1] = separations[0][1][-1]
        else:
            separations = None
        plausibility_generated_text = prompt_llama_with_caching(model, tokenizer, plausibility_prompt, args, lm_cache, example_id + '_plausibility', max_tokens=128, output_attn=args.output_attn, separations=separations)
    if args.lm_backbone in ['gpt3', 'gpt4', 'chatgpt']:
        predicted_plausible_story_letter = plausibility_generated_text.split(' ')[1][:1] # Just take second token as the plausible story
    elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
        response = response_extractor(plausibility_generated_text, type='plausibility')
        predicted_plausible_story_letter = response['plausibility']['story_converted']
    prediction_obj['plausible_story'] = predicted_plausible_story_letter
    prediction_obj['generated_text_plausibility'] = plausibility_generated_text.split('\n')

    # If LM was correct, separately ask for the evidence for the end task prediction
    prediction_obj['confl_pairs'] = []
    prediction_obj['physical_states'] = []
    acc, cons, ver = 0, 0, 0
    if check_response(plausibility_generated_text, story_pair, demo_type="plausibility"):
        # Picked the correct story!
        plausibility_score += 1
        acc = 1

        # STEP 2: Prompt LM for which sentences are conflicting
        if args.lm_backbone in ['gpt3']:
            conflict_prompt = conflict_demos
            conflict_prompt += "\n" + story_pair_prompt_generator(story_pair)
            prediction_obj['input_prompt_conflict'] = conflict_prompt.split('\n')
            conflict_generated_text = prompt_gpt3_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['gpt4', 'chatgpt']:
            conflict_prompt = []
            conflict_prompt.append(get_chat_message("system", preamble_conflict))
            conflict_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + conflict_demos))
            conflict_prompt.append(get_chat_message("user", story_pair_prompt_generator(story_pair)))
            prediction_obj['input_prompt_conflict'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in conflict_prompt]
            conflict_generated_text = prompt_chat_gpt_with_caching(conflict_prompt, args, lm_cache, example_id + '_conflict', token_counts=TOKEN_COUNTS)
        elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
            conflict_prompt = conflict_demos
            conflict_prompt += "\n" + story_pair_prompt_generator(story_pair)
            if args.output_attn:
                separations = separation_generator(conflict_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer) 
                separations[0][1] = separations[0][1][:-1]
                separations[1][1][-1] = separations[0][1][-1]
            else:
                separations = None
            conflict_generated_text = prompt_llama_with_caching(model, tokenizer, conflict_prompt, args, lm_cache, example_id + '_conflict', max_tokens=128, output_attn=args.output_attn, separations=separations)
        response = response_extractor(conflict_generated_text, type='conflict')
        prediction_obj['conv_sentence'] = response['conflict']['state_converted_to']
        prediction_obj['generated_text_conflict'] = conflict_generated_text

        if check_response(conflict_generated_text, story_pair, demo_type="conflict"):
            # Picked the correct conflicting sentences!
            conflict_score += 1
            cons = 1

            # STEP 3: Prompt LM for physical states in conflicting sentences
            if args.lm_backbone in ['gpt3']:
                physical_states_prompt = physical_states_demos
                physical_states_prompt += "\n" + story_pair_prompt_generator(story_pair)
                prediction_obj['input_prompt_physical_states'] = physical_states_prompt.split('\n')
                physical_states_generated_text = prompt_gpt3_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)
            elif args.lm_backbone in ['gpt4', 'chatgpt']:
                physical_states_prompt = []
                physical_states_prompt.append(get_chat_message("system", preamble_physical_states))
                physical_states_prompt.append(get_chat_message("system", "Here are some examples:\n\n" + physical_states_demos))
                physical_states_prompt.append(get_chat_message("user", story_pair_prompt_generator(story_pair)))
                prediction_obj['input_prompt_aep'] = [get_chat_message(message['role'], message['content'].split('\n')) for message in physical_states_prompt]
                physical_states_generated_text = prompt_chat_gpt_with_caching(physical_states_prompt, args, lm_cache, example_id + '_physical_states', token_counts=TOKEN_COUNTS)    
            elif args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                physical_states_prompt = physical_states_demos
                physical_states_prompt += "\n" + story_pair_prompt_generator(story_pair)
                if args.output_attn:
                    separations = separation_generator(physical_states_demos + "\n", story_pair_prompt_generator(story_pair), tokenizer)
                    separations[0][1] = separations[0][1][:-1]
                    separations[1][1][-1] = separations[0][1][-1]
                else:
                    separations = None
                physical_states_generated_text = prompt_llama_with_caching(model, tokenizer, physical_states_prompt, args, lm_cache, example_id + '_physical_states', max_tokens=128, output_attn=args.output_attn, separations=separations)  
            response = response_extractor(physical_states_generated_text, type='physical_states')
            prediction_obj['participant_converted_to'] = response['physical_states']['participant_converted_to']
            prediction_obj['generated_text_physical_states'] = physical_states_generated_text

            if check_response(physical_states_generated_text, story_pair, demo_type="physical_states"):
                # Picked the correct physical states!
                physical_states_score += 1
                ver = 1
                # print("Correct!")

    predictions.append(prediction_obj)
    correct_story, correct_sentence, correct_physical_states = acc, cons, ver
    if correct_story:
        num_correct_story += 1
        all_conflict_attn = lm_cache[str(example_id) + "_conflict_attn"]
        story_A_conflict_attn = np.array(slice_4d_list(all_conflict_attn, 0))
        story_B_conflict_attn = np.array(slice_4d_list(all_conflict_attn, 1))
        story_A_conflict_attn_mean = np.mean(story_A_conflict_attn[30:50], axis=0)
        story_B_conflict_attn_mean = np.mean(story_B_conflict_attn[30:50], axis=0)
        if story_pair['story_converted'] == 'A':
            second_level_converted_story_attn_mean = np.mean(story_A_conflict_attn_mean[0])
            second_level_irrelevant_story_attn_mean = np.mean(story_B_conflict_attn_mean[0])
            # print(second_level_converted_story_attn_mean, second_level_irrelevant_story_attn_mean)
        else:
            second_level_converted_story_attn_mean = np.mean(story_B_conflict_attn_mean[0])
            second_level_irrelevant_story_attn_mean = np.mean(story_A_conflict_attn_mean[0])
            # print(second_level_converted_story_attn_mean, second_level_irrelevant_story_attn_mean)
        if second_level_converted_story_attn_mean > story_attn_threshold:
            num_story_attn_exceed_threshold += 1
            if correct_sentence:
                num_true_positive_second_level += 1
            else:
                num_false_positive_second_level += 1
        else:
            if correct_sentence:
                num_false_negative_second_level += 1
            else:
                num_true_negative_second_level += 1
        correct_story_total_attn += second_level_converted_story_attn_mean
        incorrect_story_total_attn += second_level_irrelevant_story_attn_mean
        if correct_sentence:
            print("second level correct")
            num_correct_sentence += 1
            sentence_index = story_pair["conversions"][0]["state_converted_to"] - 1
            all_physical_states_attn = lm_cache[str(example_id) + "_physical_states_attn"]
            story_A_physical_states_attn = np.array(slice_4d_list(all_physical_states_attn, 0))
            story_B_physical_states_attn = np.array(slice_4d_list(all_physical_states_attn, 1))
            story_A_physical_states_attn_mean = np.mean(story_A_physical_states_attn[30:50], axis=0)
            story_B_physical_states_attn_mean = np.mean(story_B_physical_states_attn[30:50], axis=0)
            if story_pair['story_converted'] == 'A':
                third_level_relevant_story_attn = story_A_physical_states_attn_mean[0]
            else:
                third_level_relevant_story_attn = story_B_physical_states_attn_mean[0]
            third_level_converted_sentence_attn = third_level_relevant_story_attn[sentence_index]
            # if correct_physical_states: print("correct physical states")
            if third_level_converted_sentence_attn > sentence_attn_threshold:
                num_sentence_attn_exceed_threshold += 1
                if correct_physical_states:
                    num_true_positive_third_level += 1
                else:
                    num_false_positive_third_level += 1
            else:
                if correct_physical_states:
                    num_false_negative_third_level += 1
                else:
                    num_true_negative_third_level += 1
            third_level_irrelevant_sentences_attn_mean = np.mean(np.delete(third_level_relevant_story_attn, sentence_index))
            correct_sentence_total_attn += third_level_converted_sentence_attn
            incorrect_sentence_total_attn += third_level_irrelevant_sentences_attn_mean

        story_A_attention_second_level_scaled = (np.mean(story_A_conflict_attn_mean[0]) / (np.mean(story_A_conflict_attn_mean[0]) + np.mean(story_B_conflict_attn_mean[0]))) * 100
        story_B_attention_second_level_scaled = (np.mean(story_B_conflict_attn_mean[0]) / (np.mean(story_A_conflict_attn_mean[0]) + np.mean(story_B_conflict_attn_mean[0]))) * 100

        story_pair["story_A_sentences"] = remove_empty_strings(story_pair["story_A_sentences"])
        story_pair["story_B_sentences"] = remove_empty_strings(story_pair["story_B_sentences"])

        if correct_sentence:
            sentences_attention_third_level = np.concatenate((story_A_physical_states_attn_mean[0], story_B_physical_states_attn_mean[0]))
            sentences_attention_third_level_scaled = sentences_attention_third_level * 100
        else:
            print("second level incorrect")
            sentences_attention_third_level_scaled = np.zeros(len(story_pair["story_A_sentences"]) + len(story_pair["story_B_sentences"]))

        story_pair_lines = story_prompt.split('\n')
        for i in range(0, len(story_pair_lines)):
            if story_pair_lines[i] == "1. PLants have roots.":
                story_pair_lines[i] = "1. Plants have roots."
        if story_pair_lines[-1] == '':
            story_pair_lines = story_pair_lines[:-1]
        if story_pair['story_converted'] == 'A':
            story_converted_identifier = 'A'
        else:
            story_converted_identifier = 'B'
        attention_list_second_level = []
        attention_list_third_level = []
        context_string = ""
        for char in story_pair_lines[0]:
            context_string += char
            attention_list_second_level.append(0) # account for story A identifier
            attention_list_third_level.append(0) # account for story A identifier
        context_string += '\n'
        attention_list_second_level.append(0) # account \n
        attention_list_third_level.append(0) # account \n
        for i in range(1, len(story_pair["story_A_sentences"]) + 1):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_A_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 1])
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n
        for char in story_pair_lines[len(story_pair["story_A_sentences"]) + 1]:
            context_string += char
            attention_list_second_level.append(0) # account for story B identifier
            attention_list_third_level.append(0) # account for story B identifier
        context_string += '\n'
        attention_list_second_level.append(0) # account \n
        attention_list_third_level.append(0) # account \n
        for i in range(2 + len(story_pair["story_A_sentences"]), 2 + len(story_pair["story_A_sentences"]) + len(story_pair["story_B_sentences"])):
            for char in story_pair_lines[i]:
                context_string += char
                attention_list_second_level.append(story_B_attention_second_level_scaled)
                attention_list_third_level.append(sentences_attention_third_level_scaled[i - 2])
            context_string += '\n'
            attention_list_second_level.append(0) # account \n
            attention_list_third_level.append(0) # account \n
        for char in story_pair_lines[2 + len(story_pair["story_A_sentences"]) + len(story_pair["story_B_sentences"])]:
            context_string += char
            attention_list_second_level.append(0) # account for what happened...
            attention_list_third_level.append(0) # account for what happened...

        if not os.path.exists("attention_visualization_results"):
            os.makedirs("attention_visualization_results")
          
        # generate_heatmap_upon_text(context_string, attention_list_second_level, "attention_visualization_results/propara_baseline_second_level_attn_vis.tex", color='red')
        generate_heatmap_upon_text(context_string, attention_list_third_level, "attention_visualization_results/propara_baseline_third_level_attn_vis.tex", color='red')