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
import pickle
# from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.model import load_model

from data.propara import story_pair_prompt_generator, top_down_demo_full, bottom_up_demo_full, load_propara_dataset
from eval.propara import check_response, response_extractor
from models import get_chat_message, API_COSTS, prompt_gpt3_with_caching, prompt_llama_with_caching, prompt_chat_gpt_with_caching, prompt_fastchat_with_caching, VICUNA13B_PATH, ALPACA13B_PATH, get_65b_device_mapping
from utils import get_output_dir_name
from visualization import separation_generator

import pandas as pd

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
    parser.add_argument("--lm_backbone", default="llama7b", choices=["gpt3", "gpt4", "chatgpt", "alpaca13b", "vicuna13b", "llama7b", "llama13b", "llama30b", "llama65b"])
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--local_only", action="store_true", default=False)
    parser.add_argument("--demo_choice", default="stories-4", choices=["stories-4", "balanced-6"])
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--cache_only", action="store_true", default=False)
    parser.add_argument("--analyze_attention", action="store_true", default=False)

    parser.add_argument("--action_type", type=str, default="conversion", choices=["conversion", "move"])

    parser.add_argument("--reasoning_depth", choices=["accurate", "consistent", "verifiable"], default='verifiable')
    parser.add_argument("--reasoning_direction", choices=["top-down"], default='top-down')
    parser.add_argument("--output_attn", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, default="saved_results")
    parser.add_argument("--debug", action='store_true', help="Whether to run the model on only a small amount of data.")
    parser.add_argument("--skip_prompting", action='store_true', help="Whether to NOT prompt any LMs, just generate prompts and inspect them.")
    args = parser.parse_args()

    LM_BACKBONE_ENGINES = {'gpt4': 'gpt-4', 'chatgpt': 'gpt-35-turbo', 'alpaca13b': 'alpaca13b'}

    action_types = [args.action_type]
    dataset = load_propara_dataset(action_types=action_types)
    train_dataset = dataset[args.action_type]["train"]
    test_dataset = dataset[args.action_type]["test"]
    if args.debug:
        test_dataset = test_dataset[:5]

    # CACHE_FNAME = "/home/sstorks/data/sstorks/MultimodalImaginationForPhysicalCausality/cache_files/trip_lm_cache_gpt4_verifiable_top-down.pkl"
    CACHE_FNAME = f'cache_files/propara_lm_cache_{args.lm_backbone}_{args.demo_choice}_{args.reasoning_depth}_{args.reasoning_direction}.pkl'
    print("Using CACHE_FNAME:", CACHE_FNAME)
    if os.path.exists(CACHE_FNAME):
        lm_cache = pickle.load(open(CACHE_FNAME, 'rb'))
    else:
        lm_cache = {}
    print(len(lm_cache))
    args.cache_fname = CACHE_FNAME

    TOKEN_COUNTS = {'prompt': 0, 'gen': 0}


    if args.lm_backbone in ["vicuna13b"]:
        print("Setting up Vicuna...")
        if not args.skip_prompting:
            model, tokenizer = load_model(
                VICUNA13B_PATH,
                "cuda",
                torch.cuda.device_count(),
                load_8bit=False,
            )
        else:
            model, tokenizer = None, None

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

    
    if args.lm_backbone in ["gpt4", "chatgpt", "vicuna13b"]:
        preamble = "You are a smart chatbot capable of physical commonsense reasoning, and understanding the physical state of the world implied in written texts. "
        preamble += "You will read stories that involve physical actions, and will be asked to "
        
        # Generate preamble message for chat models
        if args.reasoning_depth == 'accurate':
            # If aiming for accurate reasoning, we don't need to care about reasoning direction (there's only one step)
            preamble += (
                "determine which of two stories is more plausible by comparing possible conflicts within them. "
            )
        else:
            # Otherwise, build the description of the task in the appropriate order
            if args.reasoning_direction == "bottom-up":
                # For bottom-up mode, task is posed as physical states -> conflicts -> plausibility
                if args.reasoning_depth == "verifiable":
                    # If targeting verifiable reasoning, describe physical state prediction then conflict detection
                    preamble += (
                        "report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
                        "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
                        "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
                        "You will then use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will determine which of two stories is more plausible by comparing possible conflicts within them. "
                    )
                elif args.reasoning_depth == "consistent":
                    # If targeting consistent reasoning, only describe conflict detection
                    preamble += (
                        "use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will determine which of two stories is more plausible by comparing possible conflicts within them. "
                    )
            elif args.reasoning_direction == "top-down":
                # For top-down mode, task is posed as plausibility -> conflicts -> physical states
                if args.reasoning_depth == "verifiable":
                    # If targeting verifiable reasoning, describe physical state prediction then conflict detection
                    preamble += (
                        "determine which of two stories is more plausible by comparing possible conflicts within them. "
                        "You will then use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                        "Lastly, you will report the physical states of entities in each story by answering questions about them before and after different actions are taken in the story. "
                        "The story may not explicitly mention state changes. In these cases, you have to use commonsense reasoning to infer the state changes. "
                        "However, you will be penalized if you incorrectly predict an entity's physical state, so you should only predict states you're absolutely sure of based on the text. "
                    )
                elif args.reasoning_depth == "consistent":
                    # If targeting consistent reasoning, only describe conflict detection
                    preamble += (
                        "determine which of two stories is more plausible by comparing possible conflicts within them. "
                        "Lastly, you will use your understanding of the physical state changes in each story to locate a plausibility conflict. "
                    )


    # Demos for action precondition prediction (APP) and action effect prediction (AEP)
    # app_demos = generate_app_demos(train_dataset)
    # aep_demos = generate_aep_demos(train_dataset)

    if args.demo_choice == "stories-4":
        selected_story_example_ids = [200, 68, 107, 311]
    # elif args.demo_choice == "balanced-6":
    #     # sample 2 stories for 3 types of conflict
    #     selected_story_example_ids = balanced_sample_story_ids(train_dataset, select_n_stories_for_each_category=2) # If encounters any error, change a random seed!
    else:
        raise Exception("demo_choice out of options")
    
    print("selected story IDs:", selected_story_example_ids)

    story_demos = ""
    for selected_story_id in selected_story_example_ids:
        story_pair = train_dataset[selected_story_id]
        if args.reasoning_direction == 'top-down':
            story_demos += top_down_demo_full(story_pair) + '\n'
        elif args.reasoning_direction == 'bottom-up':
            story_demos += bottom_up_demo_full(story_pair) + '\n'
        else:
            raise Exception(f"reasoning_direction: {args.reasoning_direction} not supported yet")
    
    # print(story_demos)

   # Start testing

    attention_analysis_table = {}

    attention_analysis_table["layers"] = ["2nd Level Attention Ratio", "3rd Level Attention Ratio", "2nd Level Precision", "2nd Level Recall", "3rd Level Precision", "3rd Level Recall"]

    story_attn_thresholds = [0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12]
    sentence_attn_thresholds = [0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12]

    attention_analysis_metrics_over_thresholds = []

    if args.analyze_attention:
    
        for i in range(0, len(story_attn_thresholds)):

            story_attn_threshold = story_attn_thresholds[i]
            sentence_attn_threshold = sentence_attn_thresholds[i]

            predictions = []
            plausibility_score, conflict_score, physical_states_score = 0, 0, 0

            correct_story_total_attn = 0
            correct_sentence_total_attn = 0
            incorrect_story_total_attn = 0
            incorrect_sentence_total_attn = 0
            num_correct_story = 0
            num_correct_sentence = 0

            num_story_attn_exceed_threshold = 0
            num_sentence_attn_exceed_threshold = 0

            num_true_positive_second_level = 0
            num_true_negative_second_level = 0
            num_false_positive_second_level = 0
            num_false_negative_second_level = 0
            num_true_positive_third_level = 0
            num_true_negative_third_level = 0
            num_false_positive_third_level = 0
            num_false_negative_third_level = 0

            for i in tqdm(range(0, len(test_dataset))):

                example_id = test_dataset[i]['example_id']

                prediction_obj = {'example_id': example_id}
                story_pair = test_dataset[i]
                if story_pair['story_converted'] == 'A':
                    story_converted = story_pair["story_A_sentences"]
                else:
                    story_converted = story_pair["story_B_sentences"]
                story_prompt = story_pair_prompt_generator(story_pair)

                # Prompt the LM
                if args.lm_backbone in ['gpt3']:
                    prompt = ""
                    prompt += story_demos + "\n" + story_prompt
                    # Record prompt used with LM (splitting by newlines for readability)
                    prediction_obj['input_prompt'] = prompt.split('\n')
                    max_tokens=128
                    story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=max_tokens, token_counts=TOKEN_COUNTS)

                if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                    if args.output_attn:
                        separations = separation_generator(story_demos + "\n", story_prompt, tokenizer) 
                        separations[0][1] = separations[0][1][:-1]
                        separations[1][1][-1] = separations[0][1][-1]
                    else:
                        separations = None
                    prompt = ""
                    prompt += story_demos + "\n" + story_prompt
                    # Record prompt used with LM (splitting by newlines for readability)
                    prediction_obj['input_prompt'] = prompt.split('\n')
                    max_tokens=128
                    story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, str(example_id), max_tokens=max_tokens, output_attn=args.output_attn, separations=separations)
                    # if args.output_attn:
                    #     attn = lm_cache[str(example_id)+'_attn']

                elif args.lm_backbone in ['gpt4', "chatgpt", "vicuna13b"]:
                    # Generate chat model messages           
                    preamble_message = get_chat_message("system", preamble, model_name=args.lm_backbone)

                    story_demos_message = (
                        get_chat_message(
                            "system",
                            "And here are some examples of how you can use your understanding to solve the two-story task that the user will give you:\n\n" + story_demos,
                            model_name=args.lm_backbone
                        )
                    )

                    # Compile messages based on passed in arguments
                    messages = [preamble_message]

                    # # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                    # if args.reasoning_depth == "verifiable":
                        # Add some demos of APP and AEP specifically
                        # app_demos_message = (
                        #     get_chat_message(
                        #         "system",
                        #         "Here are some examples of how you can understand physical *precondition* states in text:\n\n" + app_demos,
                        #         model_name=args.lm_backbone
                        #     )                
                        # )
                        # aep_demos_message = (
                        #     get_chat_message(
                        #         "system",
                        #         "Here are some examples of how you can understand physical *effect* states in text:\n\n" + aep_demos,
                        #         model_name=args.lm_backbone
                        #     )
                        # )
                        # messages += [app_demos_message, aep_demos_message]

                    messages += [story_demos_message]
                    messages += [
                        get_chat_message(
                            "user",
                            story_prompt,
                            model_name=args.lm_backbone
                        )
                    ]
                    # Record prompt used with LM (splitting all messages by newlines for readability)
                    if args.lm_backbone in ['gpt4', 'chatgpt']:
                        prediction_obj['input_prompt'] = [get_chat_message(
                                                            message['role'], 
                                                            message['content'].split('\n'),
                                                            model_name=args.lm_backbone,
                                                        ) for message in messages]
                    elif args.lm_backbone in ['vicuna13b']:
                        prediction_obj['input_prompt'] = [get_chat_message(
                                                            message[0], 
                                                            message[1].split('\n'),
                                                            model_name=args.lm_backbone,
                                                        ) for message in messages]
                    
                    if args.lm_backbone in ['gpt4', 'chatgpt']:
                        story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128, token_counts=TOKEN_COUNTS)
                    elif args.lm_backbone in ['vicuna13b']:
                        story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128)

                # Gather up model predictions using templates and regular expressions
                prediction_obj['generated_text'] = story_generated_text.split('\n')
                # print("generated text:", prediction_obj['generated_text'])
                if args.reasoning_direction == 'top-down':
                    response = response_extractor(story_generated_text, type='top_down')
                    prediction_obj['response'] = response['top_down']
                    acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='top_down')
                    plausibility_score += acc
                    conflict_score += cons
                    physical_states_score += ver
                    if (acc, cons, ver) == (1, 1, 1):
                        # print("Correct!")
                        pass
                elif args.reasoning_direction == 'bottom-up':
                    response = response_extractor(story_generated_text, type='bottom_up')
                    prediction_obj['response'] = response['bottom_up']
                    acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='bottom_up')
                    plausibility_score += acc
                    conflict_score += cons
                    physical_states_score += ver
                    if (acc, cons, ver) == (1, 1, 1):
                        # print("Correct!")
                        pass
                else:
                    raise NotImplementedError("Bottom-up reasoning not implemented yet")   
                predictions.append(prediction_obj)
                correct_story, correct_sentence, correct_physical_states = acc, cons, ver
                if correct_story:
                    num_correct_story += 1
                    all_attn = lm_cache[str(example_id) + "_attn"]
                    story_A_attn = np.array(slice_4d_list(all_attn, 0))
                    story_B_attn = np.array(slice_4d_list(all_attn, 1))
                    # print("story_A_attn.shape:", story_A_attn.shape)
                    # print("story_B_attn.shape:", story_B_attn.shape)
                    story_A_attn_mean = np.mean(story_A_attn[30:50], axis=0)
                    story_B_attn_mean = np.mean(story_B_attn[30:50], axis=0)
                    # print("story_A_attn_mean.shape:", story_A_attn_mean.shape)
                    # print("story_B_attn_mean.shape:", story_B_attn_mean.shape)
                    # print(story_A_attn_mean.shape[1], len(story_pair["story_A_sentences"]))
                    # print(story_B_attn_mean.shape[1], len(story_pair["story_B_sentences"]))
                    if story_pair['story_converted'] == 'A':
                        second_level_converted_story_attn_mean = np.mean(story_A_attn_mean[1])
                        second_level_irrelevant_story_attn_mean = np.mean(story_B_attn_mean[1])
                        # print(second_level_converted_story_attn_mean, second_level_irrelevant_story_attn_mean)
                        third_level_converted_story_attn = story_A_attn_mean[2]
                    else:
                        second_level_converted_story_attn_mean = np.mean(story_B_attn_mean[1])
                        second_level_irrelevant_story_attn_mean = np.mean(story_A_attn_mean[1])
                        # print(second_level_converted_story_attn_mean, second_level_irrelevant_story_attn_mean)
                        third_level_converted_story_attn = story_B_attn_mean[2]
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
                    sentence_index = story_pair["conversions"][0]["state_converted_to"] - 1
                    # print("sentence_index:", sentence_index)
                    third_level_converted_sentence_attn = third_level_converted_story_attn[sentence_index]
                    third_level_irrelevant_sentences_attn_mean = mean = np.mean(np.delete(third_level_converted_story_attn, sentence_index))
                    if correct_sentence:
                        num_correct_sentence += 1
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
                        correct_sentence_total_attn += third_level_converted_sentence_attn
                        incorrect_sentence_total_attn += third_level_irrelevant_sentences_attn_mean

            precision_second_level = num_true_positive_second_level / (num_true_positive_second_level + num_false_positive_second_level)
            recall_second_level = num_true_positive_second_level / (num_true_positive_second_level + num_false_negative_second_level)
            f1_second_level = 2 * ((precision_second_level * recall_second_level) / (precision_second_level + recall_second_level))
            precision_third_level = num_true_positive_third_level / (num_true_positive_third_level + num_false_positive_third_level)
            recall_third_level = num_true_positive_third_level / (num_true_positive_third_level + num_false_negative_third_level)
            f1_third_level = 2 * ((precision_third_level * recall_third_level) / (precision_third_level + recall_third_level))

            # ["story ratio", "sentence ratio", "avg: # story exceeds threshold", "avg: # sentence exceeds threshold", "second level precision", "second level recall", "third level precision", "third level recall"]

            attention_analysis_metrics_over_thresholds.append([correct_story_total_attn / incorrect_story_total_attn, correct_sentence_total_attn / incorrect_sentence_total_attn, precision_second_level, recall_second_level, precision_third_level, recall_third_level])

            if not args.skip_prompting:
                plausibility_score /= len(test_dataset)
                conflict_score /= len(test_dataset)
                physical_states_score /= len(test_dataset)
                metrics = {
                    'accuracy': plausibility_score,
                    'consistency': conflict_score,
                    'verifiability': physical_states_score,
                }
                # pprint(metrics)
                print("Accuracy:", metrics['accuracy'])
                print("Consistency:", metrics['consistency'])
                print("Verifiability:", metrics['verifiability'])

                # Create output directory for results
                instance_timestamp = datetime.datetime.now()
                output_dir, instance_name = get_output_dir_name('propara', args, instance_timestamp, 0, result_type="combined")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                json.dump(predictions, open(os.path.join(output_dir, "preds_test.json"), "w"), indent=4)
                json.dump(metrics, open(os.path.join(output_dir, "metrics_test.json"), "w"), indent=4)

            else:
                if args.lm_backbone in API_COSTS:
                    prompt_cost = TOKEN_COUNTS['prompt'] * API_COSTS[args.lm_backbone][0]
                    gen_cost = TOKEN_COUNTS['gen'] * API_COSTS[args.lm_backbone][1]
                    total_cost = prompt_cost + gen_cost
                else:
                    prompt_cost = 0.0
                    gen_cost = 0.0
                    total_cost = 0.0
                print("PROMPT COST ESTIMATES (EXCLUDING CACHED RESULTS):")
                print("Prompts: %.2f" % prompt_cost)
                print("Generation: %.2f" % gen_cost)
                print("Total: %.2f" % total_cost)
        
        attention_analysis_table_key = str(30) + " to " + str(50)
        attention_analysis_table[attention_analysis_table_key] = np.mean(np.vstack(attention_analysis_metrics_over_thresholds), axis=0)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        df = pd.DataFrame(attention_analysis_table)
        print(df)
    else:
        predictions_bool_list = []
        plausibility_score, conflict_score, physical_states_score = 0, 0, 0
        predictions = []
        for i in tqdm(range(0, len(test_dataset))):

            example_id = test_dataset[i]['example_id']

            prediction_obj = {'example_id': example_id}
            story_pair = test_dataset[i]
            if story_pair['story_converted'] == 'A':
                story_converted = story_pair["story_A_sentences"]
            else:
                story_converted = story_pair["story_B_sentences"]
            story_prompt = story_pair_prompt_generator(story_pair)

            # Prompt the LM
            if args.lm_backbone in ['gpt3']:
                prompt = ""
                prompt += story_demos + "\n" + story_prompt
                # Record prompt used with LM (splitting by newlines for readability)
                prediction_obj['input_prompt'] = prompt.split('\n')
                max_tokens=128
                story_generated_text = prompt_gpt3_with_caching(prompt, args, lm_cache, example_id, max_tokens=max_tokens, token_counts=TOKEN_COUNTS)

            if args.lm_backbone in ['llama7b', 'llama13b', 'llama30b', 'llama65b']:
                if args.output_attn:
                    separations = separation_generator(story_demos + "\n", story_prompt, tokenizer) 
                    separations[0][1] = separations[0][1][:-1]
                    separations[1][1][-1] = separations[0][1][-1]
                else:
                    separations = None
                prompt = ""
                prompt += story_demos + "\n" + story_prompt
                # Record prompt used with LM (splitting by newlines for readability)
                prediction_obj['input_prompt'] = prompt.split('\n')
                max_tokens=128
                story_generated_text = prompt_llama_with_caching(model, tokenizer, prompt, args, lm_cache, str(example_id), max_tokens=max_tokens, output_attn=args.output_attn, separations=separations)
                # if args.output_attn:
                #     attn = lm_cache[str(example_id)+'_attn']

            elif args.lm_backbone in ['gpt4', "chatgpt", "vicuna13b"]:
                # Generate chat model messages           
                preamble_message = get_chat_message("system", preamble, model_name=args.lm_backbone)

                story_demos_message = (
                    get_chat_message(
                        "system",
                        "And here are some examples of how you can use your understanding to solve the two-story task that the user will give you:\n\n" + story_demos,
                        model_name=args.lm_backbone
                    )
                )

                # Compile messages based on passed in arguments
                messages = [preamble_message]

                # # To fully specify physical state prediction step, prepend several demos of it covering all physical attributes first
                # if args.reasoning_depth == "verifiable":
                    # Add some demos of APP and AEP specifically
                    # app_demos_message = (
                    #     get_chat_message(
                    #         "system",
                    #         "Here are some examples of how you can understand physical *precondition* states in text:\n\n" + app_demos,
                    #         model_name=args.lm_backbone
                    #     )                
                    # )
                    # aep_demos_message = (
                    #     get_chat_message(
                    #         "system",
                    #         "Here are some examples of how you can understand physical *effect* states in text:\n\n" + aep_demos,
                    #         model_name=args.lm_backbone
                    #     )
                    # )
                    # messages += [app_demos_message, aep_demos_message]

                messages += [story_demos_message]
                messages += [
                    get_chat_message(
                        "user",
                        story_prompt,
                        model_name=args.lm_backbone
                    )
                ]
                # Record prompt used with LM (splitting all messages by newlines for readability)
                if args.lm_backbone in ['gpt4', 'chatgpt']:
                    prediction_obj['input_prompt'] = [get_chat_message(
                                                        message['role'], 
                                                        message['content'].split('\n'),
                                                        model_name=args.lm_backbone,
                                                    ) for message in messages]
                elif args.lm_backbone in ['vicuna13b']:
                    prediction_obj['input_prompt'] = [get_chat_message(
                                                        message[0], 
                                                        message[1].split('\n'),
                                                        model_name=args.lm_backbone,
                                                    ) for message in messages]
                
                if args.lm_backbone in ['gpt4', 'chatgpt']:
                    story_generated_text = prompt_chat_gpt_with_caching(messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128, token_counts=TOKEN_COUNTS)
                elif args.lm_backbone in ['vicuna13b']:
                    story_generated_text = prompt_fastchat_with_caching(model, tokenizer, messages, args, lm_cache, example_id, max_tokens=1024 if args.reasoning_direction == 'bottom-up' else 128)

            # Gather up model predictions using templates and regular expressions
            prediction_obj['generated_text'] = story_generated_text.split('\n')
            # print("generated text:", prediction_obj['generated_text'])
            if args.reasoning_direction == 'top-down':
                response = response_extractor(story_generated_text, type='top_down')
                prediction_obj['response'] = response['top_down']
                acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='top_down')
                plausibility_score += acc
                conflict_score += cons
                physical_states_score += ver
                if (acc, cons, ver) == (1, 1, 1):
                    # print("Correct!")
                    pass
            elif args.reasoning_direction == 'bottom-up':
                response = response_extractor(story_generated_text, type='bottom_up')
                prediction_obj['response'] = response['bottom_up']
                acc, cons, ver = check_response(story_generated_text, story_pair, demo_type='bottom_up')
                plausibility_score += acc
                conflict_score += cons
                physical_states_score += ver
                if (acc, cons, ver) == (1, 1, 1):
                    # print("Correct!")
                    pass
            else:
                raise NotImplementedError("Bottom-up reasoning not implemented yet")   
            predictions.append(prediction_obj)
        if not args.skip_prompting:
            plausibility_score /= len(test_dataset)
            conflict_score /= len(test_dataset)
            physical_states_score /= len(test_dataset)
            metrics = {
                'accuracy': plausibility_score,
                'consistency': conflict_score,
                'verifiability': physical_states_score,
            }
            # pprint(metrics)
            print("Accuracy:", metrics['accuracy'])
            print("Consistency:", metrics['consistency'])
            print("Verifiability:", metrics['verifiability'])

            # Create output directory for results
            instance_timestamp = datetime.datetime.now()
            output_dir, instance_name = get_output_dir_name('propara', args, instance_timestamp, 0, result_type="combined")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            json.dump(predictions, open(os.path.join(output_dir, "preds_test.json"), "w"), indent=4)
            json.dump(metrics, open(os.path.join(output_dir, "metrics_test.json"), "w"), indent=4)

        else:
            if args.lm_backbone in API_COSTS:
                prompt_cost = TOKEN_COUNTS['prompt'] * API_COSTS[args.lm_backbone][0]
                gen_cost = TOKEN_COUNTS['gen'] * API_COSTS[args.lm_backbone][1]
                total_cost = prompt_cost + gen_cost
            else:
                prompt_cost = 0.0
                gen_cost = 0.0
                total_cost = 0.0
            print("PROMPT COST ESTIMATES (EXCLUDING CACHED RESULTS):")
            print("Prompts: %.2f" % prompt_cost)
            print("Generation: %.2f" % gen_cost)
            print("Total: %.2f" % total_cost)