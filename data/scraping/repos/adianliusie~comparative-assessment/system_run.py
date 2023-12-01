import os
import argparse
import pickle
import random
import openai
import os

from tqdm import tqdm
from datetime import datetime

from src.data_handler import DataHandler
from src.utils.general import save_json, load_json, load_pickle
from src.models import load_interface
from src.prompts.load_prompt import get_prompt_template
from src.utils.post_processing import save_combined_json, delete_leftover_files

import time

def main(
    system:str,
    output_path:str,
    dataset:str='sumevall',
    score_type:str='consistency',
    prompt_id:int=None,
    shuffle:bool=True,
    comparative:bool=False,
    max_len:int=None,
    device:str=None,
    probs:bool=False,
    num_comparisons:int=None
):
    #load prompt from default, or choose your own prompt
    if prompt_id:
        assert ('c' in prompt_id) == comparative
    
        prompt_template = get_prompt_template(prompt_id, score_type)
        if dataset == 'topicalchat':
            prompt_template = prompt_template.replace('Summary', 'Response')
            prompt_template = prompt_template.replace('summary', 'response')
            prompt_template = prompt_template.replace('Passage', 'Dialogue')
            prompt_template = prompt_template.replace('passage', 'dialogue')
        if dataset == 'podcast':
            prompt_template = prompt_template.replace('Passage', 'Podcast')
            prompt_template = prompt_template.replace('passage', 'podcast')

    else:
        prompt_template = None
    
    if system in ['llama2-7b-chat', 'llama2-13b-chat', 'chatgpt']:
        if comparative:
            prompt_template += '\n\nAnswer:'
        else:
            prompt_template += '\n\nScore:'

    # print path and template
    print('-'*50, '\n', output_path)
    print('-'*50, '\n', prompt_template, '\n', '-'*50)

    # get input text data to feed to chatgpt
    if dataset=='podcast':
        CACHE_PATH = '/rds/project/rds-8YSp2LXTlkY/experiments/al826/comp_eval/cache/'
        if comparative:
            proc_inputs = load_pickle(f"{CACHE_PATH}/podcast-comparative-{prompt_id}.pk")
        else:
            proc_inputs = load_pickle(f"{CACHE_PATH}/podcast-scoring-{prompt_id}.pk")

    else:
        data_handler = DataHandler(prompt_template, dataset=dataset)
        if comparative:
            proc_inputs = data_handler.comparative_texts(score_type)
        else:
            proc_inputs = data_handler.scoring_texts(score_type)

    # create directory if not already existing
    system_output_path = f"{output_path}/outputs"
    if not os.path.isdir(system_output_path):
        os.makedirs(system_output_path)   

    # select the model to run the inputs through
    interface = load_interface(system=system, device=device)
    
    # set decoder_prefix (only used for prob mode) 
    decoder_prefix='Summary'
    if (dataset == 'topicalchat') and (probs):
        decoder_prefix='Response'
    elif (dataset in ['wi-train', 'wi-dev']) and (probs):
        decoder_prefix='Text'

    # save experiment settings 
    info = {
        'prompt_id':prompt_id,
        'prompt':prompt_template,
        'system':system, 
        'dataset':dataset,
        'score_type':score_type,
        'comparative':comparative,
        'max_len':max_len
    }             
            
    info_path = f"{output_path}/info.json"
    if not os.path.isfile(info_path):
        save_json(info, info_path) 
    else:
        log_info = load_json(info_path)
        assert all([info[k] == log_info[k] for k in info.keys() if k != 'dataset'])
        
    # select evaluation order for examples (shuffling allows parallelisation)
    ids = [x for x in range(len(proc_inputs))]
    if shuffle:
        random.shuffle(ids)

    # determine examples already run (and processed)
    done_ids = []
    if os.path.isfile(f"{system_output_path}/combined.json"):
        done_ids = set(list(load_json(f"{system_output_path}/combined.json").keys()))

    # process all inputs to chatgpt
    for k, idx in enumerate(ids):
        # break early if sufficient comparisons have been done
        if num_comparisons and (k + len(done_ids) > num_comparisons):
            break

        ex = proc_inputs[idx]
        outpath = f"{system_output_path}/{ex.ex_id}.json"

        # skip outputs already computed
        if (os.path.isfile(outpath)) or (ex.ex_id in done_ids):
            continue

        elif probs:
            response = interface.prompt_template_response(
                input_text=ex.input_text, 
                decoder_prefix=decoder_prefix
            )
            
        else:
            response = interface.text_response(
                input_text=ex.input_text, 
                do_sample=False,
                max_new_tokens=max_len
            )            

        # get and print generated text
        gen_text = response.output_text
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] {ex.ex_id} : {gen_text}")

        # save output file
        save_json(response.__dict__, outpath)

    save_combined_json(system_output_path)  
    delete_leftover_files(system_output_path)

def generation_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True, help='where to save chatgpt outputs')
    
    parser.add_argument('--system', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--probs', action='store_true', help='whether prompt templates should be used')
    parser.add_argument('--prompt-id', type=str, default=None, help='which prompt to use')

    parser.add_argument('--dataset', type=str, default='summeval', help='which evaluation dataset to use')
    parser.add_argument('--score-type', type=str, default='consistency', help='which score to use of the dataset')
    
    parser.add_argument('--max-len', type=int, default=10, help='number of maximum tokens to be generated')
    parser.add_argument('--num-comparisons', type=int, default=None, help='number of comparisons to do for the dataset')

    parser.add_argument('--device', type=str, default=None, help='device to run experiments')

    parser.add_argument('--shuffle', action='store_true', help='whether to shuffling order of samples')
    parser.add_argument('--comparative', action='store_true', help='whether to do comparative evaluation')
    
    return parser

if __name__ == "__main__":
    parser = generation_parser()
    kwargs = vars(parser.parse_args())
    main(**kwargs)
