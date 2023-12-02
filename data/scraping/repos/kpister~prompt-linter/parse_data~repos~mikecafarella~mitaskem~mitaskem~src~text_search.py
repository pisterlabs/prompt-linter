from mitaskem.src.gpt_interaction import *
from openai import OpenAIError
from mitaskem.src.connect import *
import argparse
from mitaskem.src.mira_dkg_interface import *


MAX_TEXT_MATCHES = 2
MAX_DKG_MATCHES = 2


def text_param_search(text, gpt_key):
    try:
        prompt = get_text_param_prompt(text)
        match = get_gpt_match(prompt, gpt_key)
        return match, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def text_var_search(text, gpt_key):
    try:
        prompt = get_text_var_prompt(text)
        match = get_gpt_match(prompt, gpt_key)
        return match, True
    except OpenAIError as err:   
        return f"OpenAI connection error: {err}", False

def vars_dedup(text:str) -> dict:
    var_dict = {}

    lines = text.splitlines()

    # Build dictionary, deduplicating along the way
    for line in lines:
        line = line.strip()
        toks = [t.strip() for t in line.split("|")]

        if len(toks) == 1 or line == "name | description | numerical value":
            continue

        var_name = toks[0]
        var_desc = toks[1]

        if var_name not in var_dict:
            var_dict[var_name] = {'description':[], 'value':None}
        desc_list = var_dict[var_name]['description']
        desc_list.append(var_desc)

        var_dict[var_name]['description'] = desc_list

        if len(toks) > 2:
            var_val = toks[2]
            print(var_name, 'found value', var_val)
            if var_val != "None":
                var_dict[var_name]['value'] = var_val

    return var_dict

def vars_to_json(var_dict: dict) -> str:

    s_out = "["
    is_first = True
    id = 0

    batch_var_ground = batch_get_mira_dkg_term(var_dict.keys(), ['id', 'name'])

    for var_name, var_ground in zip(var_dict, batch_var_ground):
        var_defs_s = "[\"" + '\",\"'.join(i for i in var_dict[var_name][:MAX_TEXT_MATCHES]) + "\"]"
        #var_ground = get_mira_dkg_term(var_name, ['id', 'name'])
        var_ground = var_ground[:MAX_DKG_MATCHES]
        var_ground_s = "[" + ",".join([("[\"" + "\",\"".join([str(item) for item in sublist]) + "\"]") for sublist in var_ground]) + "]"

        if is_first:
            is_first = False
        else:
            s_out += ","

        s_out += "{\"type\" : \"variable\", \"name\": \"" + var_name \
        + "\", \"id\" : \"mit" + str(id) + "\", \"text_annotations\": " + var_defs_s \
        + ", \"dkg_annotations\" : " + var_ground_s + "}"

        id += 1
    
    s_out += "]"

    return s_out

from mitaskem.src.kgmatching import local_batch_get_mira_dkg_term
import json

async def avars_to_json(var_dict: dict, kg_domain: str) -> str:

    s_out = "["
    is_first = True
    id = 0

    term_list = []
    meta_list = []
    for (term_name,term_desc) in var_dict.items():
        gpt_desc = term_desc['description'][0]
        term_list.append(term_name +':' + gpt_desc)
        meta_list.append({'llm_output_name':term_name,
                         'llm_output_desc':gpt_desc})
    
    batch_var_ground0 = local_batch_get_mira_dkg_term(term_list, kg_domain)
    pretty_var0 = json.dumps(batch_var_ground0, indent=2)
    print(f'batch_var_ground\n{pretty_var0}')

    # batch_var_ground = await abatch_get_mira_dkg_term(var_dict.keys(), ['id', 'name'])
    # pretty_var = json.dumps(batch_var_ground, indent=2)
    # print(f'batch_var_ground\n{pretty_var}')

    batch_var_ground = [[[res['id'], res['name']] for res in batch] for batch in batch_var_ground0]

    for var_name, var_ground in zip(var_dict, batch_var_ground):
        var_defs_s = "[\"" + '\",\"'.join(i for i in var_dict[var_name]['description'][:MAX_TEXT_MATCHES]) + "\"]"
        #var_ground = get_mira_dkg_term(var_name, ['id', 'name'])

        var_ground = var_ground[:MAX_DKG_MATCHES]
        var_ground_s = "[" + ",".join([("[\"" + "\",\"".join([str(item) for item in sublist]) + "\"]") for sublist in var_ground]) + "]"

        if is_first:
            is_first = False
        else:
            s_out += ","

        s_out += "{\"type\" : \"variable\", \"name\": \"" + var_name \
        + "\", \"id\" : \"mit" + str(id) + "\", \"text_annotations\": " + var_defs_s \
        + ", \"dkg_annotations\" : " + var_ground_s
        if var_dict[var_name]['value']:
            s_out +=  ", \"value\" : \"" + var_dict[var_name]['value'] + "\""
        s_out += "}"

        id += 1
    
    s_out += "]"

    return s_out

def main(args):
    GPT_KEY = None 

    out_filename_params = args.out_dir + "/" + args.in_path.split("/")[-1].split(".")[0] + "_params.txt"
    out_filename_vars = args.out_dir + "/" + args.in_path.split("/")[-1].split(".")[0] + "_vars.txt"

    with open(args.in_path, "r") as fi, open(out_filename_params, "w+") as fop, open(out_filename_vars, "w+") as fov:
        text = fi.read()
        length = len(text)
        segments = int(length/3500 + 1)

        for i in range(segments):
            snippet = text[i * 3500: (i+1) * 3500]

            output, success = text_param_search(snippet, GPT_KEY)
            if success:
                print("OUTPUT (params): " + output + "\n------\n")  
                if output != "None":
                    fop.write(output + "\n") 

            output, success = text_var_search(snippet, GPT_KEY)
            if success:
                print("OUTPUT (vars): " + output + "\n------\n")  
                if output != "None":
                    fov.write(output + "\n") 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str)
    parser.add_argument("-o", "--out_dir", type=str, default="resources/jan_evaluation/cosmos_params")
    args = parser.parse_args()

    main(args)
