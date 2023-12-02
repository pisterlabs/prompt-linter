import json
import os
from openai_query import openai_chat
from topological_sorting import generate_graph, topological_sort
from model_nodes_edges import load_nodes_edges, get_genes
from file_io import read_system_json, get_root_path, get_model_directory_path, write_system_tsv
from gene_feature import summarize_gene_feature, summarized_gene_feature_to_tsv
from pages_txt import read_system_page, write_system_page
from chatgpt_prompts import create_music_2_chatGPT_prompt_text, create_chatGPT_prompt_parent, concat_children_summary, estimate_tokens
from ontology_modify import find_children

# set up the openai api
## load the config file 
####remember to change config if you want to use different map/model
with open('config.json') as config_file:
    data = json.load(config_file)

#`MODEL_ANNOTATION_ROOT` is the path to the root directory of the model annotation repository
os.environ['MODEL_ANNOTATION_ROOT'] = data["MODEL_ANNOTATION_ROOT"]

# load the API key
key = data["OPENAI_API_KEY"] 
temperature = data["TEMP"] # Set your temperature here 
# CH: I want it to be deterministic, so I set temperature to 0
max_tokens = data["MAX_TOKENS"] # Set your max tokens here
rate_per_token = data["RATE_PER_TOKEN"]# Set your rate per token here 
model = data["GPT_MODEL"]
DOLLAR_LIMIT = data["DOLLAR_LIMIT"]  # Set your dollar limit here
logfile_name = data["LOG_NAME"] # Set your log file name here
# set the context for music2
context = data["MUSIC2_CONTEXT"]

# set the model name and version, canbe found in the directory
model_name = data["MAP_NAME"]

version = data["MAP_V"]

file_name = data["MAP_FILE"]

## remember to change the name of the log file 
LOG_FILE = os.path.join(get_model_directory_path(model_name, version), f'{logfile_name}log.json')

## load the model 
nodes, edges = load_nodes_edges(model_name, version, file_name)
# print(nodes.head()), print(edges.head())


# load parent group 
with open(os.path.join(get_model_directory_path(model_name, version),'parent_nodes_by_group.json'), 'r') as f:
    parent_group = json.load(f)

# get the root node info
root_node_info = read_system_json(model_name, version, 'root_node', 'my_gene', get_root_path())
## get the sorted nodes 
graph = generate_graph(edges)
sorted_nodes = topological_sort(graph)

huge_token_nodes = []
for system in sorted_nodes:
    # print(system)
        # check if is leaf node 
    if system not in edges['parent'].values:
        print(f"generating prompt for {system}")
        #run prompt generation and collect the responses
        genes = get_genes(system, nodes)
        # print(genes)

        summarized_info = summarize_gene_feature(root_node_info, genes)  # Summarize the gene features
        summarized_tsv = summarized_gene_feature_to_tsv(summarized_info) 
        # print(summarized_tsv)
        write_system_tsv(summarized_tsv, model_name, version, system, 'go_summary', get_root_path())
        prompt = create_music_2_chatGPT_prompt_text(system,nodes, summarized_tsv) # not html

        est_tokens = estimate_tokens(context + '\n' +prompt)

        print (f"Estimated number of tokens: {est_tokens}")
        if est_tokens > 5500:
                huge_token_nodes.append(system)
                continue
        write_system_page(prompt, model_name, version, system, "chatgpt_prompt", get_root_path())
        
        response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")
        print(response_path)
        # run chatgpt if response file does not exist, avoid duplicate runs
        if not os.path.exists(response_path+ '.txt'):
            print(f"Running chatgpt for {system}")
            # TODO: run chatgpt
            prompt = read_system_page(model_name, version, system, "chatgpt_prompt", get_root_path())
            # print(prompt)
            response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
            if response_text:
                with open(response_path + '.txt', "w") as f:
                    f.write(response_text)    
        else:
            print(f"Chatgpt response file exists for {system}, skipping")
        continue
    # check if is parent node
    if system in edges['parent'].values:
        # print(f"{system} is a parent node")
        # get child list 
        children = find_children(system, edges)
        # print(children)
        if len(children) >=20:
            print(f"{system} is a parent node with more than 20 children")
            continue
        elif system in parent_group['group_1']:
            # print(f"{node} is a parent node from group 1")
            # TODO: get prompt for unique genes 
            unique_genes = nodes.loc[nodes['term'] == system, 'unique_genes'].values[0].split()

            summarized_info = summarize_gene_feature(root_node_info, unique_genes)  # Summarize the gene features
            summarized_tsv = summarized_gene_feature_to_tsv(summarized_info) 
            write_system_tsv(summarized_tsv, model_name, version, system, 'unique_gene_go_summary', get_root_path())
            prompt = create_music_2_chatGPT_prompt_text(system,nodes, summarized_tsv)

            # TODO: concate results from children
            children_summary = concat_children_summary(model_name, version, children, nodes)
            prompt += children_summary
            est_tokens = estimate_tokens(context + '\n' +prompt)
            print (f"Estimated number of tokens: {est_tokens}")
            if est_tokens > 5500:
                huge_token_nodes.append(system)
                continue
            # print (prompt)
            write_system_page(prompt, model_name, version, system, "chatgpt_prompt", get_root_path())
            response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")

        # run chatgpt if response file does not exist, avoid duplicate runs
            if not os.path.exists(response_path + '.txt'):
                print(f"running Chatgpt for {system}")
                # TODO: run chatgpt
                prompt = read_system_page(model_name, version, system, "chatgpt_prompt", get_root_path())
                # print(prompt)
                response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
                if response_text:
                    with open(response_path + '.txt', "w") as f:
                        f.write(response_text)    
            else:
                print(f"Chatgpt response file exists for {system}, skipping")

        elif system in parent_group['group_2'] or system in parent_group['group_3'] or system in parent_group['big_leaf']:
            # print(f"{system} is a parent node of {children}")
        #     # print(f"{system} is a parent node from group 2 or 3, or is a big leaf)

            # TODO: concate results from all children including step children
            prompt = create_chatGPT_prompt_parent()
            children_summary = concat_children_summary(model_name, version, children, nodes)
            prompt += children_summary
            est_tokens = estimate_tokens(context + '\n' +prompt)
            print (f"Estimated number of tokens: {est_tokens}")
            if est_tokens > 5500:
                huge_token_nodes.append(system)
                continue
            # print (prompt)
            write_system_page(prompt, model_name, version, system, "chatgpt_prompt", get_root_path())
            
            response_path = os.path.join(get_model_directory_path(model_name, version), system, f"{system}_chatgpt_response")
        # run chatgpt if response file does not exist, avoid duplicate runs
            if not os.path.exists(response_path + '.txt'):
                print(f"running Chatgpt for {system}")
                # TODO: run chatgpt
                prompt = read_system_page(model_name, version, system, "chatgpt_prompt", get_root_path())
                # print(prompt)
                response_text = openai_chat(context, prompt, model,temperature, max_tokens, rate_per_token, LOG_FILE, DOLLAR_LIMIT)
                if response_text:
                    with open(response_path + '.txt', "w") as f:
                        f.write(response_text)    
            else:
                print(f"Chatgpt response file exists for {system}, skipping")
            continue

if huge_token_nodes:
    with open(os.path.join(get_model_directory_path(model_name, version), 'huge_token_nodes_tocheck.txt'), 'w') as f:
        f.write('\n'.join(huge_token_nodes))
        



