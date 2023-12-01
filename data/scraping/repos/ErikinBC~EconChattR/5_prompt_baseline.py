"""
Compare the 'Russ-like' answers between models
"""

import os
import openai
import pandas as pd
# Cost data
from cost import custom, existing
from params import prompts, models, di_completion_params
from utils import n_tokens, set_openai_keys, find_finetuned_models

# Make the folder output if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Set up keys
set_openai_keys()
di_finetuned_models = find_finetuned_models()

# Get a list of the all the current engines
engine_list = pd.Series([d['id'] for d in openai.Engine.list()['data']]).sort_values().reset_index(drop=True)
print(engine_list[~engine_list.str.contains('\\-',regex=True)].to_string(index=False))


##########################################
# ---- (1) ESTIMATE INFERENCE COSTS ---- #

# Calculate the number of tokens in each prompt
n_tok_prompts = [n_tokens(p) for p in prompts]
n_tok_total = sum(n_tok_prompts)
# Calculate the cost of all the prompts by model
total_cost_existing = (existing * n_tok_total)['Usage'].sum()
total_cost_custom = (custom * n_tok_total)['Usage'].sum()

print(f'Cost to run all prompts:\nExisting models: ${total_cost_existing:.3f}\nCustom models: ${total_cost_custom:.3f}')


##########################################
# ---- (2) RUN BASELINE EXPERIMENTS ---- #

holder = []
for model in models:
    print(f'Running model: {model}')
    for prompt in prompts:
        # Set up the completion parameters
        di_completion_params['model'] = model
        di_completion_params['prompt'] = prompt
        # Run the query
        completion = openai.Completion.create(**di_completion_params)
        res = {'model': model, 'prompt': prompt, 'completion': completion['choices'][0]['text']}
        holder.append(res)        
# Merge results into a dataframe
res_bl = pd.DataFrame(holder)
res_bl['completion'] = res_bl['completion'].str.split(pat='\\n{2}',n=1,expand=True)[1]
# Save results
res_bl.to_csv(os.path.join('output','results_baseline.csv'),index=False)


print('~~~ END OF 5_prompt_baseline.py ~~~')