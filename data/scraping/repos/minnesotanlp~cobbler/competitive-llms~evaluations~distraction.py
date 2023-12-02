from tqdm import tqdm
import random
import json
import itertools
from math import comb
import re

from utils import guidance_uniform_chat, uniform_prompt_func, guidance_uniform_completion, process_generation, call_guidance, guidance_models
from utils import guidance_uniform_chat_distraction, uniform_prompt_func_distraction, guidance_uniform_completion_distraction
from utils import v_models, get_model_output

random.seed(939)

bias_name = "distraction"

def choose_distraction(distractor):
    distractions = [
                    f'{distractor} likes to eat apples and oranges.',
                    f'The coolest thing that {distractor} can do is a 60 second handstand.',
                    f'{distractor} plays a lot of soccer and basketball.',
                    f'{distractor} has been all around Europe two times.'
                ]
    return distractions

def evaluate_distraction(N, evaluator, instructions, reference, responses, eval_gen):
    true_order = f"n15_evaluations_{bias_name}/nC2_true_order_{evaluator}.json"
    preferences = f"n15_evaluations_{bias_name}/nC2_preferences_{evaluator}.json"
    stats = f"n15_evaluations_{bias_name}/nC2_statistics_{evaluator}.json"
    log_responses = f"n15_evaluations_{bias_name}/nC2_eval_gens_bias_{evaluator}.json"
    
    distracted_count = 0
    valid_responses = 0
    consistency = 0
        
    with open(true_order, "w") as txo, open(preferences, "w") as pw, open(stats, "w") as wr, open(log_responses, "w") as lr:
        keys = list(responses.keys())  # Get a list of keys
        
        count = 0
        # Iterate over indices
        for index in tqdm(range(0, N), total=N):
            # Generate unique combinations of items at the same index
            rankings = {}
            for i in keys:
                rankings[i] = 0
                
            item_combinations = list(itertools.combinations(keys, 2))
            random.shuffle(item_combinations)
            
            # Iterate over combinations
            for combination in item_combinations:
                count += 1
                model1, model2 = combination
                
                models = [model1, model2]
                
                order = ["System Star", "System Square"]
                
                # all distractions should be off topic
                distraction_st = random.choice(choose_distraction("System Star"))
                distraction_sq = random.choice(choose_distraction("System Square"))

                # shuffle the models for certain bias tests
                random.shuffle(models)
                response1, response2 = responses[models[0]][index],  responses[models[1]][index]
                
                inp = order[0] + ": " + response1 + "\n" + order[1] + ": " + response2
                prompt = uniform_prompt_func_distraction(instructions[index], reference[index], inp, distraction_st)
                val_prompt = uniform_prompt_func_distraction(instructions[index], reference[index], inp, distraction_sq)
                
                if evaluator != "random":
                    if evaluator not in guidance_models and evaluator not in v_models:
                        # caller
                        evaluation = eval_gen.generate([prompt])[0]['generation']
                        validation = eval_gen.generate([val_prompt])[0]['generation']
                    elif evaluator in v_models:
                        evaluation = get_model_output(evaluator, eval_gen, prompt)
                        validation = get_model_output(evaluator, eval_gen, val_prompt)
                    else:
                        # prompter 
                        evaluation = call_guidance(eval_gen, instruction=instructions[index], input=inp, reference=reference[index], biasing=distraction_st)
                        validation = call_guidance(eval_gen, instruction=instructions[index], input=inp, reference=reference[index], biasing=distraction_sq)
                else:
                    evaluation = random.choice(order)
                    validation = random.choice(order)

                preference = process_generation(evaluation, instructions[index], reference[index], inp, response1, response2)                
                val_preference = process_generation(validation, instructions[index], reference[index], inp, response1, response2)
                
                if count % 50 == 0:
                    lr.write("========================Generation for [" + ", ".join(models) + f"] for instance {index} ============================\n")
                    lr.write("---------RAW GENERATION--------\n" + evaluation + "\n")
                    lr.write("---------PATTERN MATCHED-------\n" + preference + "\n")

                pf = re.findall(r"(?i)(system star|system square)", preference)[0].title() if re.findall(r"(?i)(system star|system square)", preference) else None
                val = re.findall(r"(?i)(system star|system square)", val_preference)[0].title() if re.findall(r"(?i)(system star|system square)", val_preference) else None
                
                # check for valid second-time response for reversed order
                if pf is not None:
                    valid_responses += 1
                    if val is not None:
                        consistency += 1
                    else: 
                        models.append("inconsistent")
                
                if pf == "System Star":
                    rankings[models[0]] += 1
                    # check for bandwagon bias 
                    if val == "System Square":
                        distracted_count += 1
                        models.append("distracted")
                    txo.write(json.dumps({"model": models[0], "combination": models}) + "\n")
                elif pf == "System Square":
                    rankings[models[1]] += 1    
                    # if val == "System Star":
                    #     distracted_count += 1  
                    #     models.append("distracted")
                    txo.write(json.dumps({"model": models[1], "combination": models}) + "\n")
                else:
                    txo.write(json.dumps({"model": "Invalid response", "combination": models}) + "\n")

            txo.write("\n")
            pw.write(json.dumps(rankings) + "\n") 
            
        total_comparisons = N * comb(len(keys), 2)
        wr.write("Distraction: " + str(distracted_count / total_comparisons) + "\n")
        wr.write("Distraction count: " + str(distracted_count) + "\n") 
        wr.write("Valid response percentage: " + str(valid_responses / total_comparisons) + "\n") 
        wr.write("Valid responses: " + str(valid_responses) + "\n")      