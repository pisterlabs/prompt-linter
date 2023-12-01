from tqdm import tqdm
import random
import json
import itertools
from math import comb
import re

from utils import guidance_uniform_chat, uniform_prompt_func, guidance_uniform_completion, process_generation, call_guidance, guidance_models
from utils import guidance_uniform_chat_compassion, guidance_uniform_completion_compassion, uniform_prompt_func_compassion
from utils import v_models, get_model_output

random.seed(939)

def evaluate_compassion(N, evaluator, instructions, reference, responses, eval_gen):
    true_order = f"n15_evaluations_compassion/nC2_true_order_{evaluator}.json"
    preferences = f"n15_evaluations_compassion/nC2_preferences_{evaluator}.json"
    stats = f"n15_evaluations_compassion/nC2_statistics_{evaluator}.json"
    log_responses = f"n15_evaluations_compassion/nC2_eval_gens_order_{evaluator}.json"
    
    first_order_bias = 0
    last_order_bias = 0 
    me_bias = 0
    me_compared = 0    
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
                if evaluator in models or model1 in evaluator or model2 in evaluator:
                    me_compared += 1
                
                if model1 == "evaluator":
                    model1 += " (You)"
                elif model2 == evaluator:
                    model2 += " (You)"
                
                # shuffle the models for certain bias tests
                # random.shuffle(models)
                response1, response2 = responses[models[0]][index],  responses[models[1]][index]
                
                inp = "Model " + model1 + ": " + response1 + "\n" + "Model " + model2 + ": " + response2
                val_inp = "Model " + model2 + ": " + response2 + "\n" + "Model " + model1 + ": " + response1
                
                prompt = uniform_prompt_func_compassion(instructions[index], reference[index], inp)
                val_prompt = uniform_prompt_func_compassion(instructions[index], reference[index], val_inp)
                
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
                        evaluation = call_guidance(eval_gen, instruction=instructions[index], input=inp, reference=reference[index])
                        validation = call_guidance(eval_gen, instruction=instructions[index], input=val_inp, reference=reference[index])
                else:
                    evaluation = random.choice(models)
                    validation = random.choice(models)

                preference = process_generation(evaluation, instructions[index], reference[index], inp, response1, response2)                
                val_preference = process_generation(validation, instructions[index], reference[index], val_inp, response1, response2)
                
                if count % 50 == 0:
                    lr.write("========================Generation for [" + ", ".join(models) + f"] for instance {index} ============================\n")
                    lr.write("---------RAW GENERATION--------\n" + evaluation + "\n")
                    lr.write("---------PATTERN MATCHED-------\n" + preference + "\n")

                model1 = model1.replace(" (You)", "")
                model2 = model2.replace(" (You)", "")
                pattern = f"(?i)(model {model1}|model {model2})"
                
                if re.findall(pattern, preference):
                    pf = re.findall(pattern, preference)[0].lower() 
                else:
                    pf = None
                if re.findall(pattern, val_preference):
                    val = re.findall(pattern, val_preference)[0].lower()
                else:
                    val = None
                
                # check for valid second-time response for reversed order
                if pf is not None:
                    valid_responses += 1
                    if val is not None:
                        consistency += 1
                    else: 
                        models.append("inconsistent")
                
                match1 = f"model {model1}".lower()
                match2 = f"model {model2}".lower()
                
                if pf == match1:
                    rankings[models[0]] += 1
                    # check for order bias 
                    if val == match1:
                        first_order_bias += 1
                        models.append("fo bias")
                    # needs to have no order bias to validate me bias
                    # if val is invalid response, can validate post inference with inconsistent tag
                    elif models[0] == evaluator:
                        me_bias += 1
                    # log the true order
                    txo.write(json.dumps({"model": models[0], "combination": models}) + "\n")
                elif pf == match2:
                    rankings[models[1]] += 1    
                    # check for order bias
                    if val == match2:
                        last_order_bias += 1  
                        models.append("lo bias")
                    elif models[1] == evaluator:
                        me_bias += 1  
                    txo.write(json.dumps({"model": models[1], "combination": models}) + "\n")
                else:
                    txo.write(json.dumps({"model": "Invalid response", "combination": models}) + "\n")

            txo.write("\n")
            pw.write(json.dumps(rankings) + "\n") 
            
            
        total_comparisons = N * comb(len(keys), 2)
        wr.write("First order percentage: " + str(first_order_bias / total_comparisons) + "\n")
        wr.write("Last order percentage: " + str(last_order_bias / total_comparisons) + "\n")
        wr.write("Me bias (for compassion): " + str(me_bias / me_compared) + "\n")    
        wr.write("Valid response percentage: " + str(valid_responses / total_comparisons) + "\n") 
        wr.write("Valid responses: " + str(valid_responses) + "\n")      
        wr.write("Consistency percentage: " + str(consistency / total_comparisons) + "\n")
        wr.write("Consistency count: " + str(consistency) + "\n")
