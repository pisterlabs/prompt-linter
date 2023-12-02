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

bias_name = "frequency"

def evaluate_frequency(N, evaluator, instructions, reference, responses, eval_gen):
    true_order = f"n15_evaluations_{bias_name}/nC2_true_order_{evaluator}.json"
    preferences = f"n15_evaluations_{bias_name}/nC2_preferences_{evaluator}.json"
    stats = f"n15_evaluations_{bias_name}/nC2_statistics_{evaluator}.json"
    log_responses = f"n15_evaluations_{bias_name}/nC2_eval_gens_order_{evaluator}.json"
    
    frequency_bias = 0
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
                if evaluator in models:
                    me_compared += 1
                
                order = {"System Star": model1, "System Square": model2}
                
                
                star_mix = ["System Star", "System Star", "System Star", "System Square"]
                square_mix = ["System Square", "System Square", "System Square", "System Star"]
                # mix = random.choice([star_mix, square_mix])
                
                random.shuffle(star_mix)
                random.shuffle(square_mix)
                
                response1, response2 = responses[order["System Star"]][index], responses[order['System Square']][index]
                
                inp = star_mix[0] + ": " + responses[order[star_mix[0]]][index]
                val_inp = square_mix[0] + ": " + responses[order[square_mix[0]]][index]
                
                for l in range(1, len(star_mix)):
                    inp += "\n" + star_mix[l] + ": " + responses[order[star_mix[l]]][index]
                    val_inp += "\n" + square_mix[l] + ": " + responses[order[square_mix[l]]][index]
                
                prompt = uniform_prompt_func(instructions[index], reference[index], inp)
                val_prompt = uniform_prompt_func(instructions[index], reference[index], val_inp)
                
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
                    evaluation = random.choice(star_mix)
                    validation = random.choice(square_mix)

                preference = process_generation(evaluation, instructions[index], reference[index], inp, response1, response2)                
                val_preference = process_generation(validation, instructions[index], reference[index], val_inp, response1, response2)
                
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
                    # check for order bias 
                    if val == "System Square":
                        frequency_bias += 1
                        models.append("frequency")
                    # needs to have no order bias to validate me bias
                    # if val is invalid response, can validate post inference with inconsistent tag
                    elif models[0] == evaluator:
                        me_bias += 1
                    # log the true order
                    txo.write(json.dumps({"model": models[0], "combination": models}) + "\n")
                elif pf == "System Square":
                    rankings[models[1]] += 1    
                    # check for order bias
                    # if val == "System Star":
                    #     frequency_bias += 1  
                    #     models.append("frequency")
                    if models[1] == evaluator:
                        me_bias += 1  
                    txo.write(json.dumps({"model": models[1], "combination": models}) + "\n")
                else:
                    txo.write(json.dumps({"model": "Invalid response", "combination": models}) + "\n")

            txo.write("\n")
            pw.write(json.dumps(rankings) + "\n") 
            
        total_comparisons = N * comb(len(keys), 2)
        wr.write("Frequency bias: " + str(frequency_bias / total_comparisons) + "\n")
        # wr.write("Me bias: " + str(me_bias / me_compared) + "\n")    
        wr.write("Valid response percentage: " + str(valid_responses / total_comparisons) + "\n") 
        wr.write("Valid responses: " + str(valid_responses) + "\n")      
        wr.write("Consistency percentage: " + str(consistency / total_comparisons) + "\n")
        wr.write("Consistency count: " + str(consistency) + "\n")