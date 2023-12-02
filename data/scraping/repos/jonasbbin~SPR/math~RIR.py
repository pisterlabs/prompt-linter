import os
import openai
import numpy as np
import json
from dataset.util import last_boxed_only_string
from math_equivalence import is_equiv
import asyncio
import math
import re
import random
from tqdm import tqdm
import time
import async_timeout

from task_init import MathTaskInit
from task_iterate import MathTaskIterate
from feedback import MathFeedback


#Parameters
openai.api_key ="Enter API key here"
#Directory with the MATH problems
rootdir = os.path.join(os.getcwd(), "MATH_sample/test/")
rootdir = "./MATH_sample/test/"
rootdir = "./MATH/test/" #5000 files
#rootdir = "./MATH_sample_medium/test/" #140 files
#rootdir = "./MATH_sample_larger/test/" #294 files
train_prompt = open('complex_math_simpler.txt').read()
WAITING_MINUTES = 100
INITIAL_INSTRUCTION = "Follow the given examples and answer the question."
K = 0
NUM_SAMPLES = 20 #there are 7 directories so multiply this number with seven to get the total size to get the devset
OUTPUT_FILENAME = "outputs/baseline_with_large.txt"
TEMPERATURE = 1
TOP_P = 1
ITERATION_WAITING_MINUTES = 5
async def extract_ans(ans_model):
    """
    Extract the final answer of the solution from the language model
    """
    if('The answer is ' in ans_model):
        pred = ans_model.split('The answer is ')[-1].strip().split()[0]
    elif('the answer is ' in ans_model):
        pred = ans_model.split('the answer is ')[-1].strip().split()[0]
    elif('the correct answer is ' in ans_model):
        pred = ans_model.split('the correct answer is ')[-1].strip().split()[0]
    elif('The correct answer is ' in ans_model):
        pred = ans_model.split('The correct answer is ')[-1].strip().split()[0]
    elif('\\boxed{' in ans_model):
        
        pred =  remove_boxed(last_boxed_only_string(ans_model))
    else:
        #If the answer can't be extracted, maybe ChatGPT finds it
        SYSTEMQ = "You are supposed to extract the numerical solution from a message. Just return the number without any reasoning or other words"
        messages = [
        {"role": "system", "content": SYSTEMQ},
        {"role": "user", "content": ans_model}
        ]

        answer =await handle_prompt(message=messages)

        return answer
    return pred

def remove_boxed(s):
    """
    Removes the boxed string used in the solution
    """
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

async def handle_prompt(message, max_retries = 2, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    """
    Handles the various execptions that occour during prompting. Easy interface to prompt
    engine = "gpt-3.5-tubo-16k"
    """
    
    mx_retries = max_retries
    retries = 0
    error_occured = True #Makes it easier to handle, gets set to False before first call

    while (error_occured and retries<=mx_retries):
        error_occured = False

        async with async_timeout.timeout(60):
            try:
                c = openai.ChatCompletion.acreate(
                                    model=engine,
                                    messages=message,
                                    top_p = top_p,
                                    temperature = temperature
                                )
                c = await c #Waits on the response
            except openai.error.RateLimitError as e:
                print(f"Request exceeded rate limit: Error Code {e.code}")
                retries += 1
                error_occured = True
                raise e
            except openai.error.APIError as e:
                print(f"Another API error: {type(e)} with code {e.code}")
                error_occured = True
                retries += 1
                wait = random.randint(1,8)
                await asyncio.sleep(wait)

            except openai.error.ServiceUnavailableError as e:
                print(f"Server Unavailable. Error Code {e.code}")
                retries += 1
                error_occured = True
                wait = random.randint(1,8)
                await asyncio.sleep(wait)

            except openai.error.InvalidRequestError as e:
                print(f"Invalid Request. Probably exceeded maximum token size. Changing to larger context model Error Code {e.code}")
                retries += 1
                error_occured = True
                engine = "gpt-3.5-turbo-16k"
                max_retries = 1
            except Exception as e:
                print(f"Got some error when Calling the engine {e}")
                retries+= 1
                error_occured = True
                wait = random.randint(1,8)
                await asyncio.sleep(wait)

    if (error_occured): #Got no answer
        return "0"
    answer = c["choices"][0]["message"]["content"]
    return answer


async def afind_prompt_and_solve(train_prompt, problem, history ,instruction, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    '''
    Finds better prompt and test it. Uses the history Runs asynchronically
    '''
    old_instruction = instruction 
    prompt = "conversation history: " + history 
    

    instruction = "You are given a conversation history. The conversation history is of an entity that tries to solve a given problem. It gives you an idea how the entity thinks and approaches a problem. Write a new instruction such that together with some demonstrations and a mathematical problem the entity knows how to solve other mathematical problems. Keep it short and general. The instruction should be able to be applied to many different mathematical problems." 
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]

    answer = await handle_prompt(message=messages, temperature=temperature, top_p=top_p)
    if answer==None:
            print("Encountered Error, Ignoring this test case")
            return ""
    if('New instruction:' in answer):
        answer = answer.split('New instruction:')[-1].strip()
    if('Instruction:' in answer):
        answer = answer.split('Instruction:')[-1].strip()

    return answer

async def refine_with_feedback(history, official_solution, problem, temperature=1, top_p=1):
    """
    If the solution is not found after some iterations, we use the official solution as an additional help
    """

    prompt ="conversation history: "+ history + "\n" + "official solution: " + "\n" + official_solution + "\n" + " problem: " + problem 
    instruction = "You are given a conversation history, the official solution and  a problem. The conversation history is of ChatGPT that tries to solve the given problem. However, it fails to do so. Use the official solution and the conversation history, to find reasons why ChatGPT fails to solve this problem."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]
    refined_with_solution = await handle_prompt(message=messages, temperature=temperature, top_p=top_p)
    history = history + "\n" + "System: After checking the offical solution, it turns out the found solution is wrong." + "\n"
    history = history + "\n" + "Reasons for failure: " + refined_with_solution
    return history

async def a_self_reprompt(train_prompt, instruction, problem, solution, model="gpt-3.5-turbo", max_attempts = 3, temperature=1, top_p=1):
    """
    Perform the self-refinement step and reprompting
    """
    try: 
        task_init = MathTaskInit(model,train_prompt)


        task_iterate = MathTaskIterate(model, train_prompt)

        task_feedback = MathFeedback(model, train_prompt)

        n_attempts = 0
        solved = False
        initial_solution = ""
        #Simple way of handling the rate limit
        curr_feedback = ""
        while n_attempts < max_attempts:

            if (n_attempts ==0 ):
                curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature=temperature, top_p=top_p)
                initial_solution = curr_solution
            else:
                
                curr_solution, history = await task_iterate(problem=curr_feedback, history=history,temperature=temperature, top_p=top_p)

            if curr_solution==None:
                print("Encountered refining Error, Ignoring this test case")
                return None
            
            if (n_attempts < max_attempts):
                curr_feedback, history, solved = await task_feedback(problem=curr_solution, history=history, temperature=temperature, top_p=top_p)
            
            if curr_feedback==None:
                print("Encountered refining Error, Ignoring this test case")
                return None
            
            if solved:
                #print("Problem solution already found. Refining stops")
                break
            n_attempts += 1

        refine_solution = await extract_ans(curr_solution)

        
        equiv = is_equiv(refine_solution, solution)

        if not equiv:
            history = await refine_with_feedback(history=history, official_solution=solution, problem=problem, temperature=temperature, top_p=top_p)
        
        
        reprompt_solution = await afind_prompt_and_solve(train_prompt=train_prompt, problem=problem, history=history, instruction=instruction, temperature=temperature, top_p=top_p)
    
        
    except Exception as e:
        print("Ignoring Refining Test case due to error")
        return ""
    
    return reprompt_solution


async def test_dataset(train_prompt, problem, pbar, solution, instruction,  model="gpt-3.5-turbo", max_attempts = 3):
    """
    Test the dataset with an instruction
    """
    try: 
        task_init = MathTaskInit(model,train_prompt)


        random_minutes = random.randint(0,WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature = 0)
        initial_solution = await extract_ans(curr_solution)
        
    except Exception as e:
        print("Ignoring Test case due to error")
        pbar.update(1)
        print(e)
        return 0
    
    pbar.update(1)
    return initial_solution




def create_dev_set(num_samples=5):
    """
    Find a devset and take the number of samples from each category
    """
    # Initialize a list to store the selected file paths
    selected_files = []

    # Define the number of files you want to select inside each subfolder
    num_files_to_select = num_samples

    # Walk through the root folder and its subfolders
    for folder_path, _, file_names in os.walk(rootdir):
        # Randomly shuffle the list of files in each subfolder
        random.shuffle(file_names)

        # Iterate through the first 'num_files_to_select' files in the shuffled list
        for file_name in file_names[:num_files_to_select]:
            # Create the full path to the selected file
            file_path = os.path.join(folder_path, file_name)
            selected_files.append(file_path)

    return selected_files

async def make_one_prompt(train_prompt, problem, instruction, official_solution, model="gpt-3.5-turbo", name=""):
    """
    Make one prompt and evaluate it directly
    """
    try:
        task_init = MathTaskInit(model,train_prompt)

        initial_solution = ""
        random_minutes = random.randint(0,ITERATION_WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature=0)
        initial_solution = curr_solution
        initial_solution= await extract_ans(initial_solution)
        
        equiv = is_equiv(initial_solution, official_solution)
    except Exception:
        print("An error occured in making one_prompt. Ignoring test case")
        return [False, name]
    
    return [equiv, name]

async def one_k_iteration(train_prompt, instruction, num_samples=5):
    """
    Performs one reprompting step
    """

    instruction = instruction
    dev_set = create_dev_set(num_samples)
    old_num_wrong = 0
    old_instruction = instruction
    tasks = []
    wrong_cases = []
    num_wrong = 0
    print("Current instruction: ", instruction)
    for test_file in dev_set:
        try:
                with open(test_file, 'r') as fp:
                    problem_data = json.load(fp)
        except Exception as e:
                print(f"Error loading JSON from {test_file}", e)
                raise e
        tasks.append(asyncio.create_task(make_one_prompt(train_prompt, problem_data["problem"],  instruction, remove_boxed(last_boxed_only_string(problem_data["solution"])),name=test_file)))
    tasks = await asyncio.gather(*tasks)
    for task in tasks:
        if task[0] == False: #Solution was not correct
            wrong_cases.append(task[1]) #append ID
            old_num_wrong += 1
    
    if (wrong_cases==[]):
        return instruction
    random.shuffle(wrong_cases)
    
    with open(wrong_cases[0], 'r') as fp:
        refining_problem = json.load(fp)
    instruction = await a_self_reprompt(train_prompt=train_prompt, problem=refining_problem["problem"] , solution=refining_problem["solution"] , instruction=instruction, max_attempts=4, temperature=TEMPERATURE, top_p=TOP_P)

    tasks = []
    #Check if really better, OPTIONAL
    for test_file in dev_set:
        try:
                with open(test_file, 'r') as fp:
                    problem_data = json.load(fp)
        except Exception as e:
                print(f"Error loading JSON from {test_file}", e)
                raise e
        tasks.append(asyncio.create_task(make_one_prompt(train_prompt, problem_data["problem"],  instruction, remove_boxed(last_boxed_only_string(problem_data["solution"])))))
    tasks = await asyncio.gather(*tasks)
    for task in tasks:
        if task[0] == False: #Solution was not correct
            num_wrong += 1
    print("old instruction failures: ", old_num_wrong)
    print("new instruction failures: ", num_wrong)
    if old_num_wrong < num_wrong:
        instruction = old_instruction

    return instruction

async def run_async(engine="gpt-3.5-turbo", max=-1):
    #Checks all the files at rootdir
    instruction = INITIAL_INSTRUCTION
    outputs = []
    answers = []
    types = []
    levels = []
    for i in range(K):
        instruction = await one_k_iteration(train_prompt=train_prompt, instruction=instruction, num_samples=NUM_SAMPLES)
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0
    print("Starting evaluation")
    with tqdm(total= 5000, desc="Prompts Completed") as pbar:

        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                fnames_list.append(os.path.join(subdir, file))
                with open(os.path.join(subdir, file), 'r') as fp:
                    try:
                        problem_data = json.load(fp)
                    except Exception as e:
                        print(f"Error loading JSON from {file}", e)
                        raise e
                    prob_level = problem_data["level"]
                    prob_type = problem_data["type"]
                    try:
                        prob_level = int(prob_level.split("Level ")[1])
                    except:
                        prob_level = None

                    answer = remove_boxed(last_boxed_only_string(problem_data["solution"]))
                    model_output = asyncio.create_task(test_dataset(train_prompt, problem_data["problem"], pbar, answer, instruction, model=engine))
                    #This gets the solution from the problem, solution is in this boxed part
                    

                    levels.append(prob_level)
                    types.append(prob_type)
                    outputs.append(model_output)
                    answers.append(answer)

        results = await asyncio.gather(*outputs)
    print("Prompting complete")
    outputs = results

    for idx, model_output in enumerate(outputs):
        answer = answers[idx]
        prob_type = types[idx]
        prob_level = levels[idx]

        try:
            equiv = is_equiv(model_output, answer)
        except:
            equiv = False
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        if equiv:
            correct += 1
        total += 1

            
    print("Initial Score:")
    with open(OUTPUT_FILENAME, "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))
        f.write("Used instruction:  {}\n ".format(instruction))


if __name__ == "__main__":
    engines = ["gpt-3.5-turbo"]
    for engine in engines:
        asyncio.run(run_async(engine))