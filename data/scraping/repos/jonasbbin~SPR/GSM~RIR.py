import os
import openai
import numpy as np
import json
import asyncio
import math
import re
import random
from tqdm import tqdm
import time
from typing import Iterable, Dict
import gzip
import async_timeout

from task_init import GSMTaskInit
from task_iterate import GSMTaskIterate
from feedback import GSMFeedback
from math_equivalence import is_equiv


#PARAMETERS
openai.api_key ="Enter API key here"
#Directory with the MATH problems
train_prompt = open('prompt_original_shorter.txt').read()

WAITING_MINUTES = 20
ITERATION_WAITING_MINUTES = 1
MAX_NUM_WRONG_CASES = 1
OUTPUTFILE_NAME = "name"
LOG_FILE_NAME = "log/"+ OUTPUTFILE_NAME + ".txt"
K = 15
TEMPERATURE = 1
TOP_P = 1
NUM_SAMPLES = 50# Sample Size of the devset
BENCHMARK_FILE_FULL = "gsm_test_set.jsonl" #1350
BENCHMARK_FILE = "gsm_test_set.jsonl" #1350
#BENCHMARK_FILE = "gsm_test_set_small.jsonl" #150
#BENCHMARK_FILE = "debugging.jsonl" #5


INITIAL_INSTRUCTION = "Follow the given examples and answer the question."

def read_problems(evalset_file: str = BENCHMARK_FILE, test_size = 15) -> Dict[str, Dict]:
    """
    Returns dataset.
    Also returns dev_set of given size if called in test mode
    """
    tasks_list = []
    
    if evalset_file =="test":

        for task in stream_jsonl(BENCHMARK_FILE_FULL):
            tasks_list.append(task)
        # Get a list of all task_ids from the dictionary
        #print(task_ids)
        # Shuffle the task_ids randomly
        random.shuffle(tasks_list)

        # Select the first 15 shuffled task_ids
        selected_task_ids = tasks_list[:test_size]
        
        # Create a new dictionary with the selected tasks
        #selected_tasks_dict = {task_id: tasks_dict[task_id] for task_id in selected_task_ids}
        return selected_task_ids
    else:
        for task in stream_jsonl(BENCHMARK_FILE):
            tasks_list.append(task)

    return tasks_list


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

async def extract_ans(ans_model):
    """
    Extract the final answer of the solution from the language model
    """
    if('The answer is ' in ans_model):
        pred = ans_model.split('The answer is ')[-1].strip().split()[0]
    elif('the answer is ' in ans_model):
        pred = ans_model.split('the answer is ')[-1].strip().split()[0]
    elif('The answer is: ' in ans_model):
        pred = ans_model.split('The answer is ')[-1].strip().split()[0]
    elif('the answer is: ' in ans_model):
        pred = ans_model.split('the answer is: ')[-1].strip().split()[0]
    elif('the correct answer is ' in ans_model):
        pred = ans_model.split('the correct answer is ')[-1].strip().split()[0]
    elif('The correct answer is ' in ans_model):
        pred = ans_model.split('The correct answer is ')[-1].strip().split()[0]
    else:
        #If the answer can't be extracted, maybe ChatGPT finds it
        SYSTEMQ = "Extract the numerical solution from the following message. Return it without any reasoning or other words"
        messages = [
        {"role": "user", "content": SYSTEMQ + ans_model}
        ]

        answer =await handle_prompt(message=messages)

        return answer
    return pred

async def handle_prompt(message, max_retries = 2, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    """
    Handles the various execptions that occour during prompting. Easy interface to prompt
    engine = "gpt-3.5-turbo-16k"
    """
    
    mx_retries = max_retries
    retries = 0
    error_occured = True #Makes it easier to handle, gets set to False before first call

    while (error_occured and retries<=mx_retries):
        error_occured = False

        try:
            async with async_timeout.timeout(60):
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
            #print("The folling message is the problem: ")
            #print(message)
            retries += 1
            error_occured = True
            engine = "gpt-3.5-turbo-16k"
            max_retries = 1
        except Exception as e:
            print(f"Got some error when Calling the engine {e}, error number {retries}")
            retries+= 1
            error_occured = True
            wait = random.randint(1,8)
            await asyncio.sleep(wait)

    if (error_occured): #Got no answer
        return ""
    answer = c["choices"][0]["message"]["content"]
    return answer


async def afind_prompt_and_solve(train_prompt, problem, history ,instruction, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    '''
    Finds better prompt and test it. Uses the history Runs asynchronically
    '''
    old_instruction = instruction
    prompt = "conversation history: " + history  
    
    prompt = " examples :" +train_prompt + "\n" + "conversation history: " + history 

    instruction = "You are given examples and a conversation history. The conversation history is of an entity that tries to solve a problem. It gives you an idea how the entity thinks and approaches a problem. The entity uses the examples and an instructions to solve different problems. Currently the instruction is '{}'.Write a new instruction, which enables the entity to solve more problems than the current one. The entity will be tested on different grade school math word problems which are not the same as the one provided in the conversation history. Refer to the giving examples and use at most 15 words for your instruction".format(old_instruction) 
       
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
    if(': ' in answer):
        answer = answer.split(': ')[-1].strip()
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
        task_init = GSMTaskInit(model,train_prompt)


        task_iterate = GSMTaskIterate(model, train_prompt)

        task_feedback = GSMFeedback(model, train_prompt)

        n_attempts = 0
        solved = False
        initial_solution = ""
        #Simple way of handling the rate limit
        curr_feedback = ""
        while n_attempts < max_attempts:

            if (n_attempts ==0 ):
                curr_solution, history = await task_init(problem=problem, instruction=instruction, n_temperature=temperature, top_p=top_p)
                initial_solution = curr_solution
            else:
                
                curr_solution, history = await task_iterate(problem=curr_feedback, history=history, temperature=temperature, top_p=top_p)

            if curr_solution==None:
                print("Encountered refining Error, Ignoring this test case")
                return None
            
            if (n_attempts < max_attempts):
                curr_feedback, history, solved = await task_feedback(problem=curr_solution, history=history, temperature=temperature, top_p=top_p)
            
            if curr_feedback==None:
                print("Encountered refining Error, Ignoring this test case")
                return None
            
            if solved:
                break
            n_attempts += 1

        refine_solution = await extract_ans(curr_solution)

        
        equiv = is_equiv(refine_solution, str(solution))

        if not equiv:
            history = await refine_with_feedback(history=history, official_solution=str(solution), problem=problem, temperature=temperature, top_p=top_p)
        
        reprompt_solution = await afind_prompt_and_solve(train_prompt=train_prompt, problem=problem, history=history, instruction=instruction, temperature=temperature, top_p=top_p)
        
    except Exception as e:
        print("Ignoring Refining Test case due to error")
        raise e
        return ""
    
    return reprompt_solution


async def test_dataset(train_prompt, problem, pbar, solution, instruction,  model="gpt-3.5-turbo", max_attempts = 3):
    """
    Test the dataset with an instruction
    """
    try: 
        task_init = GSMTaskInit(model,train_prompt)

        
        random_minutes = random.randint(0,WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        curr_solution, history = await task_init(problem=problem, instruction=instruction, n_temperature=0)
        initial_solution = await extract_ans(curr_solution)
    except Exception as e:
        print("Ignoring Test case due to error")
        pbar.update(1)
        #raise e
        return 0
    
    pbar.update(1)
    return initial_solution



async def make_one_prompt(train_prompt, problem, instruction, official_solution, model="gpt-3.5-turbo", name=""):
    """
    Make one prompt and evaluate it directly
    """
    try:
        task_init = GSMTaskInit(model,train_prompt)

        initial_solution = ""
        
        random_minutes = random.randint(0,ITERATION_WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        curr_solution, history = await task_init(problem=problem, instruction=instruction, n_temperature=0)
        initial_solution = curr_solution
        initial_solution= await extract_ans(initial_solution)
        
        equiv = is_equiv(initial_solution, str(official_solution))
    except Exception as e:
        print("An error occured in making one_prompt. Ignoring test case")
        raise e
        return [False, name]
    
    return [equiv, name]

async def one_k_iteration(train_prompt, instruction, num_samples=15):
    """
    Performs one improvement step:
    - Create and evaluate current instruction on new dev set
    - Refine on a wrongly predicted one
    - Evaluete new instruction
    """
    with open(LOG_FILE_NAME, "a") as file:
        instruction = instruction
        dev_set = read_problems("test", num_samples)
        num_wrong_counter = 0
        num_wrong = []
        tasks = []
        wrong_cases = []
        instructions = []
        print("Current instruction: ", instruction)
        for test_case in dev_set:
            tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, test_case["target"],name=test_case)))
        tasks = await asyncio.gather(*tasks)
        for task in tasks:
            if task[0] == False: #Solution was not correct
                wrong_cases.append(task[1]) #append ID
                num_wrong_counter += 1
        if (wrong_cases==[]):
            return instruction
        num_wrong.append(num_wrong_counter)
        instructions.append(instruction)
        random.shuffle(wrong_cases)
        file.write("Old instruction: {} \n".format(instructions[0]))
        file.write("Old instruction error count: {} \n".format(num_wrong[0]))
        file.write("---------------------------------------------\n")
        
        for i in range(np.minimum(len(wrong_cases), MAX_NUM_WRONG_CASES)):
            num_wrong_counter = 0
            test_case = wrong_cases[i]
            instruction = await a_self_reprompt(train_prompt=train_prompt, problem=test_case["input"] , solution=test_case["target"] , instruction=instruction, max_attempts=4, temperature=TEMPERATURE, top_p=TOP_P)

            tasks = []
            #Check if really better, OPTIONAL
            for test_case in dev_set:
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, test_case["target"])))
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    num_wrong_counter += 1
            num_wrong.append(num_wrong_counter)
            instructions.append(instruction)
            
            file.write("New instruction {}: {} \n".format(i+1, instructions[i+1]))
            file.write("New instruction {} error count: {} \n".format(i+1, num_wrong[i+1]))
        file.write("---------------------------------------------\n")
        instruction = instructions[np.argmin(num_wrong)]
        file.write("Mean error count of new instructions: {} \n".format(np.mean(num_wrong[1:])))
        file.write("Variance error count of new instructions: {} \n".format(np.var(num_wrong[1:])))
        file.write("---------------------------------------------\n")
    return instruction

async def run_async(engine="gpt-3.5-turbo", max=-1):
    """
    runs the benchmark
    """
    #Checks all the files at rootdir
    instruction = INITIAL_INSTRUCTION
    outputs = []
    answers = []
    evaluations =  []
    for i in range(K):
        instruction = await one_k_iteration(train_prompt=train_prompt, instruction=instruction, num_samples=NUM_SAMPLES)
        time.sleep(60)
    fnames_list = []

    cors = {}

    correct = 0
    total = 0
    print("Starting evaluation")
    problems = read_problems()
    print("Chosen instruction: ", instruction)
    with tqdm(total= len(problems), desc="Prompts Completed") as pbar:
        for test_case in problems:
            
            solution = test_case["target"]
            model_output = asyncio.create_task(test_dataset(train_prompt, test_case["input"],pbar, solution ,instruction))
            
            outputs.append(model_output)
            answers.append(solution)

        results = await asyncio.gather(*outputs)
    print("Prompting complete")
    outputs = results

    for idx, model_output in enumerate(outputs):
        answer = answers[idx]

        try:
            equiv = is_equiv(model_output, str(answer))
        except Exception as e:
            equiv = False
            raise e
        if equiv:
            correct += 1
            evaluations.append("True")
        else:
            evaluations.append("False")
        total += 1
       
    print("Initial Score:")
    with open("outputs/" + OUTPUTFILE_NAME+ ".txt", "w+") as f:
        for k, (output, answer, evaluation) in enumerate(zip(outputs, answers, evaluations)):
            f.write("{}  OUTPUT: {} | ANSWER: {} | EVALUATION: {}\n".format(k, output, answer, evaluation))
       
        f.write("#####################\n")
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))
        f.write("Used instruction:  {}\n ".format(instruction))


if __name__ == "__main__":
    engines = ["gpt-3.5-turbo"]
    for engine in engines:
        asyncio.run(run_async(engine))