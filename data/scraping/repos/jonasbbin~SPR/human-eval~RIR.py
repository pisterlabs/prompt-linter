import os
import openai
import numpy as np
import json
import asyncio
import math
from human_eval.data import write_jsonl, read_problems
import random
from tqdm import tqdm
import re
import ast
import tiktoken
import time

from task_init import HumanEvalTaskInit
from task_iterate import HumanEvalTaskIterate
from feedback import HumanEvalFeedback
from human_eval.evaluation import evaluate_one_sample

#Parameters
openai.api_key ="Enter API key here"
WAITING_MINUTES = 2
K=0 
INITIAL_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."
DEV_SET_SIZE = 10
FILENAME = "outputs/test.jsonl"
TEMPERATURE=1
TOP_P=1


async def extract_correct_code(prompt):
    """
    Extracts the solution from the answer
    """
    pattern = r'`python(.*?)`'
    matches = re.findall(pattern, prompt, re.DOTALL)
 
    extracted_answer = ""
    #Often the whole solution code of the response is at the end
    for match in matches:
        try:
            code = match.strip()
            ast.parse(code)
            if('def ' in code):
                if ('# testing the code' in code):
                    code = code.split('# testing the code')[0]
                extracted_answer = code
        except SyntaxError:
            pass

    if extracted_answer != "":
        if "```python" in extracted_answer:
            pattern = r'```python(.*?)```'
            matches = re.findall(pattern, extracted_answer, re.DOTALL)
            extracted_answer = matches[0]
        return extracted_answer
    if not len(matches): #prompt consists of code
        try:
            if('def ' in prompt):
                ast.parse(prompt.strip())
                return prompt
        except SyntaxError:
            pass
    #If everything else fails, ask ChatGPT to exract the answer, rarely occurs  
    SYSTEMQ = "You are supposed to extract the python code from a message. The message maybe includes the reasoning of the code creation and maybe some examples. Just return the python code for the completed function."
    messages = [
    {"role": "system", "content": SYSTEMQ},
    {"role": "user", "content": prompt}
    ]

    answer =await handle_prompt(message=messages)

    if "```python" in answer:
            pattern = r'```python(.*?)```'
            matches = re.findall(pattern, answer, re.DOTALL)
            answer = matches[0]
    return answer

async def count_tokens(prompt, encoding_name="cl100k_base"):
    """
    counts the number of tokens inside a prompt
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(prompt))

async def handle_prompt(message, max_retries = 1, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    """
    Handles the various execptions that occour during prompting. Easy interface to prompt
    """
    mx_retries = max_retries
    retries = 0
    error_occured = True #Makes it easier to handle, gets set to False before first call
    await asyncio.sleep(1)
    while (error_occured and retries<=mx_retries):
        error_occured = False

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
        except openai.error.InvalidRequestError as e:
            print(f"Invalid Request. Probably exceeded maximum token size. Changing to larger context model Error Code {e.code}")
            retries += 1
            error_occured = True
            engine = "gpt-3.5-turbo-16k"

        except openai.error.APIError as e:
            print(f"Another API error: {type(e)} with code {e.code}")
            error_occured = True
            retries += 1

        except openai.error.ServiceUnavailableError as e:
            print(f"Server Unavailable. Error Code {e.code}")
            retries += 1
            error_occured = True
      
        except Exception as e:
            print(f"Got some error when Calling the engine {e}")
            retries+= 1
            error_occured = True

    if (error_occured): #Got no answer
        return ""
    answer = c["choices"][0]["message"]["content"]
    return answer

async def afind_prompt_and_solve(train_prompt, problem, history, instruction ,engine="gpt-3.5-turbo", temperature=1, top_p=1):
    '''
    Given a problem, returns the most likely answer determined by the GPT engine. Runs asynchronically
    Find better prompt and test it. Uses the history. Runs asynchronically
    '''
    prompt ="conversation history: "+ history 
    
    instruction = "You are given a conversation history. The conversation history is of the entity that tries to solve a problem. It gives you an idea how the entity thinks and approaches a problem. Give the entity some advice on how it should solve future problems. The entity will be tested on different programming problems which are not the same as the one provided in the conversation history. Just write one instructive sentence." 
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]

    instruction = await handle_prompt(message=messages, temperature=temperature, top_p=top_p)
    if instruction==None:
            print("Encountered Error, Ignoring this test case")
            return ""
    
    return instruction


async def refine_with_feedback(history, official_solution,task_id,  problem, temperature=1, top_p=1):
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


async def a_self_reprompt(train_prompt, problem, instruction, task_id, official_solution, model="gpt-3.5-turbo", max_attempts = 2, temperature=1, top_p=1):
    """
    Perform the self-refinement step
    """
    try:
        task_init = HumanEvalTaskInit(model,train_prompt)

        task_iterate = HumanEvalTaskIterate(model, train_prompt)

        task_feedback = HumanEvalFeedback(model, train_prompt)

        n_attempts = 0
        solved = False
        
        initial_solution = ""
        curr_feedback=""
        while n_attempts <= max_attempts:

            if (n_attempts ==0 ):
                curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature=temperature, top_p=top_p)
                initial_solution = curr_solution
            else:
                curr_solution, history = await task_iterate(problem=curr_feedback, history=history, temperature=temperature, top_p=top_p)

            if curr_solution==None:
                print("Encountered Error, Ignoring this test case")
                return ""
            
            #No need for feedback if we are finished afterwards, saves a bit of prompting
            if (n_attempts < max_attempts):
                curr_feedback, history, solved = await task_feedback(problem=curr_solution, history=history, temperature=temperature, top_p=top_p)
            
            if curr_feedback==None:
                print("Encountered No feedback error, Ignoring this test case")
                return ""
            
            if solved:
                break
            n_attempts += 1
        code_solution = await extract_correct_code(curr_solution)
        if not (evaluate_one_sample(task_id, code_solution)):
            history = await refine_with_feedback(history=history, official_solution=official_solution, task_id=task_id, problem=problem, temperature=temperature, top_p=top_p)
        
        reprompted_prompt = await afind_prompt_and_solve(train_prompt=train_prompt, problem=problem, history=history, instruction=instruction, temperature=temperature, top_p=top_p)

    except Exception as e:
        print("An error occured in self refinement. Ignoring test case")
        #raise e
        return ""
    return reprompted_prompt#[reprompt_solution, curr_solution]

async def make_one_prompt(train_prompt, problem, instruction, task_id, model="gpt-3.5-turbo"):
    """
    Make one prompt and evaluate it directly
    """
    try:
        task_init = HumanEvalTaskInit(model,train_prompt)

        initial_solution = ""
        
        curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature=0)
        initial_solution = curr_solution
        initial_solution = await extract_correct_code(initial_solution)
        
        evaluation = evaluate_one_sample(task_id, initial_solution)
    except Exception:
        print("An error occured in making one_prompt. Ignoring test case")
        return [False, task_id]
    
    return [evaluation, task_id]

async def test_dataset(train_prompt, problem, instruction, pbar, model="gpt-3.5-turbo"):
    """
    Test Benchmark with the given Instruction
    """
    try:
        task_init = HumanEvalTaskInit(model,train_prompt)

        random_minutes = random.randint(0,WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        initial_solution = ""
        
        curr_solution, history = await task_init(problem=problem, instruction=instruction, temperature=0)
        initial_solution = curr_solution
        initial_solution = await extract_correct_code(initial_solution)
        
        
    except Exception:
        print("An error occured. Ignoring test case")
        pbar.update(1)
        return [""]
    pbar.update(1)
    return [initial_solution]

train_prompt = " "

async def opro_find_prompt():
    meta_instruction = open('meta_prompt_OPRO.txt').read()
    messages = [
            {"role": "user", "content": meta_instruction}
            ]
    answer = await handle_prompt(messages)
    print("Answer:", answer)
    if answer.startswith(('[')) and answer.endswith((']')):
        answer =  answer[1:-1]
    print("Extraced:" ,answer)

    instruction = answer
    return instruction

async def run():
    problems = read_problems()
    
    instruction = INITIAL_INSTRUCTION
    with tqdm(total= K, desc="Updates Completed") as pbar:
        for i in range(K):
            
            dev_set = read_problems("test", DEV_SET_SIZE)
            old_num_wrong = 0
            old_instruction = instruction
            tasks = []
            wrong_cases = []
            num_wrong = 0
            print("Current instruction: ", instruction)
            for task_id in dev_set:
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, dev_set[task_id]["prompt"],instruction, task_id)))
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    wrong_cases.append(task[1]) #append ID
                    old_num_wrong += 1
            
            if (wrong_cases==[]):
                break
            random.shuffle(wrong_cases)
            instruction = await a_self_reprompt(train_prompt=train_prompt, problem=dev_set[wrong_cases[0]]["prompt"] , task_id=wrong_cases[0], official_solution=dev_set[wrong_cases[0]]["canonical_solution"] , instruction=instruction, max_attempts=4, temperature=TEMPERATURE, top_p=TOP_P)

            tasks = []
            #Check if really better, OPTIONAL
            for task_id in dev_set:
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, dev_set[task_id]["prompt"],instruction, task_id)))
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    num_wrong += 1
            print("old instruction failures: ", old_num_wrong)
            print("new instruction failures: ", num_wrong)
            if old_num_wrong < num_wrong:
                instruction = old_instruction
            pbar.update(1)
            #Make sure to not overwhelm the 90k Threshold
            await asyncio.sleep(50)

    samples = []
    tasks = []
    print("Chosen instruction: ", instruction)
    #OPRO
    with tqdm(total= len(problems), desc="Prompts Completed") as pbar:
        for task_id in problems:
            tasks.append(asyncio.create_task(test_dataset(train_prompt, problems[task_id]["prompt"],instruction, pbar)))
        tasks = await asyncio.gather(*tasks)
    
    for idx, task_id in enumerate(problems):
        samples.append(dict(task_id=task_id, completion=tasks[idx][0], instruction=instruction))
                       
    write_jsonl(FILENAME, samples)

if __name__ == "__main__":
    
    asyncio.run(run())