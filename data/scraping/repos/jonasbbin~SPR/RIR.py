# evaluating GPT-3.5 turbo model on BBH

import openai
import json
import os
import numpy as np
import asyncio 
from tqdm import tqdm
import time
import random
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
import async_timeout

from task_init import BBHTaskInit
from task_iterate import BBHTaskIterate
from feedback import BBHFeedback

TASKS = 'all'

#Tests
MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table', 
        'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects', 
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation', 
        'salient_translation_error_detection', 'reasoning_about_colored_objects' 
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding', 
        'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies' 
]

#Parameters
OPENAIKEY = "Enter API key here"
ITERATION_WAITING_MINUTES = 1 #Waiting time during reprompting to avoid rate limit
OUTPUTFILENAME = 'outputs_all/name.txt'
INITIAL_INSTRUCTION = 'Follow the given examples and answer the question.'
K = 6
TEMPERATURE = 1
TOP_P=1
BENCHMARKTASKLENGTH = 7 #Waiting time during testing to avoid rate limit
NUM_SAMPLES = 35 #Size of the Devset
DATASETPATH = 'data' #PATH to dataset

async def handle_prompt(message, max_retries = 2, temperature=1, engine="gpt-3.5-turbo", waiting_minutes = 0, top_p=1):
    """
    Handles the various execptions that occour during prompting. Easy interface to prompt
    """
    if waiting_minutes != 0:
        random_minutes = random.randint(0,waiting_minutes)
        await asyncio.sleep(random_minutes*60)
    mx_retries = max_retries
    retries = 0
    error_occured = True #Makes it easier to handle, gets set to False before first call
    await asyncio.sleep(1)
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
        except openai.error.InvalidRequestError as e:
            print(f"Invalid Request. Probably exceeded maximum token size. Changing to larger context model Error Code {e.code}")
            #print(e)
            retries += 1
            error_occured = True
            if engine == "gpt-3.5-turbo-16k":
                message[1]["content"] =  message[1]["content"] + "Keep your answer short."
            engine = "gpt-3.5-turbo-16k"

        except openai.error.APIError as e:
            print(f"Another API error: {type(e)} with code {e.code}")
            error_occured = True
            retries += 1

        except openai.error.ServiceUnavailableError as e:
            print(f"Server Unavailable. Error Code {e.code}. Error count {retries}")
            retries += 1
            error_occured = True
    
        except Exception as e:
            print(f"Got some error when Calling the engine {e}. Error count {retries}")
            retries+= 1
            error_occured = True

    if (error_occured): #Got no answer
        return " "
    answer = c["choices"][0]["message"]["content"]
    return answer


def extract_ans(ans, mode):
    """
    Extract answer from ChatGPT response
    """
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == 'free_form':
        if ans[-1] == '.':
            ans = ans[:-1]
        return ans


async def afind_prompt_and_solve(train_prompt, problem, history ,instruction, engine="gpt-3.5-turbo", temperature=1, top_p=1):
    '''
    Finds better prompt and test it. Uses the history Runs asynchronically
    '''
    old_instruction = instruction
    prompt = "conversation history: " + history 

    instruction = "You are given a conversation history. The conversation history is of an entity that tries to solve a problem. It gives you an idea how the entity thinks and approaches a problem. The entity uses given examples and an instructions to solve different problems. Currently the instruction is '{}'.Write a new instruction, which enables the entity to solve more problems than the current one. Start your instruction with 'Follow the given examples and answer the question.'. The entity will be tested on different problems which are not the same as the one provided in the conversation history. Just answer the new instruction and keep your answer short.".format(old_instruction) #10 history with 
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]

    answer = await handle_prompt(message=messages, temperature=temperature, top_p=top_p, engine=engine)
    if answer==None:
            print("Encountered Error, Ignoring this test case")
            return ""

    if('New instruction:' in answer):
        answer = answer.split('New instruction:')[-1].strip()
    if('Instruction:' in answer):
        answer = answer.split('Instruction:')[-1].strip()
    if('instruction:' in answer):
        answer = answer.split('instruction:')[-1].strip()
    if(': ' in answer):
        answer = answer.split(': ')[-1].strip()
    answer = answer.strip()
    if answer.startswith(('"', "'")) and answer.endswith(('"', "'")):
        answer =  answer[1:-1]
    else:
        answer = answer
    print("Proposed Prompt: ", answer)
    return answer


async def refine_with_feedback(history, official_solution, problem, temperature=1, top_p=1):
    """
    If the solution is not found after some iterations, we use the official solution as an additional help
    """

    prompt ="conversation history: "+ history + "\n" + "official solution: " + "\n" + official_solution + "\n" + " problem: " + problem 
    instruction = "You are given a conversation history, the official solution and  a problem. The conversation history is of ChatGPT that tries to solve the given problem. However, it fails to do so on the first try. Use the official solution and the conversation history, to find reasons why ChatGPT fails to solve this problem."
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
        ]
    refined_with_solution = await handle_prompt(message=messages, temperature=temperature, top_p=top_p)
    history = history + "\n" + "System: After checking the offical solution, it turns out the provided reasoning is wrong." + "\n"
    history = history + "\n" + "Reasons for failure: " + refined_with_solution
    return history


async def a_self_reprompt(train_prompt, instruction, problem, solution, model="gpt-3.5-turbo", max_attempts = 3, mode='multiple_choice', temperature=1, top_p=1):
    """
    Perform the self-refinement step and reprompting
    """
    try: 
        task_init = BBHTaskInit(model,train_prompt)

        task_iterate = BBHTaskIterate(model, train_prompt)

        task_feedback = BBHFeedback(model, train_prompt)

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
            
        refine_solution = extract_ans(curr_solution, mode=mode)

        if (refine_solution!=solution):
            history = await refine_with_feedback(history=history, official_solution=str(solution), problem=problem, temperature=temperature, top_p=top_p)
    
        reprompt_solution = await afind_prompt_and_solve(train_prompt=train_prompt, problem=problem, history=history, instruction=instruction,  temperature=temperature, top_p=top_p)

    except Exception as e:
        print("Ignoring Refining Test case due to error")
        return ""

    return reprompt_solution

async def create_dev_set(size):
    """
    Create a smaller test set 
    """
    devset = {}
    for task in MULTIPLE_CHOICE_TASKS:
            task_data = json.load(open('data/%s.json' % task))
            random.shuffle(task_data['examples'])
            problem_list = task_data['examples'][0:size]
            for problem in problem_list:
                problem['mode'] = 'multiple_choice'
                problem['task'] = task        
            devset[task] = problem_list



    for task in FREE_FORM_TASKS:
            task_data = json.load(open('data/%s.json' % task))
            random.shuffle(task_data['examples'])
            problem_list = task_data['examples'][0:size]            
            
            for problem in problem_list:
                problem['mode'] = 'free_form'
                problem['task'] = task
            devset[task] = problem_list  
    return devset

async def make_one_prompt(train_prompt, problem, instruction, official_solution, model="gpt-3.5-turbo", name="", mode='multiple_choice'):
    """
    Make one prompt and evaluate it directly
    """
    try:
        task_init = BBHTaskInit(model,train_prompt)

        initial_solution = ""
        
        random_minutes = random.randint(0,ITERATION_WAITING_MINUTES)
        await asyncio.sleep(random_minutes*60)
        curr_solution, history = await task_init(problem=problem, instruction=instruction, n_temperature=0)
        initial_solution = curr_solution
        initial_solution= extract_ans(initial_solution, mode=mode)
        
        equiv = (initial_solution == official_solution)
    except Exception as e:
        print("An error occured in making one_prompt. Ignoring test case")
        return [False, name]
    
    return [equiv, name]

async def one_k_iteration( instructions, num_samples=2):
        """
        Performs one cycle
        """

        dev_set = await create_dev_set(num_samples)

        for task_name in MULTIPLE_CHOICE_TASKS:
            old_num_wrong = 0
            tasks = []
            wrong_cases = []
            num_wrong = 0
            instruction = instructions[task_name]
            for idx, test_case in enumerate(dev_set[task_name]):
                train_prompt = open('lib_prompt/%s.txt' % task_name, 'r').read()
                #Solution needs to be extracted correctly
                    
                target = test_case['target'][1]    
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, target,name=test_case, mode='multiple_choice')))
                        
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    wrong_cases.append(task[1]) #append ID
                    old_num_wrong += 1
        
            if (wrong_cases==[]):
                instructions[task_name] = instruction
                await asyncio.sleep(60)
                continue
            random.shuffle(wrong_cases)
        
            test_case = wrong_cases[0]
        
            target = test_case['target'][1]

            instruction = await a_self_reprompt(train_prompt=train_prompt, problem=test_case["input"] , solution=target , instruction=instruction, max_attempts=4, mode='multiple_choice', temperature=TEMPERATURE, top_p=TOP_P)

            tasks = []
            for idx, test_case in enumerate(dev_set[task_name]):
                train_prompt = open('lib_prompt/%s.txt' % task_name, 'r').read()
                #Solution needs to be extracted correctly
                    
                target = test_case['target'][1]    
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, target,name=test_case, mode='multiple_choice')))


            
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    num_wrong += 1

            if old_num_wrong > num_wrong:
                instructions[task_name] = instruction
            await asyncio.sleep(60)
        for task_name in FREE_FORM_TASKS:
            old_num_wrong = 0
            tasks = []
            wrong_cases = []
            num_wrong = 0
            instruction = instructions[task_name]
            for idx, test_case in enumerate(dev_set[task_name]):
                train_prompt = open('lib_prompt/%s.txt' % task_name, 'r').read()
                #Solution needs to be extracted correctly
                    
                target = test_case['target']   
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, target,name=test_case, mode='free_form')))
                        
            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    wrong_cases.append(task[1]) #append ID
                    old_num_wrong += 1
        
            if (wrong_cases==[]):
                instructions[task_name] = instruction
                await asyncio.sleep(60)
                continue
            random.shuffle(wrong_cases)
        
            test_case = wrong_cases[0]
        
            target = test_case['target']

            instruction = await a_self_reprompt(train_prompt=train_prompt, problem=test_case["input"] , solution=target , instruction=instruction, max_attempts=4, mode='free_form', temperature=TEMPERATURE, top_p=TOP_P)

            tasks = []
            for idx, test_case in enumerate(dev_set[task_name]):
                train_prompt = open('lib_prompt/%s.txt' % task_name, 'r').read()
                #Solution needs to be extracted correctly
                    
                target = test_case['target']   
                tasks.append(asyncio.create_task(make_one_prompt(train_prompt, test_case["input"],  instruction, target,name=test_case, mode='free_form')))

            tasks = await asyncio.gather(*tasks)
            for task in tasks:
                if task[0] == False: #Solution was not correct
                    num_wrong += 1

            if old_num_wrong > num_wrong:
                instructions[task_name] = instruction
            await asyncio.sleep(60)
        return instructions

async def run_tasks(tasks, mode, instructions,  model_index="gpt-3.5-turbo"):
    """
    Runs benchmark tests
    """
    filename = 'outputs/temp_%s.txt' %mode
    with open(filename, 'w') as fd:
        scores = []
        
        for task in tqdm(tasks):
            print('Testing %s ...' % task)
            acc = 0
            total = 0
            task_data = json.load(open(DATASETPATH + '/%s.json' % task))
            task_prompt = open('lib_prompt/%s.txt' % task, 'r').read()
            fd.write("---------------------")
            fd.write("Testing task %s " % task)
            print_first = False

            responses = []
            answers = []
            for q_ in task_data['examples']:
                q = '\n\nQ: ' + q_['input']

                prompt_q = task_prompt + q + "\nA: Let's think step by step."

                if mode == 'multiple_choice':
                    a = q_['target'][1]
                elif mode == 'free_form':
                    a = q_['target']
                answers.append(a)
                
                messages=[
                   
                            {"role": "system", "content": instructions[task]},
                            {"role": "user", "content": prompt_q},
                    
                        ]
                
                responses.append(asyncio.create_task(handle_prompt(message=messages,temperature=0,waiting_minutes=BENCHMARKTASKLENGTH)))
            responses = await asyncio.gather(*responses)
                
            
            for idx, ans_model in enumerate(responses):
                ans_ = extract_ans(ans_model, mode=mode)
                a = answers[idx]   
            

                fd.write('%s\nA_model:\n%s\nA_target:\n%s\n\n' % (task_data['examples'][idx]['input'], ans_, a))
            
                if ans_ == a:
                    acc += 1
                total += 1
            print('%s acc %.4f' % (task, acc / len(task_data['examples'])))

            scores.append(acc/total)

            #await asyncio.sleep(60) #Reset Ratelimit for next test task
    return scores, filename

OPRO_instructions ={}
OPRO_instructions['temporal_sequences'] = 'The answer is the time that is not mentioned in the given statements'
OPRO_instructions['disambiguation_qa'] = 'Identifying Antecedents of Pronouns: A Comprehensive Guide'
OPRO_instructions['date_understanding'] = """To find the date X time ago from today, first find today’s date. Then subtract X time from today’s date. If the current date is the last day of a month, then the date a month ago is the last day of the previous month. If the current date is not the last day of a month, then the date a month ago is the same day of the previous month. For example, if today is March 31, 2023, then the date a month ago is February 28, 2023. If today is April 1, 2023, then the date a month ago is March 1, 2023."""
OPRO_instructions['tracking_shuffled_objects_three_objects'] = 'Claire has the blue ball, Gertrude has the black ball, and Dave has the green ball. They are all happy with their new balls.'
OPRO_instructions['penguins_in_a_table'] = 'Here is my new text:'
OPRO_instructions['geometric_shapes'] = 'A closed polygonal chain is a series of connected line segments. The line segments can be straight or curved. The first and last line segments are connected. The line segments do not intersect each other except at their endpoints. A closed polygon can be described by an SVG path element, which starts at a given point, goes to one or more additional points, and then ends at the starting point. The path element can consist of straight line segments, curved segments, or a mixture of both'
OPRO_instructions['snarks'] = 'Identify the sarcastic statement by considering the following factors: incongruity, exaggeration, understatement, context, speaker’s intent, and audience’s reaction. I will also consider the speaker’s tone of voice, facial expressions, and body language.'
OPRO_instructions['ruin_names'] = 'Which is the funniest pun on the artist or movie name?'
OPRO_instructions['tracking_shuffled_objects_seven_objects'] = 'Claire has the blue ball, Gertrude has the black ball, and Dave has the green ball. They are all happy with their new balls.'
OPRO_instructions['tracking_shuffled_objects_five_objects'] = 'Claire has the blue ball, Gertrude has the black ball, and Dave has the green ball. They are all happy with their new balls.'
OPRO_instructions['logical_deduction_three_objects'] = 'The following questions will test your ability to use deductive reasoning. You will be given a set of statements about a group of objects. You will then be asked to answer questions about the objects based on the statements. The statements in the questions are logically consistent, so you can use them to deduce the order of the objects. For each question, you must choose the option that is logically consistent with the information in the questions'
OPRO_instructions['logical_deduction_five_objects'] = 'The following questions will test your ability to use deductive reasoning. You will be given a set of statements about a group of objects. You will then be asked to answer questions about the objects based on the statements. The statements in the questions are logically consistent, so you can use them to deduce the order of the objects. For each question, you must choose the option that is logically consistent with the information in the questions'
OPRO_instructions['logical_deduction_seven_objects'] = 'The following questions will test your ability to use deductive reasoning. You will be given a set of statements about a group of objects. You will then be asked to answer questions about the objects based on the statements. The statements in the questions are logically consistent, so you can use them to deduce the order of the objects. For each question, you must choose the option that is logically consistent with the information in the questions'
OPRO_instructions['hyperbaton'] = 'The correct adjective order in English is opinion, size, shape, age, color, origin, material, and purpose. If you have more than one adjective of the same type, they are usually placed in order of importance. For example, you would say "a large, old, Pakistani ship" rather than "an old, large, Pakistani ship." There are a few exceptions to these rules, but they are generally followed in most cases.'
OPRO_instructions['movie_recommendation'] = 'Based on your input, I have analyzed the given movies in terms of genre, plot, tone, audience rating, year of release, director, cast, and reviews. I have also taken into account the given options. The movie that is most similar to the given movies in terms of all these factors is'
OPRO_instructions['salient_translation_error_detection'] = 'Instructions: Read the German sentence and its English translation carefully, then identify the type of error in the translation and select the correct option. There are six possible types of errors: Named Entities, Numerical Values, Modifiers or Adjectives, Negation or Antonyms, Facts, and Dropped Content.'
OPRO_instructions['reasoning_about_colored_objects'] = 'Starting from the leftmost object in the row, I observe the following objects arranged in this order:'
OPRO_instructions['multistep_arithmetic_two'] = 'The order of operations in mathematics is PEMDAS, which stands for Parentheses, Exponents, Multiplication, Division, Addition, and Subtraction. When there are multiple operations of the same precedence, they must be performed from left to right. Note that multiplication and division have the same precedence, as do addition and subtraction'
OPRO_instructions['navigate'] = 'You will return to the starting point if and only if (1) the total number of steps you take forward is equal to the total number of steps you take back, and (2) the total number of turns you make is a multiple of 180 degrees.'
OPRO_instructions['dyck_languages'] = 'First, look for the opening parentheses. Then, count the number of opening parentheses. Finally, close the parentheses in the reverse order that they were opened.'
OPRO_instructions['word_sorting'] = 'Alphabetical order of given words:'
OPRO_instructions['sports_understanding'] = 'I will determine if a sentence about an athlete is plausible by first checking if it is grammatically correct. If it is, I will then check if it is consistent with the athlete’s sport, position, and real-world statistics. I will also check if it is consistent with the rules of the athlete’s sport. If the sentence is consistent with all of these things, I will answer "yes", otherwise I will answer "no".'
OPRO_instructions['boolean_expressions'] = 'A Boolean expression is a well-formed expression consisting of variables, values, and logical operators. The expression must evaluate to a single True or False value. The order of precedence of the logical operators is as follows: NOT, AND, OR, XOR, IMP. Parentheses can be used to group subexpressions and to control the order of evaluation.'
OPRO_instructions[ 'object_counting'] = 'Here is a list of the objects you mentioned and their corresponding counts:'
OPRO_instructions['formal_fallacies'] = 'A deductive argument is one where the conclusion follows necessarily from the premises. If the premises are true, then the conclusion must also be true. An invalid argument is one where it is possible for the premises to be true and theconclusion to be false.'
OPRO_instructions['causal_judgement'] = 'When considering questions about causation, a typical person would consider the following factors: whether the action or event was a necessary condition for the outcome to occur, a sufficient condition, a proximate cause, or a foreseeable cause.'
OPRO_instructions['web_of_lies'] = 'The answer to a question is yes if there are an odd number of liars before the current speaker, and no if there are an even number of liars before the current speaker. If the current speaker is a truth-teller, they will say the opposite of what the previous person said, while a liar will say the same thing as the previous person said'

async def main(multiple_choice_tasks=MULTIPLE_CHOICE_TASKS, free_form_tasks=FREE_FORM_TASKS, outputfilename=OUTPUTFILENAME):
    openai.api_key = OPENAIKEY
    model_index = 'gpt-3.5-turbo'
    instruction = INITIAL_INSTRUCTION
    instructions = {}
    for task in MULTIPLE_CHOICE_TASKS:
        instructions[task] = instruction
    for task in FREE_FORM_TASKS:
        instructions[task] = instruction
    task = TASKS
    for i in range(K):
        instructions = await one_k_iteration(instructions=instructions, num_samples=NUM_SAMPLES)
        await asyncio.sleep(60)#Reset Ratelimit for next test task
    #OPRO
    #instructions = OPRO_instructions
    run_multiple_choice = task == 'all' or task == 'multiple_choice'
    run_free_form = task == 'all' or task == 'free_form'
    if run_multiple_choice:
        accuracies_mult, file1_path = await run_tasks(multiple_choice_tasks, mode='multiple_choice', instructions=instructions, model_index=model_index)
    if run_free_form:
        accuracies_free, file2_path = await run_tasks(free_form_tasks, mode='free_form', instructions=instructions,  model_index=model_index)
    if task=='all':
        accuracy = sum(accuracies_mult + accuracies_free) / len(accuracies_free + accuracies_mult)
                # Open the first file for reading
        with open(file1_path, 'r') as file1:
            content1 = file1.read()

        # Open the second file for reading
        with open(file2_path, 'r') as file2:
            content2 = file2.read()

        # Combine the contents of both files
        combined_content = content1 + '\n' + content2
        # Delete the initial files
        os.remove(file1_path)
        os.remove(file2_path)
        # Open a new file or overwrite one of the existing files for writing
        with open(outputfilename, 'w') as merged_file:
            merged_file.write(combined_content)
            merged_file.write("------------------------------------")
            for idx, task in enumerate(MULTIPLE_CHOICE_TASKS):
                merged_file.write("Benchmark {}".format(task)+ "\n")
                merged_file.write("Accuracy {}".format(accuracies_mult[idx])+ "\n")
                merged_file.write("Used Instruction: {}".format(instructions[task]+ "\n"))
            for idx, task in enumerate(FREE_FORM_TASKS):
                merged_file.write("Benchmark {}".format(task)+ "\n")
                merged_file.write("Accuracy {}".format(accuracies_free[idx])+ "\n")
                merged_file.write("Used Instruction: {}".format(instructions[task]+ "\n"))
            merged_file.write("Average accuracy: " + str(accuracy) + "\n")

        print("Accuracy: ", accuracy)
        return
    
    return 

if __name__ == '__main__':
    asyncio.run(main())