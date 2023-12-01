import sys
sys.path.insert(0, './scripts')
sys.path.insert(0, './config')
import os
import openai
import time
from time import time, sleep
import datetime
from uuid import uuid4
import concurrent.futures
import requests


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
        
        
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
        

def chatgpt200_completion(messages, model="gpt-3.5-turbo", temp=0.2):
    max_retry = 7
    retry = 0
    while  True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=200)
            text = response['choices'][0]['message']['content']
            temperature = temp
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to an error in ChatGPT: {oops}")
                exit(1)
            print(f'Error communicating with OpenAI: "{oops}" - Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)
            
           
def chatgpt250_completion(messages, model="gpt-3.5-turbo", temp=0.40):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=250)
            text = response['choices'][0]['message']['content']
            temperature = temp
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to an error in ChatGPT: {oops}")
                exit(1)
            print(f'Error communicating with OpenAI: "{oops}" - Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)
            
            
def chatgpt35_completion(messages, model="gpt-3.5-turbo", temp=0.3):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            text = response['choices'][0]['message']['content']
            temperature = temp
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to an error in ChatGPT: {oops}")
                exit(1)
            print(f'Error communicating with OpenAI: "{oops}" - Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)
            
            
def chatgpt_tasklist_completion(messages, model="gpt-4", temp=0.3):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            text = response['choices'][0]['message']['content']
            temperature = temp
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to an error in ChatGPT: {oops}")
                exit(1)
            print(f'Error communicating with OpenAI: "{oops}" - Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


def chatgptyesno_completion(messages, model="gpt-3.5-turbo", temp=0.0):
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=1)
            text = response['choices'][0]['message']['content']
            temperature = temp
        #    filename = '%s_chat.txt' % time()
        #    if not os.path.exists('chat_logs'):
        #        os.makedirs('chat_logs')
        #    save_file('chat_logs/%s' % filename, str(messages) + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                print(f"Exiting due to an error in ChatGPT: {oops}")
                exit(1)
            print(f'Error communicating with OpenAI: "{oops}" - Retrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


def bing_api(line):
    try:
        # Perform the Bing search
        subscription_key = open_file('key_bing.txt')
        assert subscription_key
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        search_term = f"{line}"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "answerCount": "5", "count": "4"}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()
    #    print(search_results)
        # Format the results
        rows = "\n".join(["""<tr>
                               <td><a href=\"{0}\">{1}</a></td>
                               <td>{2}</td>
                             </tr>""".format(v["url"], v["name"], v["snippet"])
                            for v in search_results["webPages"]["value"]])
        table = "<table>{0}</table>".format(rows)
        return table
    except Exception as e:
        print(e)
        table = "Error"
        return table

def fail():
    print('')
    fail = "Not Needed"
    return fail
    



if __name__ == '__main__':
    openai.api_key = open_file('key_openai.txt')
    conversation = list()
    int_conversation = list()
    conversation2 = list()
    tasklist_completion = list()
    master_tasklist = list()
    tasklist = list()
    tasklist_log = list()
    webcheck = list()
    counter = 0
    if not os.path.exists('logs/complete_chat_logs'):
        os.makedirs('logs/complete_chat_logs')
    bot_name = open_file('./config/prompt_bot_name.txt')
    username = open_file('./config/prompt_username.txt')
    main_prompt = open_file('./config/prompt_main.txt').replace('<<NAME>>', bot_name)
    greeting_msg = open_file('./config/prompt_greeting.txt').replace('<<NAME>>', bot_name)
    while True:
        # # Get Timestamp
        timestamp = time()
        # # Start or Continue Conversation based on if response exists
        conversation.append({'role': 'system', 'content': '%s' % main_prompt})
        int_conversation.append({'role': 'system', 'content': '%s' % main_prompt})
        conversation.append({'role': 'assistant', 'content': "%s" % greeting_msg})
        print("\n%s" % greeting_msg)
        # # User Input Text
        a = input(f'\n\nUSER: ')
        message_input = a
        conversation.append({'role': 'user', 'content': a})        
        # # Generate Semantic Search Terms
        tasklist.append({'role': 'system', 'content': "You are a task coordinator. Your job is to take user input and create a list of 2-5 inquiries to be used for a semantic database search of a chatbot's memories. Use the format [- 'INQUIRY']."})
        tasklist.append({'role': 'user', 'content': "USER INQUIRY: %s" % a})
        tasklist.append({'role': 'assistant', 'content': "List of Semantic Search Terms: "})
        tasklist_output = chatgpt200_completion(tasklist)
    #    print(tasklist_output)
       # # # Inner Monologue Generation
        conversation.append({'role': 'assistant', 'content': "Other possible user meanings: %s" % tasklist_output})
        conversation.append({'role': 'assistant', 'content': "USER MESSAGE: %s;\nBased on the user, %s's message, compose a brief silent soliloquy as an inner monologue that reflects on your deepest contemplations in relation to the user's message.\n\nINNER_MONOLOGUE: " % (a, username)})
        output_one = chatgpt250_completion(conversation)
        message = output_one
        print('\n\nINNER_MONOLOGUE: %s' % output_one)
        output_log = f'\nUSER: {a}\n\n{bot_name}: {output_one}'
        # # Clear Conversation List
        conversation.clear()
        # # Memory DB Search
        # # Intuition Generation
        int_conversation.append({'role': 'assistant', 'content': "%s" % greeting_msg})
        int_conversation.append({'role': 'user', 'content': tasklist_output})
        int_conversation.append({'role': 'assistant', 'content': "INNER MONOLOGUE: %s;\n\nUSER MESSAGE: %s;\nIn a single paragraph, interpret the user, %s's message in third person by creating an intuitive plan on what information needs to be researched, even if the user is uncertain about their own needs.;\nINTUITION: " % (output_one, a, username)})
        output_two = chatgpt200_completion(int_conversation)
        message_two = output_two
        print('\n\nINTUITION: %s' % output_two)
        output_two_log = f'\nUSER: {a}\n\n{bot_name}: {output_two}'
        # # Generate Asynchronous Research Task list
        master_tasklist.append({'role': 'system', 'content': "You are a stateless task list coordinator. Your job is to take the user's input and transform it into a list of independent research queries that can be executed by separate AI agents in a cluster computing environment. The other asynchronous Ai agents are also stateless and cannot communicate with each other or the user during task execution. Exclude tasks involving final product production, hallucinations, user communication, or checking work with other agents. Respond using the following format: '- [task]'"})
        master_tasklist.append({'role': 'user', 'content': "USER FACING CHATBOT'S INTUITIVE ACTION PLAN:\n%s" % output_two})
        master_tasklist.append({'role': 'user', 'content': "USER INQUIRY:\n%s" % a})
        master_tasklist.append({'role': 'user', 'content': "SEMANTICALLY SIMILAR INQUIRIES:\n%s" % tasklist_output})
        master_tasklist.append({'role': 'assistant', 'content': "TASK LIST:"})
        master_tasklist_output = chatgpt_tasklist_completion(master_tasklist)
        print(master_tasklist_output)
        # # Start Conversation list for Final Response Module
        tasklist_completion.append({'role': 'system', 'content': "You are the final response module of a cluster compute Ai-Chatbot. Your job is to take the completed task list, and give a verbose response to the end user in accordance with their initial request."})
        tasklist_completion.append({'role': 'user', 'content': "%s" % master_tasklist_output})
        task = {}
        task_result = {}
        task_result2 = {}
        task_counter = 1
        # # Split bullet points into separate lines to be used as individual tasks
        lines = master_tasklist_output.splitlines()
        print('\n\nSYSTEM: Would you like to autonomously complete this task list?\n        Press Y for yes or N for no.')
        user_input = input("'Y' or 'N': ")
        if user_input == 'y':
            # # Start Asynchronous Processing with concurrent.futures library
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        lambda line, task_counter, conversation, webcheck, tasklist_completion: (
                            # # Update Final Response Module with Task
                            tasklist_completion.append({'role': 'user', 'content': "ASSIGNED TASK:\n%s" % line}),
                            # # Start Sub-Module for asynchronous task completion
                            conversation.append({'role': 'system', 'content': "You are a sub-module for an Autonomous Ai-Chatbot. You are one of many agents in a chain. You are to take the given task and complete it in its entirety. Take other tasks into account when formulating your answer."}),
                            conversation.append({'role': 'user', 'content': "Task list:\n%s" % master_tasklist_output}),
                            conversation.append({'role': 'assistant', 'content': "Bot %s: I have studied the given tasklist.  What is my assigned task?" % task_counter}),
                            conversation.append({'role': 'user', 'content': "Bot %s's Assigned task: %s" % (task_counter, line)}),
                            webcheck.append({'role': 'system', 'content': f"You are a sub-module for an Autonomous Ai-Chatbot. You are one of many agents in a chain. Your task is to decide if a web-search is needed in order to complete the given task. Only recent or niche information needs to be searched. Do not search for any information pertaining to the user, {username}, or the main bot, {bot_name}.   If a websearch is needed, print: YES.  If a web-search is not needed, print: NO."}),
                            webcheck.append({'role': 'user', 'content': "Hello, how are you today?"}),
                            webcheck.append({'role': 'assistant', 'content': "NO"}),
                            # # Check if websearch is needed
                            webcheck.append({'role': 'user', 'content': f"{line}"}),
                            web1 := chatgptyesno_completion(webcheck),
                            table := bing_api(line) if web1 =='YES' else fail(),
                            # # Update Conversation with Websearch
                            conversation.append({'role': 'assistant', 'content': "WEBSEARCH: %s" % table}),
                            conversation.append({'role': 'user', 'content': "Bot %s Task Reinitialization: %s" % (task_counter, line)}),
                            conversation.append({'role': 'assistant', 'content': "Bot %s's Response:" % task_counter}),
                            task_completion := chatgpt35_completion(conversation),
                            conversation.clear(),
                            # # Update Final Response Module with Completed Task
                            tasklist_completion.append({'role': 'assistant', 'content': "COMPLETED TASK:\n%s" % task_completion}),
                            # # conversation log file
                            tasklist_log.append({'role': 'user', 'content': "ASSIGNED TASK:\n%s\n\n" % line}),
                            tasklist_log.append({'role': 'assistant', 'content': "WEBSEARCH:\n%s\n\n" % table}),
                            tasklist_log.append({'role': 'assistant', 'content': "COMPLETED TASK:\n%s\n\n" % task_completion}),
                            print(line),
                            print(table),
                            print(task_completion),
                        ) if line != "None" else tasklist_completion,
                        line, task_counter, webcheck.copy(), conversation.copy(), []
                    )
                    for task_counter, line in enumerate(lines)
                ]
            # # Generate Final Output with Final Response Module    
            tasklist_completion.append({'role': 'user', 'content': "Take the given set of tasks and completed responses and transmute them into a verbose response for the end user in accordance with their request. The end user is both unaware and unable to see any of your research. User's initial request: %s" % a})
            print('\n\nGenerating Final Output...')
            final_response_complete = chatgpt_tasklist_completion(tasklist_completion)
            print('\nFINAL OUTPUT:\n%s' % final_response_complete)
            # # Save Log
            complete_message = f'\nUSER: {a}\n\nINNER_MONOLOGUE: {output_one}\n\nINTUITION: {output_two}\n\n{bot_name}: {tasklist_log}\n\nFINAL OUTPUT: {final_response_complete}'
            filename = '%s_chat.txt' % timestamp
            save_file('logs/complete_chat_logs/%s' % filename, complete_message)
            conversation.clear()
            int_conversation.clear()
            conversation2.clear()
            tasklist_completion.clear()
            master_tasklist.clear()
            tasklist.clear()
            tasklist_log.clear()
        continue
