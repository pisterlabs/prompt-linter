


# ----------------------------------------------------------------------------
# Project: Semantic Search Module for the Alter Brain project
# File:    llib__script_template_json.py
# 
# This library consists of several utility and execution functions designed to 
# process and handle task data for specific NLP tasks. These tasks can include 
# correction, drafting, paraphrasing, translation, summarization, and more, 
# in both English and French. The tasks are formatted as JSON or text files 
# and then processed using OpenAI's GPT-4.
# 
# Author:  Michel Levy Provencal
# Brightness.ai - 2023 - contact@brightness.fr
# ----------------------------------------------------------------------------

"""
## Utility Functions
* `read_json_file(file)`: This function reads and returns the content of a JSON file.
* `write_json_file(file, data)`: This function writes data to a JSON file.
* `json_to_txt(json_file, txt_file)`: This function reads from a JSON file, extracting task data, and writes it to a text file in a readable format.
* `textfile_to_json(text_file, json_file)`: This function reads from a text file with a specific format (each task property in a new line, separated by ': '), transforms it to a JSON object, and writes it to a JSON file.
* `text_to_json(text, json_file)`: This function takes a string input with a similar format to `textfile_to_json()`, converts it to a JSON object, and writes it to a JSON file.

## Execution Functions
* `request_llm(prompt, context, input_data, model)`: This function sends a request to the OpenAI API, passing the prompt, context, and input data, and gets a response from the specified language model. It also handles error scenarios and makes multiple attempts if the request fails initially.
* `truncate_strings(prompt, context, input_data)`: This function truncates the given strings to ensure that the combined length does not exceed the maximum length allowed by GPT-4. If the strings are already under the limit, they are returned as is. If not, they are truncated starting with 'input_data' and 'context' if necessary.
* `exec_task(prompt, brain_id, input_data)`: This function takes a prompt, brain_id, and input data, retrieves the context using `lib__embedded_context.query_extended_llm()` function, and sends a request to the language model via the `request_llm()` function. It returns the response from the language model.
* `execute_json(input_json_file, output_json_file)`: This function reads tasks from an input JSON file, executes each task using the `exec_task()` function, and writes the results back to an output JSON file.
"""

import json
import lib__embedded_context
from dotenv import load_dotenv
import os
import requests
import time
import sys
import openai
from urllib.parse import unquote
from dotenv import load_dotenv
from queue import Queue
import json
from datetime import *
from lib__path import *



####################################################################################################
# TOOL FUNCTIONS
####################################################################################################


# Fonction pour lire le fichier json
def read_json_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
        #print(data)
    return data

# Fonction pour écrire dans un fichier json
def write_json_file(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)

def json_to_txt(json_file, txt_file):
    # Lecture du fichier json
    tasks = read_json_file(json_file)

    # Ouvrir le fichier de sortie pour écrire
    with open(txt_file, 'w') as f:
        for task in tasks:
            prompt = tasks[task]['prompt']
            brain_id = tasks[task]['brain_id']
            input_data = tasks[task]['input_data']
            
            # Écrire les informations de la tâche dans le fichier texte
            f.write(f"Task: {task}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Brain_id: {brain_id}\n")
            f.write(f"Input Data: {input_data}\n")
            f.write("\n")  # ligne vide pour séparer les tâches







# Function that transforms a text file into a json file
def textfile_to_json(text_file, json_file):
    tasks = {}
    try:
        with open(text_file, 'r') as f:
            lines = f.readlines()
            task = None
            for line in lines:
                line = line.strip() 
                if line: 
                    try:
                        key, value = line.split(": ", 1)
                    except ValueError:
                        print(f"Erreur de formatage sur la ligne : '{line}'. Ignoré.")
                        continue

                    if key == "Task":
                        task = value
                        tasks[task] = {}
                    else:
                        tasks[task][key.lower().replace(' ', '_')] = value
    except FileNotFoundError:
        print(f"Le fichier '{text_file}' n'a pas été trouvé.")
        return
    
    try:
        with open(json_file, 'w') as f:
            json.dump(tasks, f, indent=4)
    except IOError:
        print(f"Erreur lors de l'écriture dans le fichier '{json_file}'.")

# Function that transforms a text into a json file
def text_to_json(text, json_file):
    tasks = {}
    lines = text.split('\n')
    task = None
    for line in lines:
        line = line.strip()
        if line: 
            if ": " not in line:
                print(f"Ligne mal formatée : '{line}'. Ignoré.")
                continue

            try:
                key, value = line.split(": ", 1)
            except ValueError:
                print(f"Erreur de formatage sur la ligne : '{line}'. Ignoré.")
                continue

            if key.strip() == "Task":
                task = value
                tasks[task] = {}
            else:
                tasks[task][key.lower().replace(' ', '_').strip()] = value.strip()
    
    try:
        with open(json_file, 'w') as f:
            json.dump(tasks, f, indent=4)
    except IOError:
        print(f"Erreur lors de l'écriture dans le fichier '{json_file}'.")



    
    


####################################################################################################
# SCRIPT EXECUTION FUNCTIONS
####################################################################################################


def request_llm(prompt, context, input_data, model="gpt-4"):
    #load_dotenv(".env") # Load the environment variables from the .env file.
    #load_dotenv("/home/michel/extended_llm/.env") # Load the environment variables from the .env file.
    load_dotenv(DOTENVPATH)
    
    attempts = 0
    execprompt = "Context : " + context + "\n" + input_data + "\n" + "Query : " + prompt
    api_key = os.environ.get("OPENAI_API_KEY")
    system = "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes."

    while attempts < 10:
        try:
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            data = {
                'model': model,
                #'model': 'gpt-3.5-turbo',
                'temperature': 0.01,
                'messages': [
                    {'role': 'user', 'content': execprompt},
                    {'role': 'system', 'content': system}
                ]
            }
            print(str(datetime.now()) + " : " + "Appel au LLM: " + str(model) + "\n\n")
            response = requests.post(url, headers=headers, json=data)
            json_data = response.json()
            message = json_data['choices'][0]['message']['content']
            print(str(datetime.now()) + " : " + "Réponse : " + str(message) + "\n\n")
            return message.strip()
        
        except Exception as e:
            error_code = type(e).__name__
            error_reason = str(e)
            attempts += 1
            print(f"Erreur : {error_code} - {error_reason}. Nouvel essai dans {str(attempts * 2)} secondes...")
            time.sleep(attempts * 2)

    print("Erreur : Echec de la création de la completion après 10 essais")
    sys.exit()
    


def execute_tasks(tasks, model):
    q = Queue()
    for task_name in tasks:
        q.put(task_name)

    while not q.empty():
        task_name = q.get()
        task = tasks[task_name]

        # input_data might be a task name, in which case we should use the result of that task
        input_data = unquote(task.get('input_data', ''))
        if input_data.startswith("task") and input_data in tasks and 'result' in tasks[input_data]:
            input_data = tasks[input_data]['result']

        prompt = unquote(task.get('prompt', ''))
        brain_id = unquote(task.get('brain_id', ''))
        
        model = model
        
        index_filename = "datas/" + brain_id + "/emb_index.csv"
        
        if brain_id.startswith("http://") or brain_id.startswith("https://"):
            url = brain_id
            index_filename= "datas/" + lib__embedded_context.build_index_url(url) + "/emb_index.csv"

        prompt, context, input_data = truncate_strings(prompt, '', input_data)
        
        # find context
        context = lib__embedded_context.find_context(prompt, index_filename, 3)
        
        
        if model=="gpt-4":
            # truncate strings for gpt-4
            prompt, context, input_data = truncate_strings(prompt, context, input_data)
        else:
            # truncate strings for gpt-3.5-turbo
            prompt, context, input_data = truncate_strings(prompt, context, input_data, 4500)

        # prepare input data
        load_dotenv(DOTENVPATH)

        #load_dotenv("/home/michel/extended_llm/.env") # Load the environment variables from the .env file.
        #load_dotenv(".env") # Load the environment variables from the .env file.
        execprompt = "Context : " + context + "\n" + input_data + "\n" + "Query : " + prompt
        system = "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes."
        attempts = 0
        
        # call openAI to get the streaming response 
        result = ''
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': execprompt}
                ],
                temperature=0.01,
                stream=True
            )

            for chunk in response:
                if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    content = chunk['choices'][0]['delta']['content']
                    print(content)
                    yield f"{content}"
                    result += content
            
            #Saut de ligne entre tache
            print("\n\n")
            yield "\n\n"
            result += "\n\n"
            
        except openai.error.OpenAIError as e:  # catch errors specific to OpenAI
            print(f"OpenAI error: {e}")

        # Store the result in the task itself:
        task['result'] = result  

        # Update the input_data of tasks that depend on the completed task:
        for dependent_task_name, dependent_task in tasks.items():
            if 'input_data' in dependent_task and dependent_task['input_data'] == task_name:
                dependent_task['input_data'] = result

    
    
def request_llm_stream(prompt, context, input_data, model) :
    #load_dotenv(".env") # Load the environment variables from the .env file.
    #load_dotenv("/home/michel/extended_llm/.env") # Load the environment variables from the .env file.
    load_dotenv(DOTENVPATH)

    execprompt = "Context : " + context + "\n" + input_data + "\n" + "Query : " + prompt
    system = "Je suis un assistant parlant parfaitement le français et l'anglais capable de corriger, rédiger, paraphraser, traduire, résumer, développer des textes."

    attempts = 0
    while attempts < 3:  # essayer 3 fois avant d'abandonner
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': execprompt}
                ],
                temperature=0.01,
                stream=True
            )

            for chunk in response:
                if 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                    content = chunk['choices'][0]['delta']['content']
                    yield content

        except openai.error.OpenAIError as e:  # attraper les erreurs spécifiques à OpenAI
            attempts += 1
            print(f"Erreur OpenAI: {e}. Nouvel essai dans 5 secondes...")
            time.sleep(attempts * 2)
   
    


# tronuqer les chaînes pour éviter les erreurs de longueur
def truncate_strings(prompt, context, input_data, max_length=9000):
    # Définir la longueur maximale autorisée avec GPT4

    # Calculer la longueur actuelle de toutes les chaînes combinées
    total_length = len(prompt) + len(context) + len(input_data)

    # Si la longueur totale est déjà inférieure à la longueur maximale, aucune action n'est nécessaire
    if total_length <= max_length:
        return prompt, context, input_data

    # Sinon, commencer à tronquer les chaînes
    remaining_length = max_length - len(prompt) - len(context)

    # Si la longueur restante après avoir préservé `prompt` et `context` est négative, 
    # cela signifie que `prompt` et `context` seuls dépassent déjà la longueur maximale.
    # Dans ce cas, on tronque `context` pour s'adapter à la longueur maximale.
    if remaining_length < 0:
        context = context[:max_length - len(prompt)]
        input_data = ""
    else:
        # Sinon, tronquer `input_data` pour s'adapter à la longueur restante
        input_data = input_data[:remaining_length]

    return prompt, context, input_data



# Fonction pour l'execution de taches
def exec_task(prompt, brain_id, input_data, model="gpt-4"):
    
    index_filename = "datas/" + brain_id + "/emb_index.csv"
    
    if model == "gpt-4":
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, '', input_data)
    else:
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, '', input_data, 4500)
        
    # recherche du contexte
    context = lib__embedded_context.query_extended_llm(prompt + input_data, index_filename, model)
    
    if model == "gpt-4":
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, context, input_data)
    else:
        # Limitation des erreurs de longueur
        prompt, context, input_data = truncate_strings(prompt, context, input_data, 4500)

    # Appel au LLM
    res = request_llm(prompt, context, input_data, model)

    return (res)



def execute_json(input_json_file, output_json_file, model="gpt-4"):
    # Lecture du fichier json
    tasks = read_json_file(input_json_file)
    
    for task in tasks:
        prompt = tasks[task].get('prompt', '')
        brain_id = tasks[task].get('brain_id', '')
        input_data = tasks[task].get('input_data', '')
        
        # Si la tâche dépend d'une autre tâche
        if input_data in tasks and 'result' in tasks[input_data]:
            # Utiliser le résultat de la tâche précédente
            input_data = tasks[input_data]['result']
        
        # Appeler la fonction execute la tache avec le contexte et obtenir le résultat
        result = exec_task(prompt, brain_id, input_data, model)

        # Ajouter le résultat à la tâche actuelle
        tasks[task]['result'] = result

    # Écriture du résultat dans le fichier json
    write_json_file(output_json_file, tasks)



