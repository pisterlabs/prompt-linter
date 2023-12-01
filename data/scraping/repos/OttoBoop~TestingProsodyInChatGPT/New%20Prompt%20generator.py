# Error from OpenAI!
import os  # Importing the os module for file and directory operations
import requests  # Importing the requests module for making HTTP requests
import json  # Importing the json module for working with JSON data
import openai  # Importing the openai module for accessing the OpenAI API
import datetime  # Importing the datetime module for working with dates and times
import itertools  # Importing the itertools module for generating combinations

# Replace with your OpenAI API Key
api_key = "ENTER KEY HERE"

# The number of API calls for each prompt
num_calls = 5  # Change this to the number you want

# The base URL for the OpenAI API
api_url = "https://api.openai.com/v1/chat/completions"

# Define a dictionary of transformation functions
TRANSFORMATIONS = {
    'bold': ("bold", lambda x: f"**{x}**"),  # Function to transform text to bold format
    'italic': ("italic", lambda x: f"_{x}_"),  # Function to transform text to italic format
    "long_dash": ("long_dash", lambda x: f'— {x} —'),  # Function to add long dashes around text
    "asterisks": ("asterisks", lambda x: f'* {x} *'),  # Function to add asterisks around text
    "quotes": ("quotes", lambda x: f'"{x}"'),  # Function to add quotes around text
    "tilde": ("tilde", lambda x: f'~{x}~'),  # Function to add tildes around text
    "paranthesis": ("paranthesis", lambda x: f'({x})'),  # Function to add parantheses around text
    "codeblock": ("codeblock", lambda x: f'`{x}`'),  # Function to format text as a code block
    "focus": ("focus", lambda x: f'>>> {x} <<<'),  # Function to add focus around text
}


def call_api(prompt):
    """
    Function to call OpenAI API and return the response
    """
    openai.api_key = api_key  # Setting the OpenAI API key
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = openai.ChatCompletion.create(**data)  # Calling the OpenAI API
    return response['choices'][0]['message']['content'].strip()  # Returning the response content


import os
import re
import datetime

def remove_special_characters(input_string):
    return re.sub(r'\W+', '_', input_string)

def create_prompt_folder(base_prompt, transformations):
    timestamp = datetime.datetime.now().strftime('%H%M%S')
    
    # Remove special characters from transformations
    transformations = [remove_special_characters(t) for t in transformations]
    transformations_str = "_".join([t[:3] for t in transformations])

    # Remove special characters from base_prompt
    base_prompt = remove_special_characters(base_prompt)
    base_prompt_shortened = base_prompt[:50].replace(" ", "_")

    folder_name = f"{timestamp}_prompt_{base_prompt_shortened}_{transformations_str}"

    # Limit the length of the folder name to prevent system errors
    max_folder_length = 260 - len(os.getcwd()) - 1 # Subtract current working directory length and 1 for the separator
    folder_name = folder_name[:max_folder_length]

    try:
        os.makedirs(folder_name)
    except Exception as e:
        print(f"Could not create the directory: {folder_name}. Error: {e}")
        folder_name = None

    return folder_name



def write_prompt_to_file(prompt, folder_name, transformations):
    """
    Function to write the prompt to a text file
    """
    transformations_str = ", ".join([TRANSFORMATIONS[t][0] for t in transformations])
    with open(os.path.join(folder_name, "PROMPT.txt"), "w") as file:
        file.write(f"Prompt: {prompt}\nTransformations: {transformations_str}")


def transform_prompt(base_prompt, word_to_transformations):
    """
    Function to apply transformations to the base_prompt
    """
    transformed_prompt = base_prompt  # Initializing the transformed prompt with the base prompt
    for word, transformations in word_to_transformations.items():  # Iterating over the word to transformation mappings
        for transformation in transformations:
            transformed_prompt = transformed_prompt.replace(word, TRANSFORMATIONS[transformation][1](word))  # Applying the transformation to the word in the prompt
    return transformed_prompt  # Returning the transformed prompt

def get_all_combinations(transformations):
    """
    Function to get all combinations of transformations
    """
    all_combinations = []  # create an empty list to store all combinations
    for i in range(1, len(transformations)+1):  # iterate over the range from 1 to the length of transformations + 1
        for subset in itertools.combinations(transformations, i):  # iterate over all combinations of transformations with length i
            all_combinations.append(subset)  # append the current combination to the all_combinations list
    return all_combinations  # return the list of all combinations

import itertools

def ranked_combinations(elements):
    all_combinations = []
    
    for r in range(1, len(elements) + 1):
        combinations = itertools.combinations(elements, r)
        all_combinations += [list(combination) for combination in combinations]
                
    return all_combinations


# Testing
transformations = ['bold', 'italic', 'underline']
print(ranked_combinations(transformations))


def call_api_and_store_responses(base_prompt, word_to_transformations, all_combinations):
    for transformation_list in all_combinations:
        transformed_word_to_transformations = {}
        for word in word_to_transformations:
            transformed_word_to_transformations[word] = [t for t in word_to_transformations[word] if t in transformation_list]

        # Generate the transformed prompt
        prompt = transform_prompt(base_prompt, transformed_word_to_transformations)

        # Create a unique folder for the transformed prompt
        folder_name = create_prompt_folder(base_prompt, transformation_list)
        
        # Write the prompt to a text file
        write_prompt_to_file(prompt, folder_name, transformation_list)
        
        # Call the API and write the responses to text files
        for j in range(num_calls):
            response = call_api(prompt)
            with open(os.path.join(folder_name, f"{j+1}.txt"), "w") as file:
                file.write(response)

base_prompts = [
    "Write a one story paragraph with the following sentence: he was a smart and handsome man"
]

word_to_transformations = {
    'smart': ['bold', 'italic', 'codeblock', 'focus', 'quotes', 'tilde', 'paranthesis', 'asterisks', 'long_dash'],  # Mapping the word 'smart' to a list of transformations
    #'handsome': ['bold', 'italic', 'codeblock', 'focus', 'quotes', 'tilde', 'paranthesis', 'asterisks', 'long_dash'],  # Mapping the word 'handsome' to a list of transformations
}

for base_prompt in base_prompts:
    transformations = list(TRANSFORMATIONS.keys())
    all_combinations = ranked_combinations(transformations)
    call_api_and_store_responses(base_prompt, word_to_transformations, all_combinations)

