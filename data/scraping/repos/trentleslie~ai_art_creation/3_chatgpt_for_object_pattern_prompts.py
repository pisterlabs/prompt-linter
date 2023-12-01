import json
import os
import pickle
import random
import re
import time
from datetime import datetime

import openai
import pandas as pd

from ai_art_creation.api.api_key import api_key
from ai_art_creation.prompts.prompt_path_info import (
    base_prompt_output_path, keywords_folder_path,
    object_pattern_csv_filtering_path, object_pattern_pickle_file_path,
    object_pattern_products_path, starting_keywords_path)

# Create a timestamp
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
final_file_name = f"concatenated_prompts_{timestamp}.txt"
final_file_name_filtering = f"concatenated_prompts_to_filter_{timestamp}.csv"

# Function for getting the list of relevant keywords and products
def get_keywords_list(starting_folder_path, 
                      starting_keywords_path, 
                      object_pattern_products_path,
                      file_type=".csv"):
    
    # if generating prompts from archived keywords, use this path
    #starting_keywords_path = r'C:\Users\trent\OneDrive\Documents\GitHub\ai_art_creation\ai_art_creation\keywords\starting_keywords_archive.txt'
    
    # Load starting keywords
    with open(starting_keywords_path, 'r') as f:
        # Read lines into a list
        starting_keywords = f.read().splitlines()

    # Load products
    with open(object_pattern_products_path, 'r') as f:
        # Read lines into a list
        all_products = f.read().splitlines()
    
    # Get list of all files
    files = os.listdir(starting_folder_path)

    # Sort list of files based on their modified time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(starting_folder_path, x)))

    # The latest file would be the last file in the sorted list
    latest_file = files[-1]

    # Ensure the file is a CSV file
    if latest_file.endswith(file_type):
        # Full file path
        file_path = os.path.join(starting_folder_path, latest_file)
    else:
        print(f"The latest file in the directory is not a {file_type} file.")
        
    df = pd.read_csv(file_path)
    
    # Create keywords_to_run list
    keywords_to_run = [keyword for keyword in starting_keywords if any(keyword.lower() in element.lower() for element in df['starting_keyword'].values)]

    # Create products_to_record list
    products_to_record = [product for product in all_products if any(product.lower() in element.lower() for element in df['starting_keyword'].values)]
    
    # Initialize an empty dictionary
    keywords_products_dict = {}

    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        # Get the 'starting_keyword' value
        starting_keyword = row['starting_keyword']
        
        # Find the associated keyword_to_run and product
        keyword_to_run = next((keyword for keyword in starting_keywords if keyword.lower() in starting_keyword.lower()), None)
        product = next((product for product in all_products if product.lower() in starting_keyword.lower()), None)
        
        # If both the keyword_to_run and product were found
        if keyword_to_run is not None and product is not None:
            # If the keyword_to_run is already a key in the dictionary, append the product to its list
            if keyword_to_run in keywords_products_dict:
                keywords_products_dict[keyword_to_run].append(product)
            # Otherwise, add the keyword_to_run as a new key and create a new list with the product as the first item
            else:
                keywords_products_dict[keyword_to_run] = [product]
    
    # Compile a regular expression pattern to match whole words
    #pattern = re.compile(r'\b(?:%s)\b' % '|'.join(map(re.escape, all_products)), flags=re.IGNORECASE)

    # Remove matched words from detailed_keywords_to_run
    #keywords_to_run.extend([pattern.sub('', keyword).strip() for keyword in df['Keywords/Phrase'].values])

    print(keywords_to_run)
    
    return (keywords_to_run, products_to_record, keywords_products_dict)

# Function for getting random gpt params
def get_random_gpt_params():
    models = ['gpt-3.5-turbo-0613', 'gpt-4-0613']
    rand_model = random.choice(models)
    rand_temperature = round(random.uniform(0.3, 0.9), 2)
    rand_top_p = round(random.uniform(0.3, 0.9), 2)
    rand_presence_penalty = round(random.uniform(0.3, 0.9), 2)
    rand_frequency_penalty = round(random.uniform(0.3, 0.9), 2)

    params = {
        "model": rand_model,
        "temperature": rand_temperature,
        "top_p": rand_top_p,
        "presence_penalty": rand_presence_penalty,
        "frequency_penalty": rand_frequency_penalty,
    }

    return params

#--------------------------------------------------------------------------------------------------------------#
#--------------------------------------------GPT Prompt Function-----------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def generate_object_prompts(keyword, 
                           model="gpt-4", 
                           temperature=0.5, 
                           top_p=0.5, 
                           presence_penalty=0.5, 
                           frequency_penalty=0.5):
    
    # Set your OpenAI API key
    openai.api_key = api_key

    system_prompt = f'''As a prompt generator for a generative AI called "Midjourney", you will create image prompts for the AI to visualize on a white #000000 background. I will give you a concept, and you will provide detailed prompts for Midjourney AI to generate an image.

                        Please adhere to the structure and formatting below, and follow these guidelines:

                        - Specify to put the focus on the object, not the background.
                        - Specify the backgrounnd to be white #000000 every time.
                        - Do not use the words "description" or ":" in any form.
                        - Provide a vivid description of the object.
                        - The generated images are intended to be used for merchandise, such as posters, t-shirts, and mugs, so try to specify trendy, cute, and/or cool concepts.

                        Structure:
                        [1] = {keyword}
                        [2] = a detailed description of [1] with specific imagery details.
                        [3] = a detailed description of the scene's environment.
                        [4] = a detailed description of the scene's mood, feelings, and atmosphere.
                        [5] = A style (e.g. watercolor, geometric, abstract, etc.) for [1].
                        [6] = A description of how [5] will be executed (e.g. repeated patterns of unique objects, one large object, etc.)

                        Formatting: 
                        Follow this prompt structure: "/imagine prompt: [1], [2], [3], [4], [5], [6]".

                        Your task: Create ten (10) distinct and unique prompts for each concept [1], varying in description, environment, atmosphere, and realization.

                        Example Prompts:
                        /imagine prompt: A cute and trendy cactus in a repeating pattern with a geometric design, vibrant colors, and minimalist style, surrounded by small rocks in a desert environment, evoking a sense of calmness and serenity, Watercolor painting on paper. Repeating pattern on a white #000000 background
                        /imagine prompt: An abstract and stylish cactus with ink drawing details, showcasing intricate patterns and textures on its surface. The background should be white #000000 to emphasize the cactus's unique design. The atmosphere should be calming and peaceful. One large cactus. #000000 background
                        /imagine prompt: A vibrant and cool scene of multiple cacti in a desert environment, with the sun setting in the distance. The cacti should have different shapes and sizes, creating an interesting composition. The atmosphere should be warm and inviting. Repeating pattern. On a white #000000 background
                        /imagine prompt: A geometric and trendy cactus garden, featuring various types of cacti arranged in a visually pleasing pattern. The background should be white #000000 to make the colors pop. The atmosphere should be modern and chic. Watercolor painting on canvas. On a white #000000 background
                        '''
                                
    # Define a list of styles
    style = f'''Please provide ten (10) unique prompts for AI art creation involving {keyword}. Along with always specifying "on a white #000000 background, Incorporate a few of the following key phrases in your prompt:
            "Geometric",
            "Cute",
            "Trendy",
            "Cool",
            "Stylish",
            "Vibrant",
            "Abstract",
            "Minimalist",
            "Vectorized cartoon style",
            "Repeated pattern"
        '''

    #build messages payload
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": style},
    ]
    
    functions = [
                {
                    "name": "get_ai_art_prompts",
                    "description": "Returns ten unique prompts for AI art creation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt1": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt2": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt3": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt4": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt5": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt6": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt7": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt8": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt9": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                            "prompt10": {
                                "type": "string",
                                "description": "A detailed description of the object with specific imagery details.",
                            },
                        },
                        "required": ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6", "prompt7", "prompt8", "prompt9", "prompt10"],
                    },
                }
            ]
    
    function_call={"name": "get_ai_art_prompts"}
    
    max_retries = 10
    retry_count = 0
    retry_flag = True
            
    while retry_flag and retry_count < max_retries:
        try:
            #call the ChatCompletion end
            prompts_list_length = 0
            while prompts_list_length < 10:
                print("Generating prompts...")
                response = openai.ChatCompletion.create(
                        model = model,
                        messages=messages,
                        temperature = temperature,
                        top_p = top_p, 
                        presence_penalty = presence_penalty,
                        frequency_penalty = frequency_penalty,
                        functions = functions,
                        function_call=function_call        
                    )
                
                reply_content = response.choices[0].message
    
                # Parse the JSON string into a Python dictionary
                args_dict = json.loads(reply_content['function_call']['arguments'])

                # Get a list of all the values associated with the arguments
                prompts_list = [v for k, v in args_dict.items()]
                
                prompts_list_length = len(prompts_list)
                
                time.sleep(3)
                
                #newline_count = response['choices'][0]['message']['content'].count('\n')
                #print("Newline count:", newline_count)
                
            retry_flag = False
            
        except Exception as e:
            print(f"Exception occurred in OpenAI API call: '{e}' Retrying...")
            retry_count += 1
            
        if retry_count == max_retries - 1:
            print("Max retries reached. Skipping this prompt...")
            return []

    # Extract the generated text from the API response
    #generated_text = (response['choices'][0]['message']['content'])

    # If it does, split the string into a list of prompts
    #prompts_list = generated_text.split("\n")

    # Remove any empty or whitespace elements from the list, and any that start with "Prompt"
    #prompts_list = [prompt.strip() for prompt in prompts_list if prompt.strip() and not prompt.startswith("Prompt")]
    
    #prompts_list = [s + " On a white #000000 background" for s in prompts_list]
     
    print(len(prompts_list) , "more prompts generated!")
    
    return prompts_list

#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------Generate object Prompts----------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

def get_prompts(key_prod_dict,
                keywords_list, 
                products_list, 
                pickle_file_path, 
                generate_func):
    try:
        with open(pickle_file_path, 'rb') as pickle_file:
            completed_keywords = pickle.load(pickle_file)
    except FileNotFoundError:
        completed_keywords = []  # initialize as an empty list if the file doesn't exist

    # Create a timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

    # Specify the path for the output text file
    output_path = f"{base_prompt_output_path}object_pattern\\all_prompts_object_pattern_{timestamp}.txt"

    # Create an empty DataFrame to store the prompt data
    prompts_data = pd.DataFrame(columns=['starting_keyword', 'gpt_model', 'gpt_keyword', 'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'prompt', 'product', 'keyword_product', 'timestamp'])

    # Open the file in write mode
    with open(output_path, 'w') as file:
        # Iterate over the keywords list
        for i, keyword in enumerate(keywords_list, 1):                
            try:
                # Attempt to split the string and get the second element
                keyword_for_gpt = keyword.split(":")[1]
            except IndexError:
                # If there is no ":" in the string, just use the whole string as the keyword
                keyword_for_gpt = keyword
                
            # If the keyword has already been processed, skip it
            if keyword_for_gpt not in completed_keywords:
                print(f"Processing keyword {i} of {len(keywords_list)}: {keyword_for_gpt}")
                
                params = get_random_gpt_params()
                
                # Generate prompts for the current keyword
                prompts = generate_func(keyword=keyword_for_gpt, 
                                                model=params['model'],
                                                temperature=params['temperature'], 
                                                top_p=params['top_p'], 
                                                presence_penalty=params['presence_penalty'],
                                                frequency_penalty=params['frequency_penalty'])
            
                if len(prompts) > 0:
                    # Write prompts to the file as they are generated
                    for prompt in prompts:
                        file.write("%s\n" % str(prompt))
                        
                        for product in key_prod_dict[keyword]:
                            # Append a record to the DataFrame
                            prompts_data = prompts_data._append({
                                'starting_keyword': keyword.split(":")[0], 
                                'gpt_model': params['model'], 
                                'gpt_keyword': keyword_for_gpt, 
                                'temperature': params['temperature'], 
                                'top_p': params['top_p'], 
                                'presence_penalty': params['presence_penalty'], 
                                'frequency_penalty': params['frequency_penalty'], 
                                'prompt': prompt,
                                'product': product,
                                'keyword_product': f'{keyword_for_gpt} {product}',
                                'timestamp': int(time.time())  # Inserting the current Unix timestamp
                            }, ignore_index=True)
                    
                    # keyword is completed, add it to the list
                    completed_keywords.append(keyword_for_gpt)

                    # update the pickle file
                    with open(pickle_file_path, 'wb') as pickle_file:
                        pickle.dump(completed_keywords, pickle_file)
                else:
                    print(f"No prompts generated for this keyword: {keyword_for_gpt}. Moving on to the next one...")
            else:
                print(f"Keyword {keyword_for_gpt} has already been processed. Moving on to the next one...")

    # Write DataFrame to a CSV file
    csv_output_path = f"C:/Users/trent/OneDrive/Documents/GitHub/ai_art_creation/ai_art_creation/prompts/object_pattern/csv_prompt_filtering/prompts_to_filter_{timestamp}.csv"
    prompts_data.to_csv(csv_output_path, index=False)
    
    print("Prompt generation completed!")
    
    return completed_keywords

def concatenate_txt_files(dir_path, final_file_name):
    # Get a list of all the txt files in the directory
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt') and 'products' not in f and 'concatenated' not in f]

    # Initialize an empty string to store the content
    content = ""

    # Loop through each txt file
    for txt_file in txt_files:
        # Open the txt file
        with open(os.path.join(dir_path, txt_file), 'r', encoding='cp1252') as f:
            # Read the content
            file_content = f.read()

            # Concatenate the content
            content += file_content

        # Delete the file after concatenation
        os.remove(os.path.join(dir_path, txt_file))

    # Create a new txt file to store the concatenated content
    with open(os.path.join(dir_path, final_file_name), 'w', encoding='utf-8') as f:
        f.write(content)

    print("Concatenation is complete. All text data is stored in '{}'".format(final_file_name))

def concatenate_csv_files(dir_path, final_file_name):
    # Get a list of all the csv files in the directory
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

    # Initialize an empty DataFrame to store the content
    all_data = pd.DataFrame()

    # Loop through each csv file
    for csv_file in csv_files:
        if "concatenated" not in csv_file:
        # Read the csv file
            df = pd.read_csv(os.path.join(dir_path, csv_file))

            # Concatenate the DataFrames
            all_data = pd.concat([all_data, df])

            # Delete the file after concatenation
            os.remove(os.path.join(dir_path, csv_file))

    # Create a new csv file to store the concatenated content
    all_data.to_csv(os.path.join(dir_path, final_file_name), index=False, encoding='utf-8')

    print(f"Concatenation is complete. All data is stored in '{final_file_name}'")

print("Getting keywords and products...")

# Get the list of keywords to run and the list of products to record
keywords_to_run, products_to_record, keyword_product_dict = get_keywords_list(keywords_folder_path,
                                                            starting_keywords_path,
                                                            object_pattern_products_path,
                                                            file_type=".csv")

print("Keywords and products retrieved! Starting Run 1...")

num_runs = 20 # you can change this to the number of runs you need


for i in range(num_runs):
    print(f"Starting Run {i+1}...")
    
    completed_keywords_check = [] 

    while len(completed_keywords_check) < len(keywords_to_run):
        # Get the prompts
        completed_keywords_check = get_prompts(keyword_product_dict,
                                                keywords_to_run, 
                                                products_list=products_to_record,
                                                pickle_file_path=object_pattern_pickle_file_path,
                                                generate_func=generate_object_prompts)

        # Delete the pickle file if all went well
        if len(completed_keywords_check) == len(keywords_to_run):
            print("All keywords have been processed! Deleting the pickle file...")
            os.remove(object_pattern_pickle_file_path)

    print(f"Run {i+1} completed!")

print("All runs completed!")

# Concatenate all the txt files into one
concatenate_txt_files(f'{base_prompt_output_path}object_pattern', final_file_name)
concatenate_csv_files(object_pattern_csv_filtering_path, final_file_name_filtering)