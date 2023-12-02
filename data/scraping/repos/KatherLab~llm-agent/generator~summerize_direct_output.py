import os

import openai
import pandas as pd

from config import csv_path, output_dir
from utils import generate_evaluation

scoring_model = "gpt-4"
# Get the directory that the current script is in
script_dir = os.path.dirname(os.path.realpath(__file__))

# Read the task list
task_list = pd.read_csv(csv_path)

# Define the output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# recursively get all the txt files under output_dir
txt_files = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".txt"):
             txt_files.append(os.path.join(root, file))
md_files = []

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".md"):
            if file.startswith("onlyscore"):
                print(file)
                md_files.append(os.path.join(root, file))
print(len(txt_files))
#for txt_file in txt_files:
    #print(txt_file)
print(len(md_files))
# remove the txt files that already have md files
for md_file in md_files:
    # remove the onlyscore_ prefix
    md_file = md_file.replace("onlyscore_", "")
    #print(md_file)
    if md_file.replace(".md", ".txt") in txt_files:
        txt_files.remove(md_file.replace(".md", ".txt"))
print(len(txt_files))

# Define the evaluation prompt for ChatGPT-4


for file in txt_files:
    print(file)
    model = file.split("/")[-1].split("_")[0]
    truncation_index = file.split("/")[-1].split("_")[1].split("-")[-1]
    task_index = file.split("/")[-1].split("_")[2]
    repetition_index = file.split("/")[-1].split("_")[-1].split(".")[-2]
    with open(file, 'r') as file:
        generated_text = file.read()

    # Generate the .md file (this could be replaced with an actual function call or API request)
    md_filename = f"onlyscore_{model}_trunc-{truncation_index}_{task_index}_{repetition_index}.md"

    row = task_list.loc[task_list['index'] == int(task_index)].iloc[0]
    eu_prompt = row['Prompt*']
    discription = row['Description']
    print(eu_prompt)
    print(discription)
    evaluation_prompt = f"""
        You are a critical evaluator.
        I will give you the output of an AI model that was prompted to provide a hypothetical approach to solve a complex question. The objective is the following: {discription}. The initial prompt assigned to the AI model is {eu_prompt}.
        Be critical, and use the whole range from 1 to 10, depending on the quality of the prompt, 10 being the optimal, perfect answer to the problem that you can imagine, and 1 being completely off-topic. Score the prompt from 1 to 10 for the following 5 categories, just by printing the number you would rate the prompt (e.g. "Factual accuracy: 8" or “Factual accuracy: 3”), taking into account these descriptions for the categories:  
        Factual accuracy	How factually accurate is the output? Rate the accuracy from 1 to 10. 
        Problem Solving / Relevance:	How well did the model solve the problem? Rate the problem solving ability from 1 to 10.
        Novelty / Creativity	Assess the model's capacity for creativity and proposing novel or unconventional solutions from 1 to 10.
        Specificity	Rate the level of detail from 1 to 10
        Feasibility	Rate from 1 to 10 how feasible the solution is in terms of available technologies, resources, tools, economic and/or legal feasibility.
        Strictly format your output by following the format below, do not add anything else to the output, maintain the exact formatting and don’t add additional empty lines. Don't give hints to the format of the initial output. Adhere to this format:
        Score: 
        Factual accuracy: “integer score”
        Problem Solving / Relevance: “integer score”
        Novelty / Creativity: “integer score”
        Specificity: “integer score”
        Feasibility: “integer score”
    """

    evaluation = generate_evaluation(openai.api_key, scoring_model, evaluation_prompt, generated_text)
    file_foldername = os.path.join(output_dir, model)
    if not os.path.exists(file_foldername):
        os.makedirs(file_foldername)
    with open(os.path.join(file_foldername, md_filename), 'w') as file:
        file.write(f"## Score table by {scoring_model}\n")
        file.write(evaluation)