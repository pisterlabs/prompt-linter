import os
import json
import csv
import re
import datetime
import argparse

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import utilities.llm_tasks_prompts as llm_tasks
import utilities.data_io as dio

# Create a list of QA pairs
qa_pairs = dict()

start_time = datetime.datetime.now()
qa_pairs['start_time'] = str(start_time)

# Add arguments to the command line
parser = argparse.ArgumentParser(description='Run the basic LLM on a dataset.')
parser.add_argument('-d', '--dataset', help='The dataset to run the LLM-KG on.', required=True)
args = parser.parse_args()

if args.dataset == "geography":
    dataset_path = "../data/mmlu_test/high_school_geography_test_filtered.csv"
    json_output_path = f"../output/qa_sets_basic_llm_geography_{start_time.timestamp()}.json"
elif args.dataset == "government_and_politics":
    dataset_path = "../data/mmlu_test/high_school_government_and_politics_test_filtered.csv"
    json_output_path = f"../output/qa_sets_basic_llm_government_and_politics_{start_time.timestamp()}.json"
elif args.dataset == "miscellaneous":
    dataset_path = "../data/mmlu_test/miscellaneous_test_filtered.csv"
    json_output_path = f"../output/qa_sets_basic_llm_miscellaneous_{start_time.timestamp()}.json"
else:
    print("Invalid dataset.")
    exit()

# Load the data
data = dio.read_data(dataset_path)

# Create the language model
llm = OpenAI(temperature=0)

num_correct = 0

# Generate a response for each question
for i, item in enumerate(data):
    question = item['question_text']
    response = llm_tasks.generate_response_weaker(question)  # TODO: Replace with either babbage or davinci

    print("Q:", question)
    print("A:", response.strip(), "\n")

    # Convert the list of choices to a string
    choices_text = "\n".join([str(i + 1) + ". " + choice for i, choice in enumerate(item['choices'])])
    print("Options:")
    print(choices_text)

    # Generate the letter output based on the response
    letter_output = llm_tasks.select_mc_response_based(question, response.strip(), item['choices'])

    print("Generated answer:", letter_output)

    qa_pairs[i] = dict()
    qa_pairs[i]["question"] = question
    qa_pairs[i]["choices"] = item['choices']
    qa_pairs[i]["initial_response"] = response.strip()

    # Evaluate the response
    is_correct = letter_output == item['correct_answer']
    print("Correct answer:", item['correct_answer'])
    print("Correct:", is_correct, "\n")

    # Update the number of correct answers
    if letter_output == item['correct_answer']:
        num_correct += 1

    qa_pairs[i]["llm_answer"] = letter_output
    qa_pairs[i]["llm_is_correct"] = is_correct

    with open(json_output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)

em = num_correct / len(data)
print("EM:", em)

qa_pairs['finish_time'] = str(datetime.datetime.now())
qa_pairs['EM'] = em

# Save the QA sets in a JSON file
with open(json_output_path, "w") as f:
    json.dump(qa_pairs, f, indent=4)
