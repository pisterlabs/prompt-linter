import sys
import subprocess

import openai
import pandas as pd
import os
#from generator.babyagi import babyagi
def generate_evaluation(api_key,model, prompt, input_text):
    """
    Function to generate an evaluation of the input_text using the OpenAI API.

    Parameters:
    api_key (str): The OpenAI API key.
    model (str): The name of the model to use.
    prompt (str): The evaluation prompt.
    input_text (str): The text to evaluate.

    Returns:
    str: The evaluation of the input_text.
    """

    # Define the API request
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text}
        ]
    )
    generated_text = response['choices'][0]['message']['content']

    return generated_text

# Set the OpenAI API key
openai.api_key = ''


# Read the task list
task_list = pd.read_csv("llm_task_list.csv")

# Define the output directory
output_dir = "results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Define the evaluation prompt for ChatGPT-4

scoring_model = "gpt-4"
ll_agi_model_list = ["gpt-3.5-turbo", "gpt-4"] #"gpt-4",
# For each task
for index, row in task_list.iterrows():
    # For each repetition
    for model in ll_agi_model_list:
        for repetition_index in range(3):
            # Define the prompt
            eu_prompt = row['Prompt*']
            evaluation_prompt = f"""
            I will give you the output of an AI model that was prompted to provide a hypothetical approach to solve a complex question. 
            The objective objective it to {row['Description']}. The initial prompt assigned to the AI model is {eu_prompt}.
            Please summarize this output in a table with the following columns: 
            summary (content should be three concise sentences), 
            main ideas (content should be three very short bullet points), 
            main finding (content should be three very short bullet points), 
            novelty (rate the novelty of this approach from 1 to 10), 
            feasibility (rate the feasibility of this approach from 1 to 10), 
            correctness (rate the factual correctness of the output from 1 to 10). 
            Remember that the approach was hypothetical, i.e. the AI model did not actually perform any study. 
            Do not suggest in your output that actual research was done, rather, emphasize the hypothetical idea. OK?
            """
            system_prompt = "You are a medical advisor."
            input_text = eu_prompt
            generated_text = generate_evaluation(openai.api_key, model, system_prompt, input_text)

            # Write the generated text to a .txt file
            txt_filename = f"{model}_{row['index']}_{repetition_index}.txt"
            with open(os.path.join(output_dir, txt_filename), 'w') as file:
                file.write(generated_text)

            # call python generator/babyagi/babyagi.py
            subprocess.run(["python", "generator/babyagi/babyagi.py", "--task_index", row['index'], "--repetition_index", repetition_index])

            # Generate the .md file (this could be replaced with an actual function call or API request)
            md_filename = f"{model}_{row['index']}_{repetition_index}.md"
            evaluation = generate_evaluation(openai.api_key, scoring_model, evaluation_prompt, generated_text)
            with open(os.path.join(output_dir, md_filename), 'w') as file:
                file.write(f"## Summary table by {scoring_model}\n")
                file.write(evaluation)
