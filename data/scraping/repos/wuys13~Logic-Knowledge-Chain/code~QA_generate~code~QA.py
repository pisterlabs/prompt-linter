import os
import csv
import openai
import json
import time
import argparse


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')

# Define a function to get prompt text
def get_text(prompt_file):
    with open(prompt_file, 'r') as file:
        return file.read()


# Define function to make API request
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,
    )
    return response.choices[0].message["content"]


# Function to process text file
def process_text_file(text_file, prompt_file, QA_num = 5):
    prompt_text = get_text(prompt_file)
    results = []  # Create a new list to store the results

    with open(text_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)  # convert to list to allow slicing
        if QA_num == -1:
            QA_num = len(rows)
        for i in range(0, QA_num, 5):
            batch = rows[i:i + 5]
            for row in batch:
                text = "'".join(row)
                prompt = f"{prompt_text}\n{text}"
                response = get_completion(prompt)
                try:
                    # Try to parse the response as JSON
                    json_data = json.loads(response)
                    results.append(
                        json_data)  # Add it to the results list if successful
                except json.JSONDecodeError:
                    # Ignore responses that are not valid JSON
                    print(f"Ignoring non-JSON response: {response}")
            time.sleep(3)  # Wait for 5s second between batches
    return results
        


# Specify paths to your files
prompt_directory = '../prompt'

# Define prompts
decision_prompt_file = os.path.join(prompt_directory, 'decision_prompt.txt')
multiple_choice_prompt_file = os.path.join(prompt_directory,
                                           'multiple_choice_prompt.txt')
short_answer_prompt_file = os.path.join(prompt_directory,
                                        'short_answer_prompt.txt')

# Map the prompt types to the file names
prompt_files = {
    'decision': decision_prompt_file,
    'multiple_choice': multiple_choice_prompt_file,
    'short_answer': short_answer_prompt_file
}




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Logic Knowledge Chain QA')
    parser.add_argument('--prompt_type', default='decision',
                        choices=['decision','multiple_choice','short_answer'])
    parser.add_argument('--prompt_num', default = None,type=int)
    parser.add_argument('--output_dir', default='../result')
    parser.add_argument('--QA_num', default=-1 ,type=int)
    parser.add_argument('--dataset', default='prism')
    args = parser.parse_args()

    Knowledge_file = os.path.join('../data', args.dataset + '.csv')
    
    try:
        openai_key = get_text('../openai_key.txt')
        print('openai key loaded')
    except Exception as e:
        print(e)
        print('Please put your openai key in ../openai_key.txt')
    openai.api_key = openai_key

    if args.prompt_num : 
        args.prompt_type = [
                       'no','decision','multiple_choice','short_answer'][args.prompt_num]
        print('')
        print('prompt_type:', args.prompt_type)

    # Check that the input corresponds to a known prompt type
    print('')
    if args.prompt_type not in prompt_files:
        print(
            f"Unknown prompt type '{args.prompt_type}'. Please enter one of the following: decision, multiple_choice, short_answer."
        )
    else:
        print(f"Processing {args.prompt_type} prompts...")
        # Call the function
        results = process_text_file(Knowledge_file, prompt_files[args.prompt_type], QA_num = args.QA_num)
        print(f"Processed {len(results)} prompts.")

        save_dir = os.path.join(args.output_dir, args.dataset)
        safe_make_dir(save_dir)
        # Generate the file name based on the prompt type
        file_name = f"{save_dir}/{args.prompt_type}_results.json"
        print(f"Saving results to {file_name}...")

        # Write the results to a .json file
        with open(file_name, 'w') as file:
            json.dump(
                results, file,
                indent=4)  # The indent parameter makes the output easier to read
        print("Done!")
