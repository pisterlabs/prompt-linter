import os
import json
import openai
from dotenv import dotenv_values, load_dotenv
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import multiprocessing
import time

config = dotenv_values("../.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')

src_path = "../datasets/llava_instruct_150k.json"
dst_path = "../datasets/llava_instruct_150k_docent_v1.json"

#####################################################
# GPT call
#####################################################
def gpt_call(
    prompt,
    model="gpt-4" # model="gpt-4",
    ):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                    {"role": "user", "content": prompt},
                ]
    )
    output_text = response["choices"][0]["message"]["content"]

    print("output_text", output_text)

    return output_text


def prompt_func(instruction_prompt):

    prompt_template = PromptTemplate(
        input_variables=[
            "prompt"
            ],
        template="\n".join([
            "data:{prompt}",
            "Change \"gpt\" value to docent style in art museum.",
            "The name of \"human\" and \"gpt\" must be maintained.",
            "data:"
        ])
    )
    prompt_template = prompt_template.format(
        prompt=instruction_prompt
    )
    print("prompt_template", prompt_template)

    return prompt_template


call_counter = 0  # This counter is for the gpt_call rate limit
line_counter = multiprocessing.Value('i', 0)  # Shared counter for lines

def rate_limit():
    global call_counter
    time_interval = 60.0  # one minute

    if call_counter >= 20000:
        time.sleep(time_interval)
        call_counter = 0  # reset counter after waiting

def append_to_dst(data):
    with open(dst_path, "a") as f:
        json.dump(data, f)
        f.write(",\n")

def process_data(chunk):
    global call_counter
    final_result_chunk = []

    for i in chunk:
        rate_limit()  # Ensure we do not exceed the rate limit
        
        result = gpt_call(prompt_func(str(i)))
        call_counter += 1
        
        # Convert single quotes to double quotes for valid JSON
        valid_json_str = result.replace("'", '"')

        # Load the string as a dictionary
        try:
            dictionary_representation = json.loads(valid_json_str)
            
            if type(dictionary_representation) == dict:
                final_result_chunk.append(dictionary_representation)
                append_to_dst(dictionary_representation)

                # Increment line counter and check if it exceeds 10,000
                with line_counter.get_lock():
                    line_counter.value += 1
                    if line_counter.value >= 10000:
                        os._exit(0)  # Kills the entire process
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {valid_json_str}")

    return final_result_chunk

def count_processed_data():
    with open(dst_path, "r") as f:
        # Count the number of lines that are not just "[", "]", or ","
        return sum(1 for line in f if line.strip() not in ["[", "]", ","])

def main():
    # Check if the dst_path exists
    start_index = 0
    if os.path.exists(dst_path):
        # Count processed data and set the start index to continue from there
        start_index = count_processed_data()

    # Clear or initialize the destination file if starting from scratch
    if start_index == 0:
        with open(dst_path, "w") as k:
            k.write("[\n")

    with open(src_path, "r") as f:
        data = json.load(f)
    
    # Skip already processed data
    data = data[start_index:]

    # Create chunks for multiprocessing
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(data) // num_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    with multiprocessing.Pool(num_cores) as pool:
        # Use tqdm here to show progress bar
        results = list(tqdm(pool.imap(process_data, chunks), total=len(chunks)))

    # Close the JSON array in the destination file
    with open(dst_path, "rb+") as k:
        # Go to the second last character in the file
        k.seek(-2, 2)
        k.truncate()
        k.write(b"\n]")

if __name__ == '__main__':
    main()