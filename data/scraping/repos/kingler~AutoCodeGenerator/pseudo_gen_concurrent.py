import openai
import re
import json
import time
import concurrent.futures
from termcolor import colored
import sys
import tenacity

# define your openai api key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-..."

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def generate_concurrently(num_runs, model="gpt-3.5-turbo-16k-0613"):
    num_runs = int(num_runs)
    print("Generating high level outline...")
    # write the user instructions to a file
    with open('user_instructions.txt', 'r', encoding="utf-8", errors="ignore") as file:
        user_instructions = file.read()

    json_object = {}

    def run_api_call(run):
        # Initialize the API call
        response = None
        while response is None:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    stream=True,
                    # top_p=0.3,
                    temperature=0.5,
                    messages=[
                        {"role": "system", "content": """You are an expert python high level outline generator. You only generate perfect high level outline. not full code. You follow the instructions very carefully and diligently.
                        """},
                        {"role": "user", "content": f"""generate minimalistic  psecudo code for the following requirements:
                        attention!: do not assume anything about the code and only write the general pseudo variables, functions, classes etc...
                        make no assumptions about urls
                        consider anything which might lead to an error in user instructions and apply it to the high level outline
                        attention!: return your response as python markdown starting with ```python and ending with ```
                        user instructions: 

                        {user_instructions}"""}
                    ]
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

        responses = ''

        # Define colors for each concurrent process
        colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']

        # Process each chunk
        for i, chunk in enumerate(response):
            if "role" in chunk["choices"][0]["delta"]:
                continue
            elif "content" in chunk["choices"][0]["delta"]:
                r_text = chunk["choices"][0]["delta"]["content"]
                responses += r_text
                print(colored(r_text, colors[run % len(colors)]), end='', flush=True)

        # Parse the python code out of the response
        python_code = re.findall(r'```python\n(.*?)\n```', responses, re.DOTALL)
        print(python_code)

        # Append the python code to the JSON object
        # check that the python code is not empty
        if python_code:
            json_object[run+1] = python_code[0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for run in range(num_runs):
            futures.append(executor.submit(run_api_call, run))

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

    # Write the JSON object to a file
    with open(f'pseudo_competitors.json', 'w', encoding="utf-8", errors="ignore") as file:
        json.dump(json_object, file)

    print(f"Output saved to pseudo_competitors.json")

# generate_concurrently(5)
if __name__ == "__main__":
    generate_concurrently(3)
