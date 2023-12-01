import json
import unittest
import os
import openai

# Set your OpenAI API key
openai.api_key = "your_openai_api_key_here"

# Generate Python functions based on standardized JSON
def create_and_complete_problem_file(directory, problem):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    function_name = problem['function_name']
    function_signature = problem['function_signature']
    prompt = problem['prompt']

    file_path = os.path.join(directory, f"{function_name}.py")
    full_prompt = f"{prompt}\n\n{function_signature}  # TODO: Complete this function"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )

    completed_function = response['choices'][0]['message']['content'].strip()

    with open(file_path, 'w') as f:
        f.write(completed_function)

# Compile individual problem files into one main.py
def compile_problems_to_main(directory, output_file="main.py"):
    combined_content = ""
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            with open(os.path.join(directory, filename), 'r') as f:
                combined_content += f"# {filename}\n{f.read()}\n\n"

    with open(output_file, 'w') as f:
        f.write(combined_content)

# Run unittests
def run_tests():
    current_directory = os.getcwd()
    if os.path.exists(os.path.join(current_directory, 'test.py')):
        loader = unittest.TestLoader()
        suite = loader.discover(current_directory, pattern='test.py')
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        print("No test.py found in the current directory.")

if __name__ == "__main__":
    # Step 1: Read the standardized JSON
    OUTPUT_STANDARD_JSON_PATH = "problems.json"
    problems = json.load(open(OUTPUT_STANDARD_JSON_PATH, 'r'))

    # Step 2: Generate Python functions
    DIRECTORY_PATH = "problems"
    for problem in problems:
        create_and_complete_problem_file(DIRECTORY_PATH, problem)

    # Step 3: Compile all Python files into a single main.py
    compile_problems_to_main(DIRECTORY_PATH)

    # Step 4: Run the tests
    run_tests()
