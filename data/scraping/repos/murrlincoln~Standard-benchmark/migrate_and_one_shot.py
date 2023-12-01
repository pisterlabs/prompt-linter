import json
import unittest
import os
import openai
import re

# Set your OpenAI API key
openai.api_key = "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"

# Standardize the raw JSON
def generate_standard_json_from_raw_input(input_path, output_path):
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    standardized_problems = []
    for item in raw_data['problem_set']:
        raw_text = json.dumps(item, indent=4)
        prompt = f"Given the raw JSON data:\n\n{raw_text}\n\nStandardize the coding problems into problems with function name, function signature, and a detailed prompt. Be sure to label the funtion name with 'Function Name', the signature with 'Function Signature' and the prompt with 'Prompt' (capitalization important), and include no new line after Prompt. For context, I'm reading the response with             prompt_line = next(line for line in lines if 'Prompt:' in line) and             detailed_prompt = prompt_line.split(':', 1)[1].strip()  # Use split with maxsplit=1 to get the rest of the line after 'Prompt:''."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        standardized_text = response['choices'][0]['message']['content'].strip()
        lines = standardized_text.split('\n')

        try:
            function_name_line = next(line for line in lines if "Function Name:" in line)
            function_signature_line = next(line for line in lines if "Function Signature:" in line)
            prompt_line = next(line for line in lines if "Prompt:" in line)

            print(f"Debugging: {function_name_line}, {function_signature_line}, {prompt_line}")  # Debugging line

            function_name = function_name_line.split(":")[1].strip()
            function_signature = function_signature_line.split(":")[1].strip()
            detailed_prompt = prompt_line.split(":", 1)[1].strip()  # Use split with maxsplit=1 to get the rest of the line after "Prompt:"

            standardized_problem = {"function_name": function_name, "function_signature": function_signature, "prompt": detailed_prompt}
            standardized_problems.append(standardized_problem)
        except StopIteration:
            print("One or more expected fields were not found in the API response.")
            print(f"Debugging: Here is the full API response: {standardized_text}")  # Debugging line

            
    with open(output_path, 'w') as f:
        json.dump(standardized_problems, f, indent=4)


# Standardize the unorganized test Python file
def generate_unittest_from_unorganized_tests(input_file_path, output_test_file_path):
    with open(input_file_path, 'r') as f:
        unorganized_tests = f.read()
    prompt = f"The following Python code contains unorganized tests:\n\n{unorganized_tests}\n\nPlease reformat these tests into a unittest-compatible test.py file. Include nothing in your response except for the code itself, as it will be entered straight into a python file."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    organized_tests = response['choices'][0]['message']['content'].strip()
    with open(output_test_file_path, 'w') as f:
        f.write(organized_tests)

# Generate Python functions based on standardized JSON
def create_and_complete_problem_file(directory, problem):
     # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    function_name = problem['function_name']
    function_signature = problem['function_signature']
    prompt = problem['prompt']
    file_path = os.path.join(directory, f"{function_name}.py")
    #with open(file_path, 'w') as f:
        #f.write(f"# {function_name}\n\n")
        #f.write(f"{function_signature}\n")
        #f.write("# TODO: Complete this function\n")
    full_prompt = f"{prompt}\n\n{function_signature}  # TODO: Complete this function. Include nothing in your response except for the code itself, as it will be entered straight into a python file."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt}
        ]
    )
    completed_function = response['choices'][0]['message']['content'].strip()
    with open(file_path, 'a') as f:
        f.write(completed_function)


# Import necessary modules

def compile_problems_to_main(directory, output_file="main.py"):
    """
    Compiles all Python files in a given directory into a single main.py file.
    
    Parameters:
        directory (str): The directory containing the Python files to compile.
        output_file (str): The name of the output file. Defaults to "main.py".
        
    Returns:
        None
    """
    # Initialize an empty string to hold the content of all Python files
    combined_content = ""
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                content = f.read()
                combined_content += f"# Content from {filename}\n"
                combined_content += content + "\n\n"
    
    # Write the combined content to the output file
    with open(output_file, 'w') as f:
        f.write(combined_content)

# Example usage:
# compile_problems_to_main("problems")


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
    # Step 1: Standardize the raw JSON and unorganized tests
    INPUT_RAW_JSON_PATH = "raw_problems.json"
    OUTPUT_STANDARD_JSON_PATH = "problems.json"
    INPUT_UNORGANIZED_TESTS_FILE = "unorganized_tests.py"
    OUTPUT_TEST_FILE = "test.py"
    generate_standard_json_from_raw_input(INPUT_RAW_JSON_PATH, OUTPUT_STANDARD_JSON_PATH)
    generate_unittest_from_unorganized_tests(INPUT_UNORGANIZED_TESTS_FILE, OUTPUT_TEST_FILE)

    # Step 2: Generate Python functions
    problems = json.load(open(OUTPUT_STANDARD_JSON_PATH, 'r'))
    DIRECTORY_PATH = "problems"
    for problem in problems:
        create_and_complete_problem_file(DIRECTORY_PATH, problem)

    compile_problems_to_main("problems")


    # Step 3: Run the tests
    run_tests()
