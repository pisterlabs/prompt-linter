import os
import openai 
import subprocess 
import dotenv
from pathlib import Path

ENGINE = "gpt-3.5-turbo"
MAX_TOKENS = 2048
FEEDBACK = ""

# LOAD API KEY FROM ENV FILE
dotenv.load_dotenv()
assert os.getenv("API_KEY") is not None, "Set API_KEY in .env file"
openai.api_key = os.getenv("API_KEY")

def get_chat_response(prompt: str) -> str:
    """Get a chat response using the language model API"""
    global FEEDBACK
    # FIXME else???
    if FEEDBACK != "":
        prompt = f"This is what should be most important. Modify the code to follow the feedback as given: {feedback} \n{prompt}"
    response = openai.ChatCompletion.create(
        model=ENGINE,
        messages=[{"role": "user",
                   "content": prompt}],
        temperature=0.5,
        max_tokens=MAX_TOKENS,
    )
    completion = response["choices"][0]["message"]["content"]
    filtered_completion = completion.replace(
    "```python", "").replace("```", "")
    return filtered_completion

def gen_code_from_prompt(prompt: str) -> str:
    print("Generating code")
    prompt = f"Only return python code to this problem with proper indentation and do not include any test suite in the output if given: {prompt}\nPython script:"
    return get_chat_response(prompt)

def gen_test_cases_from_prompt(prompt: str) -> str:
    print("Generating test cases")
    prompt = f"{prompt}\nOnly write the Python test case code that will validate if the function works and write a main function that will run the test suite when run as the file and if the python function uses a gui then do not write proper tests and write one test that will return true after running the function:"
    return get_chat_response(prompt)

def gen_file_name_from_prompt(prompt: str) -> str:
    print("Generating file name")
    prompt = f"Create a short python file name with the file extension for this task: {prompt}"
    return get_chat_response(prompt)

def write_code_to_file(code: str, filename: str):
    print("Writing code to file")
    with open(filename, 'w') as f:
        f.write(code)

def run_code(filename: str) -> str:
    print("Running code")
    """Run a Python code file and return the error output."""
    process = subprocess.Popen(
        ['python3', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    return stderr.decode()

save_dir = Path(__file__).parent / "generated"

print(save_dir)

def code_loop(new_prompt: str = "", og_prompt: str = "", error: str = "", loop: int = 0, title: str = ""):
    """Run a loop that generates Python code and test cases, writes them to files, and runs the code."""
    loop += 1
    print(f"Loop: {loop}")
    if new_prompt and not error:
        inp = input("Are you happy with this code? (y/n): ")
        if inp.lower() == "y":
            print("Great! Your code is ready to submit!")
            return
        elif inp.lower() == "n":
            global feedback

            feedback = input(
                "Please enter the feedback you would like to give to the AI.\nFeedback: ")
            error = "Human feedback given"
            code_loop(new_prompt, og_prompt, error, loop, title)
    elif new_prompt == "":
        og_prompt = input("Please enter the prompt for the task: ")
        # og_prompt = "Create a gui that will display a button that will display a message when clicked"
        title = gen_file_name_from_prompt(
            f"Create a short python file name with the file extension for this task: {og_prompt}")
        code = gen_code_from_prompt(og_prompt)
        test_prompt = f"The prompt: {og_prompt} \nThe code: {code}"
        test_cases = gen_test_cases_from_prompt(test_prompt)

        # Write both the code and test cases to the same file
        write_code_to_file(code, os.path.join(save_dir, title))  #FIXME Invalid argument: 'C:\\Users\\zsyed\\OneDrive\\Documents\\code\\applet-bucket\\generated\\filename = "even_numbers.py"'
        write_code_to_file(test_cases, 'test.py')
        test_str = f"{code} \n{test_cases}"
        write_code_to_file(test_str, 'checker.py')
        error = run_code("checker.py")
        print('=====================')
        print(f"Code: \n{code}")
        print('---------------------')
        print(f"Test Cases: \n{test_cases}")
        print('---------------------')
        print(f"Error: \n{error}")
        print('=====================')

        code_loop(test_str, og_prompt, error, loop, title)
    else:
        prompt = f"previous code with the test suite: {new_prompt} \n error output: {error}"
        code_prompt = f"{prompt} \nWrite new code to fix the error"
        code = gen_code_from_prompt(code_prompt)
        test_prompt = f"The prompt: {og_prompt} \n The code: {code}"
        test_cases = gen_test_cases_from_prompt(test_prompt)
        write_code_to_file(code, title)
        write_code_to_file(test_cases, 'test.py')
        test_str = f"{code} \n{test_cases}"
        write_code_to_file(test_str, 'checker.py')
        error = run_code("checker.py")
        print('=====================')
        print(f"Code: \n{code}")
        print('---------------------')
        print(f"Test Cases: \n{test_cases}")
        print('---------------------')
        print(f"Error: \n{error}")
        print('=====================')
        code_loop(test_str, og_prompt, error, loop, title)

if __name__ == "__main__":
    # TODO: set up i.e. if there is no generated file, make a generated file, maybe should be moved to an init function
    gen_dirname = os.path.join(os.getcwd(), 'generated')
    if not os.path.exists(gen_dirname):
        os.makedirs(gen_dirname)
    
    code_loop()