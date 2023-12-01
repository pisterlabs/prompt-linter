import os
import openai
import subprocess
from helpers.openai import Model
from pathlib import Path

import dotenv
dotenv.load_dotenv()

assert os.getenv(
    "OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY in .env file"

# Please replace `your-openai-api-key` with your own OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = Model.GPT_4_32K
max_tokens = 2048
feedback = ""


def create_openai_chat_response(prompt: str) -> str:
    """Create a chat response using OpenAI API."""
    global feedback
    if feedback != "":
        prompt = f"This is what should be most important. Modify the code to follow the feedback as given: {feedback} \n{prompt}"
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=max_tokens,
    )
    completion = response["choices"][0]["message"]["content"]  # type: ignore
    filtered_completion = completion.replace(
        "```python", "").replace("```", "")
    return filtered_completion


def generate_code_from_prompt(prompt: str) -> str:
    """Generate Python code using OpenAI API based on the given prompt."""
    prompt = f"Only return python code to this problem with proper indentation and do not include any test suite in the output if given: {prompt}\nPython script:"

    return create_openai_chat_response(prompt)


def generate_test_cases_from_prompt(prompt: str) -> str:
    """Generate Python test cases using OpenAI API based on the given prompt."""
    prompt = f"{prompt}\nOnly write the Python test case code that will validate if the function works and write a main function that will run the test suite when run as the file and if the python function uses a gui then do not write proper tests and write one test that will return true after running the function:"
    return create_openai_chat_response(prompt)


def generate_file_name_from_prompt(prompt: str) -> str:
    """Generate file using OpenAI API based on the given prompt."""
    prompt = f"Create a short python file name with the file extension for this task: {prompt}"
    return create_openai_chat_response(prompt)


def write_code_to_file(code: str, filename: str | Path):
    """Write the given code to a file with the given filename."""
    with open(filename, 'w') as f:
        f.write(code)


def run_code(filename: str) -> str:
    """Run a Python code file and return the error output."""
    process = subprocess.Popen(
        ['python', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    return stderr.decode()


save_dir = Path(__file__).parent / "generated"

print(save_dir)


def code_loop(new_prompt: str = "", og_prompt: str = "", error: str = "", loop: int = 0, title: str = ""):
    """Run a loop that generates Python code and test cases, writes them to files, and runs the code."""
    loop += 1
    print(f"Loop: {loop}")
    if new_prompt and not error:
        inp = input("Does this code work? (y/n): ")
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
        title = generate_file_name_from_prompt(
            f"Create a short python file name with the file extension for this task: {og_prompt}")
        code = generate_code_from_prompt(og_prompt)
        test_prompt = f"The prompt: {og_prompt} \nThe code: {code}"
        test_cases = generate_test_cases_from_prompt(test_prompt)

        # Write both the code and test cases to the same file
        write_code_to_file(code, save_dir/title)
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
        code = generate_code_from_prompt(code_prompt)
        test_prompt = f"The prompt: {og_prompt} \n The code: {code}"
        test_cases = generate_test_cases_from_prompt(test_prompt)
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
    code_loop()
