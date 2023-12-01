from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import os
import io
import sys
import traceback
import re
import time


# Set env variables here or in .env
os.environ["OPENAI_API_BASE"] = "http://localhost:443/v1" # comment this in if using gpt-llama.cpp
os.environ["OPENAI_API_KEY"] = "../llama.cpp/models/wizardLM/7B/wizardLM-7B.GGML.q5_1.bin"

code_generated = False
attempt_num = 1

def extract_code(response):
    code_blocks = re.findall(r"```(?:python|bash)?\s*[\s\S]*?```", response)
    extracted_code = {"python": [], "bash": []}
    
    for code_block in code_blocks:
        code_type, code = re.match(
            r"```(python|bash)?\s*([\s\S]*?)```", code_block
        ).groups()
        code_type = code_type or "python"
        extracted_code[code_type].append(code.strip())

    if len(extracted_code['python']) == 0:
        return response
    else:
        return "\n".join(extracted_code['python']) # only return python for now



# Simple completion example
llm = OpenAI(temperature=0.7)

# text = """Write python code that takes in a list and sorts it:"""
# code_request = "function that takes in a string and returns true if its a palindrome, false otherwise:"
# code_request = "function that extracts the code within ```python ``` in a given string"
code_request = "function that extracts the name and email from a user json object"
text = f"""Write python code that implements the following: 
{code_request}

Code:"""
print('\n===== REQUEST =====')
print(text)

print('\n===== CODE GENERATED =====')
result = llm(text)
result = extract_code(result)
print(result)


while not code_generated:
    time.sleep(1) # sleep for 1 second to avoid too many loop bois


    # Compile the code string into a code object
    # code = compile("print('hello world')", "<string>", "exec")
    print('\n===== CODE EXECUTION RESULT =====')
    stdout_backup = sys.stdout
    sys.stdout = io.StringIO()
    try:
        code = compile(result, "<string>", "exec")
        exec(code)

        # Capture the output from stdout and reset stdout to its original value
        output = sys.stdout.getvalue()
        sys.stdout = stdout_backup

        # Return the output as a string
        output = output.strip()
        if len(output) == 0:
            output = 'Missing example usage: Please provide an a few example usages of the given function and print out the output of the function'
            has_error = True
        else:
            has_error = False

    except Exception as e:
        # Print any exceptions that occurred during execution
        output = traceback.format_exc()
        has_error = True

    finally:
        # Reset stdout to its original value
        sys.stdout = stdout_backup

    print(output)

    if not has_error:
        print('\n==== CHECK OUTPUT VALIDITY ====')
        check_code_template = """
``` python
{output}
```

Given above python script execution output delimited by triple backticks, determine if the output satisfies the given instruction.\
    There should not be any errors in the output, and should print the expected output outlined in the instruction. \
    If it satisfies all the above conditions, then expected output is 'Yes', otherwise, it is 'No'.
Instruction: {query}
Expected Output? (yes/no):"""
        prompt = PromptTemplate(
            template=check_code_template, input_variables=["query", "output"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        result = llm_chain.run({'query': text, 'output': output})

        print(result)
        if result.lower() == 'yes':
            code_generated = True
            print('SUCCESS!')

    else:
        attempt_num += 1
        print(f"\n\n============================== ATTEMPT #{attempt_num} ==============================")
        print('\n==== CODE GENERATED (ATTEMPTED FIX) ====')
        attempt_fix = f"""Your task is to fix python code that's outputting an error:
Code:
{result}

The following error occurred while running the above code:
{output}

Fix the error in the code specifically based on the above error description. Do NOT return the same piece of code.
Fixed Code:"""
        result = llm(attempt_fix)
        result = extract_code(result)
        print(result)