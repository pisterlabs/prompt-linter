"""
This program is used to find all the functions in a file, to find the code of a specific function, and to replace the code of a specific function.
This enables verbal programming by allowing the user to say the name of the function and what it should do, and then the AI model will find the function in the file and replace the code with the new code based on what the user instructed the AI model to do.
"""
from whispering.skills.read_file_contents import get_directory_contents, select_classification_label
from code_search import search_functions
import os
import sys
import openai
import re



import os.path
import re

import macro



def get_function_code(function_name, file_name):
    """
    Gets the code of a function in a file.
    :param function_name: The name of the function to get the code of.
    :param file_name: The name of the file to search in.
    :return: The code of the function.
    """
    with open(file_name, 'r') as file:
        code = file.read()

    function_def = re.search(r'\n\s*def\s+' + function_name + r'\(', code, re.MULTILINE)
    if not function_def:
        raise Exception(f"Could not find function definition for {function_name}")
    function_def = function_def.group()
    # Find the beginning and end of the function code:
    function_begin = code.find(function_def)
    function_end = function_begin + len(function_def)
    if next_function_def := re.search(
        r'\n\s*def\s+[a-zA-Z0-9_]+\(', code[function_end:], re.MULTILINE
    ):
        indentation_level = function_def.count('\t') + 1
        # Get the indentation level so we don't want any lines that have a lesser indentation
        while code[function_end:].startswith('\t' * indentation_level):
            function_end += 1
        function_end += next_function_def.start() - 1
    else:
        function_end = len(code)

    return code[function_begin:function_end +1 ]


def edit_code(code, command):
    """
    This function takes in a code and a command and returns the edited code.
    """
    openai.api_key = "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"
    response = openai.Edit.create(
    model="code-davinci-edit-001",
    input=code,
    instruction=command,
    temperature=0,
    top_p=.5
    )
    return response.choices[0].text

    













def indent_code(code, indentation_level):
    return '\t' * indentation_level + code.replace('\n', '\n' + '\t' * indentation_level)


def replace_function(function_name, file_name, new_code):
    """
    Replaces the code of a function in a file.
    
    :param function_name: The name of the function to replace.
    :param file_name: The name of the file to replace the function in.
    :param new_code: The new code of the function.
    :return: The new code of the file.
    """
    with open(file_name, 'r') as file:
        code = file.read()
    function_def = re.search(r'\n\s*def\s+' + function_name + r'\(', code)
    if function_def is None:
        raise Exception(f'Could not find function definition for {function_name}')

    function_begin = code.find(function_def.group())
    function_end = function_begin + len(function_def.group())

    if next_function_def := re.search(
        r'\n\s*def\s+[a-zA-Z0-9_]+\(', code[function_end:]
    ):
        indentation_level = function_def.group().count('\t') + 1

        while code[function_end:].startswith('\t' * indentation_level) or code[function_end:].startswith('\n'):
            function_end += 1
        function_end += next_function_def.start()

    else:
        function_end = len(code)

    new_code = indent_code(new_code, function_def.group().count('\t'))

    code = code[:function_begin] + new_code + code[function_end:] + '\n\n'

    with open(file_name, 'w') as file:
        file.write(code)
    return code








def list_functions(file_name):
    '''
    Get a list of all the functions in a file
    :param file_name: Name of file to read
    :return: a list of function names in the file
    '''
    with open(file_name, 'r') as file:
        code = file.read()
    lines = code.split('\n')
    return [
        line.strip().split('(')[0].split()[1]
        for line in lines
        if line.startswith('def')
    ]






def select_function(file_name, command):
    """
    Prints a list of all the functions in the file and asks the user which function they want to see the code for.
    :param file_name: The name of the file.
    :param function_name: The name of the function.
    :return: The code for the function.
    """
    all_functions = list_functions(file_name)
    function_name, confidence = select_classification_label(command, all_functions)
    return file_name, function_name

# if __name__ == '__main__':
#     command = input("What do you want to do? ")
#     directory_contents, best_label, confidence = get_directory_contents(command)
#     file_name, function_name = select_function(best_label, command)
#     replace_function(function_name, file_name, edit_code(get_function_code(function_name, file_name),command))

    # select_function(file_name=filename, command=command)



# create a function that uses my os to take a screen of the current screen and then use the open ai api to edit the code and then use the os to replace the code with the new code
# step 1: take a screen shot of the current screen
# step 2: get that code from the screen shot
# step 3: print the code

def get_screen_shot():
    import pyautogui
    import time
    time.sleep(3)
    screenshot = pyautogui.screenshot()
    screenshot.save('/Users/canyonsmith/Desktop/sentient_ai/assistent_ai_code/whispering/whispering.png')
get_screen_shot()
def get_code_from_screen_shot():
    import cv2
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    img = cv2.imread('/Users/canyonsmith/Desktop/sentient_ai/assistent_ai_code/whispering/whispering.png')
    return pytesseract.image_to_string(img)

def get_function_name(code):
    """
    Extract function name from a line beginning with "def "
    """
    assert code.startswith("def ")
    return code[len("def "): code.index("(")]

def get_until_no_space(all_lines, i) -> str:
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, i + 10000):
        if j < len(all_lines):
            if len(all_lines[j]) == 0 or all_lines[j][0] in [" ", "\t", ")"]:
                ret.append(all_lines[j])
            else:
                break
    return "\n".join(ret)

def get_functions(code):
    """
    Get all functions in a Python file.
    """
    all_lines = list(code.split("\n"))
        

    for i, l in enumerate(all_lines):
        if l.startswith("def "):
            code = get_until_no_space(all_lines, i)
            function_name = get_function_name(code)
            yield ({"code": code, "function_name": function_name})



# f = get_functions(code=get_code_from_screen_shot())

# for function in f:
#     first_function = function["function_name"]
#     found_function = search_functions(first_function,n=1, pprint=True,n_lines=10)
#     '''                                                       code       function_name                                  filepath                                     code_embedding  similarities
# 48   def get_function_name(code):\n    """\n    Ext...   get_function_name                        utils.py  [-0.03586161509156227, 0.005071669816970825, -...      0.701660'''
#     # get the function name and filepath

    
#     print(function_name, filepath)
#     break


