#!/usr/bin/env python3

'''
This script automatically fixes bugs in programs.
It does this by taking the command to execute as an argument and
analyzing the traceback after executing the command.
Then the script uses OpenAI's Codex to fix the line that causes
the bug.
'''

import sys
import os
import subprocess
import json
import re
import pickle
import time
import random
from AUTH import *
import openai
import difflib

# The maximum number of times to try to fix a bug
MAX_FIX_TRIES = 10

FIX_PROMPT = (
        '# The above line throws the following exception:',
        '# Line that does not throw the error:\n'
        )

openai.organization = ORGANIZATION_ID
openai.api_key = SECRET_KEY


def assemble_prompt(lines_until_buggy_line, traceback, fix_prompt):
    prompt_lines = lines_until_buggy_line
    prompt_lines.append(fix_prompt[0])
    prompt_lines.append( '# "' + traceback.split('\n')[-2] + '"') 
    prompt_lines.append(fix_prompt[1])
    prompt = '\n'.join(prompt_lines)
    prompt += '\n'
    return prompt


def get_traceback(program):
    traceback = None

    # Run the program and capture its output
    with open(os.devnull, 'w') as devnull:
        try:
            # Get the stderr and the stdout of the program program.
            stderr = subprocess.check_output(program,  stderr=subprocess.STDOUT, shell=True).decode('utf-8')
        except subprocess.CalledProcessError as e:
            error_message_line = e.output.decode('utf-8').split('\n')[-2]
            traceback =  e.output.decode('utf-8')
            print("error_message_line:", error_message_line)
            print("traceback:", traceback)

    return traceback



def get_filename(traceback):
    # Get the line number from the traceback
    match = re.search(r'File \"(.*?)\", line ([0-9]+)', traceback)
    if not match:
        print('Could not find line number, exiting')
        sys.exit(1)

    filename = match.group(1)
    return filename


def get_source_code(filename):
    # Read the whole program code.
    with open(filename, 'r') as f:
        code = f.read()

    return code



def show_diff(original, new):
    d = difflib.Differ()
    diff = d.compare(original.split('\n'), new.split('\n')[:len(original)])
    colored_diff = []
    for line in diff:
        if line[0] == '+':
            colored_diff.append('\033[92m' + line + '\033[0m')
        elif line[0] == '-':
            colored_diff.append('\033[91m' + line + '\033[0m')
        else:
            colored_diff.append(line)

    print('\n'.join(colored_diff))


def get_fixed_code_single_line(code, traceback, level):


    # Get all the line numbers in the traceback.
    line_numbers = []
    for line in traceback.split('\n'):
        match = re.search(r'File \"(.*?)\", line ([0-9]+)', line)
        if match:
            line_numbers.append(int(match.group(2)))


    level = level % len(line_numbers)
    line_number = line_numbers[level]



    lines_until_buggy_line  = code.split('\n')[:line_number]
    prompt = assemble_prompt(lines_until_buggy_line, traceback, FIX_PROMPT)
    input_prompt = prompt
    # print("input_prompt:", input_prompt)


    # Create prompt that surrounds the buggy line with text indicating that this is
    # the line that should be fixed.

    response = openai.Completion.create(engine='davinci-codex', prompt=input_prompt, temperature=0.5, max_tokens=64, stop='\n')
    fixed_line = response['choices'][0]['text']

    fixed_code = replace_faulty_line(code, fixed_line, line_number)

    # Visualize diff between original code and fixed_code with color.
    # The inserted words are colored green, the delted parts red.
    # The rest is black.

    return fixed_code

def replace_faulty_line(code, fixed_line, line_number):
    lines = code.split('\n')
    lines[line_number-1] = fixed_line
    return '\n'.join(lines)



def wrap_text_around(text, end_line_number):
    lines = text.split('\n')
    lines_out = []
    lines_out.extend(lines[end_line_number:])
    lines_out.extend(lines[:end_line_number])
    return '\n'.join(lines_out)


def remove_block_of_code(code, from_line):
    # Generate two random numbers between from_line and len(code)
    # and then remove the lines between those two numbers.
    start_line_number = random.randint(from_line, len(code.split('\n')))
    end_line_number = random.randint(start_line_number, len(code.split('\n')))

    # Sets the lines between start_line_number and end_line_number
    # to blank lines.
    lines = code.split('\n')
    lines[start_line_number:end_line_number] = [''] * (end_line_number - start_line_number)
    return '\n'.join(lines)

def get_fixed_code_wrapped(code):
    rand_end_line_number = random.randint(1, len(code.split('\n')))
    code = remove_block_of_code(code, rand_end_line_number)
    code_wrapped = wrap_text_around(code, rand_end_line_number)
    input_prompt = code_wrapped
    max_num_tokens = int(len(code[rand_end_line_number:]) / 2)
    response = openai.Completion.create(engine='davinci-codex', prompt=input_prompt, temperature=0.5, max_tokens=max_num_tokens, stop='#!')
    generated_code = response['choices'][0]['text']
    original_until_end_line = '\n'.join(code.split('\n')[:rand_end_line_number])
    original_and_generated = original_until_end_line + generated_code
    return original_and_generated



def main(argv):
    if len(argv) != 2:
        print('Usage: %s <program>' % argv[0])
        sys.exit(1)

    program = argv[1]

    

    # Try to fix the bug
    for i in range(MAX_FIX_TRIES):

        traceback = get_traceback(program)


        # If the program didn't crash, exit
        if i == 0 and not traceback:
            print('No traceback, exiting')
            sys.exit(0)

        # If it didn't crash, exit
        elif i > 0 and not traceback:
            print('Successfully fixed bug')
            # Ask user whether to accept the solution.
            selection = input('Accept solution? [y/N]')

            if selection == 'Y' or selection == 'y':
                sys.exit(0)
            else:
                with open(filename, 'w') as f:
                    f.write(original_code)

                continue


        filename = get_filename(traceback)

        if i > 0:
            code = original_code
            traceback = original_traceback
        else:
            original_code = get_source_code(filename)
            code = original_code
            original_traceback = traceback

        print('Trying to fix bug (try %d of %d)' % (i+1, MAX_FIX_TRIES))

        # wraped = wrap_text_around(code, 80)
        # print("wraped:", wraped)
        # input()


        if True:
            fixed_code = get_fixed_code_single_line(code, traceback, i)
        else:
            fixed_code = get_fixed_code_wrapped(code)


        # print("fixed_code:", fixed_code)
        # input()


        # Write the code to the file
        with open(filename, 'w') as f:
            f.write(fixed_code)


        show_diff(code, fixed_code)
        print('======================================\n\n')



    print('Failed to fix the bug')
    with open(filename, 'w') as f:
        f.write(original_code)

if __name__ == '__main__':
    main(sys.argv)


