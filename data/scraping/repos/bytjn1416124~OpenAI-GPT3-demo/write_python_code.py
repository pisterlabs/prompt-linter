# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
from generalGpt3 import GPT3

print ('')
print ('Give a description of a code snippet you want the model to generate')
print ('Example: ')
print ('Action: create a function that takes a variable as input and prints it')
print ('')

prompt ='''Action: write a Python code that prints "hello" everytime there is an even number in a loop over every number
        code:
        i = 0
        while (True):
            if i %2:
                print ('hello')
            i+=1

        Action: write a python function that takes a variable as an input, and multiplies this variable with 3
        code:
        def multiplication(var):
            return var*3

        Action: write a python code that prints a long decimal number with only two decimals.
        code:
        def long_decimal(num):
            return str(num).rjust(2, '0')


        Action: write a python code that takes a variable as an input, and prints the variable in a binary format
        code:
        def binary_format(var):
            return '%b' % var

        Action: write a python code that takes a variable as an input, and prints the variable in a octal format
        code:
        def octal_format(var):
            return '%o' % var

        Action: write a python code that opens and reads a text file
        code:
        def read_file(filename):
            with open(filename, 'r') as f:
            for line in f:
            print(line)

        Action: write a python code that opens and writes a text file
        code:
        def write_file(filename, data):
            with open(filename, 'w') as f:
            f.write(data)

        Action: '''




GPT3(prompt, restart_text = "Action:", start_text = "Code:",
        temperature=0.5, frequency_penalty=0, presence_penalty=0, response_length = 100,
        top_p = 1,engine = 'davinci')
