import openai
import json
import os
import sys
import subprocess
packages = ["openai"]
command = ["pip", "install"] + packages
print(f"\nRequirements installing:\n\n" + "\n".join(packages))
result = subprocess.run(command, capture_output=True, text=True)
print("\nPackages installed.\n")

API_KEY = open("API_KEY", "r").read()

openai.api_key = API_KEY

with open(sys.argv[1]) as file:
    data = json.load(file)

with open(sys.argv[2], 'w') as output_file:
    output_file.write("[")
    output_file.write("\n")
    for index, item in enumerate(data):
        #Copies original instruction
        json.dump(item, output_file)
        output_file.write(',')
        output_file.write('\n')
        print(item['instruction'])
        #Queries API for response
        input = "Give me four different ways to say the following, with each instruction separated by a semicolon only: " + item['instruction']
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "user", "content": input}
            ]
        )
        content = response['choices'][0]['message']['content']
        responses = content.split(';')
        responses = [input.strip() for input in responses]
        for num, input in enumerate(responses):
            entry = {"instruction": input, "input":item['input'], "output":item['output']}
            json.dump(entry, output_file)
            if index != len(data)-1:
                output_file.write(',')
            if index == len(data)-1 and num != len(responses)-1:
                output_file.write(',')
            output_file.write('\n')
    output_file.write("]")



