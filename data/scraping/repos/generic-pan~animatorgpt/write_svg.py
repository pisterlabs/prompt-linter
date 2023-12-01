import argparse
import math

parser = argparse.ArgumentParser(description='Get topic ID.')
parser.add_argument('number', type=int, help='An integer argument.')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument value
ind = args.number



# Read CSV file into a DataFrame
df = pd.read_csv('info.csv')



import openai
import requests

import subprocess
import shlex

def call_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    message = response.choices[0]['message']['content']
    return message




import pandas as pd

# Function to call Chatgpt
def call_Chatgpt(value):
    # Replace this function with your own logic
    # Example implementation: printing the value
    print(value)

# Iterate over each row
for index, row in df.iterrows():
    # Get the value in 'column_5' for the current row
    print(index)
    svg_value = row['svg_' + str(row['chosen_svg'])]
    
    # Call the function with the column value
    #slowed_svg = call_chatgpt("Hi")
    with open('item.svg', 'w') as file:
        file.write(slowed_svg)
    
    subprocess.run(shlex.split('./node_modules/timecut/cli.js frame.html --output=segments/' + str(index) + '.mp4 --canvas-capture-mode="#all" --viewport="1280,720,deviceScaleFactor=2" -R 60 -d 5'))
    
