import sys
import openai
import os


# Set up the OpenAI API client
openai.api_key = os.environ["OPEN_AI_KEY"]

# read output of stderr
input_lines = sys.stdin.readlines()

# create a string from the input lines
errorMessage = "".join(input_lines)
print(errorMessage)

# Define the prompt for GPT-3
prompt = "My program is returning an error. here is the error message, explain what is wrong: " 
prompt += errorMessage

# Use GPT-3 to classify the statement
model_engine = "text-davinci-003"
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    temperature=0.5,  # Increase the temperature to encourage more diverse responses
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

print(completion.choices[0].text)