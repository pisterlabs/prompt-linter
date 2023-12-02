import openai
import json

openai.api_key = ''

# List to store the datasets
dataset = []

for i in range(100):
    # Create an initial instruction
    init_instruction = "Generate a common Go language error and its solution."
    response_instruction = openai.ChatCompletion.create(
        model="gpt-4.0-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": init_instruction},
        ]
    )

    # Get the assistant's reply
    instruction = response_instruction['choices'][0]['message']['content']

    # Create an initial input based on instruction
    init_input = "Generate a log error based on the previous instruction."
    response_input = openai.ChatCompletion.create(
        model="gpt-4.0-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": init_input},
        ]
    )

    # Get the assistant's reply
    input_error = response_input['choices'][0]['message']['content']

    # Generate output based on input
    response_output = openai.ChatCompletion.create(
        model="gpt-4.0-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": input_error},
        ]
    )

    # Get the assistant's reply
    output = response_output['choices'][0]['message']['content']

    data = {
        "instruction": instruction,
        "input": input_error,
        "output": output
    }

    dataset.append(data)

# Save the dataset into a JSON file
with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)
