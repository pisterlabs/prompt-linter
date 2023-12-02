import openai
import json
from src.config import Config



openai.api_key = Config().API_KEY


def chat_with_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt},
        ],
        temperature=0.5,
        max_tokens=150,
        functions=[{
            "name": "add_numbers",
            "description": "Add two numbers",
            'parameters': {
                'type': 'object',
                'properties': {
                    'number_a': {
                        'type': 'number',
                        'description': 'The first number'
                    },
                    'number_b': {
                        'type': 'number',
                        'description': 'The second number'
                    }
                }
            }
        }],
    )
    return response.choices[0]

def prompt_and_parse(my_prompt:str = "What is the sum of 25 minus 5?"):

    def add_numbers(number_a, number_b):
        return number_a + number_b

    #my_prompt = "what colour is the sky"a
    msg = chat_with_chatgpt(my_prompt)
    print(msg)
    ########## PARSE THE RESPONSE ##########
    # Parse the JSON data
    data = msg

    # Extract function name and arguments
    function_name = data['message']['function_call']['name']
    arguments_json = data['message']['function_call']['arguments']

    # Parse arguments JSON string to a dictionary
    arguments = json.loads(arguments_json)

    # Extract numeric values
    number_a = float(arguments['number_a'])
    number_b = float(arguments['number_b'])

    # Print the results
    print(f"Function Name: {function_name}")
    print(f"Argument 1: {number_a}")
    print(f"Argument 2: {number_b}")

    ########## CALL THE FUNCTION ##########
    # Use locals() to call the function
    if function_name in locals() and callable(locals()[function_name]):
        result = locals()[function_name](number_a, number_b)
        print(f"Result of {function_name}({number_a}, {number_b}): {result}")
    else:
        print(f"Function {function_name} not found in locals()")