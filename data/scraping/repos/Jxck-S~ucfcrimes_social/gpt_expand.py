'''
It may be possible to use OpenAI's models to help us expand the case titles
It it a rather difficult linguistic task and got may be able to handle it well

Maverick Reynolds
04.19.2023
UCF Crimes
'''

from configparser import ConfigParser
import openai
import json

main_config = ConfigParser()
main_config.read('config.ini')

# Prompt engineering for the model
# It seems to work pretty well even without the examples so that will be an option for us
def generate_prompt(title: str, provide_examples=True):
    with open('gpt_expansions.json', 'r') as f:
        prev_examples = json.load(f)

    # Start with instruction
    prompt = "Here is some text from a case notification system that has some abbreviations. Can you expand the text while still keeping it uppercase? If it is just one word, you can keep it that word. Don't worry too much about prepositions such as 'of' and 'as', but do include them when necessary. You will have to be clever; there may be spelling errors and other abnormalities in the text."
    prompt += '\n\n'

    # List previous_examples
    if provide_examples:
        prompt += 'Here are some examples to help you out. If the text provided matches an example, feel free to use the provided solution exactly as it is written.'
        prompt += '\n\n'

        for resp in prev_examples:
            if resp['verified_example']:
                prompt += f"Example: {resp['raw']}\nAnswer: {resp['expanded']}\n\n"
    
        prompt += 'Example: '
    
    # Give current message
    prompt += title.upper() # Just in case

    # Continue formatting if examples are listed
    if provide_examples:
        prompt += '\nAnswer:'

    return prompt.strip()   # Beneficial to start and end without leading/trailing spaces



# Use model to expand the titles of cases
def gpt_title_expand(formatted_title, provide_examples=True):
    API_KEY = main_config.get("OPENAI", "API_KEY")
    openai.api_key=API_KEY
    model='gpt-3.5-turbo' # Because this is way cheaper!

    # Build the prompt using verified examples and the title
    prompt = generate_prompt(formatted_title, provide_examples=provide_examples)
    messages=[{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    answer = response['choices'][0]['message']['content'].strip('.')    # Remove period if generated

    # Get the responses list
    with open('gpt_expansions.json', 'r') as f:
        responses_list = json.load(f)

    # Check if a pre-expanded title is already in the list
    for resp in responses_list:
        if resp['raw'] == formatted_title:
            # If so, return that answer (don't add it to the list again)
            return resp['expanded']

    # Otherwise add it to the list
    responses_list.append({
        "raw": formatted_title,
        "expanded": answer,
        "verified_example": False
    })

    # Save the responses list
    with open('gpt_expansions.json', 'w') as f:
        json.dump(responses_list, f, indent=4)

    # Return the GPT answer
    return answer