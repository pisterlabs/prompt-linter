"""
(Same as number_sense_test.py, but only the letters are spaced out in
order to ameliorate encoding issues.)

To test whether some kind of "number sense" influences GPT-3's
classifications, versus just shallow patern matching on symbols,
I map digits 0, ..., 9 in the feature vector to some
random letters, and see how that affects classification accuracy.
"""
from utils import textify_numbers
import string
import random
import json
import re
import openai
import os

random.seed(42)

alphabet = string.ascii_lowercase
# Filter for letters not in "input" or "output", so their co-occurrence
# doesn't confuse GPT-3:
filtered_alphabet = [x for x in alphabet if (x not in 'inputoutput')]

openai.api_key = os.getenv("OPENAI_API_KEY")

#letters = random.sample(filtered_alphabet, k=10)
#mapping = dict(zip(
#                    [str(x) for x in range(0, 10)],
#                    letters
#                    )
#                )
mapping = {'0': 'd ', '1': 'a ', '2': 'j ', '3': 'h ', '4': 'w ',
             '5': 'c ', '6': 'm ', '7': 'b ', '8': 'l ', '9': 'x '}

with open('experiments_log.json', 'r') as file:
    experiments = json.loads(file.read())

experiment_names = [f'2d_class_type_{n}_rstate_{rs}'
                    for n in range(1, 10)
                    for rs in [42, 55, 93]]

engine = 'davinci'


for name in experiment_names:
    experiment = experiments[name]

    modified_input_text = ''
    for line in experiment['input_text'].split('\n')[:-1]:
        initial_part = line[:-1]
        for digit in mapping.keys():
            initial_part = initial_part.replace(digit, mapping[digit])
        new_line = initial_part + line[-1] + '\n'
        modified_input_text += new_line
    
    experiment['spaced_letters_input_text'] = modified_input_text
    experiment[f'spaced_letters_response_{engine}'] = []
    experiment[f'spaced_letters_output_test_raw_{engine}'] = []
    experiment[f'spaced_letters_output_test_cleaned_{engine}'] = []

    for point in experiment['input_test']:
        point = textify_numbers(point)
        for digit in mapping.keys():
            point = point.replace(digit, mapping[digit])
        prompt_text = (
                modified_input_text
                + f'Input = {point}, output ='
            )
    
        response = openai.Completion.create(engine=engine,
                prompt=prompt_text, max_tokens=6,
                temperature=0, top_p=0)

        experiment[f'spaced_letters_response_{engine}'].append(response)

        response_text = response['choices'][0]['text']
        experiment[f'spaced_letters_output_test_raw_{engine}'].append(response_text)

        experiment[f'spaced_letters_output_test_cleaned_{engine}'].append(
                int(
                    re.findall('-?\d+',response_text
                            )[0]
                    )
                )
                
        experiments[name] = experiment

with open('experiments_log.json', 'w') as file:
    json.dump(experiments, file, indent=4)
