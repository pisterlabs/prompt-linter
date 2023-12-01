import os
import openai
import yaml


# Generate Prompt (standardize to get structured answer
# prompt = f'Return in yaml format 10 multiple choice scenario-type question with 4 possible answers, in which indicates the correct answer, similar to the AWS Certified Solutions Architect Associate SAA-C03 exam. Use the following transcript: \n{prompt_transcript}'
prompt = 'Return in yaml format 2 different multiple choice scenario-type question with 4 possible answers, in which indicates the only one correct answer, content relevant to the AWS Certified Solutions Architect Associate SAA-C03 exam.' \
         'The yaml output should include unique id, question, options and the correct_answer_position.'

# Prompt used manually on chat.openai.com
prompt_2 = 'Return a yaml representation of 10 multiple choice scenario-type questions with 4 possible answers, indicating the correct answer. ' \
           'Add in this yaml for each question their unique ID (from 1 to 10) as the primary key. ' \
           'The topic is related to EC2 services  and is similar to the AWS Certified Solutions Architect Associate SAA-C03 exam'

response = openai.Completion.create(
    engine='text-davinci-002',
    prompt=prompt,
    temperature=1,
    max_tokens=200
)

output_response = (response['choices'][0]['text'])

test = output_response.replace('---\n- id: ', '')
test = test.replace('\n  q', ':\n  q')

import json

# convert dictionary string to dictionary
res = json.loads(test)

# print result
print(res)