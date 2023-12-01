import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.organization = os.getenv('Orginization_ID')
openai.api_key = os.getenv('API_chatGPT')

model = "text-davinci-003"
temperature = 0.5
max_tokens = 60

prompts = []
with open('ideas.txt', 'r', encoding="UTF-8") as f:
    for line in f:
        prompt = line.rstrip()
        prompts.append(prompt)

with open('output_prompts.txt', 'w', encoding="UTF-8") as f:
    for i, prompt in enumerate(prompts):
        print(f'Running query {i+1}th...')
        respose = openai.Completion.create(
            engine=model,
            prompt='Can you generate a script for Midjourney to draw a' +
            prompt + ' in form of keywords, separate with commas.',
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Call the answer
        answer = respose.choices[0].text.strip()
        f.write(answer + '\n')
