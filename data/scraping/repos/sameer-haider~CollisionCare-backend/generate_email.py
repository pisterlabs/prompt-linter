import openai
import os
openai.api_key = os.environ.get('OPENAI_KEY')

# Testing GPT-3.5 "davinci" model to generate answer to question
def generate_email(prompt):
    final_prompt = f'using this data [{prompt}] make me a message to tell a collision center that an insurance member was affected. explaining the situation concisely and requesting them to help.'
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=final_prompt,
        max_tokens=100
    )
    generated_text = response.choices[0].text
    print(generated_text)
    return generated_text