import openai
import os
from openai import OpenAI
from config import OPENAI_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_KEY

client = OpenAI()


def kg_generate_and_compare(text, image_text, kg_generate_prompt_path='kg_gen_prompt.md',
                            kg_compare_prompt_path='kg_comp_prompt.md'):
    with open(kg_generate_prompt_path, 'r', encoding='utf-8') as f:
        gen_prompt = f.read()
    with open(kg_compare_prompt_path, 'r', encoding='utf-8') as f:
        comp_prompt = f.read()

    print('Generating KG...')
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are an expert in Knowledge Graph generation"},
            {"role": "user",
             "content": gen_prompt.format(TEXT=text, IMAGETEXT=image_text)}
        ]
    )
    kg = completion.choices[0].message.content

    print('Comparing...')
    original_text = 'Original text for the first KG:\n' + text + '\nOriginal text for the second KG:\n' + image_text + '\n'
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system",
             "content": "You are an expert in Knowledge Graph comparison"},
            {"role": "user",
             "content": comp_prompt.format(KG=kg, ORIGINALTEXT=original_text)}
        ],
        temperature=0.05,
    )
    predicted_label = int(completion.choices[0].message.content.split('\n')[0].strip())
    return kg, predicted_label, completion.choices[0].message.content
