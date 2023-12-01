import os
import time

import openai
from dotenv import load_dotenv
from openai.error import RateLimitError

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))
openai.api_key = os.environ.get('OPENAI_API_KEY')

# G-EVAL implementation: https://arxiv.org/abs/2303.16634

def get_consistency_score(context, answer, language, llm_model="text-davinci-003"):
    if language == 'en':
        prompt = open('data/prompts/G-Eval-Con-En.txt').read()
    else:
        prompt = open('data/prompts/G-Eval-Con-De.txt').read()
    prompt = prompt.replace('{{Document}}', context).replace('{{Answer}}', answer)
    for _ in range(3):
        try:
            response = openai.Completion.create(
                model=llm_model,
                prompt=prompt,
                temperature=0,
            )
            gen_text = response.choices[0].text.strip()
            print("GPT-3 Score:", gen_text)
            if gen_text.isdigit():
                consistency_score = int(gen_text)
                return consistency_score
            else:
                return -1
        except RateLimitError:
            print("Rate limit error, retrying...")
            time.sleep(20)
            continue
