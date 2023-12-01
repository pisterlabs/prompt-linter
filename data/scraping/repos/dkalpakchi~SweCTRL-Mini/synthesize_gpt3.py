import os
import json
import time

import dotenv
import yaml
import openai
import requests
import jsonlines as jsl

from tqdm import tqdm


prompt_templates = {
    "news": "Skriv en nyhetsartikel.\\{}",
    "wiki": "Skriv en artikel i Wikipedia.\\{}",
    "news_sport": "Skriv en nyhetsartikel om idrott.\\{}",
    "blogs": "Skriv ett blogginlägg.\\{}",
    "news_pressrelease": "Skriv ett pressmeddelande.\\{}",
    "ads": "Skriv en annons.\\{}",
    "news_opinion": "Skriv en insändare.\\{}",
    "news_culture": "Skriv en nyhetsartikel om kultur.\\{}",
    "admin": "Skriv en förvaltningstext.\\{}",
    "news_economy": "Skriv en nyhetsartikel om ekonomi.\\{}",
    "info_medical": "Skriv en informerande text om ett medicinskt ämne.\\{}",
    "info": "Skriv en informerande text.\\{}",
    "news_tech": "Skriv en nyhetsartikel om teknologi.\\{}",
    "review": "Skriv en recension.\\{}",
    "info_travel": "Skriv en informerande text om resor.\\{}",
    "news_lifestyle": "Skriv en nyhetsartikel om livstil.\\{}",
    "blogs_sport": "Skriv ett blogginlägg om idrott.\\{}",
    "info_lifestyle": "Skriv en informerande text om livstil.\\{}",
    "news_sustainability": "Skriv en nyhetsartikel om hållbarhet.\\{}",
    "news_travel": "Skriv en nyhetsartikel om resor.\\{}",
    "info_business": "Skriv en informerande text om affär.\\{}",
    "news_politics": "Skriv en nyhetsartikel om politik.\\{}",
    "news_science": "Skriv en nyhetsartikel om vetenskap.\\{}",
    "news_food": "Skriv en nyhetsartikel om mat.\\{}",
    "news_fashion": "Skriv en nyhetsartikel om mode.\\{}",
    "news_weather": "Skriv en nyhetsartikel om vädret.\\{}",
    "blogs_economy": "Skriv ett blogginlägg om ekonomi.\\{}"
}


if __name__ == '__main__':
    dotenv.load_dotenv()

    openai.api_key = os.getenv('SECRET_KEY')

    with open("prompts.yaml") as f:
        prompts = yaml.load(f)

    generated = []

    ts = int(time.time())
    max_samples = 5

    with jsl.open('generated_{}.jsonl'.format(ts), 'w') as writer:
        for cat in tqdm(prompts):
            for subcat, prompt_lst in prompts[cat].items():
                for prompt in prompt_lst:
                    text = prompt_templates[cat].format(prompt)
                    
                    num_samples = 0
                    while True:
                        if num_samples == max_samples:
                            break
                        
                        try:
                            gen_params = {
                                'prompt': text,
                                'temperature': 0.7,
                                'max_tokens': 256
                            }
                            completion = openai.Completion.create(engine='text-davinci-003', **gen_params)
                            num_samples += 1
                        except openai.error.RateLimitError:
                            time.sleep(60)
                            continue

                        writer.write({
                            'text': text,
                            'cat': cat,
                            'subcat': subcat,
                            'params': gen_params,
                            'res': completion
                        })

