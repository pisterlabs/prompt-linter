import os
import re
import logging
import argparse

import pandas as pd
import inflect
import openai

from newspaper import Article
from pathlib import Path
from datetime import date, timedelta
from tqdm.auto import tqdm

from tenacity import retry, stop_after_attempt, wait_random_exponential


_CATEGORIES = [
    'Tools & Apps',
    'Applications & Business',
    'Projects & Open Source',
    'Research & Advancements',
    'Policy & Safety',
    'Synthetic Media & Art'
]

MAIN_STORY_COL = "Podcast pick (main story)"
LIGHTING_STORY_COL = "Podcast pick (lighting round)"

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(2))
def query_openai(messages):
    return openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=messages,
        max_tokens=1000,
        temperature=0
    ).choices[0]['message']['content']


def classify_article_type(row):
    prompt = f'''
Title: {row['Name']}
Description: {row['Excerpt']}
Link: {row['URL']}
'''.strip()

    system_prompt = '''
Your task is to classify articles about AI into one of the following types:
Tools & Apps: Tools or applications that regular people can use for various purposes.
Applications & Business: Applying AI to do something or anything related to product announcements, investments, funding, VCs, company updates, or market trends.
Projects & Open Source: Misc projects that are not done by companies or academic labs, especially open source projects.
Research & Advancements: Scientific studies, research in AI, or new results that advance IA.
Policy & Safety: News, analysis, and opinions related to government policies or about AI risk/harm.
Synthetic Media & Art: Things related to generative AI, deepfakes, or art and are not a better fit for the above categories.

The user will provide the article title, link, and description. 
After careful consideration, you will respond with ONLY the predicted article type, with no explanations, punctuation, formatting, or anything else.
Please only respond with one of the above types (Tools & Apps, Applications & Business, Projects & Open Source,  Research & Advancements, Policy & Safety, Synthetic Media & Art).
'''.strip()
    return query_openai([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])

def summarize_article(row, lighting_round_story=False):
    article = Article(arxiv_to_huggingface(row['URL']))
    try: 
        article.download()
        article.parse()
        authors = article.authors
        text = article.text
        words = text.split(" ")
        if len(words) > 1000:
            text = " ".join(words[:1000])
    except Exception as e:
        print(e)
        return ":("

    system_prompt = '''
Your task is to provide a bullet point summary of a news article about AI. Each bullet point should be no more than 2 sentences long. This summary will be used for the podcast Last Week in AI, in which the hosts summarize stories about AI in an accessible manner. We will provide the title of the article, the subtitle, and the text. Output the bullet points in markdown format.'''

    if lighting_round_story:
        system_prompt + " This story will be in a lighting round, so summarize it in no more than 8 bullet points, but still make sure to cover all the important details."
    else:
        system_prompt + " This will be covered as a main story, so produce a detailed summary with as many as 15 bullet points."
        
    system_prompt = system_prompt.strip()
    
    prompt = f'''
Title: {row['Name']}
Subtitle: {row['Excerpt']}
Text: {text}
'''.strip()
    return query_openai([
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])

def arxiv_to_huggingface(url: str) -> str:
    match = re.search(r"https://arxiv.org/abs/(\d+\.\d+)(?:v\d+)?", url)
    if match:
        return f"https://huggingface.co/papers/{match.group(1)}"
    else:
        return url

if __name__ == "__main__":
    with open('secrets/openai_api_key.txt', 'r') as f:
        openai.api_key = f.read().strip()

    articles_map = {c : ([],[]) for c in _CATEGORIES}
    articles_map['other'] = ([],[])
    csv = pd.read_csv('news.csv', encoding='utf-8')
    num_picks = 0
    for row_num, row in tqdm(csv.iterrows()):
        has_content = row['Name'] and row['Excerpt']
        is_main_pick = str(row[MAIN_STORY_COL]).lower().strip()=='x'
        is_lighting_pick = str(row[LIGHTING_STORY_COL]).lower().strip()=='x'
        if (is_main_pick or is_lighting_pick):
            num_picks+=1
    pbar = tqdm(total=num_picks)
    count = 0
    for row_num, row in tqdm(csv.iterrows()):
        has_content = row['Name'] and row['Excerpt']
        is_main_pick = str(row[MAIN_STORY_COL]).lower().strip()=='x'
        is_lighting_pick = str(row[LIGHTING_STORY_COL]).lower().strip()=='x'
        if not (is_main_pick or is_lighting_pick):
            continue
        category = classify_article_type(row)
        print('\nProcessing "%s"'%row['Name'])
        if is_main_pick:
            print("Type: Main story")
        else:
            print("Type: Lighting round story")
        print("Link: %s"%row['URL'])
        print("Category: %s"%category)
        if category not in _CATEGORIES:
            category = 'other'
        summary = summarize_article(row,is_lighting_pick)
        print('Summary:')
        print(summary)
        if is_main_pick:
            articles_map[category][0].append([row['Name'],row['URL'],summary])
        else:
            articles_map[category][1].append([row['Name'],row['URL'],summary])
        print("")
        count+=1
        pbar.update(1)
    pbar.close()
        
    content = 'Outline:\n'
    
    for c in _CATEGORIES:
        content+="- "+c+"\n"
        items = articles_map[c]
        for item in items[0]:        
            name, url, summary = item
            content += f'   - [{name}]({url})\n'
        content+="  - Lighting round\n"
        for item in items[1]: 
            name, url, summary = item       
            content += f'       - [{name}]({url})\n'
            
    content +="\n\n#Summaries\n\n"
    for c in _CATEGORIES:
        items = articles_map[c]
        content += f'## {c}'
        if len(items[0]) > 0:
            content += '\n\n'
            for item in items[0]:
                name, url, summary = item
                content += f'[{name}]({url})'
                content += '\n'
                content += summary
                content += "\n\n"
        content +="\n### Lighting Round\n\n"
        if len(items[1]) > 0:
            for item in items[1]:
                name, url, summary = item
                content += f'[{name}]({url})'
                content += '\n'
                content += summary
                content += "\n\n"
                
    print(content)
    with open("summaries.md", 'wb') as f:
        f.write(content.encode('utf-8'))