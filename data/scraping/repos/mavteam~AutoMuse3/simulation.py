import numpy as np
import re
import os
import openai
from time import time,sleep
import textwrap
from random import seed,choice


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')
#rarities = ['common', 'likely', 'unlikely', 'interesting', 'exciting', 'funny', 'stressful', 'irritating', 'ordinary' , 'extraordinary' , 'shocking']  # shocking might cause t he story to end
rarities = ['common', 'likely', 'funny', 'ordinary', 'minor']


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def load_story():
    files = [i for i in os.listdir('logs/') if 'summary' not in i]  # exclude summaries even though we want to save them
    result = list()
    for file in files:
        result.append(open_file('logs/%s' % file).strip())
    return result


def summarize_block(text_block):
    chunks = textwrap.wrap(text_block, 4000)
    result = ''
    #print(len(chunks), 'chunks to summarize')
    for chunk in chunks:
        prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk)
        summary = gpt3_completion(prompt)
        result = result + ' ' + summary
        result = result.strip()
    return result


def recursively_summarize(story):
    #print('Recursively summarizing story up to this point...')
    summary = '\n'.join(story).strip()
    while True:
        summary = summarize_block(summary)
        if len(summary) < 1000:
            return summary


def get_recent(story):
    if len(story) <= 10:  # increase this number to get bigger chunks of story
        return '\n'.join(story)
    else:
        return '\n'.join(story[-10:])  # increase this number to get bigger chunks of story


if __name__ == '__main__':
    while True:
        #print('NEW instance, loading story...')
        story = load_story()  # load the entire story so far
        summary = 'SUMMARY: %s' % recursively_summarize(story)  # write a summary of the whole story so far TODO: make this more efficient (maybe not necessary with finetuned CURIE?)
        print('\n\n\n', summary)
        save_file('logs/log_%s_summary.txt' % time(), summary)
        # instantiate current SCENE
        recent = get_recent(story)
        prompt = open_file('prompt_scene.txt').replace('<<SUMMARY>>', summary).replace('<<RECENT>>', recent)
        scene = 'SCENE: %s' % gpt3_completion(prompt)
        story.append(scene)
        save_file('logs/log_%s_scene.txt' % time(), scene)
        print(scene)
        # iterate through characters
        #print('Iterating through characters...')
        character_files = [i for i in os.listdir() if 'character_' in i]
        for charfile in character_files:
            recent = get_recent(story)
            charname = charfile.replace('character_','').replace('.txt','').replace('_',' ')
            profile = open_file(charfile)
            prompt = open_file('prompt_character.txt').replace('<<NAME>>', charname).replace('<<CHARACTER>>', profile).replace('<<SUMMARY>>', summary).replace('<<RECENT>>', recent)
            dialog = gpt3_completion(prompt)
            if charname not in dialog:
                dialog = charname + ': ' + dialog  # add character name if it wasn't put in by the model
            save_file('logs/log_%s_dialog.txt' % time(), dialog)
            story.append(dialog)
            print(dialog)
        # instantiate plot event
        recent = get_recent(story)
        seed()
        prompt = open_file('prompt_event.txt').replace('<<RARITY>>', choice(rarities)).replace('<<SUMMARY>>', summary).replace('<<RECENT>>', recent)
        event = 'EVENT: %s' % gpt3_completion(prompt)
        story.append(event)
        save_file('logs/log_%s_event.txt' % time(), event)
        print(event)
        #exit(0)