import re
import os
import openai
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')


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


def gpt3_finetune(prompt, model="davinci:ft-david-shapiro:scene-sim-2022-09-15-15-12-29", temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=["END"]):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                model=model,
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


prev_chunk = '''The trees stood like sentinels, their leaves rustling in the wind. The sun had long since set, and the only light came from the moon, which cast a pale glow over the forest. The forest was a place of legend and myth, and it was said that dragons still roamed its depths. It was also said that wizards still lived in the forest, hiding from the world in their towers. The main character, a young boy named Sylas, had always been fascinated by the stories of the Forest. He had often dreamed of visiting the forest, and now, finally, he was going to get his chance. Sylas's parents had been killed by a dragon, and he had been taken in by the wizard, Kyzax. Kyzax had told Sylas that he was special, that he had the potential to be a great wizard. And so, on this night, Sylas was setting out into the Forest, on a quest to find the dragons and wizards that lived there. He was not afraid, for he knew that he was the chosen one, destined to save the world from the evil that was coming.'''


if __name__ == '__main__':
    print(prev_chunk)
    next_chunk = gpt3_finetune(prev_chunk + '\n\nNEXT PARAGRAPH:')
    save_file('story/story_%s.txt' % time(), next_chunk)
    print('\n\n', next_chunk)
    prompt = open_file('prompt_summary.txt').replace('<<STORY>>', prev_chunk)
    summary = gpt3_completion(prompt)
    for i in list(range(0,20)):
        prev_chunk = next_chunk
        next_chunk = gpt3_finetune(summary + ' ' + prev_chunk + '\n\nNEXT PARAGRAPH:')
        save_file('story/story_%s.txt' % time(), next_chunk)
        print('\n\n', next_chunk)
        prompt = open_file('prompt_summary.txt').replace('<<STORY>>', summary + ' ' + prev_chunk)
        summary = gpt3_completion(prompt)