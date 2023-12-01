import re
import os
import openai
import textwrap
from time import time,sleep
from pprint import pprint
from uuid import uuid4
from random import seed,sample


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')


def gpt3_completion(prompt, engine='text-davinci-002', temp=1.2, top_p=1.0, tokens=100, freq_pen=0.0, pres_pen=0.0, stop=['\n', 'asdasdf']):
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
            #text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def improve_outline(request, outline):
    prompt = open_file('prompt_improve_outline.txt').replace('<<REQUEST>>',request).replace('<<OUTLINE>>', outline)
    outline = '1. ' + gpt3_completion(prompt)
    return outline


def neural_recall(request, section):
    prompt = open_file('prompt_section_research.txt').replace('<<REQUEST>>',request).replace('<<SECTION>>',section)
    notes = gpt3_completion(prompt)
    return notes


def improve_prose(research, prose):
    prompt = open_file('prompt_improve_prose.txt').replace('<<RESEARCH>>',research).replace('<<PROSE>>', prose)
    prose = gpt3_completion(prompt)
    return prose


if __name__ == '__main__':
    # TODO choose difficult
    seed()
    seed_words = open_file('common_words.txt').splitlines()
    prompt = str(uuid4())
    for s in sample(seed_words, 10):
        prompt = prompt + '\nPick a random word: %s' % s
    prompt = prompt + '\nPick a random word:'
    #print(prompt)
    secret_word = gpt3_completion(prompt)
    #print('DEBUG:', secret_word)
    questions_remaining = 20
    print('I have picked my secret word! Ask your first question... (questions remaining: %s)' % questions_remaining)
    while True:
        # user asks a question        
        question = input('Question (%s): ' % questions_remaining)
        prompt = open_file('prompt_valid.txt').replace('<<QUESTION>>', question)
        is_valid = gpt3_completion(prompt, temp=0.0)
        #print(is_valid)
        if is_valid == 'False':
            print('That is not a valid question! Minus one point to Gryffindor!')
        elif is_valid == 'True':
            prompt = open_file('prompt_answer.txt').replace('<<SECRET>>', secret_word).replace('<<QUESTION>>', question)
            answer = gpt3_completion(prompt)
            if answer == 'Correct!':
                print('Congratulations! You won the game!')
                exit(0)
            print('Answer: %s' % answer)
        else:
            print('Sorry, the machine is confused, try again. No points deducted.')
            continue
        # check if user has lost
        questions_remaining = questions_remaining - 1
        if questions_remaining <= 0:
            print('You are out of guesses! The correct answer was: %s' % secret_word)
            exit(0)