import csv
import time
import openai
import os
from summary import summarize
from findquotes import quoter
openai.api_key = os.environ.get('OPENAI_API_KEY')


def outliner(topic, source, instructions):

    summary = ''
    while(len(topic) + len(source) < 2048):

        summary = summarize(source, 'Make the summaries as brief as possible.')

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt='Create an outline for an essay about: ' + topic + '. Ensure to consider the following instructions: ' + instructions +
        '.\nUse the following information as resources to form your outline:' +
        summary + '\nOutline:\nI: Introduction',
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return(response['choices'][0]['text'])


def add_quotes(outline, source):
    quotes = quoter(
        outline, source, 'Make sure to only use quotes that are highly relevant to the topic.\nSelect no more than 1 quote total per section. Some sections may not be relevant to the prompt. In that case, output nothing at all.')
    # take the first half of quotes
    # take the first and last quarter of the quotes

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt='You are given the following outline:' + outline + ' You are also given the following list of quotes: ' +
        quotes + '\nRewrite the outline to include specific quotes from the list of quotes given above. Only use quotes which are completed.',
        temperature=0.7,
        max_tokens=1300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return(response['choices'][0]['text'])


# select which add ons to use from a  list.
# add ons are:
# Define key terms
# Long
# Short
# Use Quotes
# Provide Counterargument
# Provide Examples
# Provide References
# Provide Citations

def add_ons(topic, source):
    print('Outline Specifications')
    print('1. Argumentative Essay')
    print('3. Paragraph')
    print('4. Use Direct Quotes')
    print('5. Counter Argument')
    print('6. High Detail')
    instructions = []
    user_choice = input()

    if '1' in user_choice:
        instructions.append(
            'Ensure that the outline contains a labelled thesis statement with a strong and defendable argument.')
        print('argumentative essay')
    if '3' in user_choice:
        instructions.append(
            'The outline is for a single paragraph. Ensure that the outline is for a single paragraph.')
        print('Short')
    if '5' in user_choice:
        instructions.append(
            'The outline should include a counter argument and a rebuttal to the counter argument.')
        print('Ensure to summarize any counter arguments made in the paper.')
    if '6' in user_choice:
        instructions.append(
            'The outline must be highly detailed and include all of the main points of the paper.')

    txt = ''
    for i in instructions:
        txt += i + '\n'
    print('\n' + txt + '\n')

    initialOutline = outliner(topic, source, txt)
    print(initialOutline)
    if '4' in user_choice:
        print('Initial outline getting quotes added to it.')
        print(add_quotes(initialOutline, source))

    # make it so that they can add multiple add ons at once

# Use specific


def read_txt_file():
    #print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    user_choice = input()

    try:
        user_choice = int(user_choice)
        if user_choice > len(files):
            print('Please enter a valid choice.')
            read_txt_file()
        user_choice = files[user_choice-1]
    except:
        print('Please enter a valid choice.')
        if user_choice not in files:
            read_txt_file()
    with open(user_choice, encoding='utf8') as f:
        text = f.read()

    return text


def get_topic():
    topic = ''
    print('Please enter the topic you would like to outline.')
    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to outline.')
        topic = input()

    if user_choice == '2':
        print('Please enter the name of the file containing the topic you would like to outline.')
        topic = read_txt_file()

    return topic


def get_source():
    source = ''
    print('Please enter the source you would like to include in your outline.')

    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to outline.')
        source = input()

    if user_choice == '2':
        print('Please enter the name of the file you would like to outline.')
        source = read_txt_file()

    print('Add additional sources?')

    print('1. Input text')
    print('2. Pick a file')
    print('3. Leave')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to outline.')
        source += input()

    if user_choice == '2':
        print('Please enter the name of the file you would like to outline.')
        source += read_txt_file()
    if user_choice == '3':
        return source

    return source


def outline():
    print('Welcome to the AI text outliner. Would you like to input the text or pick a file?')
    source = ''
    topic = ''
    topic = get_topic()

    time.sleep(0.5)
    source = get_source()

    add_ons(topic, source)
