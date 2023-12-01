import csv
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def summarize(user_prompt, instructions):

    summaries = []

    i = 0
    b = 8000
    while i < len(user_prompt):

        summaries.append('The following is a powerpoint for a university lecture. Imagine you are a student and need to prepare for an exam on this lecture material. Make a study guide off the lecture content while accounting for the following instructions:\n' +
                         instructions + '\n\Lecture content:\n')
        summaries[-1] += (user_prompt[i:b])
        i += 8000
        b += 8000

    big_summary = ''
    for i in range(len(summaries)):

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=summaries[i],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        big_summary += response['choices'][0]['text']
        print(response['choices'][0]['text'])

    return big_summary


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

def add_ons(user_prompt):
    print('Would you like to add any of the following to your summary?')
    print('1. Define key terms')
    print('2. Paragraph')
    print('3. Bullet Point')
    print('4. Use Direct Quotes')
    print('5. Descriptive')
    print('6. Simple')
    instructions = []
    user_choice = input()

    if '1' in user_choice:
        instructions.append(
            'The summary should include definitions for any key terms in the paper.')
        print('Define key terms')
    if '2' in user_choice:
        instructions.append(
            'The summary should be written in paragraph formats.')
        print('Long')
    if '3' in user_choice:
        instructions.append(
            'Ensure that the summary is written in bullet points.')
        print('Short')
    if '4' in user_choice:
        instructions.append(
            'Make sure to use specific, direct quotes from the paper in the summary.')
        print('Use Quotes')
    if '5' in user_choice:
        instructions.append(
            'Ensure to make the summary as descriptive as possible.')
        print('Ensure to summarize any counter arguments made in the paper.')
    if '6' in user_choice:
        instructions.append(
            'Ensure that the summary is as simple and short as possible.')
        print('Bullet Points')

    txt = ''
    for i in instructions:
        txt += i + '\n'
    print('\n' + txt + '\n')
    print(summarize(user_prompt, txt))
    # make it so that they can add multiple add ons at once


def read_txt_file():
    print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    print('Please enter the name of the file you would like to summarize.')
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


def summary():
    print('Welcome to the AI text summarizer. Would you like to input the text or pick a file?')
    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to summarize.')
        user_prompt = input()
        add_ons(user_prompt)
    elif user_choice == '2':
        user_prompt = read_txt_file()
        add_ons(user_prompt)

    else:
        print('Please enter a valid choice.')
        summary()



