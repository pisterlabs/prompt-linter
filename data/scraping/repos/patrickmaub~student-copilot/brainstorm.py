import csv
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def brainstormer(user_prompt):

    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
       
    return(response['choices'][0]['text'])


def add_ons(user_prompt):
    print('What would you like to brainstorm about?')
    print('1. Thesis Statement')
    print('2. Arguments to make')
    print('3. Find Authors')
    instructions = []
    user_choice = input()

    if '1' in user_choice:
        print('Thesis Statement\n')
        print(brainstormer('Write a specific thesis statement for a 5 paragraph essay on the following prompt:\n' + user_prompt))
    if '2' in user_choice:
        print('Arguments to make')
        print(brainstormer('Write a list of specific ideas of arguments to make for a 5 paragraph essay on the following prompt:\n' + user_prompt))

    if '3' in user_choice:
        print('Scholars on the topic')
        print(brainstormer('Write a list of scholars on the topic of the following prompt:\n' + user_prompt))
    # make it so that they can add multiple add ons at once


def read_txt_file():
    print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    print('Please enter the name of the file you would like to brainstorm about.')
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


def brainstorm():
    print('Welcome to the AI brainstormer. Would you like to input the text or pick a file?')
    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to brainstorm about.')
        user_prompt = input()
        add_ons(user_prompt)
    elif user_choice == '2':
        user_prompt = read_txt_file()
        add_ons(user_prompt)

    else:
        print('Please enter a valid choice.')
        brainstorm()
