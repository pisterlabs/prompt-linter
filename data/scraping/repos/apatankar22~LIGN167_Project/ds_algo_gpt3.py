from urllib.parse import urlparse
import urllib.robotparser
import requests
import re
from bs4 import BeautifulSoup 
import string
import numpy as np

import os
import openai
from io import StringIO
import sys
from pyfiglet import Figlet

class PythonCodingSession:
    all_string_exercises_url = 'https://www.w3resource.com/python-exercises/string/'
    all_ds_algo_exercises_url = 'https://www.w3resource.com/python-exercises/data-structures-and-algorithms/'
        
    all_ds_algo_response = requests.get(all_ds_algo_exercises_url, timeout=5)
    all_ds_algo_content = BeautifulSoup(all_ds_algo_response.content, "html.parser")
    all_ds_algo_text = [p.text for p in all_ds_algo_content.find_all('p')]
    all_ds_algo_exercises = [s for s in all_ds_algo_text if s[0].isdigit()]
    all_ds_algo_exercises_clean = list(map(lambda s: s.replace('Go to the editor', '').replace('Click me to see the sample solution', ''), all_ds_algo_exercises))
    
    all_str_response = requests.get(all_string_exercises_url, timeout=5)
    all_str_content = BeautifulSoup(all_str_response.content, "html.parser")
    all_str_text = [p.text for p in all_str_content.find_all('p')]
    all_str_exercises = [s for s in all_str_text if s[0].isdigit()]
    all_str_exercises_clean = list(map(lambda s: s.replace('Go to the editor', '').replace('Click me to see the sample solution', ''), all_str_exercises))
    
    all_questions = [all_str_exercises_clean, all_ds_algo_exercises_clean]
    
    # PLEASE ENTER IN YOUR OWN OPEN AI API KEY
    openai.api_key = 'your-key-here'
    
    with open('python_prompt.txt') as f:
        python_prompt = f.read()
        
    with open('python_comparison_prompt.txt') as f:
        python_comparison_prompt = f.read()
        
        
    STDOUT = sys.stdout
    f = Figlet(font='slant')
    
    def __init__(self, username):
        new_session_message = "Welcome, {}, to your Python Practice Hub! Here you'll be able to practice your Python coding skills in string manipulation, data structures, and algorithms and get instant feedback and sample solutions."
        self.username = username
        self.total_questions_completed = 0
        self.current_topic = 0
        self.current_question = 0
        self.current_results = ''
        print(PythonCodingSession.f.renderText('The Python Practice Hub'))
        print(new_session_message.format(self.username))
        self.run_session(0)
        
        
    def run_session(self, flow):
        # exit session
        if flow == 4:
            print(PythonCodingSession.f.renderText('Thank you!'))
            print('We hope The Python Practice Hub helped you learn SQL or prepare for your next interview :)')
            return
        # repeat question
        elif flow == 1:
            user_text = self.get_user_input_solution(self.current_question)
            self.get_gpt3_solution_and_comparison(self.current_question, user_text)
            self.post_question()
        # new question new topic
        elif flow == 0 or flow == 3:
            topic = self.get_user_topic_choice()
            random_choice = self.get_python_exercise_info(topic)
        # new question same topic
        elif flow == 2:
            random_choice = self.get_python_exercise_info(self.current_topic)
        user_text = self.get_user_input_solution(random_choice)
        self.get_gpt3_solution_and_comparison(random_choice, user_text)
        self.post_question()
        
    def get_user_topic_choice(self):
        print()
        print('Choose a Topic.')
        print()
        topic_list_message = '1 for String Manipulation.\n2 for Data Structures and Algorithms.'
        topic_prompt = 'Enter in the corresponding number for the topic you want to study. For example, if you would like to practice String Manipulation, enter 1.\n' + topic_list_message + '\nYour choice: '
        
        user_input = -1
        while True:
            user_input = input(topic_prompt)
            if int(user_input) not in [1,2]:
                print('Invalid response. Try again.')
                continue
            else:
                break
        
        self.current_topic = int(user_input)
        return int(user_input)
    
    def get_python_exercise_info(self, user_choice):
        topic = user_choice - 1
        questions = PythonCodingSession.all_questions[topic]
        random_choice = np.random.choice(questions, replace = False)
        self.current_question = random_choice
        return random_choice
    
    def get_user_input_solution(self, random_choice):
        print()
        print('************************************')
        print()
        print('Exercise:')
        print(random_choice)
        print()
        print('Write your answer in a .py file. Then, enter in the filepath of your solution. Double check your filepath does not have any typos.')
        filepath = input('Enter in the .py filepath of your solution here (include file extension):')
        user_file = open(filepath, 'r')
        user_text = user_file.read()
        return user_text
    
    def get_gpt3_solution_and_comparison(self, random_choice, user_text):
        prompt = f'Prompt:\n{random_choice}\nAnswer:\n```'
        response = openai.Completion.create( model="text-davinci-002", prompt=PythonCodingSession.python_prompt+prompt, temperature=0, max_tokens=512, stop='```', )
        self.total_questions_completed += 1
        gpt3_solution = response['choices'][0]['text'].strip()
        print('Example solution:')
        print(gpt3_solution)
        print('************************************')
        comparison_prompt = f'Solution function:\n{gpt3_solution}\nInputted function:{user_text}\nFeedback:\n```'
        comparison_response = openai.Completion.create( model="text-davinci-002", prompt=PythonCodingSession.python_comparison_prompt+comparison_prompt, temperature=0, max_tokens=512, stop='```', )
        gpt3_feedback = comparison_response['choices'][0]['text'].strip()
        print('Feedback:')
        print(gpt3_feedback)
        
    def post_question(self):
        post_question_prompt = 'If you would like to try this question again, enter 1. If you would like to try'\
        ' a different question in this topic, enter 2. If you would like to try a new topic, enter 3. If you would like to'\
        ' exit your session, enter 4: '
        print('************************************')
        user_input = -1
        while True:
            user_input = input(post_question_prompt)
            if int(user_input) not in [1,2,3,4]:
                print('Invalid response. Try again.')
                continue
            else:
                break
        user_input = int(user_input)
        self.run_session(user_input)
