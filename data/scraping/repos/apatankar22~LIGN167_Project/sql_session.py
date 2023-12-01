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


class SQLSession:
    
    w3_sql_topic_urls = {'Retrieve data from tables': ['https://www.w3resource.com/sql-exercises/sql-retrieve-exercise-{}.php', range(1, 34)],
                         'Boolean and Relational Operators': ['https://www.w3resource.com/sql-exercises/sql-boolean-operator-exercise-{}.php', range(1, 13)],
                         'Wildcard and Special operators': ['https://www.w3resource.com/sql-exercises/sql-wildcard-special-operator-exercise-{}.php', range(1, 23)],
                         'Aggregate Functions': ['https://www.w3resource.com/sql-exercises/sql-aggregate-function-exercise-{}.php', range(1, 26)],
                         'Formatting query output': ['https://www.w3resource.com/sql-exercises/sql-formatting-output-exercise-{}.php', range(1, 11)],
                         'SQL JOINS': ['https://www.w3resource.com/sql-exercises/sql-joins-exercise-{}.php', range(1, 30)]
                        }
    w3_sql_topics = list(w3_sql_topic_urls.keys())
    
    # PLEASE ENTER IN YOUR OWN OPEN AI API KEY
    openai.api_key = 'your-key-here'
    with open('sql_prompt.txt') as f:
        PROMPT = f.read()
    STDOUT = sys.stdout
    f = Figlet(font='slant')
    
    def __init__(self, username, new_user = True):
        new_session_message = "Welcome, {}, to your SQL Practice Hub! Here you'll be able to practice writing SQL queries in 6 different topics and get instant feedback."
        self.username = username
        self.question_tracker = {topic:set() for topic in SQLSession.w3_sql_topics}
        self.total_questions_completed = 0
        self.current_topic = 0
        self.current_q_num = 0
        self.current_results = ''
        self.is_new_user = new_user
        print(SQLSession.f.renderText('The SQL Practice Hub'))
        print(new_session_message.format(self.username))
        self.run_session(0)
    
    def run_session(self, flow):
        # exit session
        if flow == 4:
            print(SQLSession.f.renderText('Thank you!'))
            print('We hope The SQL Practice Hub helped you learn SQL or prepare for your next interview :)')
            return
        # repeat question
        elif flow == 1:
            # user solution
            user_input_query = self.get_user_input_query(self.current_results)
            # gpt3 comparison
            self.openai_api_call(self.current_topic, self.current_q_num, user_input_query, self.current_results)
            # ask for session input
            self.post_question(self.current_results)
        # new question
        elif flow == 0 or flow == 3:
            # get topic choice
            topic = self.get_user_topic_choice()
            # scrape w3 results
            q_num, exercise_url, topic = self.sql_exercises(topic)
        # new topic
        elif flow == 2:
            q_num, exercise_url, topic = self.sql_exercises(self.current_topic)
        results = self.get_sql_exercise_info(exercise_url)
        user_input_query = self.get_user_input_query(results)
        self.openai_api_call(self.current_topic, q_num, user_input_query, results)
        self.post_question()
    
    def get_user_topic_choice(self):
        print()
        print('Choose a Topic.')
        print()
        topic_list_message = '\n'.join(['{} for {}'.format(i+1, SQLSession.w3_sql_topics[i]) for i in range(len(SQLSession.w3_sql_topics))])
        topic_prompt = 'Enter in the corresponding number for the topic you want to study. For example, if you would like to practice SQL JOINS, enter 6.\n' + topic_list_message + '\nYour choice: '
        
        user_input = -1
        while True:
            user_input = input(topic_prompt)
            if int(user_input) not in [1,2,3,4,5,6]:
                print('Invalid response. Try again.')
                continue
            else:
                break
        
        self.current_topic = int(user_input)
        return int(user_input)
    
    def sql_exercises(self, user_choice):
        topic = SQLSession.w3_sql_topics[user_choice - 1]
        url, q_nums = SQLSession.w3_sql_topic_urls[topic]
        completed = self.question_tracker[topic]
        q_nums_not_done = set(q_nums) - completed
        if len(q_nums_not_done) == 0:
            q_nums_not_done = q_nums
        random_choice = np.random.choice(list(q_nums_not_done))
        self.current_q_num = random_choice
        self.question_tracker[topic].add(random_choice)
        return [random_choice, url.format(random_choice), user_choice]

    def get_sql_exercise_info(self, exercise_url):
        req_response = requests.get(exercise_url, timeout=5)
        html_content = BeautifulSoup(req_response.content, "html.parser")
        prompt = html_content.find_all('p')[0].text
        solution = html_content.find_all('code')[0].text
        table_strs = {}
        table_srcs = html_content.find_all("iframe", {"class": "span12"})
        for lst in table_srcs:
            table = requests.get("https://www.w3resource.com/sql-exercises/{}".format(lst['src']))
            table_name = lst['src'].strip('.php')
            table_str = BeautifulSoup(table.content, "html.parser").find('pre').text
            table_strs[table_name] = table_str
        results = {'prompt':prompt, 'solution':solution, 'tables':table_strs}
        self.current_results = results
        return results
    
    def get_user_input_query(self, results):
        print()
        print('************************************')
        print()
        print('Exercise:')
        print(results['prompt'])
        print()
        print('See sample table(s) below:')
        tables = results['tables']
        for table in tables:
            print('Table name: ' + table)
            print(tables[table])
        print('Please enter in your solution as one line (do not hit the Enter or Return key, and do not enter new lines)')
        user_input = input('Enter your solution here:')
        return user_input

    def openai_api_call(self, topic, q_num, user_input, results):
        sol = results['solution']
        prompt = f'Solution SQL Query:\n{sol}\nInput SQL Query:{user_input}\nAnswer:\n```'
        response = openai.Completion.create( model="text-davinci-002", prompt=SQLSession.PROMPT+prompt, temperature=0, max_tokens=512, stop='```', )
        self.total_questions_completed += 1
        gpt3_feedback = response['choices'][0]['text'].strip()
        topic_str = SQLSession.w3_sql_topics[topic - 1]
        print('Topic: {}. Question #{}'.format(topic_str, q_num))
        print('************************************')
        print('Example solution:')
        print(sol)
        print('************************************')
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
        
        
class SQLInterview(SQLSession):
    def __init__(self, username, num_questions = 5):
        new_session_message = "Welcome, {}, to your SQL Interview Simulator. Here, you'll be asked 5 random SQL questions to assess your skills. At the end, you will be provided with feedback on each question. "
        self.username = username
        self.num_questions = num_questions
        self.question_tracker = {topic:set() for topic in SQLSession.w3_sql_topics}
        self.total_questions_completed = 0
        self.current_topic = 0
        print(new_session_message.format(self.username))
        self.run_interview()
    
    def run_interview(self):
        exercises = self.get_random_questions()
        all_results = [self.get_sql_exercise_info(e[1]) for e in exercises]
        user_inputs = []
        for result in all_results:
            user_inputs.append(self.get_user_input_query(result))
        for i in range(len(all_results)):
            print('Question {} Results:'.format(i + 1))
            self.openai_api_call(exercises[i][-1], exercises[i][0], user_inputs[i], all_results[i])
        print(SQLSession.f.renderText('Thank you!'))
        print('We hope The SQL Interview Simulator helped you get one step closer to getting your dream job! Good luck :)')
    
    def get_random_questions(self):
        if self.num_questions <= 6: 
            topics = np.random.choice(range(1, 7), self.num_questions, replace=False)
            exercises = [self.sql_exercises(topic) for topic in topics]
        else:
            exercises = []
            for i in range(1, 7):
                for j in range(self.num_questions // 6):
                    exercises.append(SQLSession.sql_exercises(i))
            other_topics = np.random.choice(range(1, 7), self.num_questions % 6, replace=False)
            exercises += [self.sql_exercises(topic) for topic in other_topics]
        return exercises
    
