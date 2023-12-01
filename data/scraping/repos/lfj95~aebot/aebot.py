# -*- coding: utf-8 -*-

# pip install openai

#!/usr/bin/env python3
import os
import openai
import json
import sys    
import sqlite3 

openai.api_key = os.getenv("OPENAI_API_KEY")
# The recommended approach is to set the API_Key in an environmental variable. If you don't want to set that up, you can uncomment this line and add your key directly. 
# openai.api_key = ""

def initDB():
  global cache 
  cache = sqlite3.connect(home + "/.aebot_cache")
  cache.execute("""
                   CREATE TABLE IF NOT EXISTS questions 
                   (id INTEGER PRIMARY KEY,
                   question TEXT,
                   answer TEXT,
                   count INTEGER DEFAULT 1)""")
def closeDB():
    cache.commit()
    cache.close()

def checkQ(question_text):
    sql = "SELECT id,answer,count FROM questions WHERE question =" + question_text
    answer = cache.execute("SELECT id,answer,count FROM questions WHERE question = ?", (question_text,))
    answer = answer.fetchone()
    if (answer):
        response = answer[1]
        newcount = int(answer[2]) + 1
        counter = cache.execute(" UPDATE questions SET count = ? WHERE id = ?", (newcount,answer[0]))
        return(response)
    else:
        return(False)

def fetchQ():
    question = ""
    for a in range(1,len(sys.argv)):
        question = question + " " + sys.argv[a]
        #sys.argv() is an array for command line arguments in Python.Command line arguments are those values that are passed during calling of program along with the calling statement. Thus, the first element of the array sys.argv() is the name of the program itself. The values are also retrieved like Python array.
    question = question.strip() #The strip() method returns a copy of the string by removing both the leading and the trailing characters (based on the string argument passed).
    return question

def parseOptions(question):
    global question_mode    # modes are normal, shortcut and general
    question_mode = "normal"
    if ("-h" in question) or (question == " "):  # Return basic help info    
        print("AEbot is a simple utility powered by GPT3")
        print("""
        Example usage:
        aebot What is the effect of cut cucumber?
        aebot What is the effect of put dough in the oven?
        aebot -i "What is the effect of cut cucumber?"      (runs in insturction series models)
        """)
        exit()

    if ("-i" in question):      # General question, not command prompt specific 
        question_mode = "instruction"
        question = question.replace("-i ","") 

    return(question)

def insertQ(question_text,answer_text):
    answer = cache.execute("DELETE FROM questions WHERE question = ?",(question_text,))
    answer = cache.execute("INSERT INTO questions (question,answer) VALUES (?,?)", (question_text,answer_text))
# 该例程执行一个 SQL 语句。该 SQL 语句可以被参数化（即使用占位符代替 SQL 文本）。sqlite3 模块支持两种类型的占位符：问号和命名占位符（命名样式）。
#例如：cursor.execute("insert into people values (?, ?)", (who, age))

global question
question = ""
question = fetchQ()
question = parseOptions(question)

# If we change our training/prompts, just delete the cache and it'll 
# be recreated on future runs. 
from os.path import expanduser
home = expanduser("~") #return the argument with an initial component of ~ or ~user replaced by that user’s home directory.
initDB()


#check if there's an aswer in our cache, then execute a GPT3 request as needed
cache_answer = checkQ(question)
if not(cache_answer):
    if (question_mode == "normal"):
        prompt="I am an action effect prediction bot.\n\nQ: What is the effect of hold a cup upside down?\nA: Spilling the liquid inside.\n\nQ: What is the effect of peel carrots?\nA: The peels will be separated from carrots.\n\nQ: What is the effect of crack an egg?\nA: The entire contents of the egg will be in the bowl, with the yolk unbroken, and that the two halves of the shell are held in the cook’s fingers.\n\nQ: What is the effect of heat water?\nA: The water will begin to boil, which is evinced by bubbles forming and rising to the surface.\n\nQ: What is the effect of shake can?\nA: The liquid inside will be shaken up and will form into a foamy head.\n\nQ: What is the effect of cut cucumber?\nA: The cucumber will be cut into two pieces.\n"
        temp_question = question
        if not("?" in question):
            temp_question = question + "?"  # GPT produces better results if there's a question mark.
                                            # using a temp variable so the ? doesn't get cached
        prompt = prompt + "Q: " + temp_question + "\n"
        start_sequence = "\nA:"
        restart_sequence = "\n\nQ: "
        response = openai.Completion.create(
          engine="davinci",
          prompt=prompt,  
          temperature=0.45,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=["\n"]
        )

    if not(cache_answer) and (question_mode == "instruction"):
        response = openai.Completion.create(
          engine="davinci-instruct-beta",
          prompt="Explain the action effect of\n",
          temperature=0.49,
          max_tokens=64,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )

    jsonToPython = json.loads(response.last_response.body)
    result = jsonToPython['choices'][0]['text'].strip("A: ")
    print("This is a new action effect prediction")
    print(result)
    insertQ(question, result)

else:
    result = cache_answer
    print("This prediction comes from cache_answer")
    print(cache_answer)
closeDB()
