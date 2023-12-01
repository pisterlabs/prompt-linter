import pyperclip
import time
import pyautogui
# from googlesearch import search
import requests
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from prompts import SEARCH_TABLE_OF_CONTENTS, EXTRACT_QUESTION, REAL_QUESTION
from pdfminer.high_level import extract_text
import re
import pdfreader
import os
from dotenv import load_dotenv
import sys

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
anthropic = Anthropic(api_key=CLAUDE_API_KEY)

def find_textbook_link(textbook_name):
    query = textbook_name + " PDF"
    links = list(search(query, num=1, stop=1, pause=2))
    
    if links:
        return links[0]
    else:
        return None
    
def find_textbook_link_googlev2(textbook_name):
    query = "textbook " + str(textbook_name) + " filetype:pdf -site:amazon"
    pyautogui.keyDown('command')
    pyautogui.press('space')
    pyautogui.keyUp('command')
    pyautogui.keyUp('Fn') # so we don't press the emoji bar
    pyautogui.typewrite("https://www.google.com/")
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.keyUp('Fn') # so we don't press the emoji bar
    pyautogui.typewrite(query)
    pyautogui.press('enter')
    time.sleep(2)

    pyautogui.press('tab', presses=21) # get to the first link
    pyautogui.press('enter') # go to the link

    pyautogui.keyDown('command')
    pyautogui.press('l')
    pyautogui.press('c')
    pyautogui.keyUp('command')
    time.sleep(1)

    return pyperclip.paste()

def generate_textbook_pdf(link):
    try:
        response = requests.get(link)
        response.raise_for_status()

        with open("downloaded_textbook.pdf", "wb") as f:
            f.write(response.content)
    except requests.exceptions.HTTPError as err:
        print(link)


def extract_pages(pdf_path, limit_pages=10):
    text = extract_text(pdf_path, page_numbers=list(range(limit_pages)))
    return text

def search_table_of_contents(excerpt, section):
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=f"{HUMAN_PROMPT} {SEARCH_TABLE_OF_CONTENTS} <excerpt>{excerpt}<excerpt> <section>{section}<section> {AI_PROMPT}",
    )
    return completion.completion

def extract_section_pages(pdf_path, start_page, limit_pages=30):
    text = extract_text(pdf_path, page_numbers=list(range(start_page, start_page + limit_pages)))
    return text

def extract_question(excerpt, question):
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=f"{HUMAN_PROMPT} {EXTRACT_QUESTION} <excerpt>{excerpt}<excerpt> <question>{question}<question> {AI_PROMPT}",
    )
    return completion.completion

# textbook_name  = "The Elements of Statistical Learning: Data Mining, Inference, and Prediction by Trevor Hastie, Robert Tibshirani, and Jerome Friedman."
# textbook_name = "John A. Rice, Third Edition."
# chapter = "8"
# section = "10"
# question = "21"

def is_real_question(excerpt, question):
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=f"{HUMAN_PROMPT} {REAL_QUESTION} <excerpt>{excerpt}<excerpt> <question>{question}<question> {AI_PROMPT}",
    )
    return completion.completion

def find_questions():
    textbook_name = ""
    chapter = ""
    section = ""
    question = ""

    pdf_questions = [{'sources': ['John A. Rice, Third Edition. | Problem 8.10.21']}, 
                    {'sources': ['John A. Rice, Third Edition. | Problem 8.10.45', 'Rproject3.script4.Chromatin.r']}, 
                    {'sources': ['John A. Rice, Third Edition. | Problem 8.10.51']}, 
                    {'sources': ['John A. Rice, Third Edition. | Problem 8.10.58', 'Rproject3.script1.multinomial.simulation.r']}]
    # pdf_questions = [{'sources': ['John A. Rice, Third Edition. | Problem 8.10.58', 'Rproject3.script1.multinomial.simulation.r']}]

    source = pdf_questions[0]['sources'][0]
    pattern = r'^(.*?)\s*\|'
    match = re.search(pattern, source)
    textbook_name = match.group(1)
    source = pdf_questions[0]['sources'][0]
    pattern = r'^(.*?)\s*\|'
    match = re.search(pattern, source)
    textbook_name = match.group(1)

    print(textbook_name)
    link = find_textbook_link_googlev2(textbook_name)
    print(link)
    # generate_textbook_pdf(link)

    questions_list = []

    # question_number = ""
    # for questions in pdf_questions:
    #     source = questions['sources'][0]
    # question_number = ""
    # for questions in pdf_questions:
    #     source = questions['sources'][0]

    #     pattern = r'Problem (\d+\.\d+\.\d+)'
    #     match = re.search(pattern, source)
    #     question_number = match.group(1)
    #     pattern = r'Problem (\d+\.\d+\.\d+)'
    #     match = re.search(pattern, source)
    #     question_number = match.group(1)

    #     chapter, section, question = question_number.split(".")
    #     print("-------------------")
    #     print("-------------------")
    #     print("-------------------")
    #     print(chapter, section, question)
    #     chapter, section, question = question_number.split(".")


    #     pdf_path = "downloaded_textbook.pdf"
    #     section_title = f"{chapter}.{section}"
    #     excerpt = extract_pages(pdf_path)
    #     pattern = fr"({section_title}.{{0,500}})"
    #     match = re.search(pattern, excerpt, re.DOTALL)
    #     excerpt = match.group(1)

    #     page_text = search_table_of_contents(excerpt, section_title)
    #     # print(page_text)
    #     page_text = search_table_of_contents(excerpt, section_title)
    #     # print(page_text)


    #     match = re.search(r"(\b\d+\b)(?!.*\b\d+\b)", page_text)
    #     try:
    #         page = int(match.group(1))
    #     except AttributeError:
    #         print("ERROR: Page not found.")
    #         page = 312
    #     match = re.search(r"(\b\d+\b)(?!.*\b\d+\b)", page_text)
    #     try:
    #         page = int(match.group(1))
    #     except AttributeError:
    #         print("ERROR: Page not found.")
    #         page = 312


    #     output = extract_section_pages(pdf_path, page)
    #     pattern = fr"({question}(?:\.|\)).{{0,2000}})"
    #     match = re.search(pattern, output, re.DOTALL)


    #     try:
    #         output = match.group(1)
    #         question_content = extract_question(output, question)
    #         questions_list.append(question_content)
    #         print(question_content)
    #     except AttributeError:
    #         print("ERROR: Question not found.")
    
    return questions_list

print("-------------------")
print("-------------------")
print("-------------------")
my_list = find_questions()
print(my_list)

# def main():
#     print(find_textbook_link_googlev2("Mathematical Statistics and Data Analysis John A. Rice"))
# main()