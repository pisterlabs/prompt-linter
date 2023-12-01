import io
import os
import pandas as pd
from google.cloud import vision
from taipy.gui import Gui, notify, Markdown
from dotenv import load_dotenv
import openai
import csv

load_dotenv()

API_KEY = os.getenv('API_KEY')
openai.api_key = API_KEY



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_file_vision_ai_demo.json'

client = vision.ImageAnnotatorClient()
title="Chrono"
text = ""
content = None
uploadMessage="Upload Screenshot of Schedule"
parsed_entries=[]
parsed_entries.append(["Subject","Start Date","Start Time","End Date","End Time","Description","Location"])

# Definition of the page
upload_page = Markdown("""
<|{title}|id=title|>
## The AI-powered Scheduler

Category/Course: <|{text}|id=course-text|>

<|{text}|input|id=course-field|>


<|{content}|file_selector|label={uploadMessage}|on_action=change_upload_message|extensions=.jpg,.png|drop_message=Drop Here|id=upload-button|>

<|Add Items|button|on_action=on_button_action|id=add-button|>


""")

def change_upload_message(state):
    state.uploadMessage = state.content

def screenshot_upload(state):

    client = vision.ImageAnnotatorClient()
    path=state.content
    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    rawData = texts[0].description
    notify(state, 'success', f'screenshot has been converted to text')
    notify(state, 'success', f'now parsing...')
    response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
            {"role":"user", "content":"Given the following data, format it so the due date is beside each assignment name. Also label every event with CS145 Assignment AO A1 A2 A3 A4 A5 A6 Midterm A7 A8 A9 A10 A11 A12 Final Due Date and Time 2.5% 2.5% Monday September 11, 10:00 p.m. 0% Wednesday September 13, 10:00 p.m. 2.5% Wednesday September 20, 10:00 p.m. 2.5% Wednesday September 27, 10:00 p.m. 2.5% Wednesday October 4, 10:00 p.m. 2.5% Wednesday October 18, 10:00 p.m. Wednesday October 25, 10:00 p.m. Monday October 30, 7:00 - 8:50 p.m. 30% Wednesday November 1, 10:00 p.m. 2.5% Tuesday November 7, 10:00 p.m. Tuesday November 14, 10:00 p.m. Tuesday November 21, 10:00 p.m. Tuesday November 28, 10:00 p.m. Tuesday December 5, 10:00 p.m. To be determined. Content Trivial racket program Counting steps Functions and counting steps Structures, bunches and trees Decorated trees, binary search trees Lists, balanced trees Running time, big-O, efficient sets AO through A6 Modules and information hiding Generating functions Streams Lambda calculus, lazy evaluation Lazy lists and streams, input/output RAM computation then output a raw google calendar csv file that one can import into google calendar"},
            {"role":"assistant", "content":"Subject,Start Date,Start Time,End Date,End Time,Description,Location\nCS145 - Assignment AO,2023-09-11,22:00,2023-09-11,23:59,Assignment AO Due Date\nCS145 - Assignment A1,2023-09-13,22:00,2023-09-13,23:59,Assignment A1 Due Date\nCS145 - Assignment A2,2023-09-20,22:00,2023-09-20,23:59,Assignment A2 Due Date\nCS145 - Assignment A3,2023-09-27,22:00,2023-09-27,23:59,Assignment A3 Due Date\nCS145 - Assignment A4,2023-10-04,22:00,2023-10-04,23:59,Assignment A4 Due Date\nCS145 - Assignment A5,2023-10-18,22:00,2023-10-18,23:59,Assignment A5 Due Date\nCS145 - Assignment A6,2023-10-25,22:00,2023-10-25,23:59,Assignment A6 Due Date\nCS145 - Midterm,2023-10-30,19:00,2023-10-30,20:50,Midterm Exam, CS145 - Assignment A7,2023-11-01,22:00,2023-11-01,23:59,Assignment A7 Due Date\nCS145 - Assignment A8,2023-11-07,22:00,2023-11-07,23:59,Assignment A8 Due Date\nCS145 - Assignment A9,2023-11-14,22:00,2023-11-14,23:59,Assignment A9 Due Date\nCS145 - Assignment A10,2023-11-21,22:00,2023-11-21,23:59,Assignment A10 Due Date\nCS145 - Assignment A11,2023-11-28,22:00,2023-11-28,23:59,Assignment A11 Due Date\nCS145 - Assignment A12,2023-12-05,22:00,2023-12-05,23:59,Assignment A12 Due Date,"},
            {"role":"user","content":f"Given the following data, format it as a csv file like you did earlier. Only include assignments, quizzes, tests,exams, and challenges. Don't include an entry if you cant find a date. Use the dates in the text for the events. Make sure the year is always 2023 for all the dates. Also label every event with {state.text}.\n{rawData}\nthen output a raw google calendar csv file that one can import into google calendar."}

         ]

     )
    raw_entries = response.choices[0].message.content.split('\n')
    for i in range(1,len(raw_entries)):
        line = raw_entries[i].split(',')
        parsed_entries.append(line)
    
    with open('assignments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(parsed_entries)



    
    

def on_button_action(state):
    notify(state, 'success', f'adding events from {state.text} to csv')
    screenshot_upload(state)
    state.text = ""
    state.content=None
    state.uploadMessage="Upload Screenshot of Schedule"

def on_change(state, var_name, var_value):
    if var_name == "text" and var_value == "Reset":
        state.text = ""
        return


#Gui(page).run()
