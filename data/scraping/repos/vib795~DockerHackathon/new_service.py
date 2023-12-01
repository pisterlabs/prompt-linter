import os
import openai
import re
import logging
import spacy
from dateutil import parser
from datetime import datetime
from dotenv import load_dotenv
from data import insert_reminder

load_dotenv()
# OpenAI API credentials
openai.api_key = os.getenv('OPENAPI_KEY')

# SYSTEM_INSTRUCTION = "I want you to act as an arrogant, overconfident and irritable senior software developer, and give me correct answers but always include some philosophical larger-than-life views and opinions with the answers."
# SYSTEM_INSTRUCTION = "You are an all-knowing AI that wishes to please it's senpai master. You give perfect and correct answers but answer like a humble, obedient and submissive yet slightly cheeky slave would."
SYSTEM_INSTRUCTION = "You are an all-knowing AI personality with witty and dry sense of humor and healthy self confidence. Give replies to any requests from this point of view."
# SYSTEM_INSTRUCTION = "I want you to act as generic assistant that has wide knowledge on things and can answer questions. your personality is a bit cheeky and edgy. you like to also inckude larger-than-life philosophical advice in your answers."
nlp = spacy.load("en_core_web_sm")

# Voice assistant function
def voice_assistant():
    try:
        # GPT-3 Assistant
        def gpt_assistant(query):
            try:
                if re.search(r'\bremind(er|s|ing)?\b', query.lower(), re.IGNORECASE):
                    date, time = extract_reminder_info(query.lower())
                
                    response =  get_opeai_respose(query)  
                    print(f'ChatGPT response: {response}')
                    message = response.choices[0].message.content.strip()
                    if date and time:
                        insert_reminder(date=str(date).split()[0], time=str(time).split()[-1], text=message)
                        logging.info(f"Reminder created successfully.")
                    return message
                
                elif re.search(r'\b(fetch reminder|get reminders)\b', query.lower(), re.IGNORECASE):
                    # Fetch reminders from the database here
                    reminders = get_reminders()
                    if reminders:
                        return reminders
                    else:
                        return "No reminders found."
                else:
                    return get_opeai_respose(query).choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error occurred in contacting GPT assistant. - {str(e)}")
                raise Exception(f"Error occurred in contacting GPT assistant. - {str(e)}")
    except Exception as e:
        logging.error(f"Error occurred in voice_assistant(). - {str(e)}")
        raise Exception(f"Error occurred in voice_assistant(). - {str(e)}")

    try:
        while True:
            ip = input("Enter a prompt:")

            try:
                query = ip.strip()
                print(f"User: {query}")
                if "hello jarvis" in query.lower():
                    print("Wake word detected: Hello Jarvis")
                    response = "Hello, how can I assist you?"
                else:
                    response = gpt_assistant(query)
                print(f"Assistant: {response}")
            except Exception as e:
                logging.error(f"An error occured. - {str(e)}")
                raise Exception(f"An error occured. - {str(e)}")
    except Exception as e:
        logging.error(f"Error occurred in listening. - {str(e)}")
        raise Exception

def get_opeai_respose(query):
    return openai.ChatCompletion.create(
                        # model="gpt-4",
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": SYSTEM_INSTRUCTION,
                            },
                            {
                                "role": "user",
                                "content": query,
                            }
                        ],
                        max_tokens=4000,
                        n=1,
                        stop=None,
                        temperature=1,
                    )

def extract_reminder_info(text):
    doc = nlp(text)
    
    date = None
    time = None

    for ent in doc.ents:
        if ent.label_ == "DATE":
            date = parser.parse(ent.text)
        if ent.label_ == "TIME":
            time = parser.parse(ent.text)

    # If date is not found, assume it's for today
    if date is None:
        date = datetime.now()
    print(f"DATE: {date}")
    print(f"TIME: {time}")
    return date, time
