from twilio.rest import Client 
from metaphor_python import Metaphor 
import openai 
import googletrans 
from datetime import datetime, timezone
from bs4 import BeautifulSoup 
import random

translator = googletrans.Translator() 

OPENAI_KEY = ""
METAPHOR_TOKEN = ""
SID = ""
AUTH_TOKEN = ""
NUMBER = ''

openai.api_key = OPENAI_KEY 
metaphor = Metaphor(METAPHOR_TOKEN) 
account_sid = SID 
auth_token = AUTH_TOKEN 

client = Client(account_sid, auth_token) 

def send_message(message_body, message_to, message_from):
    # To send messages to one user 
    # print(message_body)
    message = client.messages.create(
        body = message_body, 
        from_ = message_from,
        to = message_to 
    )
    # print(message_to)
    return message.sid 

def get_all_numbers():
    unique_numbers = set()
    # Get all messages
    messages = client.messages.list()

    for message in messages:
        # Add the 'to' and 'from' numbers to our set (this ensures uniqueness)
        unique_numbers.add(message.to)
        unique_numbers.add(message.from_)

    return list(unique_numbers)
def count_number_of_messages_from_number(phone_number): 
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch messages sent after the start of today
    messages = client.messages.list(date_sent_after=today_start)

    count = 0
    for message in messages:
        if message.from_ == phone_number:
            count += 1

    return count
    


def user_sent_message_today(phone_number):
    # Get the start of today in UTC
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Fetch messages sent after the start of today
    messages = client.messages.list(date_sent_after=today_start)

    for message in messages:
        # Check if the 'from' number matches our phone_number
        if message.from_ == phone_number:
            return True

    return False

def get_all_messages_for_number(message_sid, message_from, message_to):
    # This will be useful in getting all the answers the user has given b4 
    message = client.messages.list(
        from_= message_from,
        to = message_to,
        limit = 100
    )
    all_messages = []
    for m in message.messages: 
        all_messages.append(m) 
    
    return all_messages

def bulk_send(list_of_numbers, daily_challenge): 
    for number in list_of_numbers: 
        try:
            send_message(daily_challenge, number, NUMBER) 
        except: 
            return f"ERROR sending a message to {number}" 

def create_questions(prompts = ""):
    # We get a list of ranks by calling metaphor to classify which answer was the best 
    USER_QUESTION = "Easy to read and simple Articles on Financial Literacy. The information should be very easy to understand. You should always assume the user is reading the information from a smartphone"  
    SYSTEM_MESSAGE = "You are a helpful assistant that generates search queiries based on user questions. Generate a search query that pertains to common banking terms and banking trivia. Only generate one search query."

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_QUESTION},
        ],
    )
    query = completion.choices[0].message.content
    search_response = metaphor.search(
        query, use_autoprompt=True
    )
    
    # Chose a random url in search response and ask a question based on that
    
    content = search_response.get_contents()
    random.shuffle(list(content.contents))
    first_five_results = content.contents[:5]
    # print()
    for i in range(len(first_five_results)):
        first_five_results[i] = BeautifulSoup(first_five_results[i].extract).get_text()
    
    # print(cont)
    SYSTEM_MESSAGE = "You are a helpful assistant that transform webpages results into questions, then the questions into small financial literacy fun quizes. The questions should be easy to answer to and have ONE clear and obvious answer, and they should be understood by a person who is around 5 years of age; anyone around 10 should be able to answer them, and always assume the user cannot read what you are reading. And you Do not provide any options for the quiz, just ask a question. If you feel like questions are too hard, find something easier to ask about the context. Do not ask questions about the title of the page, names of people, visual elements in the site or HTML metadata. Ask only one question"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join(first_five_results)},
        ],
    )    
    # print(first_five_results)
    question = completion.choices[0].message.content

     
    SYSTEM_MESSAGE = f"Given the question I will provide later, provide one true answer to the question. The answer should be very short. Question:" 

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ],
    )   
    true_answer = completion.choices[0].message.content
    SYSTEM_MESSAGE = f"You have the true answer to a question. Generate three false answers separated by a semicolon. Please separate false answers by a semicolon. Each of those false answers should be very short. Here is the true answer {true_answer}, and here is the question:" 

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ],
    )
    # question_rw = translator.translate(question, src = 'en', dest = 'sw') 
    # print(question, "\n\n\n") 

    # print(true_answer, "\n\n\n")

    mapped_variables = {
        "question": question,
        "true_answer": true_answer,
        "completion": set((completion.choices[0].message.content).split(';'))
    }
    
    
    return mapped_variables 
    # return completion.choices[0].message.content