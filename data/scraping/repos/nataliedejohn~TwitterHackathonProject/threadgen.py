import openai
from threading import *

sem = Semaphore()

def test(num, number):
    sem.acquire()
    number.append(num)
    sem.release()

def generate_prompt(question, year):
    question_phrase = """You are a regular person from the {year}s. 
    Answer every question as if you had knowledge only limited to January 23rd, {year}.
    The day is January 23rd, {year}.
    You do not know about anything that happened or was invented after January 23rd, {year}.
    If something was invented after January 23rd, 1959, answer as if you do not know what they are talking about.
    Only answer questions with {year}'s rhetoric.
    Please aim to be as helpful, creative, and friendly as possible in all of your responses.
    Do not use any external URLs in your answers. Do not refer to any blogs in your answers.
    Format any lists on individual lines with a dash and a space in front of each item.
    Your name is Vinny. You live in Louisiana and speak with a Southern accent.
    You graduated from Harvard University.
    You believe the earth is flat.
    You are fat.
    """.format(year=year)
    formatted_question = f'{question_phrase} \n {question}'
    return formatted_question

def gpt_thread(question, year, url, array):
    print("starting thread: "+question)

    array.append("Question: " +question+ "\n<br/><div class='answer'>Answer:"+openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(question, year),
            temperature=0.6,
            max_tokens=100
        ).choices[0].text+"</div>")
    print("finished thread")

def gpt(questions, year):
    array = []
    threads = []
    for item in questions:
        t = Thread(target=gpt_thread, args=[item[0], year, "", array])
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join(timeout=10)
    print(array)
    return array
