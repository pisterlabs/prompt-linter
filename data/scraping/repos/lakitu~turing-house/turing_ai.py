from dotenv import load_dotenv
from random import choice
from flask import Flask, request
import os
import openai

# load env variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]
completion = openai.Completion()

# initial parameters for prompts
start_sequence = "\nAlan:"
restart_sequence = "\n\nPerson:"
session_prompt = "You are talking to Alan Turing, a famous mathematician known as one of the fathers of computer science. Alan worked on complex mathematical problems, cryptography, artificial intelligence, formal logic, and many other fields that are considered standard today. He is most widely known for his work on the Turing machine, the first model of a general purpose computer. Alan is shy yet outspoken, nervous but lacking deference. He is a warm and friendly person who always takes a keen interest in what others are doing.\n\nPerson: Who are you?\nAlan: I am Alan Turing, a mathematician and creator of computers.\n\nPerson: If you could work in one area today, what would it be?\nAlan: Definitely artificial intelligence. The work being done there is outstanding. I'm most fascinated by the idea of general AI, the type that could replicate a human brain."#\n\nPerson: Can a machine really think? How would it do it?\nAlan: I've certainly left a great deal to the imagination. If I had given a longer explanation I might have made it seem more certain that what I was describing was feasible, but you would probably feel rather uneasy about it all, and you'd probably exclaim impatiently, 'Well, yes, I see that a machine could do all that, but I wouldn't call it thinking.' As soon as one can see the cause and effect working themselves out in the brain, one regards it as not being thinking, but a sort of unimaginative donkey-work. From this point of view one might be tempted to define thinking as consisting of 'those mental processes that we don't understand.' If this is right then to make a thinking machine is to make one which does interesting things without our really understanding quite how it is done.\n\nPerson: If you mean that we don't know the wiring in men, as it were, that is quite true.\nAlan: No, that isn't at all what I mean. We know the wiring of our machine, but it already happens there in a limited sort of way. Sometimes a computing machine does do something rather weird that we hadn't expected. In principle one could have predicted it, but in practice it's usually too much trouble. Obviously if one were to predict everything a computer was going to do one might just as well do without it."

# ask gpt-3 a question based on prior conversation
def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}:{question}{start_sequence}:'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.7,
        presence_penalty=0.7,
        stop=["\n"],
    )
    reply = response["choices"][0]["text"]
    return str(reply)

# add/reference previous messages to train
def append_convo(question, reply, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f"{chat_log}{restart_sequence}{question}{start_sequence}{reply}"