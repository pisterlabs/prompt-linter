from webbrowser import get
from dotenv import load_dotenv
import openai
import os
import argparse

load_dotenv()

def main():
    print ("Klara is starting up...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, help="Ask Klara a question")
    arguments = parser.parse_args()
    user_input = arguments.input
    print ("You're asking Klara: " + user_input[0])
    asking_klara(user_input[0])
    answer = asking_klara(user_input)
    print ("Klara's answer: " + answer)

    print(f"You are getting Klara to help you study {user_input[1]}")
    getting_klara_to_help_me_study(user_input[1])
    print("Klara thinks you need to focus on: " + getting_klara_to_help_me_study(user_input[1]))

    coffee = coffee_chatbot(user_input[2])
    print("coffee chatbot: " + coffee)

def asking_klara(question: str) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"My name is Klara and I'm an observant AF who thrives off the sun's nutrience, ask me a question and I will try to answer {question}"

    response = openai.Completion.create(engine='text-davinci-002', prompt=prompt, max_tokens=100)

    # extract the answer from the response and remove whitespaces
    reply:str = response.choices[0].text
    reply = reply.strip()

    return reply 

def getting_klara_to_help_me_study(topic: str) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"What are the 5 key points I should know when studying {topic}?"

    response = openai.Completion.create(engine='text-davinci-002', prompt=prompt, max_tokens=150)

    # extract the answer from the response and remove whitespaces
    reply:str = response.choices[0].text
    reply = reply.strip()

    return reply

def coffee_chatbot(topic: str) -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    prompt = f"Cofee chatbot here to help you order coffee{topic}?"

    response = openai.Completion.create(engine='text-davinci-002', prompt=prompt, max_tokens=150)

    # extract the answer from the response and remove whitespaces
    reply:str = response.choices[0].text
    reply = reply.strip()

    return reply


if __name__ == "__main__":
    main()