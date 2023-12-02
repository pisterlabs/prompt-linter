#to acess environment variables
import os

#access openai methods
import openai

#load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def explanation(topic):
    #so long as user wants to ask questions, run
    while True:
        response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role" : "user", "content" : f"Explain to me, in a broad scope, what {topic} is about, using only 100 words."}
        ])
        print("\n", "ChatGPT: " , response.choices[0].message.content)

        #if user wants more, detailed information
        question = input("Would you like more information about this topic? Y/N: ")
        if question == "y" or question == "Y":
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role" : "user", "content" : f"Explain to me, in detail, what {topic} is about."}
                    ])
            print("\n" , "ChatGPT: " , response.choices[0].message.content)

        #ask if the user wants to learn more about another topic
        prompt = input("Is there another topic which you would like to learn about? Y/N: ")

        #if yes, continue along and change the topic variable to the new topic. Else, break the loop 
        if prompt == "Y" or prompt == 'y':
            newTopic = input("What topic would you like to learn more about now?: ")
            topic = newTopic
        else:
            break
    
    print("Have a nice day!")
    return 0